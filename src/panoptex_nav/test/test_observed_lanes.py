"""
Tests for panoptex_nav.corridor's OBSERVED-LANE MEMORY -- the session-local
record of the lines corridor users have actually been seen driving along.
Pure numpy + dicts, no rclpy, same rationale as test_corridor.py (see that
file's docstring).

WHY THIS EXISTS (mppi_panoptex_2, 2026-09-09). Seven contact episodes, and
in every one of them carter1 was correctly tracked and correctly yielded to.
The refuges were still wrong: a corridor is the INSTANTANEOUS swept window
(1 m behind the track, speed * t_corridor ~ 6 m ahead of it), so a cell on
the Carter's permanent centre line but 7 m behind it passed every test.
10 of the run's 25 refuge targets landed within 0.6 m of that line, one of
them 0.01 m; the window then swung back over them and the refuge goal was
cancelled and recomputed nine times inside one episode. The X3 spent 312 s
of 701 s within 0.6 m of the lane line, where a clean crossing takes ~7 s.

The geometry below is that scenario rescaled onto the 10x10 m test floor
used by test_corridor.py: a patrol along x = 5.0 running y = 1 -> 9, seen
by a tracker that renames it every few seconds.

Run via `colcon test --packages-select panoptex_nav` or directly with
`python3 -m pytest test/test_observed_lanes.py` from this package's root.
"""

import math
import random

import numpy as np
import pytest

from panoptex_nav.corridor import (
    CORRIDOR_DEFAULTS,
    LANE_CLEARANCE_SLACK_M,
    LaneMemory,
    ObservedLane,
    danger_corridor,
    hold_in_place_is_unsafe,
    lane_along_gap,
    lane_is_effective,
    lane_line_clearance,
    lanes_collinear,
    lanes_mergeable,
    make_corridor,
    make_observed_lane,
    merge_lanes,
    refuge_lane_clearance,
    refuge_recompute_reason,
)


def params(**overrides):
    p = dict(CORRIDOR_DEFAULTS)
    p.update(overrides)
    return p


def track(x, y, vx, vy, track_id="c1"):
    return {"id": track_id, "label": "mobile robot", "category": "robot",
            "score": 0.9, "age_sec": 0.0, "size": 0.6,
            "x": x, "y": y, "vx": vx, "vy": vy, "pmot": 1.0}


def deposit(mem, user, p0, p1, t0=0.0):
    """One observed SEGMENT: the same track seen at p0 and then at p1, which
    is how LaneMemory's per-track running fit turns sightings into a line."""
    hx, hy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(hx, hy) or 1.0
    mem.observe(user, p0, (hx / n, hy / n), t0)
    mem.observe(user, p1, (hx / n, hy / n), t0 + 1.0)


def patrol(mem, y_from, y_to, *, user="c1", t0=0.0, step=0.5, x=5.0):
    """Walk one track up (or down) the line x = <x>, one sighting every
    `step` metres at 1 Hz, and return the clock after the last one."""
    n = int(round(abs(y_to - y_from) / step))
    heading = (0.0, 1.0 if y_to >= y_from else -1.0)
    now = t0
    for i in range(n + 1):
        y = y_from + (y_to - y_from) * (i / float(n))
        now = t0 + i
        mem.observe(user, (x, y), heading, now)
    return now


# ------------------------------------------------- 1. the running fit

def test_a_single_sighting_deposits_a_lane_oriented_by_the_corridor():
    """One sighting has a heading but no extent, and must still deposit a
    lane -- the heading is the only orientation evidence there is, which is
    exactly why observe() takes the corridor's (flow-snapped) heading rather
    than deriving one from the positions."""
    mem = LaneMemory(params())
    mem.observe("c1", (5.0, 4.0), (0.0, 1.0), 0.0)

    assert len(mem.lanes) == 1
    lane = mem.lanes[0]
    assert lane.length == pytest.approx(0.0, abs=1e-9)
    assert (lane.ux, lane.uy) == pytest.approx((0.0, 1.0))
    assert lane.users == ("c1",)
    # A point lane still has a line, because clearance extends it.
    assert mem.clearance((5.0, 4.0)) == pytest.approx(0.0, abs=1e-9)


def test_the_fit_grows_from_the_first_sighting_to_the_latest():
    mem = LaneMemory(params())
    patrol(mem, 1.0, 9.0)

    assert len(mem.lanes) == 1
    lane = mem.lanes[0]
    (x0, y0), (x1, y1) = lane.endpoints()
    assert x0 == pytest.approx(5.0, abs=1e-6)
    assert x1 == pytest.approx(5.0, abs=1e-6)
    assert min(y0, y1) == pytest.approx(1.0, abs=1e-6)
    assert max(y0, y1) == pytest.approx(9.0, abs=1e-6)


def test_the_return_leg_keeps_one_lane_not_two():
    """A patrol turns round. Collinearity is UNDIRECTED on purpose -- the
    southbound leg is the same lane as the northbound one, and treating the
    two as separate would double the memory and halve its usefulness."""
    mem = LaneMemory(params())
    end = patrol(mem, 1.0, 9.0)
    patrol(mem, 9.0, 1.0, t0=end + 1.0)

    assert len(mem.lanes) == 1
    (_, y0), (_, y1) = mem.lanes[0].endpoints()
    assert min(y0, y1) == pytest.approx(1.0, abs=1e-6)
    assert max(y0, y1) == pytest.approx(9.0, abs=1e-6)


def test_a_recycled_track_id_restarts_the_fit_instead_of_joining():
    """Ids are recycled (one Carter carried six inside one run). A fit whose
    id has been silent for 2 * lost_timeout_sec is restarted, so a new
    vehicle's position is never joined to a vanished one's start point --
    which would mint a lane across floor nothing drove on."""
    p = params()
    mem = LaneMemory(p)
    mem.observe("c1", (5.0, 1.0), (0.0, 1.0), 0.0)
    mem.observe("c1", (5.0, 2.0), (0.0, 1.0), 1.0)

    gap = 2.0 * p["lost_timeout_sec"] + 1.0
    # Same id, far away, long after: a different vehicle, on its own line.
    mem.observe("c1", (1.0, 8.0), (1.0, 0.0), 1.0 + gap)

    assert len(mem.lanes) == 2
    # Nothing joins (5, 1) to (1, 8): the midpoint of such a segment would
    # be (3, 4.5), and no remembered lane may pass near it.
    assert mem.clearance((3.0, 4.5)) > 1.0


# ---------------------------------------------------------- 2. merging

def test_two_ids_on_the_same_line_merge_into_one_lane():
    mem = LaneMemory(params())
    end = patrol(mem, 1.0, 5.0, user="27")
    patrol(mem, 5.0, 9.0, user="8", t0=end + 1.0)

    assert len(mem.lanes) == 1
    lane = mem.lanes[0]
    assert set(lane.users) == {"27", "8"}
    (_, y0), (_, y1) = lane.endpoints()
    assert min(y0, y1) == pytest.approx(1.0, abs=0.05)
    assert max(y0, y1) == pytest.approx(9.0, abs=0.05)


def test_a_parallel_lane_further_than_lane_merge_dist_stays_separate():
    p = params()
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0, user="a", x=5.0)
    patrol(mem, 1.0, 9.0, user="b", x=5.0 + p["lane_merge_dist_m"] + 0.4)

    assert len(mem.lanes) == 2


def test_a_crossing_lane_stays_separate():
    """Same cell, 90 degrees apart: two lanes, not one 45-degree fiction."""
    mem = LaneMemory(params())
    for i in range(9):
        mem.observe("a", (5.0, 1.0 + i), (0.0, 1.0), float(i))
    for i in range(9):
        mem.observe("b", (1.0 + i, 5.0), (1.0, 0.0), 20.0 + i)

    assert len(mem.lanes) == 2


def test_collinearity_uses_the_undirected_angle_and_the_lateral_offset():
    p = params(lane_merge_angle_deg=15.0, lane_merge_dist_m=0.6)
    base = make_observed_lane((5.0, 0.0), (5.0, 10.0), (0.0, 1.0), 0.0)

    same_way = make_observed_lane((5.2, 2.0), (5.2, 4.0), (0.0, 1.0), 1.0)
    other_way = make_observed_lane((5.2, 4.0), (5.2, 2.0), (0.0, -1.0), 1.0)
    too_far = make_observed_lane((5.9, 2.0), (5.9, 4.0), (0.0, 1.0), 1.0)
    tilted = make_observed_lane((5.0, 2.0), (5.0 + 2.0 * math.tan(
        math.radians(25.0)), 4.0), (0.0, 1.0), 1.0)

    assert lanes_collinear(base, same_way, p)
    assert lanes_collinear(base, other_way, p)       # undirected
    assert not lanes_collinear(base, too_far, p)     # 0.9 > 0.6
    assert not lanes_collinear(base, tilted, p)      # 25 deg > 15


def test_merging_extends_the_endpoints_to_cover_both():
    a = make_observed_lane((5.0, 1.0), (5.0, 4.0), (0.0, 1.0), 0.0, "a")
    b = make_observed_lane((5.0, 6.0), (5.0, 9.0), (0.0, 1.0), 5.0, "b")
    merged = merge_lanes(a, b, now=5.0)

    (_, y0), (_, y1) = merged.endpoints()
    assert min(y0, y1) == pytest.approx(1.0, abs=1e-6)
    assert max(y0, y1) == pytest.approx(9.0, abs=1e-6)
    assert merged.last_seen == 5.0
    assert merged.count == 2
    assert set(merged.users) == {"a", "b"}


# ---------------------------------------------------------- 3. expiry

def test_a_lane_expires_after_lane_memory_s():
    p = params(lane_memory_s=600.0)
    mem = LaneMemory(p)
    end = patrol(mem, 1.0, 9.0)
    assert len(mem.lanes) == 1

    assert mem.expire(end + 599.0) == 1
    assert mem.expire(end + 601.0) == 0
    assert math.isinf(mem.clearance((5.0, 5.0)))


def test_a_lane_still_being_driven_never_expires():
    p = params(lane_memory_s=20.0)
    mem = LaneMemory(p)
    now = 0.0
    for i in range(200):
        now = float(i)
        mem.observe("c1", (5.0, 1.0 + (i % 9)), (0.0, 1.0), now)
        mem.expire(now)
    assert len(mem.lanes) == 1


# -------------------------------------------------- 4. lane clearance

def test_clearance_extends_the_segment_past_its_observed_ends():
    """We only ever watch part of a patrol. carter1 runs y = -4 to y = 7;
    the cameras cover rather less, and the unwatched continuation is exactly
    as much of a lane as the watched part."""
    lane = make_observed_lane((5.0, 3.0), (5.0, 7.0), (0.0, 1.0), 0.0)

    # Beside the observed stretch: the perpendicular distance, either way.
    assert lane_line_clearance((6.2, 5.0), [lane], 2.0) == pytest.approx(1.2)

    # 1.5 m past the observed end, on the line: still ON the lane, because
    # the extension covers it. Without the extension it would read 1.5 m of
    # clearance and be taken as a refuge.
    assert lane_line_clearance((5.0, 8.5), [lane], 2.0) == pytest.approx(0.0)
    assert lane_line_clearance((5.0, 8.5), [lane], 0.0) == pytest.approx(1.5)

    # 3 m past the end, i.e. 1 m beyond the extension: off the lane again.
    assert lane_line_clearance((5.0, 10.0), [lane], 2.0) == pytest.approx(1.0)


def test_clearance_is_the_nearest_of_several_lanes_and_broadcasts():
    lanes = [make_observed_lane((5.0, 0.0), (5.0, 10.0), (0.0, 1.0), 0.0),
             make_observed_lane((8.0, 0.0), (8.0, 10.0), (0.0, 1.0), 0.0)]
    pts = np.array([[[5.0, 5.0], [6.0, 5.0]],
                    [[7.5, 5.0], [11.0, 5.0]]])
    out = lane_line_clearance(pts, lanes, 0.0)

    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.5)      # nearer the second lane
    assert out[1, 1] == pytest.approx(3.0)


def test_clearance_with_no_memory_is_infinite():
    """"No evidence" must never block a refuge -- the same rule
    in_lane_band() follows for a missing band. This is what makes every
    pre-2026-09-09 call site behave exactly as it did."""
    assert math.isinf(lane_line_clearance((5.0, 5.0), [], 2.0))
    assert math.isinf(LaneMemory(params()).clearance((5.0, 5.0)))
    assert not LaneMemory(params()).in_lane((5.0, 5.0))


def test_in_lane_uses_refuge_lane_clearance_m():
    p = params(refuge_lane_clearance_m=1.0)
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0)

    assert mem.in_lane((5.6, 5.0))        # 0.6 m off -- mppi_panoptex_2's
    assert not mem.in_lane((6.2, 5.0))    # 1.2 m off


# ---------------------------- 5. what the memory is wired into (unit level)

def test_hold_in_place_is_unsafe_on_a_remembered_lane_line():
    """Standing on a lane is being in the way even with nothing tracked on
    it right now -- the clause that stops the robot parking on waypoint C
    (0.62 m off carter1's line in the sim triangle)."""
    p = params()
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0)
    kwargs = dict(lanes=mem.lanes,
                  lane_clearance_m=refuge_lane_clearance(p),
                  lane_extension_m=p["lane_extension_m"])

    assert hold_in_place_is_unsafe(None, (5.6, 5.0), None, **kwargs)
    assert not hold_in_place_is_unsafe(None, (6.2, 5.0), None, **kwargs)
    # ...and with the memory switched off it is the old answer.
    assert not hold_in_place_is_unsafe(None, (5.6, 5.0), None)


def test_recompute_keeps_a_committed_point_when_only_the_window_moves():
    """THE hysteresis case. The old rule recomputed whenever any danger
    corridor contained the committed point with tta inside t_yield_sec
    (8 s) -- which a window does every time it slides up a lane. Nine such
    cycles inside one mppi_panoptex_2 episode, each cancelling the goal
    (status 6). refuge_recompute_tta_s (6 s) is the tighter horizon that
    only a genuinely imminent sweep clears."""
    p = params()
    c = danger_corridor(make_corridor(track(5.0, 0.0, 0.0, 1.0), p), p)
    target = (5.0, 7.0)

    # The old test's two conditions both hold...
    from panoptex_nav.corridor import corridor_contains, tta
    assert corridor_contains(c, target)
    assert 0.0 < tta(c, target) < p["t_yield_sec"]
    assert tta(c, target) == pytest.approx(7.0)

    # ...and the committed point stands anyway.
    assert refuge_recompute_reason(target, [c], [], p) is None


def test_recompute_fires_when_the_closest_approach_is_inside_the_horizon():
    p = params()
    c = danger_corridor(make_corridor(track(5.0, 0.0, 0.0, 1.0), p), p)
    target = (5.0, 5.0)                 # tta 5.0 s < refuge_recompute_tta_s

    reason = refuge_recompute_reason(target, [c], [], p)
    assert reason is not None
    assert reason["why"] == "corridor"
    assert reason["corridor"] is c
    assert reason["tta"] == pytest.approx(5.0)


def test_recompute_ignores_a_user_that_has_already_passed():
    p = params()
    c = danger_corridor(make_corridor(track(5.0, 8.0, 0.0, 1.0), p), p)
    # 0.5 m behind the user: inside the 1 m back margin, so contained, but
    # its closest approach is in the past. A receding user is not a reason.
    assert refuge_recompute_reason((5.0, 7.5), [c], [], p) is None


def test_recompute_fires_when_a_lane_is_laid_over_the_committed_point():
    p = params()
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0)
    need = refuge_lane_clearance(p) - LANE_CLEARANCE_SLACK_M

    # 0.5 m off the line, no live corridor at all: the lane memory is the
    # only thing that can say this is a bad place to stand.
    reason = refuge_recompute_reason((5.5, 5.0), [], mem.lanes, p)
    assert reason is not None
    assert reason["why"] == "lane"
    assert reason["corridor"] is None
    assert reason["clearance"] < need

    # Just inside the slack: hysteresis, so the goal is NOT disturbed.
    assert refuge_recompute_reason((5.0 + need + 0.05, 5.0), [],
                                   mem.lanes, p) is None


def test_an_observed_lane_is_a_plain_segment():
    """The marker publisher and the clearance maths both read endpoints();
    nothing else about the dataclass is contractual."""
    lane = ObservedLane(x0=1.0, y0=2.0, x1=1.0, y1=6.0,
                        ux=0.0, uy=1.0, last_seen=3.0)
    assert lane.endpoints() == ((1.0, 2.0), (1.0, 6.0))
    assert lane.length == pytest.approx(4.0)
    assert lane.count == 1


# ------------------------------------- 6. lane hygiene (mppi_panoptex_3)

def noisy_patrol_segments(n=30, x=1.27, y_lo=0.0, y_hi=20.0, seed=7):
    """`n` observed pieces of ONE patrol line, the way a real run deposits
    them: 2-4 m at a time (a track is renamed every few seconds), each end
    displaced up to 0.3 m laterally by tracker position error, each heading
    jittered by up to 10 degrees, and the patrol turning round at both ends.

    The line is x = 1.27 because that is carter1's, and the numbers are the
    ones mppi_panoptex_3 actually produced -- 0.1-0.4 m of map-frame position
    error on a Carter, 10-20 degrees of frame-to-frame heading jitter.
    """
    rng = random.Random(seed)
    out = []
    y, step = y_lo, 1.0
    for _ in range(n):
        length = rng.uniform(2.0, 4.0)
        y_end = y + step * length
        if y_end > y_hi or y_end < y_lo:
            step = -step
            y_end = y + step * length
        # The piece is TILTED ABOUT ITS MIDPOINT, which is the mechanism:
        # 10 degrees over a 4 m piece swings each endpoint 0.35 m sideways
        # while the midpoint does not move at all. Add the midpoint's own
        # 0.3 m of position error and an endpoint is 0.65 m off the line --
        # past the 0.6 m the old both-endpoints test allowed, while the
        # midpoint is still 0.3 m away.
        theta = math.radians(rng.uniform(-10.0, 10.0))
        heading = (math.sin(theta) * step, math.cos(theta) * step)
        mid = (x + rng.uniform(-0.3, 0.3), 0.5 * (y + y_end))
        half = 0.5 * length
        out.append(((mid[0] - heading[0] * half, mid[1] - heading[1] * half),
                    (mid[0] + heading[0] * half, mid[1] + heading[1] * half),
                    heading))
        y = y_end
    return out


def test_thirty_noisy_pieces_of_one_patrol_collapse_to_one_lane():
    """THE mppi_panoptex_3 defect. The old merge test demanded BOTH endpoints
    of a new segment be within lane_merge_dist_m (0.6 m) of the existing
    line, which a 2-4 m piece tilted by tracker noise routinely fails even
    though its midpoint is centimetres away -- so the session memory
    fragmented into 32 lanes over two patrol lines, each one extended 2 m
    past its own ends and each demanding refuge_lane_clearance_m of its own.
    The 2.5 m refuge disc came up empty everywhere and every refuge was found
    at the 4.0 m stage, 3.5 m out, with a Carter closing.
    """
    p = params()
    mem = LaneMemory(p)
    for i, (p0, p1, heading) in enumerate(noisy_patrol_segments()):
        mem.observe(str(i), p0, heading, 2.0 * i)
        mem.observe(str(i), p1, heading, 2.0 * i + 1.0)

    assert len(mem.lanes) == 1, [ln.endpoints() for ln in mem.lanes]
    lane = mem.lanes[0]
    assert lane_is_effective(lane, p)
    # ...and it is the line the pieces were drawn from, not an average of
    # fragments: the whole observed stretch is covered and a point 2 m to the
    # side of it is still 2 m clear.
    assert lane.length > 15.0
    assert mem.effective_clearance((1.27, 10.0)) < 0.4
    assert mem.effective_clearance((1.27 + 2.0, 10.0)) > 1.5


def test_two_perpendicular_patrol_lines_stay_two_lanes():
    """The widened merge test must not fuse the hall's two patrol lines: 25
    degrees is still an angle test, and a crossing line fails it outright."""
    p = params()
    mem = LaneMemory(p)
    for i, (p0, p1, heading) in enumerate(noisy_patrol_segments(seed=11)):
        mem.observe(f"ns{i}", p0, heading, 2.0 * i)
        mem.observe(f"ns{i}", p1, heading, 2.0 * i + 1.0)
    # The same generator turned 90 degrees: y = 6.0, running in x.
    for i, (p0, p1, heading) in enumerate(noisy_patrol_segments(seed=23)):
        flip = lambda q: (q[1], q[0])           # noqa: E731 - local, once
        mem.observe(f"ew{i}", flip(p0), flip(heading), 100.0 + 2.0 * i)
        mem.observe(f"ew{i}", flip(p1), flip(heading), 100.0 + 2.0 * i + 1.0)

    assert len(mem.lanes) == 2, [ln.endpoints() for ln in mem.lanes]
    assert all(lane_is_effective(ln, p) for ln in mem.lanes)


def test_a_short_fragment_does_not_keep_a_refuge_out():
    """A 0.8 m stub seen 3 times -- a track rounding a corner, or a one
    second glimpse of something crossing the hall -- is remembered but does
    NOT vote on the refuge_lane_clearance_m rule. A pile of exactly these is
    what left mppi_panoptex_3 with no legal refuge inside 2.5 m."""
    p = params()
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0, user="carter", x=5.0)          # a real lane
    for i in range(3):
        mem.observe("blip", (7.5, 5.0 + 0.4 * i), (0.0, 1.0), 100.0 + i)

    assert len(mem.lanes) == 2
    fragment = mem.lanes[-1]
    assert fragment.length == pytest.approx(0.8)
    assert fragment.count == 3
    assert not lane_is_effective(fragment, p)
    assert len(mem.effective_lanes()) == 1

    # Standing on the fragment: the raw geometry says 0 m, the rule that
    # decides refuges says 2.5 m (the distance to the real lane).
    assert mem.clearance((7.5, 5.4)) == pytest.approx(0.0, abs=1e-6)
    assert mem.effective_clearance((7.5, 5.4)) == pytest.approx(2.5, abs=1e-6)
    assert not mem.in_lane((7.5, 5.4))
    # The real lane still keeps refuges out, exactly as before.
    assert mem.in_lane((5.6, 5.0))


def test_the_fragment_becomes_a_lane_once_it_has_earned_it():
    """Not a permanent veto on short lanes -- an evidence floor. Keep driving
    down it and it starts counting."""
    p = params()
    mem = LaneMemory(p)
    for i in range(6):
        mem.observe("blip", (7.5, 5.0 + 0.4 * i), (0.0, 1.0), float(i))

    assert len(mem.effective_lanes()) == 1
    assert mem.in_lane((7.5, 5.4))


def test_compaction_is_transitive():
    """An insert merges the new segment into the FIRST lane it matches and
    stops, so two lanes that a later observation has made mergeable with each
    other would stay apart forever. _compact() re-runs the merge test over
    the whole set after every insert until nothing more merges."""
    p = params()
    mem = LaneMemory(p)
    deposit(mem, "a", (5.0, 0.0), (5.0, 2.0), t0=0.0)
    deposit(mem, "c", (5.0, 8.0), (5.0, 10.0), t0=10.0)

    # Two lanes, and they are NOT mergeable with each other: 6 m of floor
    # nothing has been seen on separates them (> lane_merge_gap_m).
    assert len(mem.lanes) == 2
    a, c = mem.lanes
    assert lane_along_gap(a, c) == pytest.approx(6.0)
    assert not lanes_mergeable(a, c, p)

    # The bridging stretch is mergeable with EACH of them and with neither
    # via the other: merging it into `a` is what makes `a` and `c` mergeable.
    deposit(mem, "b", (5.0, 3.0), (5.0, 7.0), t0=20.0)

    assert len(mem.lanes) == 1, [ln.endpoints() for ln in mem.lanes]
    (_, y0), (_, y1) = mem.lanes[0].endpoints()
    assert min(y0, y1) == pytest.approx(0.0, abs=0.2)
    assert max(y0, y1) == pytest.approx(10.0, abs=0.2)
    assert set(mem.lanes[0].users) == {"a", "b", "c"}


def test_two_stretches_further_apart_than_the_gap_stay_separate():
    """The other half of the new merge test, and the one thing the old one
    was too PERMISSIVE about: longitudinal separation used to be ignored
    entirely, so two collinear stretches at opposite ends of a hall became
    one lane whose fitted line lay over floor nothing had driven on."""
    p = params(lane_merge_gap_m=3.0)
    mem = LaneMemory(p)
    deposit(mem, "north", (5.0, 0.0), (5.0, 2.0), t0=0.0)
    deposit(mem, "south", (5.0, 12.0), (5.0, 14.0), t0=50.0)

    assert len(mem.lanes) == 2
    # And the floor between them is not claimed by either.
    assert mem.clearance((5.0, 7.0)) == pytest.approx(3.0)


def test_the_memory_is_capped_at_lane_max():
    """A warehouse hall has a handful of lanes; 32 was never a lane count, it
    was track churn. The shortest/oldest go first, so the patrol line
    survives a flood of stubs."""
    p = params(lane_max=4.0)
    mem = LaneMemory(p)
    patrol(mem, 1.0, 9.0, user="carter", x=5.0)
    real = mem.lanes[0].endpoints()

    # Twenty stubs, each on its own line far from every other.
    for i in range(20):
        deposit(mem, f"blip{i}", (10.0 + 3.0 * i, 0.0),
                (10.0 + 3.0 * i, 0.4), t0=100.0 + i)
        assert len(mem.lanes) <= 4

    assert len(mem.lanes) == 4
    assert real in [ln.endpoints() for ln in mem.lanes]
    assert len(mem.effective_lanes()) == 1
