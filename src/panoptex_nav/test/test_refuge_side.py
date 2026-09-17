"""
NEVER CROSS A LANE TO REACH A REFUGE -- panoptex_nav.corridor's crossing
guards, the derived lane-clearance floor and the wall-hug fallback. Pure
numpy + dicts, no rclpy, same rationale as test_corridor.py (see that
file's docstring).

WHY THIS EXISTS (mppi_panoptex_4, 2026-09-09, x3_nav.log ~line 3850). The
refuge search always PREFERRED the robot's own side of the lane, but only
as a preference, ranked below the lane-line clearance -- and the sim aisle
made it lose:

    refuge (2.73, 1.93) found at refuge_radius_m=2.5 m (clearance 1.46 m)
    YIELD/refuge #6: user 2 enters our lane in tta=6.5 s -- stepping aside
      to (2.73, 1.93), ..., other side of the lane
    ...
    speed cap 60% -- track 2 reason=closing {'t_cpa_s': 0.24, ...}
    release after 3.8 s: ... resuming from (1.27, 1.85)   <- ON the line

carter1 patrols x = 1.27 at ~0.6 m/s, a shelf runs at x ~ 0.30, and the
strip between them is 0.97 m wide -- narrower than the flat 1.0 m
`refuge_lane_clearance_m` of the day, so NOTHING on the robot's own side
could ever be legal and the only "legal" refuge was across the lane.
Crossing 1.9 m of danger band at <= 0.26 m/s takes >= 7 s; the Carter was
6.5 s out.

The geometry below is that scene at its real coordinates: the robot at
(0.82, 0.57), waypoint C at (0.65, 0.50), the shelf occupying x <= 0.35,
and carter1 on the line x = 1.27 heading +y at 0.6 m/s, placed so its tta
to the robot is whatever the test needs.

Run via `colcon test --packages-select panoptex_nav` or directly with
`python3 -m pytest test/test_refuge_side.py` from this package's root.
"""

import math

import numpy as np
import pytest

from panoptex_nav.corridor import (
    CORRIDOR_DEFAULTS,
    LaneGuard,
    LaneMemory,
    crossing_rejected,
    danger_corridor,
    find_refuge,
    find_wall_hug,
    guard_blocks,
    guard_crossing_time,
    lane_guards,
    lane_line_clearance,
    lateral,
    make_corridor,
    make_observed_lane,
    refuge_lane_clearance,
)

RES = 0.05
LANE_X = 1.27          # carter1's patrol line
WALL_X = 0.35          # everything at or below this column is shelf
ROBOT = (0.82, 0.57)   # where the run was standing when it chose to cross
WAYPOINT_C = (0.65, 0.50)
V_CARTER = 0.6


def params(**overrides):
    p = dict(CORRIDOR_DEFAULTS)
    p.update(overrides)
    return p


def track(x, y, vx, vy, track_id="carter1"):
    return {"id": track_id, "label": "mobile robot", "category": "robot",
            "score": 0.9, "age_sec": 0.0, "size": 0.6,
            "x": x, "y": y, "vx": vx, "vy": vy, "pmot": 1.0}


def _map_info():
    return {"resolution": RES, "origin_x": 0.0, "origin_y": 0.0}


def _aisle_grid(size_m=6.0):
    """6 x 6 m of free floor with the shelf wall at x <= WALL_X."""
    n = int(round(size_m / RES))
    grid = np.ones((n, n), dtype=bool)
    grid[:, :int(math.ceil(WALL_X / RES))] = False
    return grid


def _carter(tta_s, p):
    """carter1 heading +y on the lane line, `tta_s` seconds from the
    robot's own y. Returned already widened -- containment question."""
    y = ROBOT[1] - tta_s * V_CARTER
    return danger_corridor(make_corridor(track(LANE_X, y, 0.0, V_CARTER), p),
                           p)


def _lane():
    """The remembered patrol line: x = 1.27 from y = -4 to y = 7."""
    return make_observed_lane((LANE_X, -4.0), (LANE_X, 7.0), (0.0, 1.0),
                              0.0, "carter1")


def _refuge(p, c, guards, radius=None, lanes=None):
    lanes = [_lane()] if lanes is None else lanes
    return find_refuge(
        _aisle_grid(), _map_info(), ROBOT, [c],
        radius=p["refuge_radius_m"] if radius is None else radius,
        clearance=p["refuge_clearance_m"], prefer_side=True,
        lanes=lanes, lane_clearance_m=refuge_lane_clearance(p),
        lane_extension_m=p["lane_extension_m"],
        guards=guards, params=p)


# ------------------------------------------- 1. the derived clearance floor

def test_the_lane_clearance_floor_is_derived_from_the_geometry():
    """0.30 (a Carter's half width) + 0.15 (the X3's robot_radius) + 0.30
    (air) = 0.75 m, not the old flat 1.0 m that was wider than the aisle."""
    p = params()
    assert p["refuge_lane_clearance_m"] == 0.0          # 0 = derive
    assert refuge_lane_clearance(p) == pytest.approx(0.75)
    # ...and it fits in the 0.97 m strip the old number did not.
    assert refuge_lane_clearance(p) < LANE_X - WALL_X


def test_the_derived_floor_never_drops_below_its_minimum():
    p = params(refuge_user_half_width_m=0.1, refuge_robot_radius_m=0.1,
               refuge_lane_margin_m=0.1)
    assert refuge_lane_clearance(p) == pytest.approx(
        p["refuge_lane_clearance_min_m"])


def test_an_explicit_clearance_overrides_the_derived_one():
    assert refuge_lane_clearance(params(refuge_lane_clearance_m=1.0)) == \
        pytest.approx(1.0)
    assert refuge_lane_clearance(params(refuge_lane_clearance_m=0.4)) == \
        pytest.approx(0.4)


def test_the_waypoint_gate_uses_the_derived_floor():
    """Waypoint C is 0.62 m off the line: inside 0.75 m, so the approach
    gate still fires -- the smaller floor must not open that hole."""
    mem = LaneMemory(params())
    mem.lanes = [_lane()]
    mem.lanes[0].count = 99                     # earns lane_is_effective
    assert mem.in_lane(WAYPOINT_C)
    assert not mem.in_lane((LANE_X + 1.0, 0.5))


# --------------------------------------------------- 2. the crossing clock

def test_crossing_time_is_both_offsets_plus_the_whole_lane():
    p = params()
    c = _carter(6.5, p)
    g = lane_guards([c], ROBOT)[0]
    cand = (2.73, 1.93)                          # the refuge the run took
    expect = ((abs(ROBOT[0] - LANE_X) + abs(cand[0] - LANE_X)
               + 2.0 * c.half_width) / p["v_cross_mps"])
    assert guard_crossing_time(g, ROBOT, cand, p) == pytest.approx(expect)
    assert expect > 15.0                         # ...against a 6.5 s tta


def test_a_guard_blocks_the_far_side_and_never_our_own():
    p = params()
    c = _carter(6.5, p)
    g = lane_guards([c], ROBOT)[0]
    assert g.tta_s == pytest.approx(6.5)
    assert guard_blocks(g, ROBOT, (2.73, 1.93), p)          # far side
    assert not guard_blocks(g, ROBOT, (0.80, 1.93), p)      # our side
    # Vectorised, same answers.
    out = crossing_rejected([g], ROBOT,
                            np.array([[2.73, 1.93], [0.80, 1.93]]), p)
    assert list(out) == [True, False]


def test_a_far_enough_user_leaves_the_far_side_legal():
    p = params()
    g = lane_guards([_carter(30.0, p)], ROBOT)[0]
    assert not guard_blocks(g, ROBOT, (2.73, 1.93), p)


def test_a_receding_user_is_not_a_reason_for_anything():
    """tta <= 0 means the user is already past -- the same convention
    decide_crossing and refuge_recompute_reason follow."""
    p = params()
    g = lane_guards([_carter(-3.0, p)], ROBOT)[0]
    assert g.tta_s < 0.0
    assert not guard_blocks(g, ROBOT, (2.73, 1.93), p)


def test_a_robot_standing_on_the_line_has_no_far_side():
    p = params()
    g = lane_guards([_carter(6.5, p)], (LANE_X, 0.57))[0]
    assert not guard_blocks(g, (LANE_X, 0.57), (2.73, 1.93), p)


# ------------------------------------------------------------- 3. guards

def test_a_remembered_lane_is_guarded_only_while_somebody_is_on_it():
    p = params()
    on_it = _carter(6.5, p)                       # driving x = 1.27
    guards = lane_guards([on_it], ROBOT, [_lane()], p["lane_extension_m"])
    assert [g.source for g in guards] == ["user", "lane"]
    assert guards[1].tta_s == pytest.approx(6.5)

    # The same lane with the only tracked user two aisles away is NOT a
    # guard: nothing on it can hit us, and walling off every learned line
    # would strand the robot.
    elsewhere = danger_corridor(
        make_corridor(track(4.5, 0.0, 0.0, V_CARTER, "other"), p), p)
    guards = lane_guards([elsewhere], ROBOT, [_lane()],
                         p["lane_extension_m"])
    assert [g.source for g in guards] == ["user"]


def test_guards_carry_the_soonest_user_on_the_lane():
    p = params()
    near, far = _carter(4.0, p), _carter(20.0, p)
    guards = lane_guards([far, near], ROBOT, [_lane()],
                         p["lane_extension_m"])
    lane_guard = next(g for g in guards if g.source == "lane")
    assert lane_guard.tta_s == pytest.approx(4.0)


# ------------------------------------------ 4. the refuge search itself

def test_the_far_side_refuge_is_refused_while_the_carter_is_close():
    """THE regression. Without guards the search still answers with the
    far-side cell the run took; with them there is no answer at all inside
    either disc, and the caller falls through to the wall hug."""
    p = params()
    c = _carter(6.5, p)
    assert abs(lateral(c, ROBOT)) > 0.0

    old = _refuge(p, c, guards=())
    assert old is not None
    assert float(old[0]) > LANE_X                 # across the lane: the bug

    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    assert _refuge(p, c, guards) is None
    assert _refuge(p, c, guards, radius=p["refuge_radius_max_m"]) is None


def test_the_far_side_refuge_is_taken_when_the_carter_is_far_away():
    """The complement: 30 s of tta is more than the ~16 s the crossing
    needs plus refuge_cross_margin_s, so crossing is allowed again -- and
    with the whole west strip inside the 0.75 m floor, it is the answer."""
    p = params()
    c = _carter(30.0, p)
    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    xy = _refuge(p, c, guards)
    assert xy is not None
    assert float(xy[0]) > LANE_X
    assert lane_line_clearance(xy, [_lane()], p["lane_extension_m"]) >= \
        refuge_lane_clearance(p)


def test_the_same_side_wins_when_a_legal_cell_exists_on_it():
    """Ranking, with the wall pushed back so the west strip is wide enough:
    same side outranks lane clearance now, so a nearer legal cell on our
    own side beats a roomier one across the lane."""
    p = params()
    grid = np.ones((120, 120), dtype=bool)       # no shelf at all
    c = _carter(30.0, p)                         # crossing would be legal
    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    xy = find_refuge(grid, _map_info(), ROBOT, [c],
                     radius=p["refuge_radius_m"],
                     clearance=p["refuge_clearance_m"], prefer_side=True,
                     lanes=[_lane()],
                     lane_clearance_m=refuge_lane_clearance(p),
                     lane_extension_m=p["lane_extension_m"],
                     guards=guards, params=p)
    assert xy is not None
    assert float(xy[0]) < LANE_X                                  # our side
    assert LANE_X - float(xy[0]) >= refuge_lane_clearance(p) - 1e-9


# --------------------------------------------------------- 5. the wall hug

def _wall_hug(p, c, radius=None):
    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    return find_wall_hug(
        _aisle_grid(), _map_info(), ROBOT, guards=guards,
        radius=p["refuge_radius_m"] if radius is None else radius,
        clearance=p["refuge_clearance_m"], lanes=[_lane()],
        lane_extension_m=p["lane_extension_m"], params=p)


def test_the_wall_hug_stands_as_far_off_the_lane_as_the_shelf_allows():
    p = params()
    c = _carter(6.5, p)
    xy = _wall_hug(p, c)
    assert xy is not None
    x = float(xy[0])
    # Our own side, and further off the line than where we were standing.
    assert x < LANE_X
    assert lateral(c, xy) * lateral(c, ROBOT) > 0.0
    assert LANE_X - x > LANE_X - ROBOT[0]
    # As close to the shelf as refuge_clearance_m allows, and no closer:
    # the wall hug relaxes the LANE floor, never the obstacle one.
    assert x >= WALL_X + p["refuge_clearance_m"] - RES
    assert x <= WALL_X + p["refuge_clearance_m"] + 2 * RES
    # ...which is inside the lane floor. That is the point: this is the
    # least-bad standing place, not a legal one.
    assert LANE_X - x < refuge_lane_clearance(p)
    assert math.hypot(xy[0] - ROBOT[0], xy[1] - ROBOT[1]) <= \
        p["refuge_radius_m"]


def test_the_wall_hug_never_crosses_the_lane_either():
    """Every cell it can reach is on our side: the guards apply here too,
    so "nowhere legal" can never quietly become "across the lane"."""
    p = params()
    c = _carter(6.5, p)
    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    xy = _wall_hug(p, c)
    assert not crossing_rejected(guards, ROBOT, xy, p)


def test_the_wall_hug_gives_up_when_there_is_no_floor_at_all():
    p = params()
    c = _carter(6.5, p)
    guards = lane_guards([c], ROBOT, [_lane()], p["lane_extension_m"])
    blocked = np.zeros((120, 120), dtype=bool)
    assert find_wall_hug(blocked, _map_info(), ROBOT, guards=guards,
                         radius=p["refuge_radius_m"],
                         clearance=p["refuge_clearance_m"],
                         params=p) is None


def test_the_wall_hug_takes_the_nearest_cell_with_nothing_to_avoid():
    """No guards, no lanes: it degenerates to "the nearest legal cell",
    which is what makes it safe to call unconditionally."""
    p = params()
    xy = find_wall_hug(np.ones((120, 120), dtype=bool), _map_info(), ROBOT,
                       radius=p["refuge_radius_m"],
                       clearance=p["refuge_clearance_m"], params=p)
    assert xy is not None
    assert math.hypot(xy[0] - ROBOT[0], xy[1] - ROBOT[1]) <= 2 * RES


def test_a_guard_can_be_built_by_hand():
    """The dataclass is contractual for the supervisor and the tests; a
    guard nobody is on (tta inf) blocks nothing."""
    g = LaneGuard(x=LANE_X, y=0.0, ux=0.0, uy=1.0, half_width=0.9,
                  tta_s=float("inf"))
    assert not guard_blocks(g, ROBOT, (2.73, 1.93), params())
