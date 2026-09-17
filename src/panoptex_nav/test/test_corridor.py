"""
Pure-geometry tests for panoptex_nav.corridor -- no rclpy, no ROS graph, no
simulator, same rationale as test_risk_speed_governor.py (see that module's
docstring): corridor.py is data in / data out by construction, so it is
tested directly rather than through a running mission_supervisor.

Run via `colcon test --packages-select panoptex_nav` (wired into
CMakeLists.txt through ament_cmake_pytest) or directly with
`python3 -m pytest test/test_corridor.py` from this package's root,
provided panoptex_nav/ is importable (PYTHONPATH at this package's source
root, or the workspace already built and sourced).

The numbers in the gap-acceptance tests are the worked example from the
WP-B design note, re-derived for the shipped half_width of 0.55 m (it was
0.65 until 2026-09-09, when the west-strip leg turned out to sit permanently
inside lane 1's corridor): lane span 2*0.55 + 0.3 = 1.4 m, so a crossing
whose entry is 1 m ahead needs (1 + 1.4) / 0.2 = 12 s to clear.
"""

import math

import numpy as np
import pytest

from panoptex_nav.corridor import (
    CORRIDOR_DEFAULTS,
    NO_PASS,
    danger_corridor,
    hold_in_place_is_unsafe,
    in_lane_band,
    loss_release_guard,
    make_lane_band,
    Corridor,
    along,
    approach_zone,
    blind_crossing_ok,
    corridor_contains,
    crossing_choice,
    crossing_decision,
    crossing_policy,
    decide_crossing,
    find_refuge,
    hold_point,
    lane_line_clearance,
    lane_speed,
    lateral,
    make_observed_lane,
    make_corridor,
    plan_corridor_crossing,
    polygon_contains,
    select_corridor_users,
    t_clear,
    tta,
    user_passed,
    zone_coverage,
    zone_users,
)


def params(**overrides):
    p = dict(CORRIDOR_DEFAULTS)
    p.update(overrides)
    return p


def track(x, y, vx, vy, track_id="c1", category="robot", pmot=1.0):
    return {"id": track_id, "label": "mobile robot", "category": category,
            "score": 0.9, "age_sec": 0.0, "size": 0.6,
            "x": x, "y": y, "vx": vx, "vy": vy, "pmot": pmot}


def straight_plan(x0, y0, x1, y1, step=0.1):
    """A polyline from (x0,y0) to (x1,y1) sampled every `step` metres --
    the shape nav2's global planner publishes on `plan`."""
    n = int(round(math.hypot(x1 - x0, y1 - y0) / step))
    ts = np.linspace(0.0, 1.0, n + 1)
    return np.stack([x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)], axis=1)


# ------------------------------------------------------- 1. geometry

def test_corridor_geometry_inside_outside_and_signs():
    """User at the origin driving +x at 1 m/s: the lane runs from 1 m
    behind it to 10 m ahead, 0.65 m either side."""
    c = make_corridor(track(0.0, 0.0, 1.0, 0.0), params())
    assert c.speed == pytest.approx(1.0)
    assert c.length_ahead == pytest.approx(10.0)
    assert (c.ux, c.uy) == pytest.approx((1.0, 0.0))

    # 2 m ahead on the centre line: inside.
    assert corridor_contains(c, (2.0, 0.0))
    # 0.7 m to the side (> half_width 0.55): outside.
    assert not corridor_contains(c, (2.0, 0.7))
    # 0.5 m behind the user, still within the 1.0 m back margin: inside.
    assert corridor_contains(c, (-0.5, 0.0))
    # 1.5 m behind: past the back margin, outside.
    assert not corridor_contains(c, (-1.5, 0.0))
    # 11 m ahead: past the 10 s horizon, outside.
    assert not corridor_contains(c, (11.0, 0.0))

    # along is measured from the USER and signed by the heading;
    # lateral is positive to the LEFT of the heading (+y here).
    assert along(c, (2.0, 0.0)) == pytest.approx(2.0)
    assert along(c, (-0.5, 0.0)) == pytest.approx(-0.5)
    assert lateral(c, (2.0, 0.4)) == pytest.approx(0.4)
    assert lateral(c, (2.0, -0.4)) == pytest.approx(-0.4)

    # Vectorised form: same answers for a stack of points.
    pts = np.array([[2.0, 0.0], [2.0, 0.7], [-0.5, 0.0]])
    assert list(corridor_contains(c, pts)) == [True, False, True]
    assert list(along(c, pts)) == pytest.approx([2.0, 2.0, -0.5])


def test_user_selection_drops_self_slow_and_wrong_category():
    """Under the default corridor_users_any_mover (true): self, too-slow
    and low-pmot tracks are still dropped, and "person" is still dropped
    (NON_CORRIDOR_CATEGORIES) -- none of these fixture tracks exercise a
    non-person, non-CORRIDOR_CATEGORIES category, so this is unaffected by
    the WP-A default change; see test_user_selection_any_mover_admits_
    furniture_category below for that."""
    p = params()
    robot = (0.0, 0.0)
    tracks = [
        track(0.3, 0.0, 0.5, 0.0, track_id="self"),        # the X3 itself
        track(5.0, 0.0, 0.05, 0.0, track_id="parked"),     # too slow
        track(5.0, 1.0, 0.6, 0.0, track_id="unsure", pmot=0.2),
        track(5.0, 2.0, 0.6, 0.0, track_id="person", category="person"),
        track(5.0, 3.0, 0.6, 0.0, track_id="carter"),      # the one real user
    ]
    assert [t["id"] for t in select_corridor_users(tracks, robot, p)] == ["carter"]


# ---------------------------------------------------- WP-A: any-mover policy

def test_user_selection_any_mover_admits_furniture_category():
    """WP-A, 2026-09-11 (default corridor_users_any_mover: true): a
    convincingly-moving "table" (category "furniture", or a tracker-
    promoted "wheeled" label "moving object") owns a corridor exactly like
    a "robot"/"wheeled" track would -- only NON_CORRIDOR_CATEGORIES
    ("person") is excluded by category now."""
    p = params()
    robot = (0.0, 0.0)
    tracks = [
        track(5.0, 0.0, 0.6, 0.0, track_id="table", category="furniture"),
        track(5.0, 1.0, 0.6, 0.0, track_id="person", category="person"),
        track(5.0, 2.0, 0.6, 0.0, track_id="carter"),
    ]
    ids = {t["id"] for t in select_corridor_users(tracks, robot, p)}
    assert ids == {"table", "carter"}


def test_user_selection_any_mover_disabled_restores_category_filter():
    """corridor_users_any_mover: false restores the pre-2026-09-11
    behaviour -- only CORRIDOR_CATEGORIES ("robot", "wheeled") may own a
    corridor, so the moving "table" is dropped."""
    p = params(corridor_users_any_mover=0.0)
    robot = (0.0, 0.0)
    tracks = [
        track(5.0, 0.0, 0.6, 0.0, track_id="table", category="furniture"),
        track(5.0, 2.0, 0.6, 0.0, track_id="carter"),
    ]
    assert [t["id"] for t in select_corridor_users(tracks, robot, p)] == ["carter"]


def test_make_corridor_carries_category_through():
    c = make_corridor(track(5.0, 0.0, 0.6, 0.0, category="furniture"), params())
    assert c.category == "furniture"


def test_crossing_choice_tie_breaks_by_category():
    """Two crossings with an EXACT S-score tie -- the robot/wheeled one
    must win over the furniture one, per _prefers_category's documented
    preference."""
    info_furniture = {
        "xy_entry": (0.0, 0.0), "xy_exit": (1.0, 0.0), "category": "furniture",
    }
    info_robot = {
        "xy_entry": (5.0, 0.0), "xy_exit": (6.0, 0.0), "category": "robot",
    }
    s_grid = {
        "values": np.zeros((20, 20)), "resolution": 0.5,
        "origin_x": -5.0, "origin_y": -5.0,
    }
    chosen = crossing_choice([info_furniture, info_robot], s_grid)
    assert chosen["category"] == "robot"


# --------------------------------------------------- 2. flow snapping

def _heading_deg(c):
    return math.degrees(math.atan2(c.uy, c.ux))


def test_flow_snaps_when_confident_and_broadly_agreeing():
    """20 deg off a confident flow cell -> snap to the flow heading."""
    p = params()
    v = 0.6
    a = math.radians(20.0)
    tr = track(0.0, 0.0, v * math.cos(a), v * math.sin(a))
    c = make_corridor(tr, p, flow_sample=(0.8, 1.0, 0.0))
    assert c.heading_source == "flow"
    assert _heading_deg(c) == pytest.approx(0.0, abs=1e-6)
    # Length still comes from the track's OWN speed, not the flow vector's.
    assert c.length_ahead == pytest.approx(v * p["t_corridor_sec"])


def test_flow_does_not_snap_when_disagreeing_or_unconfident():
    p = params()
    v = 0.6
    a = math.radians(45.0)
    tr45 = track(0.0, 0.0, v * math.cos(a), v * math.sin(a))
    c45 = make_corridor(tr45, p, flow_sample=(0.8, 1.0, 0.0))
    assert c45.heading_source == "velocity"
    assert _heading_deg(c45) == pytest.approx(45.0)

    a = math.radians(20.0)
    tr20 = track(0.0, 0.0, v * math.cos(a), v * math.sin(a))
    c_lowconf = make_corridor(tr20, p, flow_sample=(0.1, 1.0, 0.0))
    assert c_lowconf.heading_source == "velocity"
    assert _heading_deg(c_lowconf) == pytest.approx(20.0)

    # No flow subscribed at all: velocity, unconditionally.
    assert make_corridor(tr20, p).heading_source == "velocity"


# ------------------------------------------------------- 3. crossing

def test_plan_crossing_indices_and_entry_distance():
    """Corridor along +y through the origin; plan runs west->east across it
    at y = 2 in 0.1 m steps. Entry/exit are the first/last plan points with
    |x| <= 0.65, and d_entry is the arc length to the entry."""
    c = make_corridor(track(0.0, 0.0, 0.0, 1.0), params())
    plan = straight_plan(-2.0, 2.0, 2.0, 2.0, step=0.1)

    idx_entry, idx_exit, d_entry, xy_entry, xy_exit = plan_corridor_crossing(plan, c)

    step = 0.1
    assert xy_entry[0] == pytest.approx(-0.55, abs=step)
    assert xy_exit[0] == pytest.approx(0.55, abs=step)
    assert idx_entry == 15 and idx_exit == 25
    # 15 steps of 0.1 m from the plan start.
    assert d_entry == pytest.approx(1.5, abs=step)

    # A plan that never enters the lane.
    assert plan_corridor_crossing(straight_plan(-2.0, 20.0, 2.0, 20.0), c) is None


# ------------------------------------------------- 4. gap acceptance

def _crossing_case(user_y, plan_x0=-1.55):
    """User approaching the origin from -y at 0.6 m/s; plan crosses the lane
    west->east through the origin. plan_x0 = -1.55 puts the lane entry
    (x = -0.55) exactly 1.0 m along the plan from the robot."""
    p = params()
    c = make_corridor(track(0.0, user_y, 0.0, 0.6), p)
    plan = straight_plan(plan_x0, 0.0, 1.6, 0.0, step=0.1)
    return c, plan, p


def test_decide_crossing_holds_when_the_gap_is_too_small():
    c, plan, p = _crossing_case(user_y=-6.0)
    info = crossing_decision(c, plan, p)
    assert info["d_entry"] == pytest.approx(1.0, abs=0.1)
    # (1.0 + 2*0.55 + 0.3) / 0.20 = 12 s to clear...
    assert info["t_clear"] == pytest.approx(12.0, abs=0.5)
    # ...against 6 m / 0.6 m/s = 10 s until the user arrives -> hold.
    assert info["tta"] == pytest.approx(10.0, abs=0.01)
    assert info["decision"] == "hold"


def test_decide_crossing_goes_when_the_user_is_far_enough():
    """Same crossing, user twice as far: 12 m / 0.6 m/s = 20 s > 13 + 2 -> go.

    Two ways that comes out "go", both asserted here. With the shipped
    t_corridor_sec of 10 s the user's lane only reaches 6 m ahead of it, so
    at 12 m the plan is not even inside the corridor -- no crossing, nothing
    to decide. Stretch the horizon to 25 s (the lane now reaches 15 m) and
    the crossing exists but is decided "go" on the timing.
    """
    c, plan, p = _crossing_case(user_y=-12.0)
    assert crossing_decision(c, plan, p) is None

    long_p = params(t_corridor_sec=25.0)
    c_long = make_corridor(track(0.0, -12.0, 0.0, 0.6), long_p)
    info = crossing_decision(c_long, plan, long_p)
    assert info is not None
    assert info["tta"] == pytest.approx(20.0, abs=0.01)     # > 12 + 2
    assert info["t_clear"] == pytest.approx(12.0, abs=0.5)
    assert info["decision"] == "go"


def test_decide_crossing_goes_when_the_user_is_moving_away():
    """User already 2 m past the crossing and still driving away: tta < 0."""
    c, plan, p = _crossing_case(user_y=2.0)
    info = crossing_decision(c, plan, p)
    assert info is None or info["tta"] < 0.0
    if info is not None:
        assert info["decision"] == "go"
    # ...and the primitive agrees on the raw numbers.
    assert decide_crossing(-3.3, 12.0, p) == "go"
    assert decide_crossing(10.0, 12.0, p) == "hold"
    assert decide_crossing(float("inf"), 12.0, p) == "go"


def test_t_clear_matches_the_worked_example():
    c = make_corridor(track(0.0, -6.0, 0.0, 0.6), params())
    assert t_clear(1.0, c, params()) == pytest.approx(12.0)


def test_user_passed_needs_the_release_margin():
    p = params()
    c = make_corridor(track(0.0, 0.0, 0.0, 1.0), p)
    # Crossing 0.5 m behind the user: its body is still in the way.
    assert not user_passed(c, (0.0, -0.5), p)
    # 1.2 m behind (> half_width 0.55 + margin 0.3): released.
    assert user_passed(c, (0.0, -1.2), p)


# --------------------------------------------------------- 5. refuge

RES = 0.05


def _free_grid(size_m=10.0, res=RES):
    n = int(round(size_m / res))
    return np.ones((n, n), dtype=bool)


def _map_info(res=RES):
    return {"resolution": res, "origin_x": 0.0, "origin_y": 0.0}


def _cell_centre(v, res=RES):
    return (int(v / res) + 0.5) * res


def test_find_refuge_leaves_the_lane_on_the_robots_own_side():
    """10x10 m free floor with one north-south wall just east of the lane.

    The lane (user driving +y along x = 5.0, half width 0.55) runs over the
    robot at (5.2, 5.0), which therefore sits on the lane's EAST side
    (negative lateral, since +y heading puts east to the right). West of
    the lane is closer -- x <= 4.45 is only 0.75 m away -- but reaching it
    means crossing in front of the user, so the side preference must win
    and the refuge must land east of the lane AND at least 0.45 m clear of
    the wall.
    """
    grid = _free_grid()
    info = _map_info()
    wall_col = int(5.725 / RES)          # a single 5 cm column of wall
    wall_x = (wall_col + 0.5) * RES
    grid[:, wall_col] = False

    robot = (5.2, 5.0)
    c = make_corridor(track(5.0, 1.0, 0.0, 1.0), params())
    assert corridor_contains(c, robot)          # the robot really is in the lane
    assert lateral(c, robot) < 0.0              # ...on the east side

    xy = find_refuge(grid, info, robot, [c],
                     radius=params()["refuge_radius_m"],
                     clearance=params()["refuge_clearance_m"],
                     prefer_side=True)
    assert xy is not None
    assert not corridor_contains(c, xy)                 # out of the lane
    assert lateral(c, xy) < 0.0                         # robot's own side
    assert abs(float(xy[0]) - wall_x) >= 0.45 - 1e-9    # clear of the wall
    assert math.hypot(xy[0] - robot[0], xy[1] - robot[1]) <= \
        params()["refuge_radius_m"]
    # The wall is what pushes it out: the first free-of-corridor column
    # (x ~ 5.575) is inside the wall's 0.45 m keep-out, so the answer has to
    # be east of the wall, not merely east of the lane.
    assert float(xy[0]) > wall_x


def test_find_refuge_returns_none_when_everything_is_blocked():
    grid = np.zeros((200, 200), dtype=bool)
    c = make_corridor(track(5.0, 1.0, 0.0, 1.0), params())
    assert find_refuge(grid, _map_info(), (5.2, 5.0), [c],
                       radius=1.5, clearance=0.45) is None


def test_find_refuge_returns_none_when_every_free_cell_is_in_a_corridor():
    """Two crossed lanes covering the whole search disc: nowhere legal."""
    grid = _free_grid()
    p = params()
    c1 = make_corridor(track(5.0, 1.0, 0.0, 1.0, track_id="a"), p)
    # A second, very wide lane straight across the first one.
    wide = dict(p, corridor_half_width_m=6.0)
    c2 = make_corridor(track(1.0, 5.0, 1.0, 0.0, track_id="b"), wide)
    assert find_refuge(grid, _map_info(), (5.2, 5.0), [c1, c2],
                       radius=1.5, clearance=0.45) is None


# ----------------------------------------------------- 6. hold point

def test_hold_point_walks_back_along_the_plan():
    plan = straight_plan(0.0, 0.0, 5.0, 0.0, step=0.1)
    idx, xy = hold_point(plan, idx_entry=20, hold_back_m=0.7)
    assert idx == 13                       # 7 x 0.1 m steps back
    assert xy[0] == pytest.approx(1.3, abs=1e-9)


def test_hold_point_signals_hold_in_place_when_the_standoff_is_behind_us():
    plan = straight_plan(0.0, 0.0, 5.0, 0.0, step=0.1)
    # Entry only 0.2 m ahead: 0.7 m of stand-off is behind the robot.
    assert hold_point(plan, idx_entry=2, hold_back_m=0.7) is None
    # Entry at the robot itself: likewise.
    assert hold_point(plan, idx_entry=0, hold_back_m=0.7) is None


# ------------------------------------------- 7. static lane band (WP-B.2)

def _prior(cells_by_col, cols=100, rows=100, res=0.10, value=100):
    """A synthetic /risk_perception/spatial_prior grid (origin 0,0) with the
    given columns fully painted -- i.e. one straight learned lane per
    column, which is what the warehouse aisle's two AMR lanes look like."""
    grid = np.zeros((rows, cols), dtype=np.int16)
    for col in cells_by_col:
        grid[:, col] = value
    return grid, {"resolution": res, "origin_x": 0.0, "origin_y": 0.0}


def test_lane_band_thresholds_and_dilates():
    grid, info = _prior([13])                      # a lane at x in [1.3, 1.4)
    band = make_lane_band(grid, info, min_value=5.0, dilate_m=0.55)
    # ceil(0.55 / 0.10) = 6 cells either side -> x in [0.7, 2.0)
    assert in_lane_band(band, (1.35, 5.0))
    assert in_lane_band(band, (0.75, 5.0))
    assert in_lane_band(band, (1.95, 5.0))
    assert not in_lane_band(band, (0.65, 5.0))
    assert not in_lane_band(band, (2.05, 5.0))
    # Outside the grid's extent, and with no band at all: never blocked.
    assert not in_lane_band(band, (-5.0, 5.0))
    assert not in_lane_band(None, (1.35, 5.0))
    # Sub-threshold evidence is not a lane.
    faint, info2 = _prior([13], value=3)
    assert not in_lane_band(
        make_lane_band(faint, info2, min_value=5.0, dilate_m=0.55),
        (1.35, 5.0))


def test_find_refuge_rejects_the_second_lane_band():
    """The avoid_panoptex_1 regression: two lanes 0.83 m apart, only ONE of
    them with a live track. Without the band the refuge lands in the aisle
    between/on them (exactly what Isaac showed); with the band it has to
    leave the traffic pattern altogether.
    """
    p = params()
    grid = _free_grid()                       # 10 x 10 m of free floor
    info = _map_info()
    robot = (1.42, 3.0)                       # in lane 1, 0.10 m east of centre
    # Lane 1 has the live user (driving -y down x = 1.32); lane 2 (x = 2.15)
    # is quiet at this instant, which is why it is invisible to `corridors`.
    c = make_corridor(track(1.32, 6.0, 0.0, -0.6), p)
    assert corridor_contains(c, robot)
    assert lateral(c, robot) > 0.0            # robot east of the lane centre

    prior, pinfo = _prior([13, 21])           # lanes at x = 1.35 and 2.15
    band = make_lane_band(prior, pinfo, p["lane_band_min_value"],
                          p["corridor_half_width_m"])
    # Union of the two dilated lanes: x in [0.7, 2.8).
    assert in_lane_band(band, (1.98, 3.0)) and in_lane_band(band, (2.53, 3.0))

    without = find_refuge(grid, info, robot, [c],
                          radius=p["refuge_radius_m"],
                          clearance=p["refuge_clearance_m"])
    assert without is not None
    assert in_lane_band(band, without)        # ...lands in the other lane

    xy = find_refuge(grid, info, robot, [c],
                     radius=p["refuge_radius_m"],
                     clearance=p["refuge_clearance_m"],
                     lane_band=band)
    assert xy is not None
    assert not in_lane_band(band, xy)         # out of the traffic pattern
    assert not corridor_contains(c, xy)
    assert lateral(c, xy) > 0.0               # still the robot's own side
    assert float(xy[0]) >= 2.8                # the east column, not the aisle
    # Even in this scaled-down synthetic aisle, leaving the band costs
    # 1.4 m -- the old 1.5 m search disc had no headroom at all, which is
    # why refuge_radius_m is now 2.5 (on the real map the free cells east of
    # the band start further out again, behind the shelf clearance).
    assert math.hypot(xy[0] - robot[0], xy[1] - robot[1]) > 1.4
    assert math.hypot(xy[0] - robot[0], xy[1] - robot[1]) <= 2.5


def test_find_refuge_falls_back_to_the_band_when_it_swallows_everything():
    """A robot deep inside the traffic pattern must still be told where to
    go: "least bad" beats stopping dead in the aisle."""
    p = params()
    grid = _free_grid()
    prior, pinfo = _prior(list(range(0, 100)))     # the whole floor is lane
    band = make_lane_band(prior, pinfo, p["lane_band_min_value"], 0.0)
    c = make_corridor(track(1.32, 6.0, 0.0, -0.6), p)
    xy = find_refuge(grid, _map_info(), (1.42, 3.0), [c], radius=2.5,
                     clearance=0.45, lane_band=band)
    assert xy is not None
    assert not corridor_contains(c, xy)             # the hard rule still holds


def test_hold_point_walks_past_the_lane_band():
    """0.7 m back from the entry lands in the OTHER lane's band, so the
    stand-off has to keep walking back until it is out of it."""
    plan = straight_plan(0.0, 0.0, 5.0, 0.0, step=0.1)
    prior = np.zeros((40, 100), dtype=np.int16)
    # Paint x in [1.0, 1.65) at the plan's y = 0 (row 20 of a grid whose
    # origin is y = -1.0 at 0.05 m), i.e. plan indices 10..16.
    prior[:, 20:33] = 100
    band = make_lane_band(prior, {"resolution": 0.05, "origin_x": 0.0,
                                  "origin_y": -1.0},
                          min_value=5.0, dilate_m=0.0)
    assert in_lane_band(band, plan[13]) and not in_lane_band(band, plan[9])

    # Without the band: 7 steps back from the entry, as before.
    assert hold_point(plan, idx_entry=20, hold_back_m=0.7)[0] == 13
    # With it: keep walking to the first point clear of the band.
    idx, xy = hold_point(plan, idx_entry=20, hold_back_m=0.7, lane_band=band)
    assert idx == 9
    assert not in_lane_band(band, xy)

    # A plan that is inside the band all the way back to the robot has no
    # stand-off point at all -> hold in place.
    prior[:, 0:33] = 100
    all_band = make_lane_band(prior, {"resolution": 0.05, "origin_x": 0.0,
                                      "origin_y": -1.0},
                              min_value=5.0, dilate_m=0.0)
    assert hold_point(plan, idx_entry=20, hold_back_m=0.7,
                      lane_band=all_band) is None


# ------------------------------------- 8. loss-release guard (WP-B.3)

def test_loss_release_guard_blocks_on_any_id():
    """The track we were yielding for vanished, but the SAME physical Carter
    is back under a new id, still sweeping the point we yielded for."""
    p = params()
    ref = np.array([1.32, 0.0])
    reidentified = make_corridor(
        track(1.32, 3.0, 0.0, -0.6, track_id="60"), p)     # tta = 5 s
    blocker = loss_release_guard([reidentified], ref, p)
    assert blocker is not None and blocker.user_id == "60"


def test_loss_release_guard_ignores_parallel_lanes_and_departures():
    p = params()
    ref = np.array([1.32, 0.0])

    # A user on the OTHER lane (x = 2.15): its along-coordinate to the same
    # y is perfectly positive, but it never sweeps this point.
    parallel = make_corridor(track(2.15, 3.0, 0.0, -0.6, track_id="8"), p)
    assert not corridor_contains(parallel, ref)
    assert loss_release_guard([parallel], ref, p) is None

    # A user that has already driven past the point.
    gone = make_corridor(track(1.32, -2.0, 0.0, -0.6, track_id="14"), p)
    assert loss_release_guard([gone], ref, p) is None

    # A user still further out than t_yield (8 s at 0.6 m/s = 4.8 m).
    far = make_corridor(track(1.32, 5.4, 0.0, -0.6, track_id="27"), p)
    assert tta(far, ref) > p["t_yield_sec"]
    assert loss_release_guard([far], ref, p) is None

    assert loss_release_guard([], ref, p) is None


def test_loss_release_guard_picks_the_soonest_arrival():
    p = params()
    ref = np.array([1.32, 0.0])
    near = make_corridor(track(1.32, 1.2, 0.0, -0.6, track_id="near"), p)
    later = make_corridor(track(1.32, 3.0, 0.0, -0.6, track_id="later"), p)
    assert loss_release_guard([later, near], ref, p).user_id == "near"


# ------------------------------- 9. danger half width + escalation (WP-B.3)

def test_containment_differs_between_corridor_and_danger_width():
    """The avoid_panoptex_2 geometry: the robot 0.65-0.70 m off the lane
    centre. The nominal 0.55 m corridor says "not in the way" and no refuge
    is triggered; the 0.90 m danger width says otherwise, and it is the one
    that was right (ground-truth gap at the encounter: 0.15 m)."""
    p = params()
    c = make_corridor(track(0.0, 0.0, 1.0, 0.0), p)      # lane along +x
    on_the_edge = (2.0, 0.7)

    assert c.half_width == pytest.approx(0.55)
    assert not corridor_contains(c, on_the_edge)
    assert corridor_contains(c, on_the_edge, half_width=p["danger_half_width_m"])

    danger = danger_corridor(c, p)
    assert danger.half_width == pytest.approx(0.90)
    assert corridor_contains(danger, on_the_edge)
    # Beyond the danger width it really is clear.
    assert not corridor_contains(danger, (2.0, 0.95))
    # Everything else about the lane is untouched -- same line, same length,
    # so gap acceptance is unaffected by the widening.
    assert (danger.ux, danger.uy, danger.length_ahead) == \
        (c.ux, c.uy, c.length_ahead)
    assert along(danger, on_the_edge) == pytest.approx(along(c, on_the_edge))
    # A corridor already wider than the danger width keeps its own.
    wide = make_corridor(track(0.0, 0.0, 1.0, 0.0),
                         params(corridor_half_width_m=1.5))
    assert danger_corridor(wide, p).half_width == pytest.approx(1.5)


def test_hold_in_place_is_unsafe_inside_danger_or_band():
    """The escalation predicate: standing still is a yield only when where
    you stand is out of the traffic."""
    p = params()
    c = make_corridor(track(1.32, 3.0, 0.0, -0.6), p)     # lane down x = 1.32
    danger = danger_corridor(c, p)
    robot = (1.97, -3.0)                                  # the run's stop point

    # 0.65 m off the centre line: outside the nominal corridor...
    assert not corridor_contains(c, robot)
    # ...but inside the danger corridor, so holding here is not a yield.
    assert hold_in_place_is_unsafe(danger, robot)
    # Genuinely clear of the lane: holding in place is fine.
    assert not hold_in_place_is_unsafe(danger, (3.0, -3.0))
    # ...unless the static band says that spot is somebody else's lane.
    prior, pinfo = _prior([50])                           # lane at x = 5.0
    band = make_lane_band(prior, pinfo, p["lane_band_min_value"],
                          p["danger_half_width_m"])
    assert hold_in_place_is_unsafe(danger, (5.0, 3.0), band)
    assert not hold_in_place_is_unsafe(danger, (3.0, 3.0), band)
    assert not hold_in_place_is_unsafe(None, (3.0, 3.0), None)


def test_find_refuge_prefers_clearance_from_the_users_line():
    """A cell can be outside the corridor rectangle (past its back margin)
    and still sit on the user's line -- a user that stops short, reverses,
    or is tracked 0.3 m off puts it straight back in the lane. With
    line_clearance_m set, a farther cell off the line beats a nearer one on
    it."""
    p = params()
    grid = _free_grid()
    info = _map_info()
    # User driving +y along x = 5.0, currently at y = 5.0: the corridor runs
    # from y = 4.0 (back margin) to y = 15. Cells below y = 4.0 are outside
    # it but still on its line.
    c = danger_corridor(make_corridor(track(5.0, 5.0, 0.0, 1.0), p), p)
    robot = (5.0, 3.6)                     # just behind the back margin

    near = find_refuge(grid, info, robot, [c], radius=2.5, clearance=0.45,
                       prefer_side=False, line_clearance_m=0.0)
    assert near is not None
    assert abs(float(lateral(c, near))) < p["danger_half_width_m"]

    off_line = find_refuge(grid, info, robot, [c], radius=2.5,
                           clearance=0.45, prefer_side=False,
                           line_clearance_m=p["danger_half_width_m"])
    assert off_line is not None
    assert abs(float(lateral(c, off_line))) >= p["danger_half_width_m"]
    # It costs distance, and that is the point.
    assert math.hypot(off_line[0] - robot[0], off_line[1] - robot[1]) > \
        math.hypot(near[0] - robot[0], near[1] - robot[1])


# =========================== 10. crossing policy: coverage + headway (WP4)
#
# The policy's whole job is to stop the supervisor deciding "the lane is
# clear" from an absence of tracks in a strip no sensor was looking at. Its
# safety property is an INVARIANT rather than a threshold: it can only ever
# turn a "go" into a "hold" (test_crossing_policy_never_overrides_a_gap_hold
# below), so every one of these branches is a superset of the pre-WP4
# behaviour that the four avoid_panoptex_* runs exercised.

COV_RES = 0.10
COV_OX, COV_OY = -5.0, -15.0
COV_ROWS, COV_COLS = 200, 100          # x in [-5, 5), y in [-15, 5)


def _coverage_grid(fill=100):
    return {"values": np.full((COV_ROWS, COV_COLS), fill, dtype=np.int16),
            "resolution": COV_RES, "origin_x": COV_OX, "origin_y": COV_OY}


def _cov_row(y):
    return int((y - COV_OY) / COV_RES)


# ------------------------------------------------------ 10a. approach_zone

def test_approach_zone_is_the_strip_upstream_of_the_crossing():
    """Traffic runs +y through the crossing at the origin at 1 m/s; we need
    10 s to clear plus 2 s of margin, so anything within 12 m UPSTREAM
    (-y) could reach us in time and nothing beyond it can."""
    zone = approach_zone((0.0, 0.0), (0.0, 1.0), v_lane=1.0, t_clear_s=10.0,
                         t_margin_s=2.0, half_width=0.9)
    assert zone.shape == (4, 2)

    # Upstream, on the centre line and at the lateral edge: inside.
    assert polygon_contains(zone, (0.0, -6.0))
    assert polygon_contains(zone, (0.85, -6.0))
    assert polygon_contains(zone, (-0.85, -11.9))
    # Downstream of the crossing (the user has passed): outside.
    assert not polygon_contains(zone, (0.0, 1.0))
    # Beyond the 12 m reach: outside -- it cannot arrive in time.
    assert not polygon_contains(zone, (0.0, -12.5))
    # Wider than half_width: outside.
    assert not polygon_contains(zone, (1.1, -6.0))

    # Vectorised, same answers.
    pts = np.array([[0.0, -6.0], [0.0, 1.0], [0.0, -12.5]])
    assert list(polygon_contains(zone, pts)) == [True, False, False]

    # The length really is v_lane * (t_clear + t_margin): halve the speed,
    # halve the reach.
    slow = approach_zone((0.0, 0.0), (0.0, 1.0), 0.5, 10.0, 2.0, 0.9)
    assert polygon_contains(slow, (0.0, -5.9))
    assert not polygon_contains(slow, (0.0, -6.1))

    # A stationary lane has no reach at all, and must not crash.
    dead = approach_zone((0.0, 0.0), (0.0, 1.0), 0.0, 10.0, 2.0, 0.9)
    assert not polygon_contains(dead, (0.0, -0.5))


def test_lane_speed_floors_and_caps_but_never_below_the_tracked_user():
    p = params()
    # No flow evidence at all -> the floor, not zero.
    assert lane_speed(None, p) == pytest.approx(p["v_lane_min_mps"])
    # s = 0 is "no data", not "observed stationary" -> the floor again.
    assert lane_speed((0.0, 3.0, 0.0), p) == pytest.approx(p["v_lane_min_mps"])
    # A learned 0.6 m/s lane is used as-is.
    assert lane_speed((0.8, 0.0, 0.6), p) == pytest.approx(0.6)
    # ...and a wild EMA cell is capped.
    assert lane_speed((0.8, 9.0, 0.0), p) == pytest.approx(p["v_lane_max_mps"])
    # The tracked user's own speed is a floor the cap does NOT apply to:
    # the zone must always be long enough to hold the vehicle it is about.
    assert lane_speed((0.8, 0.0, 0.6), p, observed_speed=2.5) == \
        pytest.approx(2.5)


# ----------------------------------------------------- 10b. zone coverage

def test_zone_coverage_is_the_fraction_of_covered_cells():
    zone = approach_zone((0.0, 0.0), (0.0, 1.0), 1.0, 10.0, 2.0, 0.9)

    assert zone_coverage(zone, _coverage_grid(100)) == pytest.approx(1.0)
    assert zone_coverage(zone, _coverage_grid(0)) == pytest.approx(0.0)

    # Half the 12 m strip covered (y >= -6) -> ~0.5.
    half = _coverage_grid(0)
    half["values"][_cov_row(-6.0):, :] = 100
    assert zone_coverage(zone, half) == pytest.approx(0.5, abs=0.02)

    # No exposure map, and a zone off the edge of one: 0, never "assume
    # covered" -- an exposure map that failed open would be worse than none.
    assert zone_coverage(zone, None) == pytest.approx(0.0)
    far = approach_zone((100.0, 100.0), (0.0, 1.0), 1.0, 10.0, 2.0, 0.9)
    assert zone_coverage(far, _coverage_grid(100)) == pytest.approx(0.0)


def test_zone_users_finds_only_tracks_inside_the_strip():
    zone = approach_zone((0.0, 0.0), (0.0, 1.0), 1.0, 10.0, 2.0, 0.9)
    tracks = [track(0.0, -6.0, 0.0, 1.0, track_id="inside"),
              track(0.0, -12.5, 0.0, 1.0, track_id="too_far"),
              track(3.0, -6.0, 0.0, 1.0, track_id="other_lane"),
              track(0.0, 2.0, 0.0, 1.0, track_id="already_past")]
    assert [t["id"] for t in zone_users(zone, tracks)] == ["inside"]
    assert zone_users(zone, []) == []


# ------------------------------------------------- 10c. the blind-crossing rule

def test_blind_crossing_waits_one_learned_headway():
    p = params()                       # min_count 3, factor 1.0, max wait 15
    ok = dict(headway_mean=10.0, headway_count=5.0, visible_zone_empty=True,
              t_arrived=0.0, params=p)

    # 9 s since the last pass, mean gap 10 s: not yet.
    assert not blind_crossing_ok(9.0, 0.0, **ok)
    # 10 s: a full typical gap has elapsed -> cross.
    assert blind_crossing_ok(10.0, 0.0, **ok)
    # ...and doubling the required factor pushes it back out again.
    assert not blind_crossing_ok(10.0, 0.0, headway_mean=10.0,
                                 headway_count=5.0, visible_zone_empty=True,
                                 t_arrived=0.0,
                                 params=params(headway_factor=2.0))


def test_blind_crossing_never_overrules_something_we_can_see():
    p = params()
    assert not blind_crossing_ok(1000.0, 0.0, headway_mean=10.0,
                                 headway_count=99.0,
                                 visible_zone_empty=False, t_arrived=0.0,
                                 params=p)


def test_blind_crossing_falls_back_to_a_bounded_wait_without_evidence():
    """Cold prior (or a lane nothing was ever seen using): there is no
    statistic to wait for, so wait blind_wait_max_s and then go."""
    p = params()
    thin = dict(headway_mean=0.0, headway_count=1.0, visible_zone_empty=True,
                params=p)
    assert not blind_crossing_ok(5.0, 0.0, t_arrived=0.0, **thin)
    assert not blind_crossing_ok(14.9, 0.0, t_arrived=0.0, **thin)
    assert blind_crossing_ok(15.0, 0.0, t_arrived=0.0, **thin)
    # Not waiting anywhere yet -> the clock has not started.
    assert not blind_crossing_ok(1000.0, 0.0, t_arrived=-1.0, **thin)

    # Plenty of samples but no known last pass: the difference would be
    # meaningless, so it takes the bounded wait too.
    unknown = dict(headway_mean=10.0, headway_count=99.0,
                   visible_zone_empty=True, params=p)
    assert not blind_crossing_ok(14.0, NO_PASS, t_arrived=0.0, **unknown)
    assert blind_crossing_ok(15.0, NO_PASS, t_arrived=0.0, **unknown)


# --------------------------------------------------- 10d. choose where to cross

def test_crossing_choice_prefers_the_quietest_crossing():
    busy = {"xy_entry": np.array([1.35, 0.0]),
            "xy_exit": np.array([1.35, 1.0])}
    quiet = {"xy_entry": np.array([6.0, 0.0]),
             "xy_exit": np.array([6.0, 1.0])}
    prior, pinfo = _prior([13])                  # lane at x in [1.3, 1.4)
    s_grid = dict(pinfo, values=prior)

    assert crossing_choice([busy, quiet], s_grid) is quiet
    assert crossing_choice([quiet, busy], s_grid) is quiet
    # No prior, or a single candidate: the first (= nearest along the plan),
    # which is the only one the robot can act on next.
    assert crossing_choice([busy, quiet], None) is busy
    assert crossing_choice([busy], s_grid) is busy
    assert crossing_choice([], s_grid) is None


# --------------------------------------------- 10e. the policy, end to end

def _policy_case(user_y, t_corridor=10.0):
    """The gap-acceptance geometry of section 4, reused: user approaching
    the origin from -y at 0.6 m/s, plan crossing west->east through it, lane
    entry 1.0 m along the plan (so t_clear = 12 s)."""
    p = params(t_corridor_sec=t_corridor)
    tr = track(0.0, user_y, 0.0, 0.6, track_id="carter1")
    c = make_corridor(tr, p)
    plan = straight_plan(-1.55, 0.0, 1.6, 0.0, step=0.1)
    info = crossing_decision(c, plan, p)
    return c, info, [tr], p


def test_crossing_policy_without_a_coverage_map_is_the_old_behaviour():
    """The degradation path that makes coverage_mask_node optional rather
    than load-bearing: no exposure map, no new branches, same answer."""
    c, info, tracks, p = _policy_case(user_y=-6.0)
    out = crossing_policy(info, c, p)
    assert out["decision"] == info["decision"] == "hold"
    assert out["mode"] == "gap"
    assert out["zone"] is None and out["zone_cov"] is None

    c, info, tracks, p = _policy_case(user_y=-12.0, t_corridor=25.0)
    assert crossing_policy(info, c, p)["decision"] == "go"


def test_crossing_policy_crosses_when_the_zone_is_covered_and_empty():
    """tta 20 s against a 12 s clearance, and we can SEE the whole strip a
    user would have to come from: cross, and say why."""
    c, info, tracks, p = _policy_case(user_y=-12.0, t_corridor=25.0)
    out = crossing_policy(info, c, p, coverage_grid=_coverage_grid(100),
                          tracks=tracks, now=100.0)
    assert out["mode"] == "seen"
    assert out["decision"] == "go"
    assert out["zone_cov"] == pytest.approx(1.0)
    assert out["zone_users"] == []
    # The zone was sized by the tracked user's own speed, not the floor.
    assert out["v_lane"] == pytest.approx(0.6)


def test_crossing_policy_falls_back_to_gap_acceptance_with_a_user_in_view():
    c, info, tracks, p = _policy_case(user_y=-6.0)
    out = crossing_policy(info, c, p, coverage_grid=_coverage_grid(100),
                          tracks=tracks, now=100.0)
    assert out["mode"] == "gap"
    assert out["zone_users"] == ["carter1"]
    assert out["decision"] == "hold"            # tta 10 < t_clear 12 + 2


def test_crossing_policy_holds_blind_and_then_crosses_on_the_headway():
    """The case every previous run got wrong: the timing test says "go"
    because nothing is tracked in the strip -- but nothing can be tracked
    there, because no sensor covers it."""
    c, info, tracks, p = _policy_case(user_y=-12.0, t_corridor=25.0)
    blind = dict(coverage_grid=_coverage_grid(0), tracks=tracks)
    assert info["decision"] == "go"             # the old answer

    # No statistic yet and the wait has not started: hold.
    first = crossing_policy(info, c, p, now=100.0, t_arrived=-1.0, **blind)
    assert first["mode"] == "blind" and first["decision"] == "hold"
    assert first["zone_cov"] == pytest.approx(0.0)

    # With a learned headway of 10 s and the last observed pass 12 s ago,
    # more than a typical gap has elapsed: cross.
    later = crossing_policy(info, c, p, now=100.0, last_pass_time=88.0,
                            headway_mean=10.0, headway_count=5.0,
                            t_arrived=95.0, **blind)
    assert later["mode"] == "blind" and later["decision"] == "go"
    assert later["wait_s"] == pytest.approx(5.0)

    # Same statistic, but only 4 s since the last pass: keep waiting.
    early = crossing_policy(info, c, p, now=100.0, last_pass_time=96.0,
                            headway_mean=10.0, headway_count=5.0,
                            t_arrived=95.0, **blind)
    assert early["decision"] == "hold"

    # ...and the bounded wait eventually releases it even with no statistic.
    timed_out = crossing_policy(info, c, p, now=100.0, t_arrived=84.0,
                                **blind)
    assert timed_out["decision"] == "go"


def test_crossing_policy_never_overrides_a_gap_hold():
    """THE INVARIANT. The user is 6 m out (tta 10 s < t_clear 12 s + 2 s),
    the zone is uncovered so nothing in it is *visible*, and the headway
    statistic would happily wave us through -- and the answer is still
    hold, because the timing test can see this one and said no."""
    c, info, tracks, p = _policy_case(user_y=-6.0)
    assert info["decision"] == "hold"
    out = crossing_policy(info, c, p, coverage_grid=_coverage_grid(0),
                          tracks=tracks, now=100.0, last_pass_time=0.0,
                          headway_mean=1.0, headway_count=99.0,
                          t_arrived=0.0)
    assert out["mode"] == "blind"
    assert out["decision"] == "hold"


def test_crossing_policy_will_not_cross_past_a_user_it_can_see():
    """Uncovered zone overall, but the user itself sits in a covered cell:
    the visible part of the zone is not empty, so no blind crossing."""
    c, info, tracks, p = _policy_case(user_y=-6.0)
    patchy = _coverage_grid(0)
    # Cover just the two rows around the user at y = -6.
    patchy["values"][_cov_row(-6.2):_cov_row(-5.8), :] = 100
    out = crossing_policy(info, c, p, coverage_grid=patchy, tracks=tracks,
                          now=1000.0, last_pass_time=0.0, headway_mean=1.0,
                          headway_count=99.0, t_arrived=0.0)
    assert out["mode"] == "blind"
    assert out["zone_users"] == ["carter1"]
    assert out["decision"] == "hold"


# ============ 11. remembered lane lines in the refuge search (2026-09-09)
#
# mppi_panoptex_2: 7 contact episodes, carter1 tracked correctly and yielded
# to correctly in every one of them, and 10 of the run's 25 refuge targets
# still landed within 0.6 m of its permanent centre line (one at 0.01 m). A
# corridor is the INSTANTANEOUS window -- back_margin_m behind the track,
# speed * t_corridor_sec ahead of it -- so a cell on the line the Carter
# patrols but several metres behind it is outside every corridor and looks
# like a perfectly good refuge, right up until the window comes back.
#
# The fix is a memory of the LINE rather than the window; see
# test_observed_lanes.py for the memory itself. These are the refuge search's
# half of it. The lane here runs x = 5.0 from y = 1 to y = 9, and the user is
# up at the far end of it -- exactly the case the window misses.


def _patrol_lane(y0=1.0, y1=9.0, x=5.0, now=0.0):
    return make_observed_lane((x, y0), (x, y1), (0.0, 1.0), now, "carter1")


def test_find_refuge_rejects_a_cell_on_a_lane_the_window_has_moved_off():
    # The 1.0 m clearance is now the explicit OVERRIDE (the default derives
    # 0.75 m from the geometry -- see test_refuge_side.py); this test's
    # floor plan is drawn for 1.0, so it sets it and doubles as override
    # coverage.
    p = params(refuge_lane_clearance_m=1.0)
    grid = _free_grid()
    info = _map_info()
    lane = _patrol_lane()
    robot = (5.0, 3.0)

    # The Carter is at y = 8 heading north: its corridor runs y = 7 -> 13,
    # so nothing anywhere near the robot is inside it.
    c = danger_corridor(make_corridor(track(5.0, 8.0, 0.0, 0.6), p), p)
    assert not corridor_contains(c, robot)

    # OLD behaviour: the nearest legal cell is the robot's own, dead on the
    # patrol line. This is the bug, reproduced.
    old = find_refuge(grid, info, robot, [c], radius=2.5, clearance=0.45,
                      prefer_side=False)
    assert old is not None
    assert lane_line_clearance(old, [lane], p["lane_extension_m"]) < 0.6

    # NEW: the line is remembered, so no cell within refuge_lane_clearance_m
    # of it may be taken -- and a cell 1.2 m off the lane is available.
    new = find_refuge(grid, info, robot, [c], radius=2.5, clearance=0.45,
                      prefer_side=False, lanes=[lane],
                      lane_clearance_m=p["refuge_lane_clearance_m"],
                      lane_extension_m=p["lane_extension_m"])
    assert new is not None
    gap = lane_line_clearance(new, [lane], p["lane_extension_m"])
    assert gap >= p["refuge_lane_clearance_m"]
    assert math.hypot(new[0] - robot[0], new[1] - robot[1]) <= 2.5
    # It costs distance, and that is the whole point.
    assert math.hypot(new[0] - robot[0], new[1] - robot[1]) > \
        math.hypot(old[0] - robot[0], old[1] - robot[1])


def test_find_refuge_accepts_a_cell_beyond_the_lane_clearance():
    """The complement of the test above, and the shape of the ranking.

    1.0 m off the line is legal, but "legal" is only the floor: the binned
    clearance preference outranks distance up to refuge_lane_clearance_m +
    1 m, so with a free disc the answer walks all the way out to that cap
    rather than stopping at the first cell that qualifies. Standing 2 m off
    a patrol line costs one extra metre of driving and is worth it.

    (1.0 m here is the explicit override; the derived default is 0.75 m.)
    """
    p = params(refuge_lane_clearance_m=1.0)
    lane = _patrol_lane()
    robot = (5.0, 3.0)
    xy = find_refuge(_free_grid(), _map_info(), robot, [],
                     radius=2.5, clearance=0.45, prefer_side=False,
                     lanes=[lane],
                     lane_clearance_m=p["refuge_lane_clearance_m"],
                     lane_extension_m=p["lane_extension_m"])
    assert xy is not None
    gap = float(lane_line_clearance(xy, [lane], p["lane_extension_m"]))
    assert gap >= p["refuge_lane_clearance_m"] + 1.0      # the cap
    assert math.hypot(xy[0] - robot[0], xy[1] - robot[1]) <= 2.5
    # Perpendicular escape: the shortest way off a line is across it.
    assert abs(float(xy[1]) - 3.0) < 0.2


def test_find_refuge_needs_the_wider_disc_when_lanes_fill_the_narrow_one():
    """Three parallel lanes 1.8 m apart put every cell of the 2.5 m disc
    within 0.9 m of one of them, so stage 1 has no answer at all. That is
    what refuge_radius_max_m is for: "nowhere to go" has to mean "nowhere
    within 4 m", not "nowhere within 2.5", before the supervisor gives up
    and stands still.

    1.0 m is the explicit override -- three lanes 1.8 m apart leave 0.9 m
    of legal floor between them, which the derived 0.75 m default would
    happily accept.
    """
    p = params(refuge_lane_clearance_m=1.0)
    grid = _free_grid()
    info = _map_info()
    lanes = [_patrol_lane(x=x) for x in (3.2, 5.0, 6.8)]
    robot = (5.0, 3.0)
    kwargs = dict(clearance=0.45, prefer_side=False, lanes=lanes,
                  lane_clearance_m=p["refuge_lane_clearance_m"],
                  lane_extension_m=p["lane_extension_m"])

    assert find_refuge(grid, info, robot, [],
                       radius=p["refuge_radius_m"], **kwargs) is None

    wide = find_refuge(grid, info, robot, [],
                       radius=p["refuge_radius_max_m"], **kwargs)
    assert wide is not None
    assert lane_line_clearance(wide, lanes, p["lane_extension_m"]) >= \
        p["refuge_lane_clearance_m"]
    assert 2.5 < math.hypot(wide[0] - robot[0], wide[1] - robot[1]) <= 4.0


def test_find_refuge_prefers_more_lane_clearance_over_a_nearer_cell():
    """Ranking, not rejection: the search must favour a cell that stands
    further off the line over the first one that merely clears the hard
    minimum. The preference is BINNED (0.5 m), so what is guaranteed is the
    top bin below the cap (0.4 + 1.0 -> cells at or past 1.0 m), not an
    exact distance -- coarse on purpose, so a stray centimetre of clearance
    cannot buy an arbitrary detour."""
    p = params()
    lane = _patrol_lane()
    xy = find_refuge(_free_grid(), _map_info(), (5.0, 3.0), [],
                     radius=2.5, clearance=0.45, prefer_side=False,
                     lanes=[lane], lane_clearance_m=0.4,
                     lane_extension_m=p["lane_extension_m"])
    assert xy is not None
    assert float(lane_line_clearance(xy, [lane],
                                     p["lane_extension_m"])) >= 1.0


def test_find_refuge_without_lanes_is_unchanged():
    """Every pre-2026-09-09 call site passes no lanes, and must get exactly
    the answer it always did."""
    p = params()
    grid = _free_grid()
    info = _map_info()
    c = danger_corridor(make_corridor(track(5.0, 1.0, 0.0, 1.0), p), p)
    before = find_refuge(grid, info, (5.2, 5.0), [c], radius=2.5,
                         clearance=0.45, prefer_side=True)
    after = find_refuge(grid, info, (5.2, 5.0), [c], radius=2.5,
                        clearance=0.45, prefer_side=True, lanes=[],
                        lane_clearance_m=p["refuge_lane_clearance_m"],
                        lane_extension_m=p["lane_extension_m"])
    assert before is not None
    assert np.allclose(before, after)
