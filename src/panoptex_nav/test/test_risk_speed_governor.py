"""
Pure-policy tests for panoptex_nav.risk_speed_governor.compute_cap -- no
rclpy spin, no ROS graph, same rationale as risk_perception's
test_encounter_geometry.py (see that module's docstring): this logic is
plain Python data in / data out, so it is tested directly.

Run via `colcon test --packages-select panoptex_nav` (wired into
CMakeLists.txt through ament_cmake_pytest) or directly with
`python3 -m pytest test/test_risk_speed_governor.py` from this package's
root, provided panoptex_nav/ is importable (either PYTHONPATH pointing at
this package's source root, or the workspace already built/sourced).
"""

import copy

from panoptex_nav.risk_speed_governor import DEFAULT_PARAMS, compute_cap

ORIGIN = (0.0, 0.0)
STOPPED = (0.0, 0.0)


def params(**overrides):
    """DEFAULT_PARAMS with a few keys overridden for one test."""
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update(overrides)
    return p


def person_track(distance, score=0.9, track_id="p1"):
    """A person track sitting `distance` meters east of the origin,
    stationary (person policy is distance-only -- velocity is irrelevant)."""
    return {
        "id": track_id, "label": "person", "category": "person",
        "score": score, "age_sec": 0.0,
        "x": distance, "y": 0.0, "vx": 0.0, "vy": 0.0, "pmot": 0.0,
    }


def robot_track(x, y, vx, vy, pmot, score=0.9, track_id="r1",
                category="robot"):
    return {
        "id": track_id, "label": category, "category": category,
        "score": score, "age_sec": 0.0,
        "x": x, "y": y, "vx": vx, "vy": vy, "pmot": pmot,
    }


# --------------------------------------------------------------- person

def test_person_at_1_5m_gets_slow_cap():
    pct, reason = compute_cap([person_track(1.5)], ORIGIN, STOPPED,
                               params())
    assert pct == 40.0
    assert reason["reason"] == "person_slow"
    assert reason["category"] == "person"


def test_person_at_0_8m_gets_crawl_cap():
    pct, reason = compute_cap([person_track(0.8)], ORIGIN, STOPPED,
                               params())
    assert pct == 15.0
    assert reason["reason"] == "person_crawl"


def test_person_beyond_slow_radius_is_uncapped():
    pct, reason = compute_cap([person_track(2.5)], ORIGIN, STOPPED,
                               params())
    assert pct == 0.0
    assert reason is None


# --------------------------------------------------------- robot / wheeled

def test_robot_closing_gets_closing_cap():
    """Robot at (3, 0) moving at (-1, 0) m/s -- straight at a stationary
    robot at the origin. t_cpa = 3s (<= robot_ttc_s), d_cpa = 0 (<=
    robot_cpa_m) -> robot_closing_pct."""
    tracks = [robot_track(x=3.0, y=0.0, vx=-1.0, vy=0.0, pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 30.0
    assert reason["reason"] == "closing"
    assert abs(reason["t_cpa_s"] - 3.0) < 1e-9
    assert abs(reason["d_cpa_m"] - 0.0) < 1e-9


def test_robot_diverging_is_uncapped():
    """Same position, moving AWAY -- t_cpa <= 0, must not cap."""
    tracks = [robot_track(x=3.0, y=0.0, vx=1.0, vy=0.0, pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_stationary_robot_within_radius_gets_static_cap():
    """pmot below moving_pmot_min -- distance-gated, not CPA-gated."""
    tracks = [robot_track(x=0.8, y=0.0, vx=0.0, vy=0.0, pmot=0.0)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 50.0
    assert reason["reason"] == "static_slow"


def test_stationary_robot_beyond_radius_is_uncapped():
    tracks = [robot_track(x=1.5, y=0.0, vx=0.0, vy=0.0, pmot=0.0)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_wheeled_category_uses_same_policy_as_robot():
    tracks = [robot_track(x=3.0, y=0.0, vx=-1.0, vy=0.0, pmot=0.9,
                          category="wheeled")]
    pct, _ = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 30.0


# ---------------------------------------------------------- gating / misc

def test_below_score_track_is_ignored():
    tracks = [person_track(0.8, score=0.05)]  # would be crawl, but score fails
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_stale_track_is_ignored():
    tr = person_track(0.8)
    tr["age_sec"] = 10.0
    pct, reason = compute_cap([tr], ORIGIN, STOPPED,
                              params(max_track_age_sec=3.0))
    assert pct == 0.0
    assert reason is None


def test_furniture_and_unknown_are_ignored():
    tracks = [
        {"id": "f1", "label": "chair", "category": "furniture",
         "score": 0.9, "age_sec": 0.0, "x": 0.1, "y": 0.0,
         "vx": 0.0, "vy": 0.0, "pmot": 0.0},
        {"id": "u1", "label": "mystery", "category": "unknown",
         "score": 0.9, "age_sec": 0.0, "x": 0.1, "y": 0.0,
         "vx": 0.0, "vy": 0.0, "pmot": 0.0},
    ]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_multiple_tracks_take_the_minimum():
    """A crawl-range person (15%) and a closing robot (30%) at once ->
    the more restrictive 15% wins."""
    tracks = [person_track(0.8), robot_track(x=3.0, y=0.0, vx=-1.0, vy=0.0,
                                             pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 15.0
    assert reason["category"] == "person"


def test_floor_is_applied_when_a_cap_would_go_below_min_pct():
    """A cap that would otherwise compute below min_pct is raised to the
    floor, not published as-is."""
    tracks = [person_track(0.8)]
    pct, reason = compute_cap(
        tracks, ORIGIN, STOPPED,
        params(person_crawl_pct=5.0, min_pct=15.0))
    assert pct == 15.0
    assert reason["pct"] == 15.0


def test_moving_robot_outside_ttc_window_is_uncapped():
    """Closing, but t_cpa beyond robot_ttc_s -- too far out in time to cap."""
    tracks = [robot_track(x=10.0, y=0.0, vx=-1.0, vy=0.0, pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_moving_robot_outside_cpa_radius_is_uncapped():
    """Closing in time, but the miss distance is too wide."""
    tracks = [robot_track(x=3.0, y=2.0, vx=-1.0, vy=0.0, pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_robot_velocity_is_relative_to_the_moving_robot():
    """A track moving at the SAME velocity as the robot (parallel, no
    closing) must not cap even though it is nearby and pmot is high."""
    tracks = [robot_track(x=1.0, y=0.0, vx=1.0, vy=0.0, pmot=0.9)]
    pct, reason = compute_cap(tracks, ORIGIN, robot_v_map=(1.0, 0.0),
                              params=params())
    assert pct == 0.0
    assert reason is None


# ---------------------------------------------------- WP-A: any-mover policy

def other_track(x, y, vx, vy, pmot, category, label=None, score=0.9,
                track_id="o1"):
    return {
        "id": track_id, "label": label or category, "category": category,
        "score": score, "age_sec": 0.0,
        "x": x, "y": y, "vx": vx, "vy": vy, "pmot": pmot,
    }


def test_moving_table_triggers_closing_cap_under_any_mover_default():
    """A 'table' (category 'furniture', or a tracker-promoted 'wheeled'
    label 'moving object') closing on the robot at pmot 0.9 must clear the
    SAME CPA-gated closing branch a 'robot'/'wheeled' track does --
    act_on_any_mover defaults to True. Geometry mirrors
    test_robot_closing_gets_closing_cap."""
    tracks = [other_track(x=3.0, y=0.0, vx=-1.0, vy=0.0, pmot=0.9,
                          category="furniture", label="table")]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 30.0
    assert reason["reason"] == "closing"
    assert reason["category"] == "furniture"


def test_static_chair_gets_no_cap_under_any_mover_default():
    """A STATIONARY 'chair' (pmot well below moving_pmot_min) must NOT get
    static_slow even under act_on_any_mover -- that cap stays restricted to
    robot/wheeled (Nav2's own obstacle layer handles static furniture)."""
    tracks = [other_track(x=0.5, y=0.0, vx=0.0, vy=0.0, pmot=0.0,
                          category="furniture", label="chair")]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 0.0
    assert reason is None


def test_moving_table_ignored_when_any_mover_disabled():
    """act_on_any_mover: false restores the pre-2026-09-11 behaviour -- a
    non robot/wheeled category is dropped outright regardless of motion."""
    tracks = [other_track(x=3.0, y=0.0, vx=-1.0, vy=0.0, pmot=0.9,
                          category="furniture", label="table")]
    pct, reason = compute_cap(
        tracks, ORIGIN, STOPPED, params(act_on_any_mover=False))
    assert pct == 0.0
    assert reason is None


def test_promoted_wheeled_table_gets_static_slow_when_parked():
    """A tracker-promoted track (Track.update_promotion relabels category
    to 'wheeled', label 'moving object') that is currently NOT convincingly
    moving still gets static_slow, exactly like an ordinary parked cart --
    promotion already earned it the robot/wheeled category."""
    tracks = [other_track(x=0.5, y=0.0, vx=0.0, vy=0.0, pmot=0.0,
                          category="wheeled", label="moving object")]
    pct, reason = compute_cap(tracks, ORIGIN, STOPPED, params())
    assert pct == 50.0
    assert reason["reason"] == "static_slow"
