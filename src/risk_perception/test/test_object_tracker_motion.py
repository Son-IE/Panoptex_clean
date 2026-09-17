"""
Regression tests for the two tracker changes made on 2026-09-08 after the
first Panoptex-in-nav2 sim run, where one patrolling Nova Carter (0.6 m/s,
seen by the overhead cameras at ~1 Hz) produced 44 distinct tracks and never
once had p_motion > 0:

1. Track.update_beliefs: the Mahalanobis-only motion test cannot fire when
   the velocity sigma stays ~0.5-0.9 m/s (sparse updates, process_noise
   0.5). The absolute speed test (motion_speed_mps / motion_speed_score) can.
2. ObjectTrackerNode association: category-level identity + a gate that
   widens with time-since-last-update keep a moving object on ONE track
   despite GroundingDINO phrase churn and >0.6 m displacement between
   sightings. Exercised without rclpy.init() by binding the node's pure
   helper methods to a stub carrying the same attributes.

Also covers the 2026-09-10 finding (see object_tracker_node.py's module
docstring): with real perception, a label-agnostic lidar_cluster track
spawned off a shelf edge (static margin only, no cross-scan persistence)
re-associates with a DIFFERENT shelf edge between scans, and the Kalman
filter reads that jump as several m/s of velocity -- Track.enforce_speed_cap
(speed plausibility) and risk_visualization.parse_class_id's hits/age
round-trip (track maturity) are the two fixes tested below.
"""
import math
import types

import numpy as np
import pytest

from risk_perception.object_tracker_node import ObjectTrackerNode, Track
from risk_perception.risk_visualization import parse_class_id


def drive_track(object_speed, hz, seconds, q=0.5, cov=0.04,
                require_displacement=False, **belief_kw):
    """Constant-velocity object observed at `hz`, beliefs ticked at 10 Hz.

    require_displacement defaults False here (pre-WP1 legacy composition)
    so the Mahalanobis/absolute-speed tests this helper exercises keep
    deciding `moving` on their own -- see
    test_mahalanobis_only_never_fires_for_sparse_amr's docstring note."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=cov,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    next_obs = 1.0 / hz
    t = 0.0
    while t < seconds:
        t += dt_tick
        tr.predict(dt_tick, q)
        if t + 1e-9 >= next_obs:
            tr.update(object_speed * t, 0.0, cov, 0.9, 0.5, t)
            next_obs += 1.0 / hz
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=require_displacement, **belief_kw)
    return tr


def test_mahalanobis_only_never_fires_for_sparse_amr():
    """require_displacement=False (drive_track's default): the WP1
    displacement-evidence test is disabled so this exercises the legacy
    Mahalanobis-only composition in isolation, same as before WP1 landed."""
    tr = drive_track(0.6, hz=1.0, seconds=8.0)  # legacy: speed_mps=0.0
    assert tr.p_motion == 0.0
    assert math.hypot(*tr.velocity) > 0.4  # velocity IS estimated...
    assert tr.motion_score < 3.0           # ...but never "significant"


def test_speed_test_fires_for_sparse_amr():
    tr = drive_track(0.6, hz=1.0, seconds=8.0, speed_mps=0.25, speed_score=1.0)
    assert tr.p_motion == 1.0
    assert tr.p_movable > 0.9


def test_speed_test_does_not_fire_for_static_object():
    tr = drive_track(0.0, hz=1.0, seconds=8.0, speed_mps=0.25, speed_score=1.0)
    assert tr.p_motion == 0.0


def test_displacement_evidence_fires_for_sparse_amr_by_default():
    """WP1 (2026-09-10): with the new default (require_displacement=True),
    a genuinely moving, sparsely-observed AMR is still detected as moving
    -- via net displacement over the window, not the Mahalanobis score."""
    tr = drive_track(0.6, hz=1.0, seconds=8.0, require_displacement=True)
    assert tr.p_motion == 1.0
    assert tr.p_movable > 0.9


def test_displacement_evidence_does_not_fire_for_static_object():
    tr = drive_track(0.0, hz=1.0, seconds=8.0, require_displacement=True)
    assert tr.p_motion == 0.0


def make_stub(association_key="category", gate=0.6, gate_speed=1.0, gate_max=2.5,
              association_groups=("robot,wheeled",)):
    stub = types.SimpleNamespace(
        association_key=association_key, gate=gate, gate_speed=gate_speed,
        gate_max_m=gate_max,
        association_groups=[
            frozenset(s.strip().lower() for s in g.split(",") if s.strip())
            for g in association_groups
        ])
    stub._same_object_class = types.MethodType(
        ObjectTrackerNode._same_object_class, stub)
    stub._in_same_group = types.MethodType(ObjectTrackerNode._in_same_group, stub)
    stub._gate_for = types.MethodType(ObjectTrackerNode._gate_for, stub)
    return stub


def test_category_association_matches_phrase_churn():
    stub = make_stub()
    assert stub._same_object_class("mobile robot", "ground mobile robot")
    assert stub._same_object_class("cart", "forklift")          # both wheeled
    assert not stub._same_object_class("mobile robot", "person")
    assert not stub._same_object_class("banana", "cable")        # unknown: exact only
    assert stub._same_object_class("banana", "Banana")


def test_label_association_is_legacy_exact_match():
    stub = make_stub(association_key="label")
    assert not stub._same_object_class("mobile robot", "ground mobile robot")
    assert stub._same_object_class("mobile robot", "Mobile Robot")


def test_gate_widens_with_time_since_last_seen():
    stub = make_stub()
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=10.0, p_movable_prior=0.9)
    assert stub._gate_for(tr, now=10.0) == 0.6
    assert stub._gate_for(tr, now=11.5) == 0.6 + 1.5
    assert make_stub(gate_speed=0.0)._gate_for(tr, now=11.5) == 0.6


def test_gate_widening_is_capped_at_gate_max_m():
    """WP1 (2026-09-10): a long-coasting track's gate must not grow
    unbounded -- see enforce_speed_cap's "speed jump reset" finding, where
    an uncapped gate let a track re-associate 10-25 m away after a
    multi-second gap. gate_max_m defaults to 2.5 m (module docstring)."""
    stub = make_stub(gate=0.6, gate_speed=1.0, gate_max=2.5)
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=10.0, p_movable_prior=0.9)
    # Without a cap this would be 0.6 + 1.0*60 = 60.6 m.
    assert stub._gate_for(tr, now=70.0) == 2.5


def test_adopt_label_follows_confidence():
    tr = Track(x=0.0, y=0.0, label="cart", score=0.5, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.7)
    ObjectTrackerNode._adopt_label(tr, {"label": "mobile robot", "score": 0.4})
    assert tr.label == "cart"
    ObjectTrackerNode._adopt_label(tr, {"label": "mobile robot", "score": 0.9})
    assert tr.label == "mobile robot"


def test_track_coasts_through_camera_handover_gap_with_velocity_intact():
    """WP-A#3 (2026-09-08): overhead-camera hand-over/occlusion gaps run
    2-3 s, not the ~0.1-0.3 s a single dropped frame implies. A track must
    keep p_motion and confidence above ObjectTrackerNode._tick's own drop
    gate (`t.confidence >= self.min_conf and (now - t.last_seen) <=
    timeout`, `timeout = max_unseen_dynamic if t.p_movable > 0.5 else
    max_unseen_static`) through such a gap on Track.predict() alone (no
    update() calls -- exactly what happens when a moving object goes
    briefly unseen), or it gets pruned and the next sighting respawns it
    as a fresh, zero-velocity track. Drives Track's own methods directly
    (predict/update_beliefs/decay_confidence), the same three calls
    ObjectTrackerNode._tick makes on every track every tick, so this is
    exercising the exact quantities the node reads its drop decision from
    without needing rclpy.init() or a real node.
    """
    half_life = 4.0   # dynamic_half_life_s node/yaml default as of this fix
    min_conf = 0.15    # object_tracker's min_confidence default (risk_perception.yaml)
    dt_tick = 0.1      # object_tracker's update_rate default is 10 Hz
    speed = 0.6        # m/s -- the sim Carter's patrol speed

    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.5, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)

    # Phase 1: observed at 1 Hz for 4 s (matches drive_track() above) so
    # velocity/p_motion/p_movable have settled to a steady state before the
    # gap starts, the same as a track that has been tracking a while when
    # the camera hand-over happens.
    t = 0.0
    next_obs = 1.0
    while t < 4.0:
        t += dt_tick
        tr.predict(dt_tick, 0.5)
        if t + 1e-9 >= next_obs:
            tr.update(speed * t, 0.0, 0.04, 0.5, 0.5, t)
            next_obs += 1.0
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          speed_mps=0.3, speed_score=0.5)
        tr.decay_confidence(dt_tick, half_life)

    assert tr.p_movable > 0.5   # dynamic_half_life_s is the branch selected below

    # Phase 2: unobserved for 2.5 s -- no update() calls at all, only the
    # three per-tick calls _tick() makes regardless of whether a detection
    # arrived this cycle. `now` keeps advancing on the same clock as phase 1
    # (t reached 4.0 there) so update_beliefs's motion_hold_s gap check
    # (now - last_seen) sees the real 2.5 s elapsed, not a reset clock.
    unseen_for = 0.0
    while unseen_for < 2.5:
        unseen_for += dt_tick
        tr.predict(dt_tick, 0.5)
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=4.0 + unseen_for,
                          speed_mps=0.3, speed_score=0.5)
        tr.decay_confidence(dt_tick, half_life)

    assert tr.p_motion > 0.5, "track should still read as moving through a 2.5 s gap"
    assert tr.confidence > min_conf, (
        "track should not cross _tick's min_conf drop gate during the gap")
    # coasting on prediction alone: velocity survives untouched (predict()
    # only grows P, it never touches x[2:4]).
    assert math.hypot(*tr.velocity) == pytest.approx(speed, rel=0.05)


def test_measurement_age_from_header(monkeypatch):
    """_measurement_age: capture stamp -> age, clamped; 0 when disabled/unset."""
    import types as _t
    from builtin_interfaces.msg import Time as _Time
    stub = _t.SimpleNamespace(meas_time_corr=True, max_meas_age=3.0, _meas_ages=[],
                              get_logger=lambda: _t.SimpleNamespace(info=lambda *a, **k: None))
    stub._measurement_age = _t.MethodType(ObjectTrackerNode._measurement_age, stub)
    msg = _t.SimpleNamespace(header=_t.SimpleNamespace(stamp=_Time(sec=100, nanosec=0)))
    assert stub._measurement_age(msg, now=101.5) == 1.5
    assert stub._measurement_age(msg, now=110.0) == 3.0          # clamped
    msg0 = _t.SimpleNamespace(header=_t.SimpleNamespace(stamp=_Time(sec=0, nanosec=0)))
    assert stub._measurement_age(msg0, now=101.5) == 0.0         # unset stamp
    stub.meas_time_corr = False
    assert stub._measurement_age(msg, now=101.5) == 0.0          # disabled


# ---------------------------------------------------------------------------
# Speed plausibility (2026-09-10 finding) -- Track.enforce_speed_cap, run by
# ObjectTrackerNode._apply_speed_cap right after every Kalman update() (the
# labeled pass, _associate_agnostic, and _revive -- see object_tracker_node
# .py). Tested directly on Track, same style as motion_mahalanobis /
# update_beliefs above: pure state, no rclpy.
# ---------------------------------------------------------------------------

def test_speed_cap_resets_velocity_and_p_motion_on_association_jump():
    """A 4 m jump in 1 s -- e.g. a lidar_cluster track that re-associated
    with a different shelf edge between scans -- has the Kalman filter
    estimate ~3.7 m/s (a real predict()+update() cycle, not a hand-set
    velocity: predict() first builds the position/velocity cross-covariance
    a single update() needs to move the velocity state at all). That is
    past both the robot cap (1.5 m/s) and jump_reset_factor (2.0) x cap
    (3.0 m/s), so enforce_speed_cap must read it as an association jump,
    not real motion, and reset velocity/p_motion to 0 outright rather than
    merely clamping to the cap."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    tr.p_motion = 0.8   # already latched "moving" before the clamp runs
    pre_x, pre_y = tr.position
    tr.predict(1.0, 0.5)
    tr.update(4.0, 0.0, 0.04, 0.9, 0.5, 1.0)   # 4 m jump in 1 s
    assert tr.speed > 3.0   # sanity: the KF really did read this as fast

    capped, was_jump, pre_speed, displacement = tr.enforce_speed_cap(
        cap_mps=1.5, jump_reset_factor=2.0, pre_x=pre_x, pre_y=pre_y)

    assert capped and was_jump
    assert tr.velocity == (0.0, 0.0)
    assert tr.p_motion == 0.0
    assert pre_speed > 3.0
    assert displacement > 3.0


def test_speed_cap_clamps_direction_preserving_when_only_slightly_over():
    """A speed slightly above the cap (not past jump_reset_factor x cap) is
    trusted, just capped -- direction is kept, not zeroed."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    tr.x[2], tr.x[3] = 1.6, 0.0   # slightly above the 1.5 m/s robot cap

    capped, was_jump, pre_speed, displacement = tr.enforce_speed_cap(
        cap_mps=1.5, jump_reset_factor=2.0, pre_x=0.0, pre_y=0.0)

    assert capped and not was_jump
    assert tr.velocity == pytest.approx((1.5, 0.0))
    assert pre_speed == pytest.approx(1.6)


def test_speed_cap_direction_preserved_off_axis():
    """Clamping must scale, not just cap the x/y components independently
    -- direction has to survive for a non-axis-aligned velocity too."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    tr.x[2], tr.x[3] = 1.6, 1.6   # speed = 1.6*sqrt(2) ~= 2.263, cap 1.5
    tr.enforce_speed_cap(cap_mps=1.5, jump_reset_factor=2.0, pre_x=0.0, pre_y=0.0)
    assert tr.speed == pytest.approx(1.5)
    assert tr.x[2] == pytest.approx(tr.x[3])   # direction (45 deg) preserved


def test_speed_cap_is_a_noop_under_the_cap():
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    tr.x[2], tr.x[3] = 0.5, 0.0

    capped, was_jump, pre_speed, displacement = tr.enforce_speed_cap(
        cap_mps=1.5, jump_reset_factor=2.0, pre_x=0.0, pre_y=0.0)

    assert not capped and not was_jump
    assert tr.velocity == (0.5, 0.0)
    assert displacement == 0.0


# ---------------------------------------------------------------------------
# Track maturity (2026-09-10 finding) -- risk_visualization.parse_class_id's
# hits/age round-trip. object_tracker_node._publish_objects appends these
# after relbonus=; every existing caller reads this dict via
# .get(key, default), so their absence must not raise.
# ---------------------------------------------------------------------------

def test_parse_class_id_round_trips_hits_and_age():
    label, kv = parse_class_id(
        "mobile robot|pmov=0.90|pmot=0.05|vx=0.12|vy=-0.03|relbonus=0.00"
        "|hits=12|age=3.40")
    assert label == "mobile robot"
    assert kv["hits"] == 12
    assert isinstance(kv["hits"], int)
    assert kv["age"] == pytest.approx(3.40)
    assert isinstance(kv["age"], float)


def test_parse_class_id_tolerates_missing_hits_and_age():
    label, kv = parse_class_id("mobile robot|pmov=0.90|pmot=0.05|vx=0.12|vy=-0.03")
    assert label == "mobile robot"
    assert "hits" not in kv
    assert "age" not in kv
    assert kv.get("hits", 0) == 0          # the .get(key, default) pattern every
    assert kv.get("age", 0.0) == 0.0       # real caller uses still works fine
    assert kv["pmov"] == pytest.approx(0.90)   # existing keys unaffected
