"""
Pure-function tests for panoptex_nav.gt_tracks_node -- no rclpy spin, no
ROS graph, no vision_msgs/tf2_msgs import needed for the estimator/builder
functions (same rationale as test_risk_speed_governor.py / test_corridor.py:
see those modules' docstrings). Only test_to_detection3d_smoke below touches
an actual ROS message type, to catch a field-name typo in _to_detection3d.

Run via `colcon test --packages-select panoptex_nav` (wired into
CMakeLists.txt through ament_cmake_pytest) or directly with
`python3 -m pytest test/test_gt_tracks.py` from this package's root,
provided panoptex_nav/ is importable (PYTHONPATH at this package's source
root, or the workspace already built and sourced).
"""

import math

import pytest

from panoptex_nav.gt_tracks_node import (
    build_class_id,
    build_detection_fields,
    classify_motion,
    displacement_velocity,
    ema_update,
    offset_position,
    yaw_from_quaternion,
)

PARAMS = {
    "label": "mobile robot",
    "pmov": 0.90,
    "bbox_x_m": 0.75,
    "bbox_y_m": 0.50,
    "cov_m2": 0.01,
}


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


# --------------------------------------------------------- yaw_from_quaternion

def test_yaw_from_quaternion_zero():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_yaw_from_quaternion_roundtrip():
    for yaw in (0.3, 1.5708, -2.1, 3.0, -3.0):
        qx, qy, qz, qw = quat_from_yaw(yaw)
        assert yaw_from_quaternion(qx, qy, qz, qw) == pytest.approx(yaw, abs=1e-9)


# --------------------------------------------------------- displacement_velocity

def test_displacement_velocity_constant_motion():
    # 0.5s window at 0.4 m/s along +x, 60 Hz-ish samples.
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    xs = [0.4 * t for t in times]
    ys = [0.0 for _ in times]
    vx, vy = displacement_velocity(times, xs, ys)
    assert vx == pytest.approx(0.4)
    assert vy == pytest.approx(0.0)


def test_displacement_velocity_stationary():
    times = [0.0, 0.2, 0.4]
    xs = [1.0, 1.0, 1.0]
    ys = [2.0, 2.0, 2.0]
    vx, vy = displacement_velocity(times, xs, ys)
    assert (vx, vy) == (0.0, 0.0)


def test_displacement_velocity_needs_two_samples():
    assert displacement_velocity([0.0], [1.0], [1.0]) == (0.0, 0.0)
    assert displacement_velocity([], [], []) == (0.0, 0.0)


def test_displacement_velocity_degenerate_dt():
    # out-of-order / duplicate timestamps -> non-positive dt -> (0, 0)
    assert displacement_velocity([1.0, 1.0], [0.0, 5.0], [0.0, 0.0]) == (0.0, 0.0)


# --------------------------------------------------------------------- ema_update

def test_ema_update_seeds_on_first_sample():
    assert ema_update(None, (1.0, -2.0), 0.3) == (1.0, -2.0)


def test_ema_update_converges_on_constant_input():
    v = None
    for _ in range(20):
        v = ema_update(v, (0.4, 0.1), 0.3)
    assert v[0] == pytest.approx(0.4, abs=1e-6)
    assert v[1] == pytest.approx(0.1, abs=1e-6)


def test_ema_update_clamps_alpha():
    # alpha > 1 clamped to 1 -> snaps straight to raw.
    assert ema_update((0.0, 0.0), (5.0, 5.0), 2.0) == (5.0, 5.0)
    # alpha < 0 clamped to 0 -> no movement at all.
    assert ema_update((1.0, 1.0), (5.0, 5.0), -1.0) == (1.0, 1.0)


# ------------------------------------------------------------------ classify_motion

def test_classify_motion_moving():
    pmot, speed = classify_motion(0.3, 0.0, moving_speed_mps=0.15)
    assert pmot == 1.0
    assert speed == pytest.approx(0.3)


def test_classify_motion_stationary():
    pmot, speed = classify_motion(0.0, 0.0, moving_speed_mps=0.15)
    assert pmot == 0.0
    assert speed == pytest.approx(0.0)


def test_classify_motion_at_threshold_counts_as_moving():
    pmot, _ = classify_motion(0.15, 0.0, moving_speed_mps=0.15)
    assert pmot == 1.0


def test_classify_motion_just_under_threshold_is_stationary():
    pmot, _ = classify_motion(0.14, 0.0, moving_speed_mps=0.15)
    assert pmot == 0.0


# ------------------------------------------------------------------ offset_position

def test_offset_position_moving_uses_velocity_direction():
    # Moving at 0.4 m/s along +x -- offset should shift along +x regardless
    # of yaw (yaw deliberately wrong/unused here).
    x, y = offset_position(
        x=1.0, y=2.0, vx=0.4, vy=0.0, yaw=math.pi, speed=0.4,
        moving_speed_mps=0.15, offset_m=-0.23)
    assert x == pytest.approx(1.0 - 0.23)
    assert y == pytest.approx(2.0)


def test_offset_position_moving_diagonal():
    speed = math.hypot(0.3, 0.3)
    x, y = offset_position(
        x=0.0, y=0.0, vx=0.3, vy=0.3, yaw=0.0, speed=speed,
        moving_speed_mps=0.15, offset_m=0.5)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert x == pytest.approx(0.5 * inv_sqrt2, abs=1e-9)
    assert y == pytest.approx(0.5 * inv_sqrt2, abs=1e-9)


def test_offset_position_stationary_uses_yaw():
    # Stopped (speed 0 < moving_speed_mps) -- velocity direction is
    # meaningless (vx/vy left as stale/noisy values here), so the offset
    # must follow yaw = pi/2 (+y) instead.
    x, y = offset_position(
        x=5.0, y=5.0, vx=0.01, vy=-0.02, yaw=math.pi / 2.0, speed=0.02,
        moving_speed_mps=0.15, offset_m=-0.23)
    assert x == pytest.approx(5.0, abs=1e-9)
    assert y == pytest.approx(5.0 - 0.23, abs=1e-9)


def test_offset_position_zero_offset_is_noop():
    x, y = offset_position(
        x=3.0, y=-1.0, vx=1.0, vy=0.0, yaw=0.0, speed=1.0,
        moving_speed_mps=0.15, offset_m=0.0)
    assert (x, y) == pytest.approx((3.0, -1.0))


# ------------------------------------------------------------------- build_class_id

def test_build_class_id_format():
    cid = build_class_id("mobile robot", pmov=0.90, pmot=1.0, vx=0.314, vy=-0.007)
    assert cid == "mobile robot|pmov=0.90|pmot=1.00|vx=0.314|vy=-0.007|relbonus=0.000"


def test_build_class_id_parses_back_with_risk_visualization():
    from risk_perception.risk_visualization import parse_class_id
    cid = build_class_id("mobile robot", pmov=0.9, pmot=0.0, vx=0.0, vy=0.0)
    label, kv = parse_class_id(cid)
    assert label == "mobile robot"
    assert kv["pmov"] == pytest.approx(0.9)
    assert kv["pmot"] == pytest.approx(0.0)
    assert kv["vx"] == pytest.approx(0.0)
    assert kv["vy"] == pytest.approx(0.0)
    assert kv["relbonus"] == pytest.approx(0.0)


# ------------------------------------------------------------ build_detection_fields

def test_build_detection_fields_constant_motion_pmot_1():
    pmot, _speed = classify_motion(0.3, 0.0, moving_speed_mps=0.15)
    fields = build_detection_fields("carter1", x=1.0, y=2.0, vx=0.3, vy=0.0,
                                    pmot=pmot, params=PARAMS)
    assert fields["id"] == "carter1"
    assert fields["position"] == (1.0, 2.0, 0.0)
    assert fields["size"] == (0.75, 0.50, 0.5)
    assert fields["score"] == 0.95
    assert fields["cov_pos"] == pytest.approx(0.01)
    assert fields["cov_rot"] == pytest.approx(0.01)
    label, kv = _parse(fields["class_id"])
    assert label == "mobile robot"
    assert kv["pmot"] == pytest.approx(1.0)
    assert kv["vx"] == pytest.approx(0.3)


def test_build_detection_fields_stationary_pmot_0():
    pmot, _speed = classify_motion(0.0, 0.0, moving_speed_mps=0.15)
    fields = build_detection_fields("carter1", x=1.0, y=2.0, vx=0.0, vy=0.0,
                                    pmot=pmot, params=PARAMS)
    _, kv = _parse(fields["class_id"])
    assert kv["pmot"] == pytest.approx(0.0)


def _parse(class_id):
    from risk_perception.risk_visualization import parse_class_id
    return parse_class_id(class_id)


# ------------------------------------------------------------------- end-to-end

def test_end_to_end_constant_motion_offset_along_velocity():
    """Full pipeline (minus the ROS message wrap): a robot cruising at
    0.4 m/s along +x, sampled every 0.1s over a 0.5s window -- velocity
    estimate should be exact (displacement_velocity has no noise to
    smooth away for perfectly constant motion), pmot=1, and the reported
    position shifted -0.23 m along +x (the velocity direction)."""
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    x0 = 10.0
    xs = [x0 + 0.4 * t for t in times]
    ys = [3.0 for _ in times]

    vx, vy = displacement_velocity(times, xs, ys)
    assert vx == pytest.approx(0.4)
    assert vy == pytest.approx(0.0)

    pmot, speed = classify_motion(vx, vy, moving_speed_mps=0.15)
    assert pmot == 1.0

    ox, oy = offset_position(xs[-1], ys[-1], vx, vy, yaw=0.0, speed=speed,
                              moving_speed_mps=0.15, offset_m=-0.23)
    assert ox == pytest.approx(xs[-1] - 0.23)
    assert oy == pytest.approx(3.0)

    fields = build_detection_fields("carter1", ox, oy, vx, vy, pmot, PARAMS)
    _, kv = _parse(fields["class_id"])
    assert kv["pmot"] == pytest.approx(1.0)
    assert kv["vx"] == pytest.approx(0.4)


def test_end_to_end_stationary_pmot_0_no_velocity_shift():
    """A parked robot: all samples at the same point -> v=(0,0), pmot=0,
    and the offset follows yaw (here 0.0, i.e. +x) rather than the
    (undefined) velocity direction."""
    times = [0.0, 0.1, 0.2]
    xs = [4.0, 4.0, 4.0]
    ys = [-1.0, -1.0, -1.0]

    vx, vy = displacement_velocity(times, xs, ys)
    assert (vx, vy) == (0.0, 0.0)

    pmot, speed = classify_motion(vx, vy, moving_speed_mps=0.15)
    assert pmot == 0.0

    ox, oy = offset_position(4.0, -1.0, vx, vy, yaw=0.0, speed=speed,
                              moving_speed_mps=0.15, offset_m=-0.23)
    assert ox == pytest.approx(4.0 - 0.23)
    assert oy == pytest.approx(-1.0)


# --------------------------------------------------------------- ROS message wrap

def test_to_detection3d_smoke():
    """Catches a field-name typo in _to_detection3d -- the one function in
    this module that touches actual vision_msgs types."""
    vision_msgs = pytest.importorskip("vision_msgs.msg")
    from panoptex_nav.gt_tracks_node import _to_detection3d
    from builtin_interfaces.msg import Time

    fields = build_detection_fields("carter1", 1.0, 2.0, 0.3, 0.0, 1.0, PARAMS)
    stamp = Time(sec=5, nanosec=0)
    det = _to_detection3d(fields, "map", stamp)
    assert isinstance(det, vision_msgs.Detection3D)
    assert det.id == "carter1"
    assert det.header.frame_id == "map"
    assert det.bbox.center.position.x == pytest.approx(1.0)
    assert det.bbox.size.x == pytest.approx(0.75)
    assert det.results[0].hypothesis.class_id == fields["class_id"]
    assert det.results[0].pose.covariance[0] == pytest.approx(0.01)
    assert det.results[0].pose.covariance[7] == pytest.approx(0.01)
    assert det.results[0].pose.covariance[21] == pytest.approx(0.01)
    assert det.results[0].pose.covariance[28] == pytest.approx(0.01)
