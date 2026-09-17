#!/usr/bin/env python3
"""
gt_tracks_node.py -- WP-C: oracle perception. Ground-truth
vision_msgs/Detection3DArray tracks straight off `/gt_tf` (Isaac's
World -> carterN transforms, ~60 Hz -- World IS the map frame, no
localization needed), so prediction (predictive_risk_costmap_node) and
planning (mission_supervisor, the DWB/MPPI risk critics) can be exercised
without cameras, GroundingDINO, SAM2 or object_tracker_node in the loop.
Publishes onto the SAME topic (`/risk_perception/world_objects` by
default) the real perception stack uses, so nothing downstream needs to
know which perception mode is running -- see x3_nav.launch.py's
`perception:=panoptex|oracle` arg.

class_id convention
--------------------
Consumers (risk_speed_governor, predictive_risk_costmap_node,
mission_supervisor, spatial_prior_node) all parse the tracker's packed
class_id string via risk_perception.risk_visualization.parse_class_id:
"<label>|pmov=<>|pmot=<>|vx=<>|vy=<>|relbonus=<>". This node builds the
exact same string (see build_class_id below) so it is a drop-in
replacement for object_tracker_node's output. `relbonus` is always
0.000 -- that field is object_tracker_node's mask_relation inference
bonus; this oracle has no such inference to report.

Ground-truth centre-offset correction (gt_centre_offset_m)
------------------------------------------------------------
/gt_tf's child frame per robot (see warehouse/carters/build_carter_graphs.py's
`chassis_paths`, topic "gt_tf") is the Nova Carter's `chassis_link`, which
IS the drive-axle origin -- confirmed against
warehouse/nav/carter_nav_params.yaml's footprint,
`[[0.14, 0.25], [0.14, -0.25], [-0.607, -0.25], [-0.607, 0.25]]` ("origin on
the drive axle"). The footprint's body centre along the local +x (forward)
axis is therefore (0.14 + -0.607) / 2 = -0.2335 m relative to the axle --
i.e. the body centre sits ~0.23 m BEHIND the axle along the robot's
forward heading, not ahead of it as originally suspected. `gt_centre_offset_m`
defaults to -0.23: NEGATIVE shifts the reported position backward along
the heading direction (see offset_position() below for what "the heading"
means when the robot isn't moving). Set to 0.0 to report the raw /gt_tf
point (the drive axle) unshifted.

Everything below that isn't ROS plumbing is a pure function
(yaw/velocity-estimate/EMA/classify/offset/build_*), unit-tested directly
in test_gt_tracks.py with no rclpy spin -- same rationale as
risk_speed_governor's compute_cap / corridor.py (see those modules'
docstrings).
"""

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import (BoundingBox3D, Detection3D, Detection3DArray,
                              ObjectHypothesisWithPose)


# ============================================================ pure core

def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rotation about Z) from a quaternion, assuming a planar robot
    (roll = pitch = 0 -- true for /gt_tf's Carter ground-truth poses). Same
    formula as warehouse/carters/tests/patrol_monitor.py's yaw_of() and
    risk_speed_governor's yaw_from_quat()."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def displacement_velocity(
    times: List[float], xs: List[float], ys: List[float],
) -> Tuple[float, float]:
    """(vx, vy) from the straight-line displacement between the OLDEST and
    NEWEST of >= 2 samples, divided by the elapsed time between them -- a
    window-baseline estimate, not a per-sample finite difference, so ~60 Hz
    /gt_tf jitter doesn't dominate the estimate. Returns (0.0, 0.0) if
    fewer than 2 samples, or the elapsed time is non-positive (a
    degenerate or out-of-order window)."""
    if len(times) < 2:
        return 0.0, 0.0
    dt = times[-1] - times[0]
    if dt <= 0.0:
        return 0.0, 0.0
    return (xs[-1] - xs[0]) / dt, (ys[-1] - ys[0]) / dt


def ema_update(
    prev: Optional[Tuple[float, float]],
    raw: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """First-order EMA smoothing of the raw displacement-window velocity.
    `prev is None` seeds directly to `raw` (no smoothing on the very first
    sample -- there is nothing to smooth against yet). `alpha` is clamped
    to [0, 1]; 1.0 means "no smoothing, snap straight to raw"."""
    if prev is None:
        return raw
    a = max(0.0, min(1.0, alpha))
    return (prev[0] + a * (raw[0] - prev[0]),
            prev[1] + a * (raw[1] - prev[1]))


def classify_motion(vx: float, vy: float, moving_speed_mps: float) -> Tuple[float, float]:
    """(pmot, speed). pmot is 1.0 at/above moving_speed_mps, else 0.0 -- a
    hard ground-truth threshold, not a fitted probability (this is an
    oracle, not object_tracker_node's motion-mixture classifier)."""
    speed = math.hypot(vx, vy)
    return (1.0 if speed >= moving_speed_mps else 0.0), speed


def offset_position(
    x: float, y: float, vx: float, vy: float, yaw: float, speed: float,
    moving_speed_mps: float, offset_m: float,
) -> Tuple[float, float]:
    """Shift the raw /gt_tf point (x, y) by `offset_m` along the robot's
    heading, to correct for gt_centre_offset_m (see module docstring). The
    heading is the velocity direction while genuinely moving (speed >=
    moving_speed_mps and non-degenerate); below that threshold the
    velocity estimate is noise-dominated (a parked or barely-creeping
    robot's displacement-window velocity has no reliable direction), so
    the /gt_tf quaternion's yaw is used instead -- the chassis is still
    oriented somewhere even when stationary."""
    if speed >= moving_speed_mps and speed > 1e-6:
        dx, dy = vx / speed, vy / speed
    else:
        dx, dy = math.cos(yaw), math.sin(yaw)
    return x + offset_m * dx, y + offset_m * dy


def build_class_id(label: str, pmov: float, pmot: float, vx: float, vy: float) -> str:
    """Packed class_id string -- see module docstring for the convention
    risk_visualization.parse_class_id parses."""
    return "{}|pmov={:.2f}|pmot={:.2f}|vx={:.3f}|vy={:.3f}|relbonus=0.000".format(
        label, pmov, pmot, vx, vy)


def build_detection_fields(
    name: str, x: float, y: float, vx: float, vy: float, pmot: float, params: Dict,
) -> Dict:
    """Pure computation of one robot's Detection3D fields -- a plain dict,
    no ROS message types involved, so this is unit-testable with no rclpy
    or vision_msgs import needed. `x, y` are the ALREADY-OFFSET position
    (see offset_position); `pmot` is ALREADY classified (see
    classify_motion). The node wraps this dict into a real Detection3D via
    _to_detection3d() below.

    params keys used: label, pmov, bbox_x_m, bbox_y_m, cov_m2.
    """
    class_id = build_class_id(str(params["label"]), float(params["pmov"]), pmot, vx, vy)
    return {
        "id": name,
        "class_id": class_id,
        "score": 0.95,
        "position": (x, y, 0.0),
        "size": (float(params["bbox_x_m"]), float(params["bbox_y_m"]), 0.5),
        # Position variance (x, y) -- covariance[0]/[7]. Orientation
        # variance (roll, pitch) -- covariance[21]/[28] -- is a fixed 0.01
        # regardless of cov_m2: this oracle reports no orientation at all
        # (bbox/pose orientation is left at the msg default identity
        # quaternion), so that figure is just "small and nonzero", not a
        # tuned quantity the way the position covariance is.
        "cov_pos": float(params["cov_m2"]),
        "cov_rot": 0.01,
    }


def _to_detection3d(fields: Dict, frame_id: str, stamp) -> Detection3D:
    """The one non-pure step: wrap a build_detection_fields() dict into a
    real vision_msgs/Detection3D. Kept separate from build_detection_fields
    so that function stays importable/testable without vision_msgs."""
    det = Detection3D()
    det.header = Header()
    det.header.stamp = stamp
    det.header.frame_id = frame_id
    det.id = fields["id"]

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = fields["class_id"]
    hyp.hypothesis.score = fields["score"]
    px, py, pz = fields["position"]
    hyp.pose.pose.position.x = px
    hyp.pose.pose.position.y = py
    hyp.pose.pose.position.z = pz
    cov = [0.0] * 36
    cov[0] = fields["cov_pos"]
    cov[7] = fields["cov_pos"]
    cov[21] = fields["cov_rot"]
    cov[28] = fields["cov_rot"]
    hyp.pose.covariance = cov
    det.results = [hyp]

    bbox = BoundingBox3D()
    bbox.center.position.x = px
    bbox.center.position.y = py
    bbox.center.position.z = pz
    sx, sy, sz = fields["size"]
    bbox.size.x = sx
    bbox.size.y = sy
    bbox.size.z = sz
    det.bbox = bbox
    return det


# ============================================================ ROS node

class _RobotState:
    """Per-robot rolling window + EMA velocity state, kept by the node
    across /gt_tf callbacks. Not a dataclass/dict -- __slots__ keeps this
    cheap at ~60 Hz per tracked robot."""

    __slots__ = ("times", "xs", "ys", "smoothed_v", "yaw", "last_t")

    def __init__(self) -> None:
        self.times: Deque[float] = deque()
        self.xs: Deque[float] = deque()
        self.ys: Deque[float] = deque()
        self.smoothed_v: Optional[Tuple[float, float]] = None
        self.yaw: float = 0.0
        self.last_t: Optional[float] = None


class GtTracksNode(Node):
    def __init__(self) -> None:
        super().__init__("gt_tracks_node")

        self.declare_parameter("robots", ["carter1"])
        self.declare_parameter("label", "mobile robot")
        self.declare_parameter("bbox_x_m", 0.75)
        self.declare_parameter("bbox_y_m", 0.50)
        self.declare_parameter("cov_m2", 0.01)
        self.declare_parameter("vel_window_sec", 0.5)
        self.declare_parameter("moving_speed_mps", 0.15)
        self.declare_parameter("pmov", 0.90)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("output_topic", "/risk_perception/world_objects")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("gt_tf_topic", "/gt_tf")
        # See module docstring: chassis_link (the /gt_tf child frame) is the
        # drive-axle origin, ~0.23 m AHEAD of the body centre along the
        # heading -- default shifts the reported point BACK by that much.
        self.declare_parameter("gt_centre_offset_m", -0.23)

        gp = self.get_parameter
        self.robots: List[str] = [str(r) for r in gp("robots").value]
        self.params: Dict[str, float] = {
            "label": str(gp("label").value),
            "bbox_x_m": float(gp("bbox_x_m").value),
            "bbox_y_m": float(gp("bbox_y_m").value),
            "cov_m2": float(gp("cov_m2").value),
            "pmov": float(gp("pmov").value),
        }
        self.vel_window_sec = float(gp("vel_window_sec").value)
        self.moving_speed_mps = float(gp("moving_speed_mps").value)
        self.publish_rate_hz = float(gp("publish_rate_hz").value)
        self.output_topic = str(gp("output_topic").value)
        self.frame_id = str(gp("frame_id").value)
        self.gt_tf_topic = str(gp("gt_tf_topic").value)
        self.gt_centre_offset_m = float(gp("gt_centre_offset_m").value)

        self._state: Dict[str, _RobotState] = {name: _RobotState() for name in self.robots}

        self.create_subscription(TFMessage, self.gt_tf_topic, self._on_gt_tf, 50)
        self.pub = self.create_publisher(Detection3DArray, self.output_topic, 10)

        period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 0.1
        self.timer = self.create_timer(period, self._publish)

        self.get_logger().info(
            "gt_tracks up: {} -> {} robots={} label={!r} offset={:.3f}m "
            "@ {:.1f} Hz".format(
                self.gt_tf_topic, self.output_topic, self.robots,
                self.params["label"], self.gt_centre_offset_m, self.publish_rate_hz))

    def _on_gt_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            st = self._state.get(t.child_frame_id)
            if st is None:
                continue  # not one of the `robots` this instance tracks

            stamp = t.header.stamp
            now = stamp.sec + stamp.nanosec * 1e-9
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            st.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

            st.times.append(now)
            st.xs.append(x)
            st.ys.append(y)
            # Prune samples older than vel_window_sec relative to the
            # newest one -- displacement_velocity() then reads oldest vs.
            # newest survivor as the window baseline.
            while st.times and (now - st.times[0]) > self.vel_window_sec:
                st.times.popleft()
                st.xs.popleft()
                st.ys.popleft()

            raw_v = displacement_velocity(list(st.times), list(st.xs), list(st.ys))
            # EMA time-constant is vel_window_sec itself -- alpha is this
            # sample's dt as a fraction of that window, clamped in
            # ema_update. First sample for this robot (last_t is None):
            # alpha=1.0, i.e. seed straight to raw (also ema_update's own
            # prev-is-None branch does this regardless).
            dt_sample = now - st.last_t if st.last_t is not None else self.vel_window_sec
            st.last_t = now
            alpha = dt_sample / self.vel_window_sec if self.vel_window_sec > 0.0 else 1.0
            st.smoothed_v = ema_update(st.smoothed_v, raw_v, alpha)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        arr = Detection3DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.frame_id

        for name, st in self._state.items():
            if st.last_t is None or not st.xs:
                continue  # no /gt_tf sample yet for this robot

            x, y = st.xs[-1], st.ys[-1]
            vx, vy = st.smoothed_v if st.smoothed_v is not None else (0.0, 0.0)
            pmot, speed = classify_motion(vx, vy, self.moving_speed_mps)
            ox, oy = offset_position(
                x, y, vx, vy, st.yaw, speed, self.moving_speed_mps,
                self.gt_centre_offset_m)
            fields = build_detection_fields(name, ox, oy, vx, vy, pmot, self.params)
            arr.detections.append(_to_detection3d(fields, self.frame_id, stamp))

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = GtTracksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
