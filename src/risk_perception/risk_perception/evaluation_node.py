#!/usr/bin/env python3
"""
evaluation_node.py  --  live counterpart to tools/evaluate_run.py

Same metrics, same evaluation_metrics.py functions, fed in real time
instead of replayed from a bag. This node only COLLECTS raw samples during
a trial (tf-derived robot pose, nearest-object distance, risk at the
robot's position, detection/costmap timestamps) into plain lists; at
trial end it calls the exact same tested batch functions
(risk_exposure, clearance_encounters, stop_events, ...) that
evaluate_run.py calls on bag data -- there is no separate "streaming"
reimplementation of any metric to keep in sync with the tested one.

Why real time instead of always going through a bag: automating "run N
trials, one after another" is much easier when a trial's own result comes
back as a message the orchestrator can wait on, rather than a bag file it
has to close, locate, and hand to a second process after the fact. Bags
are still worth recording alongside this for anything this node doesn't
capture or for later re-analysis with different parameters (see
tools/evaluate_run.py) -- the two are complementary, not a replacement for
each other.

No custom messages or services -- trial control is a single
std_msgs/String topic pair, matching this project's existing preference
for reusing standard message types (see spatial_prior_node's per-category
Image topics) over introducing new .msg/.srv definitions for something
this small:

  /evaluation/trial_control  (std_msgs/String, subscribed)
      "start <label>"   -- clear buffers, begin collecting under this label
      "end"             -- compute the RunReport, publish it, ready for the
                            next "start"

  /evaluation/trial_result  (std_msgs/String, published)
      the just-finished trial's RunReport.summary(), JSON-encoded

Samples the FIXED reference costmap (risk_costmap_topic, default
/risk_costmap_reference -- see evaluation_reference.launch.py) at the
robot's live tf pose, not whichever costmap is actually driving Nav2 that
run -- same reasoning as evaluate_run.py's module docstring: ablations
must be scored on a common yardstick.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from vision_msgs.msg import Detection2DArray, Detection3DArray
from tf2_ros import Buffer, TransformException, TransformListener

from risk_perception import evaluation_metrics as em


class EvaluationNode(Node):
    def __init__(self) -> None:
        super().__init__("evaluation_node")

        self.declare_parameter("risk_topic", "/risk_costmap_reference")
        self.declare_parameter("world_objects_topic", "/risk_perception/world_objects")
        self.declare_parameter("detections_topic", "/risk_perception/detections_2d")
        self.declare_parameter("replan_topic", "/plan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("sample_rate_hz", 10.0)
        self.declare_parameter("configured_period_s", 0.2)
        self.declare_parameter("engagement_threshold_m", 2.0)
        self.declare_parameter("stop_speed_mps", 0.05)

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.sample_period_s = 1.0 / max(0.1, float(gp("sample_rate_hz").value))
        self.configured_period_s = float(gp("configured_period_s").value)
        self.engagement_threshold_m = float(gp("engagement_threshold_m").value)
        self.stop_speed_mps = float(gp("stop_speed_mps").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_costmap: Optional[OccupancyGrid] = None
        self.latest_world_objects: Optional[Detection3DArray] = None
        self.latest_odom: Optional[Odometry] = None

        self._reset_buffers()
        self.active = False
        self.label = ""

        self.create_subscription(
            OccupancyGrid, str(gp("risk_topic").value), self._costmap_cb, 1)
        self.create_subscription(
            Detection3DArray, str(gp("world_objects_topic").value),
            self._world_objects_cb, 10)
        self.create_subscription(
            Detection2DArray, str(gp("detections_topic").value),
            self._detections_cb, 10)
        self.create_subscription(
            Path, str(gp("replan_topic").value), self._replan_cb, 10)
        self.create_subscription(
            Odometry, str(gp("odom_topic").value), self._odom_cb, 10)
        self.create_subscription(
            String, "/evaluation/trial_control", self._control_cb, 10)
        self.result_pub = self.create_publisher(String, "/evaluation/trial_result", 10)

        self.create_timer(self.sample_period_s, self._sample_tick)

        self.get_logger().info(
            "evaluation_node ready -- waiting for 'start <label>' on "
            "/evaluation/trial_control")

    # ------------------------------------------------------------------ trial control

    def _reset_buffers(self) -> None:
        self.t0: Optional[float] = None
        self.times_s: List[float] = []
        self.xy: List[Tuple[float, float]] = []
        self.risk_samples: List[float] = []
        self.raw_risk_samples: List[float] = []
        self.distances: List[float] = []
        self.detection_stamps_s: List[float] = []
        self.costmap_stamps_s: List[float] = []
        self.replan_stamps_s: List[float] = []

    def _control_cb(self, msg: String) -> None:
        text = msg.data.strip()
        if text.startswith("start"):
            parts = text.split(maxsplit=1)
            self.label = parts[1] if len(parts) > 1 else "run"
            self._reset_buffers()
            self.active = True
            self.get_logger().info(f"trial started: {self.label}")
        elif text == "end":
            if not self.active:
                self.get_logger().warning("'end' received with no trial active -- ignoring")
                return
            self.active = False
            summary = self._finalize()
            self.result_pub.publish(String(data=json.dumps(summary)))
            self.get_logger().info(f"trial finished: {self.label} -> {summary}")
        else:
            self.get_logger().warning(f"unrecognized trial_control message: {text!r}")

    # ------------------------------------------------------------------ inputs

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self.latest_costmap = msg
        if self.active:
            stamp = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            self.costmap_stamps_s.append(stamp)

    def _world_objects_cb(self, msg: Detection3DArray) -> None:
        self.latest_world_objects = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _detections_cb(self, msg: Detection2DArray) -> None:
        if self.active:
            stamp = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            self.detection_stamps_s.append(stamp)

    def _replan_cb(self, msg: Path) -> None:
        if self.active:
            stamp = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
            self.replan_stamps_s.append(stamp)

    # ------------------------------------------------------------------ per-tick sampling

    def _robot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        return (float(tf.transform.translation.x), float(tf.transform.translation.y))

    def _risk_at(self, xy: Tuple[float, float]) -> float:
        grid = self.latest_costmap
        if grid is None:
            return 0.0
        col = int((xy[0] - grid.info.origin.position.x) / grid.info.resolution)
        row = int((xy[1] - grid.info.origin.position.y) / grid.info.resolution)
        if 0 <= row < grid.info.height and 0 <= col < grid.info.width:
            value = grid.data[row * grid.info.width + col]
            return max(0, value) / 100.0
        return 0.0

    def _nearest_distance(self, xy: Tuple[float, float]) -> float:
        msg = self.latest_world_objects
        if msg is None or not msg.detections:
            return float("inf")
        best = float("inf")
        for det in msg.detections:
            dx = det.bbox.center.position.x - xy[0]
            dy = det.bbox.center.position.y - xy[1]
            best = min(best, math.hypot(dx, dy))
        return best

    def _robot_v(self) -> Tuple[float, float]:
        """Body-frame /odom twist rotated into map frame by the current tf
        yaw -- same construction as predictive_risk_costmap_node's own
        _update_robot_state, needed so _raw_risk_at's CPA/TTC term sees the
        robot velocity the live node would. (0.0, 0.0) with no odom/tf yet."""
        if self.latest_odom is None:
            return (0.0, 0.0)
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return (0.0, 0.0)
        q = tf.transform.rotation
        yaw = em.yaw_from_quat(q.x, q.y, q.z, q.w)
        bx = float(self.latest_odom.twist.twist.linear.x)
        by = float(self.latest_odom.twist.twist.linear.y)
        return (bx * math.cos(yaw) - by * math.sin(yaw),
                bx * math.sin(yaw) + by * math.cos(yaw))

    def _tracks_now(self) -> List[Dict[str, float]]:
        """Latest world_objects, unpacked into the plain dicts
        em.reference_severity_at_point expects -- same fields
        tools/evaluate_run.py's tracks_at_samples() extracts from a bag,
        just from the live message instead."""
        msg = self.latest_world_objects
        if msg is None:
            return []
        tracks = []
        for det in msg.detections:
            if not det.results:
                continue
            h = det.results[0].hypothesis
            label, kv = em.parse_class_id(h.class_id)
            cov = det.results[0].pose.covariance
            tracks.append({
                "label": label,
                "score": float(h.score),
                "x": float(det.bbox.center.position.x),
                "y": float(det.bbox.center.position.y),
                "pmot": kv.get("pmot", 0.0),
                "vx": kv.get("vx", 0.0),
                "vy": kv.get("vy", 0.0),
                "relbonus": kv.get("relbonus", 0.0),
                "Pxx": float(cov[0]),
                "Pyy": float(cov[7]),
                "Pvx": float(cov[21]),
                "Pvy": float(cov[28]),
            })
        return tracks

    def _raw_risk_at(self, xy: Tuple[float, float]) -> float:
        vx, vy = self._robot_v()
        return em.reference_severity_at_point(self._tracks_now(), xy[0], xy[1], vx, vy)

    def _sample_tick(self) -> None:
        if not self.active:
            return
        xy = self._robot_xy()
        if xy is None:
            return  # no tf yet this tick -- skip, do not record a bogus sample
        now = self.get_clock().now().nanoseconds / 1e9
        if self.t0 is None:
            self.t0 = now
        self.times_s.append(now - self.t0)
        self.xy.append(xy)
        self.risk_samples.append(self._risk_at(xy))
        self.raw_risk_samples.append(self._raw_risk_at(xy))
        self.distances.append(self._nearest_distance(xy))

    # ------------------------------------------------------------------ finalize

    def _finalize(self) -> dict:
        if len(self.times_s) < 2:
            self.get_logger().warning(
                f"trial '{self.label}' collected only {len(self.times_s)} samples "
                "-- report will be mostly empty/zero")

        t0 = self.times_s[0] if self.times_s else 0.0
        rel_times = [t - t0 for t in self.times_s]
        dt = ([rel_times[0]] if rel_times else []) + [
            rel_times[i] - rel_times[i - 1] for i in range(1, len(rel_times))]

        exposure_total, exposure_rate = em.risk_exposure(self.risk_samples, dt)
        length = em.path_length(self.xy)
        exposure_per_m = em.risk_exposure_per_distance(exposure_total, length)

        exposure_raw_total, exposure_raw_rate = em.risk_exposure(self.raw_risk_samples, dt)
        exposure_raw_per_m = em.risk_exposure_per_distance(exposure_raw_total, length)

        encounters = em.clearance_encounters(
            self.distances, rel_times, self.engagement_threshold_m)
        clearance_stats = em.clearance_distribution_stats(encounters)

        speed = [0.0]
        for i in range(1, len(self.xy)):
            step_dt = max(1e-6, rel_times[i] - rel_times[i - 1])
            dx = self.xy[i][0] - self.xy[i - 1][0]
            dy = self.xy[i][1] - self.xy[i - 1][1]
            speed.append(math.hypot(dx, dy) / step_dt)
        stop_count, stopped_time = em.stop_events(
            speed, rel_times, self.stop_speed_mps)
        smoothness = em.velocity_smoothness(speed, rel_times)

        det0 = min(self.detection_stamps_s + self.costmap_stamps_s, default=0.0)
        latencies = em.detection_to_costmap_latency(
            [s - det0 for s in self.detection_stamps_s],
            [s - det0 for s in self.costmap_stamps_s])
        rtf = em.realtime_factor(self.costmap_stamps_s, self.configured_period_s)

        report = em.RunReport(
            label=self.label,
            time_to_goal_s=em.time_to_goal(0.0, rel_times[-1] if rel_times else 0.0),
            path_length_m=length,
            stop_count=stop_count,
            stopped_time_s=stopped_time,
            velocity_smoothness=smoothness,
            replan_count=em.replan_count(self.replan_stamps_s),
            risk_exposure_total=exposure_total,
            risk_exposure_rate=exposure_rate,
            risk_exposure_per_m=exposure_per_m,
            risk_exposure_raw_total=exposure_raw_total,
            risk_exposure_raw_rate=exposure_raw_rate,
            risk_exposure_raw_per_m=exposure_raw_per_m,
            clearance=clearance_stats,
            detection_to_costmap_latency_s=latencies,
            realtime_factor=rtf,
        )
        return report.summary()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvaluationNode()
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
