#!/usr/bin/env python3
"""
scan_cluster_detector_node.py  --  lidar dynamic-cluster detector (WP1)

Publishes vision_msgs/Detection3DArray clusters of laser-scan returns that
are NOT explained by the static map, onto the SAME shared topic the two
camera chains use (/risk_perception/detections_3d_map, see
risk_perception.yaml) -- object_tracker_node fuses them in for free, via its
label-agnostic association pass keyed on `label_agnostic_labels`
(default ["lidar_cluster"], see object_tracker_node.py).

Pipeline per scan (pure math in risk_perception/scan_clustering.py, so it is
unit-testable without ROS -- see test/test_scan_clustering.py):

  1. Static-map subtraction (scan_clustering.subtract_static): a distance-
     transform field over the latest `map` message, built once per map (see
     `_map_cb`), drops any beam return within `static_margin_m` of an
     occupied cell or landing on unknown territory.
  2. Range-adaptive Euclidean clustering (scan_clustering.cluster_points): a
     beam gap that would be tight up close is loose far away, so the gap
     threshold grows with range (`cluster_gap_m` + `cluster_gap_per_m` * r).
  3. Per-cluster summary + filters (scan_clustering.cluster_summaries):
     drops sparse clusters (`min_points`), oversized ones (`max_extent_m`,
     almost always a static-subtraction miss on a large flat surface), and
     the robot's own lidar-visible footprint (`self_radius_m`, robot
     map-frame pose via TF `map -> base_frame`).
  4. Cross-scan persistence (scan_clustering.ClusterPersistence, WP-B
     2026-09-10): a cluster is only ever published once it has matched a
     cluster in each of the last `min_consecutive_scans` scans within
     `persist_match_m` -- see that class's docstring for why. Filters out
     the shelf-edge flickers `static_margin_m` alone let through; a real
     mover keeps matching scan-to-scan and clears this in
     `min_consecutive_scans` scans' worth of time.

Covariance is deliberately SMALLER than every other source publishing on
this topic -- global_cam_projector_node's default 0.10 m^2 (ground-plane
projection) and object_tracker_node's own default_cov_m2 fallback of
0.04 m^2 (the RGB-D path's implicit zero-covariance case) -- because a
centimetre-range lidar return really is that much more precise than either
camera path; see object_tracker_node.py's module docstring for the fusion
rule this is exploiting (smaller covariance wins the Kalman update).
"""

from typing import List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from risk_perception.geometry_utils import quaternion_to_rotation_matrix
from risk_perception.scan_clustering import (
    build_static_distance_field, cluster_points, cluster_summaries,
    ClusterPersistence, subtract_static)


class ScanClusterDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_cluster_detector")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("detections_topic", "/risk_perception/detections_3d_map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("max_range_m", 12.0)
        # Raised from 0.15 (2026-09-10 finding, see this node's module
        # docstring / object_tracker_node.py's module docstring): AMCL
        # error + map-quantisation rounding let real shelf edges survive
        # inside 0.15 m of the occupied cell they belong to, sprouting a
        # lidar_cluster there that the tracker then reads as a jump between
        # shelf segments. 0.35 m clears both AMCL's typical error band and
        # a full map cell's worth of quantisation.
        self.declare_parameter("static_margin_m", 0.35)
        self.declare_parameter("self_radius_m", 0.25)
        self.declare_parameter("cluster_gap_m", 0.25)
        self.declare_parameter("cluster_gap_per_m", 0.02)
        self.declare_parameter("min_points", 3)
        self.declare_parameter("max_extent_m", 1.5)
        # Near-face centroid correction (lidar-fusion probe, 2026-09-09): a
        # lidar beam only returns off the side of an object facing the
        # sensor, so the raw mean-of-points centroid sits biased toward the
        # sensor by roughly half the object's extent -- see
        # scan_clustering.push_centroid_from_sensor. 0 disables (either
        # param).
        self.declare_parameter("centroid_push_factor", 0.5)
        self.declare_parameter("centroid_push_max_m", 0.30)
        self.declare_parameter("detection_score", 0.5)
        # Raised from 0.02 (2026-09-10 finding): a lidar cluster's
        # covariance has to stay clearly SMALLER than every camera-path
        # covariance for the KF fusion rule to work as intended (smaller
        # covariance wins), but 0.02 was small enough that even a jumpy,
        # low-confidence shelf-edge cluster could outweigh a well-tracked
        # camera detection outright. 0.05 keeps the ordering (still well
        # under global_cam_projector's 0.10 and default_cov_m2's 0.04
        # fallback) with more headroom.
        self.declare_parameter("cov_base_m2", 0.05)
        self.declare_parameter("cov_per_m", 0.002)
        # Beam decimation -- 1 = every beam. A cheap way to cut the per-scan
        # point count (and therefore clustering cost) on a dense lidar.
        self.declare_parameter("stride", 1)
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("publish_markers", False)
        # Cross-scan persistence (WP-B, 2026-09-10) -- see
        # scan_clustering.ClusterPersistence's docstring and this node's
        # module docstring, step 4.
        self.declare_parameter("min_consecutive_scans", 3)
        self.declare_parameter("persist_match_m", 0.3)

        gp = self.get_parameter
        scan_topic = str(gp("scan_topic").value)
        map_topic = str(gp("map_topic").value)
        detections_topic = str(gp("detections_topic").value)
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.max_range = float(gp("max_range_m").value)
        self.static_margin = float(gp("static_margin_m").value)
        self.self_radius = float(gp("self_radius_m").value)
        self.gap_base = float(gp("cluster_gap_m").value)
        self.gap_per_m = float(gp("cluster_gap_per_m").value)
        self.min_points = int(gp("min_points").value)
        self.max_extent = float(gp("max_extent_m").value)
        self.centroid_push_factor = float(gp("centroid_push_factor").value)
        self.centroid_push_max_m = float(gp("centroid_push_max_m").value)
        self.detection_score = float(gp("detection_score").value)
        self.cov_base = float(gp("cov_base_m2").value)
        self.cov_per_m = float(gp("cov_per_m").value)
        self.stride = max(1, int(gp("stride").value))
        self.tf_timeout = float(gp("tf_timeout_sec").value)
        self.publish_markers_enabled = bool(gp("publish_markers").value)
        self.min_consecutive_scans = int(gp("min_consecutive_scans").value)
        self.persist_match_m = float(gp("persist_match_m").value)
        self._persistence = ClusterPersistence(
            min_consecutive=self.min_consecutive_scans, match_m=self.persist_match_m)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._occ_grid_info: Optional[dict] = None
        self._dist_transform_m: Optional[np.ndarray] = None

        # RELIABLE + TRANSIENT_LOCAL + KeepLast(1), matching map_server's own
        # QoS (and mission_supervisor.py's `map` subscription) -- a late
        # subscriber still gets the one latched map message.
        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, map_qos)
        self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)

        self.publisher = self.create_publisher(Detection3DArray, detections_topic, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "~/markers", 10)

        self.get_logger().info(
            f"scan_cluster_detector: {scan_topic} (+ {map_topic}) -> {detections_topic}")

    def _map_cb(self, msg: OccupancyGrid) -> None:
        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        occ_grid_info = {
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
            "width": int(msg.info.width),
            "height": int(msg.info.height),
        }
        self._dist_transform_m = build_static_distance_field(grid, occ_grid_info)
        self._occ_grid_info = occ_grid_info

    def _resolve_robot_xy(self) -> Optional[np.ndarray]:
        """Robot map-frame (x, y) for cluster self-exclusion, or None.

        Latest available TF (no wait -- this is a best-effort filter, not
        worth blocking scan processing on), same failure mode as
        global_cam_projector_node's TF fallback: no transform yet (AMCL
        unseeded) simply disables the self_radius_m filter for this scan.
        """
        try:
            tr = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
            return np.array(
                [tr.transform.translation.x, tr.transform.translation.y],
                dtype=np.float64)
        except TransformException:
            return None

    def _scan_cb(self, msg: LaserScan) -> None:
        if self._occ_grid_info is None or self._dist_transform_m is None:
            self.get_logger().warning(
                "no static map yet -- skipping scan", throttle_duration_sec=5.0)
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, msg.header.stamp,
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException as error:
            self.get_logger().warning(
                f"cannot transform {msg.header.frame_id} -> {self.map_frame} "
                f"at scan stamp: {error}", throttle_duration_sec=5.0)
            return

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        n = ranges.shape[0]
        if n == 0:
            return

        indices = np.arange(0, n, self.stride)
        r = ranges[indices]
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= self.max_range)
        indices = indices[valid]
        r = r[valid]
        if indices.shape[0] == 0:
            self._publish(msg.header.stamp, [])
            return

        angles = msg.angle_min + indices.astype(np.float64) * msg.angle_increment
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points_scan = np.stack([xs, ys, np.zeros_like(xs)], axis=1)

        t = transform.transform.translation
        q = transform.transform.rotation
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        origin = np.array([t.x, t.y, t.z], dtype=np.float64)
        points_map = points_scan @ R.T + origin
        points_map_xy = points_map[:, :2]

        keep = subtract_static(
            points_map_xy, self._occ_grid_info, self._dist_transform_m,
            self.static_margin)
        points_map_xy = points_map_xy[keep]
        r = r[keep]
        if points_map_xy.shape[0] == 0:
            self._publish(msg.header.stamp, [])
            return

        robot_xy = self._resolve_robot_xy()

        clusters = cluster_points(points_map_xy, self.gap_base, self.gap_per_m, r)
        # `origin` is the LIDAR's own map-frame position (the transform this
        # scan's points were just projected through), not the robot's
        # base_frame footprint -- see push_centroid_from_sensor's docstring
        # for why the push has to originate there.
        summaries = cluster_summaries(
            points_map_xy, r, clusters, self.min_points, self.max_extent,
            self_radius_m=self.self_radius, robot_xy=robot_xy,
            sensor_xy=origin[:2], centroid_push_factor=self.centroid_push_factor,
            centroid_push_max_m=self.centroid_push_max_m)

        self._publish(msg.header.stamp, summaries)

    def _publish(self, stamp, summaries: List[dict]) -> None:
        # Cross-scan persistence (WP-B, 2026-09-10) -- feed EVERY scan's
        # centroids through, including an empty list on a scan with no
        # surviving points, so a candidate whose cluster genuinely
        # disappears for a scan has its streak reset like any other
        # unmatched scan (see ClusterPersistence's class docstring). Only
        # the CONFIRMED subset is ever published from here on.
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        confirmed_xy = set(self._persistence.update(
            [s["centroid"] for s in summaries], stamp_sec))
        summaries = [s for s in summaries if s["centroid"] in confirmed_xy]

        output = Detection3DArray()
        output.header.stamp = stamp
        output.header.frame_id = self.map_frame

        for s in summaries:
            cx, cy = s["centroid"]
            w, h = s["extent"]
            mean_range = s["mean_range"]
            cov = self.cov_base + self.cov_per_m * mean_range

            detection = Detection3D()
            detection.header = output.header
            detection.bbox.center.position.x = cx
            detection.bbox.center.position.y = cy
            detection.bbox.center.position.z = 0.0
            detection.bbox.center.orientation.w = 1.0
            detection.bbox.size.x = max(w, 0.05)
            detection.bbox.size.y = max(h, 0.05)
            detection.bbox.size.z = 0.3

            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = "lidar_cluster"
            result.hypothesis.score = self.detection_score
            result.pose.pose = detection.bbox.center

            covariance = [0.0] * 36
            covariance[0] = cov
            covariance[7] = cov
            result.pose.covariance = covariance

            detection.results.append(result)
            output.detections.append(detection)

        if output.detections:
            self.publisher.publish(output)

        if self.publish_markers_enabled:
            self._publish_markers(output)

    def _publish_markers(self, detections: Detection3DArray) -> None:
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        lifetime = Duration(seconds=0.5).to_msg()

        for index, detection in enumerate(detections.detections):
            marker = Marker()
            marker.header = detections.header
            marker.ns = "lidar_clusters"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = detection.bbox.center
            marker.scale.x = detection.bbox.size.x
            marker.scale.y = detection.bbox.size.y
            marker.scale.z = detection.bbox.size.z
            marker.color.r = 0.1
            marker.color.g = 0.7
            marker.color.b = 1.0
            marker.color.a = 0.6
            marker.lifetime = lifetime
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanClusterDetectorNode()
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
