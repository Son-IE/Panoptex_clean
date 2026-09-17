#!/usr/bin/env python3
"""
global_cam_projector_node.py  --  TRACK 1c  (depth-free ground-plane projection)

The RGB-D pipeline gets 3D object positions from depth. The overhead camera
has none, so this node uses the ground-plane assumption instead: every
object of interest (person, chair, cart, ...) touches the floor somewhere in
its footprint, so the BOTTOM-CENTER pixel of its SAM2 mask is (approximately)
that floor-contact point. Undistort that pixel into a camera-frame ray,
rotate the ray into `map` using the extrinsic solved by
global_cam_calibrator_node (TF `map -> <camera optical frame>`), and
intersect it with the z=0 plane.

This publishes straight onto the SAME topic the RGB-D chain uses
(/risk_perception/detections_3d_map, see risk_perception.yaml), so
object_tracker_node fuses both cameras with no changes to it -- a chair seen
by both becomes one track, not two.

Ground-plane positions are inherently less certain than RGB-D range
(grazing angle near the image edges, and it breaks for objects with
overhang -- e.g. a table gives the point where its LEGS meet the floor, not
its top surface), so this node reports a covariance LARGER than the RGB-D
path's effective default (object_tracker_node falls back to
`default_cov_m2=0.04` when a detection's covariance is zero, which is what
the RGB-D path always publishes today) -- so the tracker prefers RealSense
wherever both cameras see the same object.
"""

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from risk_perception.geometry_utils import quaternion_to_rotation_matrix
from risk_perception.risk_visualization import label_category, is_operable_machine
from risk_perception.mask_relation import tag_operator_relations


def apply_extent_offset(
    point_xy: np.ndarray,
    cam_xy: np.ndarray,
    extent: float,
) -> np.ndarray:
    """Pushes a floor point away from the camera by `extent` metres.

    The mask's bottom-band pixel (see `_mask_cb`'s floor-contact-point
    comment) is the NEAR edge of the object's footprint as seen from the
    overhead camera, not its centre -- a person standing with their feet
    toward the camera projects half a body-width short of where they
    actually are, and the effect is worse for a physically larger object
    (a Carter-class AMR's footprint is ~0.35 m in front of its centre from
    a grazing overhead angle). This corrects for that by moving the point
    further along the camera->point horizontal ray by a per-category
    half-extent (see `GlobalCamProjectorNode`'s `extent_offset_*_m`
    params), so `range_m`/covariance and the self-exclusion test downstream
    both see the corrected position, not the near-edge one.

    Pure function (no ROS/node state) so it can be unit-tested directly --
    see test/test_projector_extent.py. `extent=0.0` is exact identity
    (`other`-category objects, furniture and unknown labels, get no
    correction by default); the degenerate case where the point already
    sits on the camera's own (x, y) -- direction undefined -- also returns
    the input unchanged rather than dividing by zero.
    """
    point_xy = np.asarray(point_xy, dtype=np.float64)
    cam_xy = np.asarray(cam_xy, dtype=np.float64)

    if extent == 0.0:
        return point_xy.copy()

    delta = point_xy - cam_xy
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return point_xy.copy()

    direction = delta / norm
    return point_xy + extent * direction


class GlobalCamProjectorNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_projector")

        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("detections_topic", "/global_cam/detections_2d")
        self.declare_parameter("mask_topic", "/global_cam/instance_mask")
        self.declare_parameter("output_topic", "/risk_perception/detections_3d_map")
        self.declare_parameter("marker_topic", "/global_cam/map_markers")
        self.declare_parameter("marker_lifetime_sec", 1.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("sync_tolerance_sec", 0.2)
        self.declare_parameter("cache_size", 60)
        self.declare_parameter("transform_timeout_sec", 0.25)
        self.declare_parameter("base_cov_m2", 0.10)
        self.declare_parameter("cov_reference_range_m", 3.0)
        self.declare_parameter("max_cov_m2", 4.0)
        self.declare_parameter("bottom_band_px", 3)
        self.declare_parameter("default_footprint_m", 0.30)
        # Near-edge correction (lab+sim default, User B's decision 2026-09):
        # the mask's bottom-band pixel is the object's floor-contact point
        # NEAREST the camera, not its centre -- push it further along the
        # camera->point horizontal ray by half the object's typical extent,
        # keyed off risk_visualization.label_category() (person/robot/
        # wheeled/other-everything-else). 0.20 m for a person (half a
        # shoulder-width stance), 0.35 m for robot/wheeled (half a
        # Carter-class AMR's ~0.7 m footprint), 0.0 m (no correction) for
        # furniture/unknown -- their bottom-band pixel is usually already
        # close to their footprint centre (a table/chair leg), and an
        # unknown label has no reliable extent to assume. See
        # apply_extent_offset() above and test/test_projector_extent.py.
        self.declare_parameter("extent_offset_person_m", 0.20)
        self.declare_parameter("extent_offset_robot_m", 0.35)
        self.declare_parameter("extent_offset_wheeled_m", 0.35)
        self.declare_parameter("extent_offset_other_m", 0.0)
        # Grazing-view guard: at ~24 deg below horizontal (the sim overhead
        # mounts), a mask whose bottom pixel lands near the top of the image
        # projects to a floor point tens of metres away -- covariance alone
        # caps how confident the tracker is in that point, not whether it
        # gets published at all. 0.0 = disabled (default; the real lab
        # cameras are close/steep enough not to need this).
        self.declare_parameter("max_range_m", 0.0)
        # Marker appearance -- was hardcoded (0.25 sphere, white 0.18m text
        # with an x=/y= second line), which was oversized and unreadable
        # against a white map.
        # Self-exclusion: the overhead camera sees the robot itself, and
        # without a gate the robot becomes a track that paints risk around
        # its own position -- Nav2 then refuses to plan near the robot.
        # Drop detections whose floor point lands within an exclusion radius
        # of the robot's map-frame position (tag-0 pose preferred, TF
        # fallback; no pose at all -> no filtering, so bench mode is inert).
        self.declare_parameter("robot_pose_topic", "/global_cam/robot_pose")
        self.declare_parameter("self_exclusion_enabled", True)
        self.declare_parameter("self_exclusion_radius_m", 0.40)
        self.declare_parameter("self_exclusion_robot_radius_m", 0.80)
        self.declare_parameter("self_pose_max_age_sec", 1.5)
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("marker_sphere_scale_m", 0.15)
        self.declare_parameter("marker_text_height_m", 0.11)
        self.declare_parameter("marker_text_color", [0.05, 0.05, 0.08])
        self.declare_parameter("marker_show_coords", False)

        # Relation prior (geometric) -- see rgbd_projector_node.py's block of
        # the same three parameters for the full rationale; kept identical
        # here so both camera chains produce comparable relconf values.
        self.declare_parameter("relation_containment_threshold", 0.6)
        self.declare_parameter("relation_dilate_px", 5)
        self.declare_parameter("relation_max_distance_m", 2.5)

        gp = self.get_parameter
        camera_info_topic = str(gp("camera_info_topic").value)
        detections_topic = str(gp("detections_topic").value)
        mask_topic = str(gp("mask_topic").value)
        output_topic = str(gp("output_topic").value)
        marker_topic = str(gp("marker_topic").value)

        self.marker_lifetime = float(gp("marker_lifetime_sec").value)
        self.map_frame = str(gp("map_frame").value)
        self.sync_tolerance_ns = int(float(gp("sync_tolerance_sec").value) * 1e9)
        self.cache_size = max(10, int(gp("cache_size").value))
        self.transform_timeout = float(gp("transform_timeout_sec").value)
        self.base_cov = float(gp("base_cov_m2").value)
        self.ref_range = max(0.1, float(gp("cov_reference_range_m").value))
        self.max_cov = float(gp("max_cov_m2").value)
        self.bottom_band_px = max(1, int(gp("bottom_band_px").value))
        self.default_footprint = float(gp("default_footprint_m").value)
        self.extent_offset_by_category = {
            "person": float(gp("extent_offset_person_m").value),
            "robot": float(gp("extent_offset_robot_m").value),
            "wheeled": float(gp("extent_offset_wheeled_m").value),
        }
        self.extent_offset_other = float(gp("extent_offset_other_m").value)
        self.max_range = float(gp("max_range_m").value)
        self.marker_sphere_scale = float(gp("marker_sphere_scale_m").value)
        self.marker_text_height = float(gp("marker_text_height_m").value)
        text_color = [float(c) for c in gp("marker_text_color").value]
        self.marker_text_color = (text_color + [0.0, 0.0, 0.0])[:3]
        self.marker_show_coords = bool(gp("marker_show_coords").value)

        robot_pose_topic = str(gp("robot_pose_topic").value)
        self.self_exclusion_enabled = bool(gp("self_exclusion_enabled").value)
        self.self_exclusion_radius = float(gp("self_exclusion_radius_m").value)
        self.self_exclusion_robot_radius = float(gp("self_exclusion_robot_radius_m").value)
        self.self_pose_max_age = float(gp("self_pose_max_age_sec").value)
        self.robot_base_frame = str(gp("robot_base_frame").value)

        self.relation_containment_threshold = float(
            gp("relation_containment_threshold").value)
        self.relation_dilate_px = int(gp("relation_dilate_px").value)
        self.relation_max_distance_m = float(gp("relation_max_distance_m").value)

        self.bridge = CvBridge()
        self.detection_cache: "OrderedDict" = OrderedDict()
        self.camera_info: Optional[CameraInfo] = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose_xy: Optional[np.ndarray] = None
        self.robot_pose_stamp_ns = 0
        self.create_subscription(
            PoseWithCovarianceStamped, robot_pose_topic, self._robot_pose_cb, 10)

        self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, detections_topic, self._detections_cb, 10)
        self.create_subscription(Image, mask_topic, self._mask_cb, 10)

        self.publisher = self.create_publisher(Detection3DArray, output_topic, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, marker_topic, 10)

        self.get_logger().info(
            f"Global-cam ground-plane projector: {mask_topic} -> {output_topic} "
            f"(+ {marker_topic} for RViz)")

    @staticmethod
    def stamp_ns(msg) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _robot_pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.robot_pose_xy = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float64)
        self.robot_pose_stamp_ns = self.stamp_ns(msg)

    def _resolve_robot_xy(self, mask_stamp_ns: int) -> Optional[np.ndarray]:
        """Robot map-frame (x, y) for self-exclusion, or None to disable it.

        Prefers the tag-0 pose from the localizer: it comes from the same
        overhead image and the same map<-camera extrinsic as the projected
        detections, so it stays aligned with them even if SLAM/AMCL drifts.
        Both stamps come from the bridge clock, so the age comparison is in
        one clock domain.
        """
        if not self.self_exclusion_enabled:
            return None

        if self.robot_pose_xy is not None:
            age_sec = abs(mask_stamp_ns - self.robot_pose_stamp_ns) * 1e-9
            if age_sec <= self.self_pose_max_age:
                return self.robot_pose_xy

        # Tag 0 stale/occluded: fall back to TF. Latest transform, NO
        # timeout -- waiting would block the very executor that fills the
        # buffer (same pattern as global_cam_align_check_node).
        try:
            tr = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_base_frame, Time())
            return np.array(
                [tr.transform.translation.x, tr.transform.translation.y],
                dtype=np.float64)
        except TransformException:
            return None

    def _detections_cb(self, msg: Detection2DArray) -> None:
        key = self.stamp_ns(msg)
        self.detection_cache[key] = msg
        while len(self.detection_cache) > self.cache_size:
            self.detection_cache.popitem(last=False)

    def _nearest_detections(self, target_stamp: int) -> Optional[Detection2DArray]:
        if target_stamp in self.detection_cache:
            return self.detection_cache[target_stamp]

        if not self.detection_cache:
            return None

        nearest_key = min(self.detection_cache.keys(), key=lambda k: abs(k - target_stamp))

        if abs(nearest_key - target_stamp) > self.sync_tolerance_ns:
            return None

        return self.detection_cache[nearest_key]

    def _mask_cb(self, mask_msg: Image) -> None:
        if self.camera_info is None:
            self.get_logger().warning("No CameraInfo yet", throttle_duration_sec=5.0)
            return

        camera_frame = self.camera_info.header.frame_id or mask_msg.header.frame_id
        if not camera_frame:
            self.get_logger().warning(
                "CameraInfo/mask has no frame_id", throttle_duration_sec=5.0)
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, camera_frame, Time(),
                timeout=Duration(seconds=self.transform_timeout))
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {camera_frame} -> {self.map_frame}: {error} "
                "(is global_cam_calibrator running, and has it seen the floor tags?)",
                throttle_duration_sec=5.0)
            return

        detections_2d = self._nearest_detections(self.stamp_ns(mask_msg))
        if detections_2d is None:
            return

        try:
            instance_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="passthrough")
        except CvBridgeError as error:
            self.get_logger().error(str(error))
            return

        instance_mask = np.asarray(instance_mask, dtype=np.uint16)

        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        D = (
            np.array(self.camera_info.d, dtype=np.float64)
            if self.camera_info.d else np.zeros(5)
        )

        t = transform.transform.translation
        q = transform.transform.rotation
        R_map_cam = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        cam_origin_map = np.array([t.x, t.y, t.z], dtype=np.float64)

        output = Detection3DArray()
        output.header.stamp = mask_msg.header.stamp
        output.header.frame_id = self.map_frame

        robot_xy = self._resolve_robot_xy(self.stamp_ns(mask_msg))
        dropped_self = 0
        dropped_range = 0

        # Collected alongside the main loop below, consumed once after it by
        # tag_operator_relations -- see mask_relation.py's docstring.
        person_entries = []   # (mask: bool ndarray, xyz)
        machine_entries = []  # (mask: bool ndarray, xyz, hypothesis)

        for index, detection_2d in enumerate(detections_2d.detections):
            instance_id = index + 1
            rows, cols = np.nonzero(instance_mask == instance_id)

            if len(rows) < 10:
                continue

            # Floor-contact point: the median column across the bottommost
            # few rows of the mask (more robust to a single noisy pixel than
            # the single lowest row alone).
            bottom_row = int(rows.max())
            band = rows >= (bottom_row - self.bottom_band_px + 1)
            bottom_col = float(np.median(cols[band]))
            bottom_row_f = float(np.median(rows[band]))

            pixel = np.array([[[bottom_col, bottom_row_f]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(pixel, K, D)
            nx, ny = float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])

            direction_cam = np.array([nx, ny, 1.0], dtype=np.float64)
            direction_map = R_map_cam @ direction_cam

            if abs(direction_map[2]) < 1e-6:
                continue  # ray nearly parallel to the floor -- bad geometry

            s = -cam_origin_map[2] / direction_map[2]
            if s <= 0:
                continue  # floor intersection behind the camera -- bad geometry

            point_map = cam_origin_map + s * direction_map

            # Near-edge correction (see apply_extent_offset's docstring):
            # push the floor point away from the camera, along its own
            # horizontal viewing ray, by half the object's category extent
            # -- BEFORE the self-exclusion test below, so the X3's own
            # detections (pushed toward the robot's actual centre, since
            # the tag-0/TF pose apply_extent_offset compares against IS the
            # robot's centre) are excluded more reliably, and so
            # range_m/covariance reflect the corrected position too.
            raw_label = (detection_2d.results[0].hypothesis.class_id
                         if detection_2d.results else "")
            bare_label = str(raw_label).split("|")[0]
            extent = self.extent_offset_by_category.get(
                label_category(bare_label), self.extent_offset_other)
            point_map[:2] = apply_extent_offset(
                point_map[:2], cam_origin_map[:2], extent)

            if robot_xy is not None:
                # Anything inside the tight radius IS the robot, whatever
                # GroundingDINO called it (cart/chair mislabels included) --
                # no real object can share the robot's footprint. Robot-
                # vocabulary labels get a looser radius to absorb bottom-
                # point projection smear at grazing angles.
                label_lower = bare_label.lower()
                radius = (self.self_exclusion_robot_radius if "robot" in label_lower
                          else self.self_exclusion_radius)
                if float(np.linalg.norm(point_map[:2] - robot_xy)) <= radius:
                    dropped_self += 1
                    continue

            range_m = float(np.linalg.norm(point_map[:2] - cam_origin_map[:2]))

            if self.max_range > 0.0 and range_m > self.max_range:
                dropped_range += 1
                continue

            cov = self.base_cov * max(1.0, (range_m / self.ref_range) ** 2)
            cov = min(cov, self.max_cov)

            detection_3d = Detection3D()
            detection_3d.header = output.header
            detection_3d.id = detection_2d.id
            detection_3d.bbox.center.position.x = float(point_map[0])
            detection_3d.bbox.center.position.y = float(point_map[1])
            detection_3d.bbox.center.position.z = 0.0
            detection_3d.bbox.center.orientation.w = 1.0
            detection_3d.bbox.size.x = self.default_footprint
            detection_3d.bbox.size.y = self.default_footprint
            detection_3d.bbox.size.z = self.default_footprint

            if detection_2d.results:
                source = detection_2d.results[0].hypothesis
                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = source.class_id
                result.hypothesis.score = source.score
                result.pose.pose = detection_3d.bbox.center

                covariance = [0.0] * 36
                covariance[0] = cov
                covariance[7] = cov
                result.pose.covariance = covariance

                detection_3d.results.append(result)

                # bare_label already computed above (same source string --
                # source.class_id IS detection_2d.results[0].hypothesis.class_id).
                xyz = (float(point_map[0]), float(point_map[1]), 0.0)
                mask_bool = (instance_mask == instance_id)
                if label_category(bare_label) == "person":
                    person_entries.append((mask_bool, xyz))
                elif is_operable_machine(bare_label):
                    machine_entries.append((mask_bool, xyz, result.hypothesis))

            output.detections.append(detection_3d)

        if dropped_self:
            self.get_logger().info(
                f"self-exclusion: dropped {dropped_self} detection(s) near robot",
                throttle_duration_sec=5.0)

        if dropped_range:
            self.get_logger().info(
                f"max_range_m={self.max_range:.1f}: dropped {dropped_range} "
                "detection(s) beyond range (grazing-angle projection)",
                throttle_duration_sec=5.0)

        tag_operator_relations(
            person_entries, machine_entries,
            containment_threshold=self.relation_containment_threshold,
            dilate_outer_px=self.relation_dilate_px,
            max_distance_m=self.relation_max_distance_m,
        )

        if output.detections:
            self.publisher.publish(output)

        self._publish_markers(output)

    def _publish_markers(self, detections: Detection3DArray) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        lifetime = Duration(seconds=self.marker_lifetime).to_msg()

        for index, detection in enumerate(detections.detections):
            label = "object"
            score = 0.0

            if detection.results:
                hypothesis = detection.results[0].hypothesis
                label = str(hypothesis.class_id)
                score = float(hypothesis.score)

            sphere = Marker()
            sphere.header = detections.header
            sphere.ns = "global_cam_objects"
            sphere.id = index * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = detection.bbox.center

            sphere.scale.x = self.marker_sphere_scale
            sphere.scale.y = self.marker_sphere_scale
            sphere.scale.z = self.marker_sphere_scale

            # Orange, distinct from map_frame_projector_node's green RGB-D
            # markers -- lets you tell the two raw streams apart in RViz
            # before object_tracker_node fuses them.
            sphere.color.r = 1.0
            sphere.color.g = 0.6
            sphere.color.b = 0.0
            sphere.color.a = 0.9
            sphere.lifetime = lifetime

            text = Marker()
            text.header = detections.header
            text.ns = "global_cam_object_labels"
            text.id = index * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = detection.bbox.center.position.x
            text.pose.position.y = detection.bbox.center.position.y
            # Just clear of the sphere's top, plus a small gap -- scales with
            # both so a smaller sphere/text pairing does not leave a big gap.
            text.pose.position.z = (
                detection.bbox.center.position.z
                + self.marker_sphere_scale / 2.0 + self.marker_text_height + 0.03
            )
            text.pose.orientation.w = 1.0

            text.scale.z = self.marker_text_height
            text.color.r = self.marker_text_color[0]
            text.color.g = self.marker_text_color[1]
            text.color.b = self.marker_text_color[2]
            text.color.a = 1.0
            text.lifetime = lifetime

            text.text = f"{label} {score:.2f}"
            if self.marker_show_coords:
                text.text += (
                    f"\nx={detection.bbox.center.position.x:.2f}, "
                    f"y={detection.bbox.center.position.y:.2f}"
                )

            marker_array.markers.extend([sphere, text])

        self.marker_publisher.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamProjectorNode()
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
