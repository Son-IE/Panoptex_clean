#!/usr/bin/env python3

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from risk_perception.risk_visualization import label_category, is_operable_machine
from risk_perception.mask_relation import tag_operator_relations


class RgbdProjectorNode(Node):
    """Projects instance masks into 3D using aligned depth and CameraInfo."""

    def __init__(self) -> None:
        super().__init__("rgbd_projector")

        self.declare_parameter(
            "depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera/color/camera_info",
        )
        self.declare_parameter(
            "detections_topic",
            "/risk_perception/detections_2d",
        )
        self.declare_parameter(
            "mask_topic",
            "/risk_perception/instance_mask",
        )
        self.declare_parameter(
            "output_topic",
            "/risk_perception/detections_3d",
        )

        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("sync_tolerance_sec", 0.05)
        self.declare_parameter("cache_size", 180)
        self.declare_parameter("max_points_per_instance", 5000)
        self.declare_parameter("mask_erosion_pixels", 3)

        # Relation prior (geometric): person-mask-inside-machine-mask
        # containment, replacing gdino_detector's language-phrase grounding
        # (found empirically to parse per-noun rather than a bound relation
        # -- see the relation-prior playbook). Pure geometry lives in
        # risk_perception.mask_relation; this node only supplies the
        # per-instance masks + 3D centroids it already computes.
        self.declare_parameter("relation_containment_threshold", 0.6)
        self.declare_parameter("relation_dilate_px", 5)
        self.declare_parameter("relation_max_distance_m", 2.5)

        depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        detections_topic = str(
            self.get_parameter("detections_topic").value
        )
        mask_topic = str(
            self.get_parameter("mask_topic").value
        )
        output_topic = str(
            self.get_parameter("output_topic").value
        )

        self.depth_scale = float(
            self.get_parameter("depth_scale").value
        )
        self.min_depth_m = float(
            self.get_parameter("min_depth_m").value
        )
        self.max_depth_m = float(
            self.get_parameter("max_depth_m").value
        )
        self.sync_tolerance_ns = int(
            float(
                self.get_parameter(
                    "sync_tolerance_sec"
                ).value
            )
            * 1e9
        )
        self.cache_size = max(
            10,
            int(self.get_parameter("cache_size").value),
        )
        self.max_points = max(
            100,
            int(
                self.get_parameter(
                    "max_points_per_instance"
                ).value
            ),
        )
        self.erosion_pixels = max(
            0,
            int(
                self.get_parameter(
                    "mask_erosion_pixels"
                ).value
            ),
        )

        self.relation_containment_threshold = float(
            self.get_parameter("relation_containment_threshold").value
        )
        self.relation_dilate_px = int(
            self.get_parameter("relation_dilate_px").value
        )
        self.relation_max_distance_m = float(
            self.get_parameter("relation_max_distance_m").value
        )

        self.bridge = CvBridge()

        # Set the first time mask_callback resizes depth to match the mask
        # (Isaac Sim: 320x240 32FC1 depth vs 640x480 masks) -- logged once,
        # not every frame, since the shape relationship doesn't change once
        # the sim is running at a fixed render size.
        self._depth_resize_logged = False

        self.depth_cache = OrderedDict()
        self.detection_cache = OrderedDict()

        self.camera_info: Optional[CameraInfo] = None

        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self.detection_callback,
            10,
        )

        self.mask_subscription = self.create_subscription(
            Image,
            mask_topic,
            self.mask_callback,
            10,
        )

        self.publisher = self.create_publisher(
            Detection3DArray,
            output_topic,
            10,
        )

        self.get_logger().info(
            f"Depth topic: {depth_topic}"
        )
        self.get_logger().info(
            f"CameraInfo topic: {camera_info_topic}"
        )
        self.get_logger().info(
            f"Detections topic: {detections_topic}"
        )
        self.get_logger().info(
            f"Mask topic: {mask_topic}"
        )
        self.get_logger().info(
            f"3D output topic: {output_topic}"
        )

    @staticmethod
    def stamp_ns(msg) -> int:
        return (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

    def trim_cache(self, cache: OrderedDict) -> None:
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def depth_callback(self, msg: Image) -> None:
        key = self.stamp_ns(msg)
        self.depth_cache[key] = msg
        self.trim_cache(self.depth_cache)

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:
        self.camera_info = msg

    def detection_callback(
        self,
        msg: Detection2DArray,
    ) -> None:
        key = self.stamp_ns(msg)
        self.detection_cache[key] = msg
        self.trim_cache(self.detection_cache)

    def nearest_message(
        self,
        cache: OrderedDict,
        target_stamp: int,
    ):
        if target_stamp in cache:
            return cache[target_stamp]

        if not cache:
            return None

        nearest_key = min(
            cache.keys(),
            key=lambda key: abs(key - target_stamp),
        )

        if (
            abs(nearest_key - target_stamp)
            > self.sync_tolerance_ns
        ):
            return None

        return cache[nearest_key]

    def convert_depth_to_meters(
        self,
        depth_msg: Image,
    ) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

        depth = np.asarray(depth)

        if depth_msg.encoding == "16UC1":
            return depth.astype(np.float32) * self.depth_scale

        if depth_msg.encoding == "32FC1":
            return depth.astype(np.float32)

        raise ValueError(
            "Unsupported depth encoding: "
            f"{depth_msg.encoding}"
        )

    def mask_callback(self, mask_msg: Image) -> None:
        if self.camera_info is None:
            self.get_logger().warning(
                "CameraInfo has not been received yet",
                throttle_duration_sec=5.0,
            )
            return

        target_stamp = self.stamp_ns(mask_msg)

        depth_msg = self.nearest_message(
            self.depth_cache,
            target_stamp,
        )
        detections_2d = self.nearest_message(
            self.detection_cache,
            target_stamp,
        )

        if depth_msg is None:
            # Fires at the full ~15 Hz mask rate whenever depth briefly lags
            # (or is entirely absent, e.g. robot-cam chain off) -- throttle
            # or this drowns everything else in the log.
            self.get_logger().warning(
                "No timestamp-matched aligned depth frame",
                throttle_duration_sec=5.0,
            )
            return

        if detections_2d is None:
            self.get_logger().warning(
                "No timestamp-matched Detection2DArray",
                throttle_duration_sec=5.0,
            )
            return

        try:
            instance_mask = self.bridge.imgmsg_to_cv2(
                mask_msg,
                desired_encoding="passthrough",
            )
            depth_m = self.convert_depth_to_meters(
                depth_msg
            )
        except (CvBridgeError, ValueError) as error:
            self.get_logger().error(str(error))
            return

        instance_mask = np.asarray(
            instance_mask,
            dtype=np.uint16,
        )

        if instance_mask.shape != depth_m.shape:
            mask_h, mask_w = instance_mask.shape[:2]
            depth_h, depth_w = depth_m.shape[:2]

            # Isaac Sim publishes depth at 320x240 32FC1 while the colour
            # image (and therefore the mask, which is sized off it) is
            # 640x480 -- a uniform 2x scale, not a different sensor/FOV. When
            # both axes scale by the same ratio, upsample depth onto the
            # mask's pixel grid with nearest-neighbour (no interpolation
            # across a depth discontinuity) rather than reject the frame.
            # The colour CameraInfo K used below still applies unchanged
            # after this resize: depth is rendered/aligned to the SAME
            # camera and FOV as colour in Isaac (unlike the real Astra,
            # which needs depth_registration:=true for the same reason) --
            # only the sampling grid density differed, not the optics.
            width_ratio = mask_w / float(depth_w)
            height_ratio = mask_h / float(depth_h)

            if depth_w > 0 and depth_h > 0 and abs(width_ratio - height_ratio) < 1e-3:
                depth_m = cv2.resize(
                    depth_m,
                    (mask_w, mask_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                if not self._depth_resize_logged:
                    self.get_logger().info(
                        "Depth "
                        f"{depth_w}x{depth_h} != mask {mask_w}x{mask_h} "
                        f"but uniform ratio ({width_ratio:.3f}x) -- "
                        "nearest-neighbour resizing depth onto the mask's "
                        "grid (Isaac Sim: depth renders at half colour "
                        "resolution)."
                    )
                    self._depth_resize_logged = True
            else:
                self.get_logger().error(
                    "Mask/depth shape mismatch: "
                    f"{instance_mask.shape} vs {depth_m.shape}"
                )
                return

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error(
                "Invalid camera intrinsics"
            )
            return

        output = Detection3DArray()
        output.header = mask_msg.header

        # Collected alongside the main loop below, consumed once after it by
        # tag_operator_relations -- see that function's docstring.
        person_entries = []   # (mask: bool ndarray, xyz)
        machine_entries = []  # (mask: bool ndarray, xyz, hypothesis)

        # Prefer the calibrated optical frame from CameraInfo.
        if self.camera_info.header.frame_id:
            output.header.frame_id = (
                self.camera_info.header.frame_id
            )

        for index, detection_2d in enumerate(
            detections_2d.detections
        ):
            instance_id = index + 1

            binary_mask = (
                instance_mask == instance_id
            ).astype(np.uint8)

            if binary_mask.sum() == 0:
                continue

            if self.erosion_pixels > 0:
                kernel_size = (
                    self.erosion_pixels * 2 + 1
                )
                kernel = np.ones(
                    (kernel_size, kernel_size),
                    dtype=np.uint8,
                )

                eroded = cv2.erode(
                    binary_mask,
                    kernel,
                    iterations=1,
                )

                # Fall back if erosion removes a small object.
                if eroded.sum() > 20:
                    binary_mask = eroded

            valid = (
                (binary_mask > 0)
                & np.isfinite(depth_m)
                & (depth_m >= self.min_depth_m)
                & (depth_m <= self.max_depth_m)
            )

            rows, cols = np.nonzero(valid)

            if len(rows) < 20:
                # Per instance per frame otherwise -- 449 lines in a 12 min
                # sim run, drowning everything else.
                self.get_logger().warning(
                    f"Instance {instance_id}: "
                    "insufficient valid depth pixels",
                    throttle_duration_sec=5.0,
                )
                continue

            if len(rows) > self.max_points:
                selected = np.linspace(
                    0,
                    len(rows) - 1,
                    self.max_points,
                    dtype=np.int64,
                )
                rows = rows[selected]
                cols = cols[selected]

            z = depth_m[rows, cols]

            median_z = float(np.median(z))

            # Remove background/foreground outliers.
            depth_band = np.abs(z - median_z) < 0.40

            rows = rows[depth_band]
            cols = cols[depth_band]
            z = z[depth_band]

            if len(z) < 20:
                continue

            x = (cols.astype(np.float32) - cx) * z / fx
            y = (rows.astype(np.float32) - cy) * z / fy

            points = np.column_stack((x, y, z))

            center = np.median(points, axis=0)

            lower = np.percentile(
                points,
                5.0,
                axis=0,
            )
            upper = np.percentile(
                points,
                95.0,
                axis=0,
            )
            size = np.maximum(
                upper - lower,
                0.01,
            )

            detection_3d = Detection3D()
            detection_3d.header = output.header
            detection_3d.id = detection_2d.id

            detection_3d.bbox.center.position.x = float(
                center[0]
            )
            detection_3d.bbox.center.position.y = float(
                center[1]
            )
            detection_3d.bbox.center.position.z = float(
                center[2]
            )
            detection_3d.bbox.center.orientation.w = 1.0

            detection_3d.bbox.size.x = float(size[0])
            detection_3d.bbox.size.y = float(size[1])
            detection_3d.bbox.size.z = float(size[2])

            if detection_2d.results:
                source = (
                    detection_2d.results[0].hypothesis
                )

                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = (
                    source.class_id
                )
                result.hypothesis.score = (
                    source.score
                )

                result.pose.pose.position.x = float(
                    center[0]
                )
                result.pose.pose.position.y = float(
                    center[1]
                )
                result.pose.pose.position.z = float(
                    center[2]
                )
                result.pose.pose.orientation.w = 1.0

                detection_3d.results.append(result)

                # bare label, in case a class_id already carries a "|..."
                # suffix from elsewhere -- category lookup must not see it.
                bare_label = str(source.class_id).split("|")[0]
                xyz = (float(center[0]), float(center[1]), float(center[2]))
                mask_bool = binary_mask.astype(bool)
                if label_category(bare_label) == "person":
                    person_entries.append((mask_bool, xyz))
                elif is_operable_machine(bare_label):
                    machine_entries.append((mask_bool, xyz, result.hypothesis))

            output.detections.append(detection_3d)

        tag_operator_relations(
            person_entries, machine_entries,
            containment_threshold=self.relation_containment_threshold,
            dilate_outer_px=self.relation_dilate_px,
            max_distance_m=self.relation_max_distance_m,
        )

        self.publisher.publish(output)

        if output.detections:
            summary = ", ".join(
                (
                    f"{d.results[0].hypothesis.class_id}: "
                    f"{d.bbox.center.position.z:.2f} m"
                )
                if d.results
                else (
                    f"object: "
                    f"{d.bbox.center.position.z:.2f} m"
                )
                for d in output.detections
            )

            self.get_logger().info(
                f"Projected {len(output.detections)} "
                f"objects — {summary}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbdProjectorNode()

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
