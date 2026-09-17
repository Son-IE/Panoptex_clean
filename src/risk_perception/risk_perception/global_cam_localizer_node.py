#!/usr/bin/env python3
"""
global_cam_localizer_node.py  --  TRACK 1d  (tag-0 robot pose)

Watches for AprilTag id 0 (mounted on top of the Yahboom robot) in the
overhead camera and turns its detection into a robot pose in `map`:

    T_map_tag0 = T_map_cam (from global_cam_calibrator_node's TF)
                 * T_cam_tag0 (solved per-frame via solvePnP on tag 0's own
                               corners, same technique as the extrinsic
                               calibration, just for a moving tag)
    T_map_base = T_map_tag0 * T_tag0_base   (fixed mechanical offset,
                                              config/floor_tags.yaml)

This is the overhead camera's payoff for localization: it can see the robot
even when the robot's own lidar/odometry would drift or momentarily lose
track, so this pose is a candidate absolute correction fused alongside
odom/imu/lidar (see the plan's Workstream B -- a second, map-frame
robot_localization EKF taking this topic as a `pose0` input).

By itself this node ONLY PUBLISHES the pose -- it does not touch any TF or
existing localization. Compare it against the SLAM-estimated robot pose in
RViz first, before wiring it into a fusion filter.
"""

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError as exc:
    raise ImportError(
        "apriltag_msgs not found -- install apriltag_ros "
        "(ros-humble-apriltag-ros)."
    ) from exc

from risk_perception.geometry_utils import (
    quaternion_to_rotation_matrix,
    rotation_to_quaternion,
    yaw_to_rotation_z,
)

ROBOT_TAG_ID = 0


class GlobalCamLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_localizer")

        self.declare_parameter("floor_tags_yaml", "")
        self.declare_parameter("detections_topic", "/global_cam/apriltag/detections")
        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("output_topic", "/global_cam/robot_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("base_cov_m2", 0.02)
        self.declare_parameter("base_cov_yaw_rad2", 0.05)
        self.declare_parameter("transform_timeout_sec", 0.25)

        gp = self.get_parameter

        floor_tags_yaml = str(gp("floor_tags_yaml").value)
        if not floor_tags_yaml:
            raise FileNotFoundError(
                "floor_tags_yaml parameter is required -- point it at "
                "config/floor_tags.yaml (needs tag_size_m + "
                "tag0_to_base_footprint)."
            )
        with open(floor_tags_yaml) as f:
            floor_tags = yaml.safe_load(f)

        self.tag_size = float(floor_tags["tag_size_m"])
        # Tag 0 (the one this node solves) may be a different physical size
        # from the floor tags -- see floor_tags.yaml. Using the floor tags'
        # size here would scale every solved robot position.
        self.tag0_size = float(floor_tags.get("tag0_size_m", self.tag_size))
        offset = floor_tags["tag0_to_base_footprint"]
        self.offset_xyz = np.array(
            [float(offset["x"]), float(offset["y"]), float(offset["z"])])
        self.offset_yaw = float(offset["yaw"])

        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.base_cov = float(gp("base_cov_m2").value)
        self.base_cov_yaw = float(gp("base_cov_yaw_rad2").value)
        self.transform_timeout = float(gp("transform_timeout_sec").value)

        self.camera_info: Optional[CameraInfo] = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, str(gp("camera_info_topic").value), self._camera_info_cb, 10)
        self.create_subscription(
            AprilTagDetectionArray, str(gp("detections_topic").value),
            self._detections_cb, 10)

        output_topic = str(gp("output_topic").value)
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, output_topic, 10)

        self.get_logger().info(
            f"Watching for tag {ROBOT_TAG_ID} on the robot -> {output_topic} "
            f"({self.base_frame} pose in {self.map_frame})")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _detections_cb(self, msg: AprilTagDetectionArray) -> None:
        if self.camera_info is None:
            return

        detection = next(
            (d for d in msg.detections if int(d.id) == ROBOT_TAG_ID), None)
        if detection is None:
            return

        camera_frame = self.camera_info.header.frame_id or msg.header.frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, camera_frame, Time(),
                timeout=Duration(seconds=self.transform_timeout))
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {camera_frame} -> {self.map_frame}: {error}",
                throttle_duration_sec=5.0)
            return

        # Solve tag-0's pose in the CAMERA frame from its own corners (a
        # local square of tag_size, centered at the origin, z=0 in the
        # tag's own frame) -- same technique global_cam_calibrator_node uses
        # for the floor tags, just for a tag whose world pose is unknown
        # (that's what we're solving for) rather than surveyed.
        s = self.tag0_size / 2.0
        object_points = np.array(
            [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])
        image_points = np.array(
            [[c.x, c.y] for c in detection.corners], dtype=np.float64)

        if image_points.shape != (4, 2):
            return

        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        D = (
            np.array(self.camera_info.d, dtype=np.float64)
            if self.camera_info.d else np.zeros(5)
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return

        R_cam_tag0, _ = cv2.Rodrigues(rvec)
        t_cam_tag0 = tvec.reshape(3)

        t_c = transform.transform.translation
        q_c = transform.transform.rotation
        R_map_cam = quaternion_to_rotation_matrix(q_c.x, q_c.y, q_c.z, q_c.w)
        t_map_cam = np.array([t_c.x, t_c.y, t_c.z])

        R_map_tag0 = R_map_cam @ R_cam_tag0
        t_map_tag0 = R_map_cam @ t_cam_tag0 + t_map_cam

        # The robot tag lies flat on top of the robot -- only its YAW about
        # the floor's z-axis is meaningful for base_footprint's heading.
        tag_yaw = math.atan2(R_map_tag0[1, 0], R_map_tag0[0, 0])
        base_yaw = tag_yaw - self.offset_yaw

        R_map_base = yaw_to_rotation_z(base_yaw)
        t_map_base = t_map_tag0 - R_map_base @ self.offset_xyz
        t_map_base[2] = 0.0  # base_footprint is on the floor by definition

        msg_out = PoseWithCovarianceStamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = self.map_frame
        msg_out.pose.pose.position.x = float(t_map_base[0])
        msg_out.pose.pose.position.y = float(t_map_base[1])
        msg_out.pose.pose.position.z = 0.0

        qx, qy, qz, qw = rotation_to_quaternion(R_map_base)
        msg_out.pose.pose.orientation.x = qx
        msg_out.pose.pose.orientation.y = qy
        msg_out.pose.pose.orientation.z = qz
        msg_out.pose.pose.orientation.w = qw

        covariance = [0.0] * 36
        covariance[0] = self.base_cov        # x
        covariance[7] = self.base_cov        # y
        covariance[35] = self.base_cov_yaw   # yaw
        msg_out.pose.covariance = covariance

        self.publisher.publish(msg_out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamLocalizerNode()
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
