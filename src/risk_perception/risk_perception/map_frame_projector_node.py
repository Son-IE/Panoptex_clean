#!/usr/bin/env python3

import copy
from typing import Optional

import numpy as np
import rclpy

from geometry_msgs.msg import Pose
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_to_rotation_matrix(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    """Convert a normalized quaternion into a 3x3 rotation matrix."""

    norm = np.sqrt(x * x + y * y + z * z + w * w)

    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product q1 * q2 using x, y, z, w ordering."""

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class MapFrameProjectorNode(Node):
    """Transform camera-frame Detection3DArray messages into map."""

    def __init__(self) -> None:
        super().__init__("map_frame_projector")

        self.declare_parameter(
            "input_topic",
            "/risk_perception/detections_3d",
        )
        self.declare_parameter(
            "output_topic",
            "/risk_perception/detections_3d_map",
        )
        self.declare_parameter(
            "marker_topic",
            "/risk_perception/map_markers",
        )
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("transform_timeout_sec", 0.25)
        self.declare_parameter("marker_lifetime_sec", 1.0)

        input_topic = str(
            self.get_parameter("input_topic").value
        )
        output_topic = str(
            self.get_parameter("output_topic").value
        )
        marker_topic = str(
            self.get_parameter("marker_topic").value
        )

        self.target_frame = str(
            self.get_parameter("target_frame").value
        )
        self.transform_timeout = float(
            self.get_parameter(
                "transform_timeout_sec"
            ).value
        )
        self.marker_lifetime = float(
            self.get_parameter(
                "marker_lifetime_sec"
            ).value
        )

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=20.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.subscription = self.create_subscription(
            Detection3DArray,
            input_topic,
            self.detection_callback,
            10,
        )

        self.detection_publisher = self.create_publisher(
            Detection3DArray,
            output_topic,
            10,
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            marker_topic,
            10,
        )

        self.get_logger().info(
            f"Input detections: {input_topic}"
        )
        self.get_logger().info(
            f"Map-frame detections: {output_topic}"
        )
        self.get_logger().info(
            f"RViz markers: {marker_topic}"
        )
        self.get_logger().info(
            f"Target frame: {self.target_frame}"
        )

    @staticmethod
    def transform_pose(pose: Pose, transform) -> Pose:
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        rotation_matrix = quaternion_to_rotation_matrix(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        source_position = np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ],
            dtype=np.float64,
        )

        target_position = (
            rotation_matrix @ source_position
            + np.array(
                [
                    translation.x,
                    translation.y,
                    translation.z,
                ],
                dtype=np.float64,
            )
        )

        output = copy.deepcopy(pose)

        output.position.x = float(target_position[0])
        output.position.y = float(target_position[1])
        output.position.z = float(target_position[2])

        target_quaternion = quaternion_multiply(
            (
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
        )

        quaternion_norm = np.linalg.norm(target_quaternion)

        if quaternion_norm > 1e-12:
            target_quaternion = tuple(
                value / quaternion_norm
                for value in target_quaternion
            )

        (
            output.orientation.x,
            output.orientation.y,
            output.orientation.z,
            output.orientation.w,
        ) = target_quaternion

        return output

    def detection_callback(
        self,
        msg: Detection3DArray,
    ) -> None:
        source_frame = msg.header.frame_id

        if not source_frame:
            self.get_logger().warning(
                "Detection3DArray has an empty frame_id"
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(), # Use the latest available transform (for initial run, static camera)
                # Time.from_msg(msg.header.stamp) # Use the transform at the time of the message (for dynamic camera)
                timeout=Duration(
                    seconds=self.transform_timeout
                ),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {source_frame} → "
                f"{self.target_frame}: {error}"
            )
            return

        output = copy.deepcopy(msg)
        output.header.frame_id = self.target_frame

        for detection in output.detections:
            detection.header = copy.deepcopy(output.header)

            detection.bbox.center = self.transform_pose(
                detection.bbox.center,
                transform,
            )

            for result in detection.results:
                result.pose.pose = self.transform_pose(
                    result.pose.pose,
                    transform,
                )

                # Current projector does not estimate covariance.
                result.pose.covariance = [0.0] * 36

        self.detection_publisher.publish(output)
        self.publish_markers(output)

    def publish_markers(
        self,
        detections: Detection3DArray,
    ) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        lifetime = Duration(
            seconds=self.marker_lifetime
        ).to_msg()

        for index, detection in enumerate(
            detections.detections
        ):
            label = "object"
            score = 0.0

            if detection.results:
                hypothesis = (
                    detection.results[0].hypothesis
                )
                label = str(hypothesis.class_id)
                score = float(hypothesis.score)

            sphere = Marker()
            sphere.header = detections.header
            sphere.ns = "risk_objects"
            sphere.id = index * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = detection.bbox.center

            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25

            sphere.color.r = 0.1
            sphere.color.g = 0.8
            sphere.color.b = 0.2
            sphere.color.a = 0.9
            sphere.lifetime = lifetime

            text = Marker()
            text.header = detections.header
            text.ns = "risk_object_labels"
            text.id = index * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = copy.deepcopy(
                detection.bbox.center
            )
            text.pose.position.z += 0.35

            text.scale.z = 0.18
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.lifetime = lifetime

            text.text = (
                f"{label} {score:.2f}\n"
                f"x={detection.bbox.center.position.x:.2f}, "
                f"y={detection.bbox.center.position.y:.2f}"
            )

            marker_array.markers.extend(
                [sphere, text]
            )

        self.marker_publisher.publish(marker_array)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = MapFrameProjectorNode()

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
