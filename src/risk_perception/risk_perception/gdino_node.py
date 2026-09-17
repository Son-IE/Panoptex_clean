#!/usr/bin/env python3

from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class GdinoNode(Node):
    """Temporary image-stream test before Grounding DINO integration."""

    def __init__(self) -> None:
        super().__init__('gdino_node')

        self.declare_parameter(
            'image_topic',
            '/camera/color/image_raw',
        )
        self.declare_parameter(
            'annotated_topic',
            '/risk_perception/annotated_image',
        )
        self.declare_parameter(
            'inference_stride',
            1,
        )

        image_topic = str(
            self.get_parameter('image_topic').value
        )
        annotated_topic = str(
            self.get_parameter('annotated_topic').value
        )

        self.inference_stride = max(
            1,
            int(self.get_parameter('inference_stride').value),
        )

        self.bridge = CvBridge()
        self.frame_count = 0

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.annotated_publisher = self.create_publisher(
            Image,
            annotated_topic,
            10,
        )

        self.get_logger().info(
            f'Subscribing to {image_topic}'
        )
        self.get_logger().info(
            f'Publishing annotated images to {annotated_topic}'
        )

    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1

        if self.frame_count % self.inference_stride != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except CvBridgeError as error:
            self.get_logger().error(
                f'cv_bridge conversion failed: {error}'
            )
            return

        if frame is None or frame.size == 0:
            self.get_logger().warning(
                'Received an empty image'
            )
            return

        # Temporary visualization proving that ROS image I/O works.
        cv2.putText(
            frame,
            'risk_perception ROS wrapper active',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        try:
            output_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8',
            )
            output_msg.header = msg.header
            self.annotated_publisher.publish(output_msg)

        except CvBridgeError as error:
            self.get_logger().error(
                f'Output conversion failed: {error}'
            )


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GdinoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
