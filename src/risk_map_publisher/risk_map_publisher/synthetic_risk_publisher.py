#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class SyntheticRiskPublisher(Node):
    """Creates a Gaussian risk field using the geometry of the incoming /map."""

    def __init__(self) -> None:
        super().__init__('synthetic_risk_publisher')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('risk_topic', '/risk_map')
        self.declare_parameter('center_x', 1.0)
        self.declare_parameter('center_y', 0.0)
        self.declare_parameter('sigma', 0.50)
        self.declare_parameter('cutoff_radius', 1.50)
        self.declare_parameter('peak_value', 80)
        self.declare_parameter('publish_rate', 1.0)

        map_topic = str(self.get_parameter('map_topic').value)
        risk_topic = str(self.get_parameter('risk_topic').value)
        publish_rate = float(self.get_parameter('publish_rate').value)

        publisher_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        subscriber_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            OccupancyGrid, risk_topic, publisher_qos)
        self._subscription = self.create_subscription(
            OccupancyGrid, map_topic, self._map_callback, subscriber_qos)
        self._latest_risk_map: Optional[OccupancyGrid] = None
        self._timer = self.create_timer(1.0 / max(publish_rate, 0.1), self._timer_callback)
        self.get_logger().info(
            f'Waiting for {map_topic}; risk output will be published on {risk_topic}')

    def _map_callback(self, map_msg: OccupancyGrid) -> None:
        center_x = float(self.get_parameter('center_x').value)
        center_y = float(self.get_parameter('center_y').value)
        sigma = float(self.get_parameter('sigma').value)
        cutoff_radius = float(self.get_parameter('cutoff_radius').value)
        peak_value = max(0, min(100, int(self.get_parameter('peak_value').value)))
        if sigma <= 0.0:
            self.get_logger().error('sigma must be greater than zero')
            return

        risk_msg = OccupancyGrid()
        risk_msg.header.frame_id = map_msg.header.frame_id
        risk_msg.info = map_msg.info
        width = int(map_msg.info.width)
        height = int(map_msg.info.height)
        resolution = float(map_msg.info.resolution)
        origin = map_msg.info.origin
        q = origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        values = [0] * (width * height)

        for gy in range(height):
            local_y = (gy + 0.5) * resolution
            for gx in range(width):
                local_x = (gx + 0.5) * resolution
                world_x = origin.position.x + cos_yaw * local_x - sin_yaw * local_y
                world_y = origin.position.y + sin_yaw * local_x + cos_yaw * local_y
                distance = math.hypot(world_x - center_x, world_y - center_y)
                if distance > cutoff_radius:
                    continue
                value = peak_value * math.exp(-0.5 * (distance / sigma) ** 2)
                values[gy * width + gx] = int(round(value))

        risk_msg.data = values
        self._latest_risk_map = risk_msg
        self._publish_latest()
        self.get_logger().info(
            f'Generated risk map: frame={risk_msg.header.frame_id}, '
            f'size={width}x{height}, center=({center_x:.2f}, {center_y:.2f}), '
            f'peak={peak_value}')

    def _timer_callback(self) -> None:
        if self._latest_risk_map is not None:
            self._publish_latest()

    def _publish_latest(self) -> None:
        assert self._latest_risk_map is not None
        self._latest_risk_map.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._latest_risk_map)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticRiskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
