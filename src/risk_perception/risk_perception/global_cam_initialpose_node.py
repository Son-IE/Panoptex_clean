#!/usr/bin/env python3
"""
global_cam_initialpose_node.py  --  seed AMCL from the overhead camera

Removes the "2D Pose Estimate" click from every session. AMCL starts with no
idea where the robot is; the overhead camera already knows, because tag 0 on
the robot is visible in a frame whose `map` pose was solved once by
global_cam_map_align. This relays that answer to AMCL's /initialpose.

`global_cam_localizer` already publishes exactly the message AMCL wants --
PoseWithCovarianceStamped in `map`, with a real covariance -- so this node is
only the glue plus the safety interlocks that make an automatic seed safe:

  * WAIT FOR A STABLE READING. Tag 0 is small and the robot may still be
    settling; a single frame can be metres out. Seeding AMCL with that is
    worse than not seeding it, because a confidently wrong prior takes longer
    to recover from than no prior at all. So require `settle_samples`
    consecutive readings agreeing within `max_spread_m`.
  * WAIT FOR AMCL TO EXIST. Publishing before AMCL subscribes goes nowhere;
    the message is not latched and there is no retry in AMCL. Hold until
    /initialpose has a subscriber.
  * SEED ONCE. After AMCL converges, the laser scan match is the better
    estimate -- it uses the whole room, not one 76 mm tag at 4 m. Continuing
    to publish would repeatedly yank a converged filter back to the noisier
    source. Set `repeat:=true` only to debug.

    ros2 run risk_perception global_cam_initialpose

Needs the overhead chain (global_cam.launch.py) up alongside AMCL. If tag 0
is out of view at startup nothing is published and you simply set the pose by
hand, as before -- this never blocks, it only saves a step.
"""

import sys
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class GlobalCamInitialPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_initialpose")

        self.declare_parameter("robot_pose_topic", "/global_cam/robot_pose")
        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("settle_samples", 10)
        self.declare_parameter("max_spread_m", 0.05)
        self.declare_parameter("max_spread_rad", 0.10)
        self.declare_parameter("repeat", False)

        gp = self.get_parameter
        self.settle_samples = max(1, int(gp("settle_samples").value))
        self.max_spread = float(gp("max_spread_m").value)
        self.max_spread_rad = float(gp("max_spread_rad").value)
        self.repeat = bool(gp("repeat").value)

        self.recent: List[np.ndarray] = []
        self.seeded = False
        self.pending: Optional[PoseWithCovarianceStamped] = None

        # TRANSIENT_LOCAL so a seed published before AMCL finishes activating
        # is still delivered when it subscribes -- the subscriber-count check
        # below covers the common case, but activation is not instantaneous.
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, str(gp("initialpose_topic").value),
            QoSProfile(depth=1,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        self.create_subscription(
            PoseWithCovarianceStamped, str(gp("robot_pose_topic").value),
            self._pose_cb, 10)
        self.create_timer(0.5, self._try_publish)

        self.get_logger().info(
            f"waiting for {self.settle_samples} stable tag-0 readings, then "
            "seeding AMCL -- no 2D Pose Estimate needed")

    @staticmethod
    def _yaw(msg: PoseWithCovarianceStamped) -> float:
        q = msg.pose.pose.orientation
        return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        if self.seeded and not self.repeat:
            return

        self.recent.append(np.array([msg.pose.pose.position.x,
                                     msg.pose.pose.position.y,
                                     self._yaw(msg)]))
        if len(self.recent) > self.settle_samples:
            self.recent.pop(0)
        if len(self.recent) < self.settle_samples:
            return

        stack = np.array(self.recent)
        spread_xy = float(np.linalg.norm(
            stack[:, :2].max(axis=0) - stack[:, :2].min(axis=0)))
        # unwrap before measuring, so readings straddling +/-pi don't look wild
        yaws = np.unwrap(stack[:, 2])
        spread_yaw = float(yaws.max() - yaws.min())

        if spread_xy > self.max_spread or spread_yaw > self.max_spread_rad:
            self.get_logger().info(
                f"tag-0 pose still moving ({spread_xy * 100:.1f} cm / "
                f"{np.degrees(spread_yaw):.1f} deg across "
                f"{self.settle_samples} samples) -- holding",
                throttle_duration_sec=5.0)
            self.pending = None
            return

        self.pending = msg

    def _try_publish(self) -> None:
        if self.pending is None or (self.seeded and not self.repeat):
            return
        if self.publisher.get_subscription_count() == 0:
            self.get_logger().info(
                "have a stable tag-0 pose; waiting for AMCL to subscribe to "
                "/initialpose", throttle_duration_sec=10.0)
            return

        msg = self.pending
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        self.seeded = True
        self.get_logger().info(
            f"seeded AMCL at ({msg.pose.pose.position.x:+.3f}, "
            f"{msg.pose.pose.position.y:+.3f}) yaw "
            f"{np.degrees(self._yaw(msg)):+.1f} deg from the overhead camera. "
            "Drive a short distance to let the laser refine it."
            + ("" if self.repeat else " Not publishing again -- once AMCL has "
               "converged, the laser scan match beats one small tag."))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamInitialPoseNode()
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
