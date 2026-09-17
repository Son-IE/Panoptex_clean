#!/usr/bin/env python3
"""
global_cam_align_check_node.py  --  accept/reject the §6 calibration

The alignment's RMS reprojection error is in pixels, which is not the unit any
decision downstream is made in. This node reports the error in metres, where
it matters: it compares the overhead camera's independent estimate of the
robot's pose (`/global_cam/robot_pose`, from global_cam_localizer's view of
tag 0) against SLAM/AMCL's own `map -> base_footprint`.

That difference IS the end-to-end calibration error -- the same error every
overhead object detection will inherit, because both ride on the identical
`map -> global_cam_optical_frame` extrinsic. Drive the robot around the
working area and watch it; the worst value over the whole area is the number
to judge, not the value at one spot.

    ros2 run risk_perception global_cam_align_check

For scale: global_cam_projector already declares base_cov_m2=0.10 for its own
object positions -- a ~32 cm standard deviation at its 3 m reference range,
because it back-projects a mask's bottom pixel onto the floor plane -- and
object_tracker_node associates tracks with a gate_distance_m of 0.6 m. An
extrinsic error well under those is not the limiting term in this pipeline.
"""

import sys
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

# Thresholds are set against what the rest of the pipeline can actually
# resolve (see the module docstring), not against surveying practice.
GOOD_M = 0.05
USABLE_M = 0.15


class GlobalCamAlignCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_align_check")

        self.declare_parameter("robot_pose_topic", "/global_cam/robot_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.worst = 0.0
        self.worst_at: Optional[np.ndarray] = None
        self.count = 0
        self.sum_err = 0.0

        self.create_subscription(
            PoseWithCovarianceStamped, str(gp("robot_pose_topic").value),
            self._pose_cb, 10)

        self.get_logger().info(
            "comparing the overhead camera's robot pose against SLAM's. "
            "Drive around the whole working area; watch the WORST value.")

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        try:
            # Latest, non-blocking -- a stamped lookup with a timeout would
            # block the executor that fills this very buffer.
            tr = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except TransformException:
            self.get_logger().warning("no map -> base_footprint yet",
                                      throttle_duration_sec=5.0)
            return

        cam = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        slam = np.array([tr.transform.translation.x, tr.transform.translation.y])
        err = float(np.linalg.norm(cam - slam))

        self.count += 1
        self.sum_err += err
        if err > self.worst:
            self.worst, self.worst_at = err, slam

        verdict = ("GOOD" if err < GOOD_M
                   else "USABLE" if err < USABLE_M else "TOO BIG")
        sys.stdout.write(
            f"\roverhead ({cam[0]:+.2f},{cam[1]:+.2f})  slam "
            f"({slam[0]:+.2f},{slam[1]:+.2f})  err {err * 100:5.1f} cm "
            f"[{verdict}]   mean {self.sum_err / self.count * 100:5.1f}  "
            f"worst {self.worst * 100:5.1f} cm    ")
        sys.stdout.flush()

    def report(self) -> None:
        sys.stdout.write("\n")
        if not self.count:
            self.get_logger().warning(
                "no samples -- was tag 0 ever visible, and is "
                "global_cam_localizer running?")
            return
        self.get_logger().info(
            f"{self.count} samples: mean {self.sum_err / self.count * 100:.1f} cm, "
            f"worst {self.worst * 100:.1f} cm"
            + (f" at ({self.worst_at[0]:+.2f}, {self.worst_at[1]:+.2f})"
               if self.worst_at is not None else ""))
        if self.worst < GOOD_M:
            self.get_logger().info(
                "GOOD -- far below the pipeline's own noise floor. Ship it.")
        elif self.worst < USABLE_M:
            self.get_logger().info(
                "USABLE -- under object_tracker's 0.6 m association gate, so "
                "the two cameras will still fuse into single tracks. Fine "
                "unless you need better than decimetre placement.")
        else:
            self.get_logger().warning(
                "TOO BIG -- re-run the alignment with wider-spread poses and "
                "more varied headings.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamAlignCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
