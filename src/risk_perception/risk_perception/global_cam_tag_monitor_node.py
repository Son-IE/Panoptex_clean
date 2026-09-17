#!/usr/bin/env python3
"""
global_cam_tag_monitor_node.py  --  operator aid for the §6 calibration

Answers one question continuously: "can the overhead camera see all four tags
right now?" -- so the calibration operator does not have to keep running
`ros2 topic echo /global_cam/apriltag/detections --once` and scroll for tag 0.

Two outputs, both optional:

  * a live one-line console readout, rewritten in place:

        tag0 OK  28px | tag1 OK  49px | tag2 OK  34px | tag3 OK  36px  ->
        ALL 4 (steady 4.2s)

    `steady` is how long ALL expected tags have been continuously visible,
    which is what actually matters: global_cam_map_align needs tag 0 held for
    the whole ~6 s of a capture, and the robot's tag drops out under motion
    blur the moment the robot moves.

  * an annotated image on `overlay_topic`, for RViz or rqt_image_view:

        ros2 run rqt_image_view rqt_image_view /global_cam/apriltag/overlay

Corners are drawn from apriltag_ros's OWN detections rather than re-running a
detector here, so the overlay shows exactly what the calibration nodes are
being fed -- if a tag is missing from the overlay, it is genuinely missing
from the pipeline.

Read-only: subscribes, never publishes TF and never touches floor_tags.yaml.
Safe to leave running alongside anything else.
"""

import sys
import time
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError as exc:
    raise ImportError(
        "apriltag_msgs not found -- install apriltag_ros "
        "(ros-humble-apriltag-ros)."
    ) from exc

# tag36h11 is 10 cells across including its border and the decoder needs ~2 px
# per cell, so ~20 px of edge is the floor and ~40 px is where it stops being
# flaky. Below MARGINAL_EDGE_PX a tag will come and go.
MARGINAL_EDGE_PX = 40.0
MIN_EDGE_PX = 20.0


class GlobalCamTagMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_tag_monitor")

        self.declare_parameter("detections_topic", "/global_cam/apriltag/detections")
        self.declare_parameter("image_topic", "/global_cam/image_raw/compressed")
        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("overlay_topic", "/global_cam/apriltag/overlay")
        self.declare_parameter("expected_ids", [0, 1, 2, 3])
        self.declare_parameter("tag_size_m", 0.115)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("console", True)
        # A tag counts as "visible" until this long after its last sighting --
        # without it, a single dropped frame reads as the tag vanishing and
        # the steady timer never accumulates.
        self.declare_parameter("visible_hold_sec", 0.5)
        self.declare_parameter("sync_tolerance_sec", 0.2)
        self.declare_parameter("cache_size", 30)

        gp = self.get_parameter
        self.expected = [int(i) for i in gp("expected_ids").value]
        self.tag_size = float(gp("tag_size_m").value)
        self.publish_overlay = bool(gp("publish_overlay").value)
        self.console = bool(gp("console").value)
        self.visible_hold = float(gp("visible_hold_sec").value)
        self.sync_tolerance = float(gp("sync_tolerance_sec").value)
        self.cache_size = int(gp("cache_size").value)

        self.bridge = CvBridge()
        self.fx: Optional[float] = None
        self.image_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        self.last_seen: dict = {}   # id -> monotonic time of last sighting
        self.last_edge: dict = {}   # id -> edge length in px
        self.all_visible_since: Optional[float] = None
        self.detection_count = 0

        self.overlay_pub = self.create_publisher(
            Image, str(gp("overlay_topic").value), qos_profile_sensor_data)

        self.create_subscription(
            CameraInfo, str(gp("camera_info_topic").value), self._info_cb, 10)
        self.create_subscription(
            CompressedImage, str(gp("image_topic").value),
            self._image_cb, qos_profile_sensor_data)
        self.create_subscription(
            AprilTagDetectionArray, str(gp("detections_topic").value),
            self._detections_cb, 10)

        if self.console:
            self.create_timer(0.2, self._print_status)

        self.get_logger().info(
            f"watching tags {self.expected} on "
            f"{gp('detections_topic').value}"
            + (f"; overlay -> {gp('overlay_topic').value}"
               if self.publish_overlay else ""))

    # -- inputs ------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo) -> None:
        self.fx = float(msg.k[0])

    def _image_cb(self, msg: CompressedImage) -> None:
        if not self.publish_overlay:
            return
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        key = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.image_cache[key] = frame
        while len(self.image_cache) > self.cache_size:
            self.image_cache.popitem(last=False)

    def _nearest_image(self, stamp) -> Optional[np.ndarray]:
        if not self.image_cache:
            return None
        key = stamp.sec * 1_000_000_000 + stamp.nanosec
        nearest = min(self.image_cache, key=lambda k: abs(k - key))
        if abs(nearest - key) > self.sync_tolerance * 1e9:
            return None
        return self.image_cache[nearest]

    def _detections_cb(self, msg: AprilTagDetectionArray) -> None:
        self.detection_count += 1
        now = time.monotonic()
        frame = self._nearest_image(msg.header.stamp) if self.publish_overlay else None
        if frame is not None:
            frame = frame.copy()

        for detection in msg.detections:
            tag_id = int(detection.id)
            corners = np.array(
                [[c.x, c.y] for c in detection.corners], dtype=np.float64)
            if corners.shape != (4, 2):
                continue

            edge = float(np.mean([
                np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)
            ]))
            self.last_seen[tag_id] = now
            self.last_edge[tag_id] = edge

            if frame is not None:
                self._draw(frame, tag_id, corners, edge)

        # Track how long EVERY expected tag has been continuously visible.
        if all(self._is_visible(i, now) for i in self.expected):
            if self.all_visible_since is None:
                self.all_visible_since = now
        else:
            self.all_visible_since = None

        if frame is not None:
            self._annotate_banner(frame, now)
            out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out.header = msg.header
            self.overlay_pub.publish(out)

    # -- rendering ---------------------------------------------------------
    def _colour(self, edge: float):
        if edge >= MARGINAL_EDGE_PX:
            return (0, 220, 0)        # green  -- comfortable
        if edge >= MIN_EDGE_PX:
            return (0, 200, 255)      # amber  -- will be flaky
        return (0, 0, 255)            # red    -- below the decode floor

    def _draw(self, frame, tag_id: int, corners: np.ndarray, edge: float) -> None:
        colour = self._colour(edge)
        cv2.polylines(frame, [corners.astype(int)], True, colour, 3)
        # A short stub on the first edge shows the tag's orientation, which is
        # what a wrong tag0_to_base_footprint yaw would show up as.
        p0, p1 = corners[0].astype(int), corners[1].astype(int)
        cv2.circle(frame, tuple(p0), 4, (255, 0, 255), -1)
        cv2.line(frame, tuple(p0), tuple(p1), (255, 0, 255), 2)

        centre = corners.mean(axis=0).astype(int)
        label = f"{tag_id}  {edge:.0f}px"
        if self.fx:
            label += f"  {self.fx * self.tag_size / edge:.1f}m"
        cv2.putText(frame, label, (centre[0] - 45, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, label, (centre[0] - 45, centre[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    def _annotate_banner(self, frame, now: float) -> None:
        missing = [i for i in self.expected if not self._is_visible(i, now)]
        if missing:
            text = f"MISSING {missing}"
            colour = (0, 0, 255)
        else:
            steady = now - (self.all_visible_since or now)
            text = f"ALL {len(self.expected)} VISIBLE  (steady {steady:.1f}s)"
            colour = (0, 220, 0)
        cv2.putText(frame, text, (18, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 6)
        cv2.putText(frame, text, (18, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, colour, 2)

    # -- console -----------------------------------------------------------
    def _is_visible(self, tag_id: int, now: float) -> bool:
        return now - self.last_seen.get(tag_id, -1e9) <= self.visible_hold

    def _print_status(self) -> None:
        now = time.monotonic()
        if not self.detection_count:
            sys.stdout.write("\rwaiting for detections ...")
            sys.stdout.flush()
            return

        parts = []
        for tag_id in self.expected:
            if self._is_visible(tag_id, now):
                parts.append(f"tag{tag_id} OK {self.last_edge.get(tag_id, 0):3.0f}px")
            else:
                parts.append(f"tag{tag_id} --      ")

        if self.all_visible_since is not None:
            verdict = f"ALL {len(self.expected)} (steady {now - self.all_visible_since:4.1f}s)"
        else:
            missing = [i for i in self.expected if not self._is_visible(i, now)]
            verdict = f"MISSING {missing}"

        sys.stdout.write("\r" + " | ".join(parts) + "  ->  " + verdict + "    ")
        sys.stdout.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamTagMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
