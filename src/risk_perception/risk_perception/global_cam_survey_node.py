#!/usr/bin/env python3
"""
global_cam_survey_node.py  --  TRACK 1b-bootstrap  (run ONCE, then never again)

Turns three tape-measure numbers into a complete floor_tags.yaml.

The problem this solves: global_cam_calibrator_node needs each floor tag's
full pose (x, y, AND yaw) in the world frame. Position you can measure with
a tape; yaw you cannot -- nobody is protractor-ing a paper tag on a lab
floor to within a degree. So instead you measure only the three
center-to-center distances, and this tool derives the yaws from what the
camera actually sees.

How it works:
  1. Each tag's CENTER in camera coordinates comes from a per-tag solvePnP
     on its 4 corners. Centers are the robust part of that solve -- being a
     centroid, the center barely moves even if the corner ORDER convention
     is off, unlike the orientation.
  2. The measured distances place the 3 tags in a world frame
     (tag 1 at origin, tag 2 on +x, tag 3 at +y).
  3. Kabsch aligns observed centers -> world positions. Three non-collinear
     point pairs give an exact, unique rigid transform: the camera pose.
  4. Mirror check: a triangle's 3 distances don't say which side of the
     floor the camera is on, so if the solve puts the camera UNDER the
     floor, flip the world frame's handedness and redo.
  5. With the camera pose known, run each tag's measured orientation back
     through it to recover that tag's world yaw.

Averages over `num_samples` detections first, since the tags are static and
the noise is not. Writes floor_tags.yaml (backing up any existing one) and
exits.

Run it with the robot OFF -- this step has nothing to do with SLAM or the
robot; it only relates the camera to the tags.

The frame this writes is defined by the tags themselves (tag 1 at the
origin), NOT the robot's SLAM `map` frame -- if the robot is available, use
global_cam_map_align_node.py instead, which solves the same floor-tag poses
directly in `map` via tag 0 and gets you a one-time calibration in a single
step. Use this node only for a camera-only bring-up (checking tag IDs,
verifying the geometry) when the robot isn't available yet.
"""

import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError as exc:
    raise ImportError(
        "apriltag_msgs not found -- install apriltag_ros "
        "(ros-humble-apriltag-ros)."
    ) from exc

from risk_perception.geometry_utils import (
    kabsch,
    orthonormalize,
    tag_object_points,
    triangle_from_distances,
)

FLOOR_TAG_IDS = (1, 2, 3)


class GlobalCamSurveyNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_survey")

        self.declare_parameter("floor_tags_yaml", "")
        self.declare_parameter("detections_topic", "/global_cam/apriltag/detections")
        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("num_samples", 30)
        self.declare_parameter("distance_tolerance_m", 0.02)

        gp = self.get_parameter

        self.yaml_path = str(gp("floor_tags_yaml").value)
        if not self.yaml_path:
            raise FileNotFoundError(
                "floor_tags_yaml parameter is required -- point it at "
                "config/floor_tags.yaml (must contain tag_size_m and a "
                "distances_m block)."
            )

        with open(self.yaml_path) as f:
            self.floor_tags = yaml.safe_load(f)

        self.tag_size = float(self.floor_tags["tag_size_m"])

        try:
            distances = self.floor_tags["distances_m"]
            self.d12 = float(distances["d_1_2"])
            self.d13 = float(distances["d_1_3"])
            self.d23 = float(distances["d_2_3"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "floor_tags.yaml needs a distances_m block with d_1_2, "
                "d_1_3 and d_2_3 (measured center-to-center, in metres)."
            ) from exc

        self.num_samples = max(1, int(gp("num_samples").value))
        self.distance_tolerance = float(gp("distance_tolerance_m").value)

        # tag id -> list of per-sample (center_in_camera, R_cam_tag)
        self.samples: Dict[int, List[tuple]] = {i: [] for i in FLOOR_TAG_IDS}
        self.camera_info: Optional[CameraInfo] = None
        self.done = False

        self.create_subscription(
            CameraInfo, str(gp("camera_info_topic").value), self._camera_info_cb, 10)
        self.create_subscription(
            AprilTagDetectionArray, str(gp("detections_topic").value),
            self._detections_cb, 10)

        self.get_logger().info(
            f"Surveying from distances d12={self.d12} d13={self.d13} "
            f"d23={self.d23} (tag size {self.tag_size} m). "
            f"Collecting {self.num_samples} samples with all of "
            f"{list(FLOOR_TAG_IDS)} visible...")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _detections_cb(self, msg: AprilTagDetectionArray) -> None:
        if self.done or self.camera_info is None:
            return

        by_id = {int(d.id): d for d in msg.detections}
        missing = [i for i in FLOOR_TAG_IDS if i not in by_id]

        if missing:
            self.get_logger().warning(
                f"waiting -- floor tags {missing} not visible "
                f"(seeing {sorted(by_id)})", throttle_duration_sec=3.0)
            return

        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        D = (
            np.array(self.camera_info.d, dtype=np.float64)
            if self.camera_info.d else np.zeros(5)
        )
        object_points = tag_object_points(self.tag_size)

        for tag_id in FLOOR_TAG_IDS:
            detection = by_id[tag_id]
            image_points = np.array(
                [[c.x, c.y] for c in detection.corners], dtype=np.float64)

            if image_points.shape != (4, 2):
                return

            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return

            R_cam_tag, _ = cv2.Rodrigues(rvec)
            self.samples[tag_id].append((tvec.reshape(3), R_cam_tag))

        collected = len(self.samples[FLOOR_TAG_IDS[0]])
        if collected % 10 == 0:
            self.get_logger().info(f"  {collected}/{self.num_samples} samples")

        if collected >= self.num_samples:
            self.done = True
            self._solve_and_write()

    def _solve_and_write(self) -> None:
        # Tags are static, so averaging is pure noise reduction.
        centers_cam = np.array(
            [np.mean([s[0] for s in self.samples[i]], axis=0) for i in FLOOR_TAG_IDS])
        rotations_cam = [
            orthonormalize(np.mean([s[1] for s in self.samples[i]], axis=0))
            for i in FLOOR_TAG_IDS
        ]

        # --- consistency check: does what the camera sees match the tape? ---
        observed = {
            "d_1_2": float(np.linalg.norm(centers_cam[0] - centers_cam[1])),
            "d_1_3": float(np.linalg.norm(centers_cam[0] - centers_cam[2])),
            "d_2_3": float(np.linalg.norm(centers_cam[1] - centers_cam[2])),
        }
        measured = {"d_1_2": self.d12, "d_1_3": self.d13, "d_2_3": self.d23}

        self.get_logger().info("distance check (camera vs. your tape measure):")
        worst = 0.0
        for key in ("d_1_2", "d_1_3", "d_2_3"):
            error = observed[key] - measured[key]
            worst = max(worst, abs(error))
            self.get_logger().info(
                f"  {key}: camera {observed[key]:.4f} m  vs  measured "
                f"{measured[key]:.4f} m   (off by {error * 100:+.1f} cm)")

        if worst > self.distance_tolerance:
            self.get_logger().error(
                f"MISMATCH of {worst * 100:.1f} cm exceeds the "
                f"{self.distance_tolerance * 100:.0f} cm tolerance. Likely causes, "
                "in order: (1) tag_size_m does not match the actual printed tag "
                "(printers rescale -- measure the black square's outer edge); "
                "(2) the camera stream resolution differs from the resolution the "
                "intrinsics were calibrated at; (3) a mistyped distance. "
                "Writing the file anyway, but the calibration will be off.")

        # --- solve camera pose, then fix handedness if it lands underground ---
        world_points = triangle_from_distances(self.d12, self.d13, self.d23)
        R_world_cam, t_world_cam = kabsch(centers_cam, world_points)

        if t_world_cam[2] < 0.0:
            # The 3 distances alone can't say which side of the floor the
            # camera is on; a camera below the floor means we guessed the
            # mirror image, so flip +y and re-solve.
            self.get_logger().info(
                "camera solved below the floor -- mirroring the world frame")
            world_points[:, 1] *= -1.0
            R_world_cam, t_world_cam = kabsch(centers_cam, world_points)

        self.get_logger().info(
            f"camera position in world: "
            f"({t_world_cam[0]:.3f}, {t_world_cam[1]:.3f}, {t_world_cam[2]:.3f}) m "
            f"-- that z is the camera's height above the floor, sanity-check it")

        # --- derive each tag's world yaw from its measured orientation ---
        entries = {}
        for index, tag_id in enumerate(FLOOR_TAG_IDS):
            R_world_tag = R_world_cam @ rotations_cam[index]
            yaw = float(np.arctan2(R_world_tag[1, 0], R_world_tag[0, 0]))
            entries[tag_id] = {
                "x": float(world_points[index, 0]),
                "y": float(world_points[index, 1]),
                "z": 0.0,
                "yaw": yaw,
            }
            self.get_logger().info(
                f"  tag_{tag_id}: x={entries[tag_id]['x']:.4f} "
                f"y={entries[tag_id]['y']:.4f} yaw={np.degrees(yaw):+.1f} deg")

        self._write_yaml(entries, observed)
        rclpy.shutdown()

    def _write_yaml(self, entries: dict, observed: dict) -> None:
        path = Path(self.yaml_path)

        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup)
            self.get_logger().info(f"backed up previous file -> {backup}")

        offset = self.floor_tags.get(
            "tag0_to_base_footprint", {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text = f"""# floor_tags.yaml -- GENERATED by global_cam_survey on {stamp}
#
# Do not hand-edit the tag_N blocks; re-run the survey instead:
#   ros2 run risk_perception global_cam_survey --ros-args \\
#     -p floor_tags_yaml:=<this file>
#
# The world frame here is defined BY THE TAGS: tag 1 is the origin, tag 2
# lies on +x, tag 3 is on the +y side, +z is up. That is NOT yet the robot's
# SLAM map frame -- relating the two is a separate, later step
# (see global_cam_localizer_node / the tag-0 alignment).
#
# Positions came from the measured distances below; yaws were derived from
# what the camera actually observed. Camera-observed distances at survey
# time were d_1_2={observed['d_1_2']:.4f}, d_1_3={observed['d_1_3']:.4f},
# d_2_3={observed['d_2_3']:.4f} m.

tag_family: "tag36h11"
tag_size_m: {self.tag_size}          # outer edge of the black square, metres

# Measured center-to-center distances (survey input -- edit these, then
# re-run the survey, if you ever move the tags).
distances_m:
  d_1_2: {self.d12}
  d_1_3: {self.d13}
  d_2_3: {self.d23}

# Solved tag poses in the tag-defined world frame.
tag_1: {{x: {entries[1]['x']:.5f}, y: {entries[1]['y']:.5f}, z: 0.0, yaw: {entries[1]['yaw']:.5f}}}
tag_2: {{x: {entries[2]['x']:.5f}, y: {entries[2]['y']:.5f}, z: 0.0, yaw: {entries[2]['yaw']:.5f}}}
tag_3: {{x: {entries[3]['x']:.5f}, y: {entries[3]['y']:.5f}, z: 0.0, yaw: {entries[3]['yaw']:.5f}}}

# base_footprint -> tag 0, in the ROBOT's own frame (forward=+x, left=+y,
# up=+z). Only x/y (lateral centering) and yaw (mounting rotation) affect
# the robot pose; height does not -- see global_cam_localizer_node.
tag0_to_base_footprint: {{x: {float(offset.get('x', 0.0))}, y: {float(offset.get('y', 0.0))}, z: {float(offset.get('z', 0.0))}, yaw: {float(offset.get('yaw', 0.0))}}}
"""

        path.write_text(text)
        self.get_logger().info(f"wrote {path}")
        self.get_logger().info(
            "Survey complete. global_cam_calibrator can now run for real.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamSurveyNode()
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
