#!/usr/bin/env python3
"""
global_cam_calibrator_node.py  --  TRACK 1b  (extrinsic calibration)

Solves the overhead camera's pose in the `map` frame using the three floor
AprilTags (tag36h11 IDs 1, 2, 3) whose positions were surveyed into
config/floor_tags.yaml. Publishes the result as a transform
`map -> <camera optical frame>`, which global_cam_projector_node (object
positions) and global_cam_localizer_node (robot pose via tag 0) both read
back from tf2.

Method: for each visible floor tag, take its 4 detected image corners
(apriltag_msgs/AprilTagDetection.corners) and the corresponding 4 corner
positions in `map`, computed from the surveyed tag center + yaw + known
tag_size (geometry_utils.tag_corners_world). Stack all correspondences (up
to 12 points across up to 3 tags) and solve ONE cv2.solvePnP for T_cam_map,
then invert for T_map_cam. Solving jointly across all visible tags -- rather
than averaging 3 independent single-tag PnPs -- uses the full spread of
points across the image and is far less sensitive to any single tag's
detection noise.

Re-solves on every frame with >= min_tags floor tags visible (a few Hz), so
a bumped camera self-corrects rather than needing a restart. The transform
is re-broadcast (not latched-once) for the same reason -- treat "static" as
a QoS/latching convenience, not a promise that it never changes.

FROZEN EXTRINSIC: the camera is bolted down, so its pose in `map` really is
a constant, and needing the floor tags present forever is an accident of the
above rather than a requirement. Set `extrinsic_yaml` and good solves get
written there (see _save_extrinsic); set `use_saved_extrinsic` and that file
is loaded at startup and broadcast immediately, so the whole overhead chain
comes up with the tags lifted off the floor and apriltag_node not running at
all. A live solve still overwrites the loaded value the moment tags reappear,
so this is a cold-start seed, not a mode switch. The saved file is only
valid for the map origin and floor_tags.yaml it was solved against -- both
are recorded in its header, because a fresh SLAM session silently
invalidates it (README §6).

If REPROJECTION ERROR (logged) is persistently large, check the corner
ordering assumed in geometry_utils.tag_corners_world against your installed
apriltag_ros version's convention -- that's the most common cause.
"""

import datetime
import os
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformBroadcaster

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError as exc:
    raise ImportError(
        "apriltag_msgs not found -- install apriltag_ros "
        "(ros-humble-apriltag-ros) and its apriltag_msgs dependency."
    ) from exc

from risk_perception.geometry_utils import (
    rotation_to_quaternion,
    tag_corners_world,
)

FLOOR_TAG_IDS = {1, 2, 3}


class GlobalCamCalibratorNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_calibrator")

        self.declare_parameter("floor_tags_yaml", "")
        self.declare_parameter("detections_topic", "/global_cam/apriltag/detections")
        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("min_tags", 2)
        # Rate at which the last good extrinsic is re-sent while no new solve
        # is available (see _republish). 0 disables holding entirely.
        self.declare_parameter("republish_rate_hz", 10.0)
        self.declare_parameter("stale_warn_sec", 5.0)
        self.declare_parameter("max_reprojection_error_px", 5.0)
        # Frozen-extrinsic support -- see the module docstring.
        self.declare_parameter("extrinsic_yaml", "")
        self.declare_parameter("save_extrinsic", True)
        self.declare_parameter("save_every_n_solves", 30)
        self.declare_parameter("use_saved_extrinsic", False)

        gp = self.get_parameter

        floor_tags_yaml = str(gp("floor_tags_yaml").value)
        if not floor_tags_yaml:
            raise FileNotFoundError(
                "floor_tags_yaml parameter is required -- point it at "
                "config/floor_tags.yaml (surveyed tag positions)."
            )
        with open(floor_tags_yaml) as f:
            self.floor_tags = yaml.safe_load(f)

        self.floor_tags_yaml = floor_tags_yaml
        self.tag_size = float(self.floor_tags["tag_size_m"])
        self.map_frame = str(gp("map_frame").value)
        self.min_tags = int(gp("min_tags").value)
        self.republish_rate = float(gp("republish_rate_hz").value)
        self.stale_warn_sec = float(gp("stale_warn_sec").value)
        self.max_reproj_err = float(gp("max_reprojection_error_px").value)
        self.extrinsic_yaml = str(gp("extrinsic_yaml").value)
        self.save_extrinsic = bool(gp("save_extrinsic").value)
        self.save_every_n = max(1, int(gp("save_every_n_solves").value))
        self.use_saved_extrinsic = bool(gp("use_saved_extrinsic").value)

        self.camera_info: Optional[CameraInfo] = None
        self.camera_frame_id: Optional[str] = None
        self.last_rms_error: Optional[float] = None
        self.solve_count = 0
        # True while the only extrinsic we have came off disk. Cleared by the
        # first live solve; drives the wording of the staleness warning, which
        # would otherwise cry wolf for the entire life of a frozen run.
        self.extrinsic_from_file = False

        self.tf_broadcaster = TransformBroadcaster(self)

        # The camera is bolted in place, so its extrinsic does not expire when
        # the tags briefly stop being detected -- and tags 2/3 sit near the
        # decode floor, so momentary dropouts are normal. Without this, every
        # dropout deletes map -> <camera frame> from TF, which takes the whole
        # overhead chain (global_cam_projector's lookups, the RViz frame) down
        # with it for no physical reason. Re-send the last good solve at a
        # steady rate; a fresh solve simply overwrites it.
        self.last_transform: Optional[TransformStamped] = None
        self.last_solve_time: Optional[RclpyTime] = None

        if self.use_saved_extrinsic:
            self._load_extrinsic()

        self.create_timer(1.0 / max(1e-3, self.republish_rate), self._republish)

        self.create_subscription(
            CameraInfo, str(gp("camera_info_topic").value), self._camera_info_cb, 10)
        self.create_subscription(
            AprilTagDetectionArray, str(gp("detections_topic").value), self._detections_cb, 10)

        if self.extrinsic_from_file:
            self.get_logger().info(
                "Broadcasting the SAVED extrinsic; a live floor-tag solve will "
                "override it if tags come back into view")
        else:
            self.get_logger().info(
                f"Waiting for camera_info + >={self.min_tags} of floor tags "
                f"{sorted(FLOOR_TAG_IDS)} to solve map -> <camera optical frame>")

    def _load_extrinsic(self) -> None:
        """Seed last_transform from a previously saved solve.

        Deliberately non-fatal: a missing or malformed file leaves the node in
        plain solve-only mode, which is still useful whenever the tags are
        down. Failing hard here would take the whole overhead chain out over a
        file that is, by construction, only an optimisation.
        """
        if not self.extrinsic_yaml:
            self.get_logger().error(
                "use_saved_extrinsic is set but extrinsic_yaml is empty -- "
                "falling back to solving from floor tags")
            return

        try:
            with open(self.extrinsic_yaml) as f:
                saved = yaml.safe_load(f)
            t = saved["translation"]
            q = saved["rotation"]
            child_frame = str(saved["child_frame_id"])
        except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
            self.get_logger().error(
                f"Could not load saved extrinsic from {self.extrinsic_yaml}: "
                f"{error} -- falling back to solving from floor tags")
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(t["x"])
        transform.transform.translation.y = float(t["y"])
        transform.transform.translation.z = float(t["z"])
        transform.transform.rotation.x = float(q["x"])
        transform.transform.rotation.y = float(q["y"])
        transform.transform.rotation.z = float(q["z"])
        transform.transform.rotation.w = float(q["w"])

        self.last_transform = transform
        self.last_solve_time = self.get_clock().now()
        self.extrinsic_from_file = True
        # The camera frame normally arrives with CameraInfo. Seed it too, so a
        # frozen run with the bridge slow to start still broadcasts onto the
        # right child frame from the first timer tick.
        self.camera_frame_id = child_frame

        self.get_logger().info(
            f"Loaded saved extrinsic {self.map_frame} -> {child_frame} from "
            f"{self.extrinsic_yaml}: t=({t['x']:.3f}, {t['y']:.3f}, {t['z']:.3f}), "
            f"solved {saved.get('solved_at', 'at an unrecorded time')}")

    def _save_extrinsic(self, transform: TransformStamped, rms_error: float,
                        seen_ids) -> None:
        """Persist a good solve so a later run can start without the tags.

        Written atomically -- the republish timer of a second process reading
        this file mid-write would otherwise get a truncated document. The
        header mirrors floor_tags.yaml's convention so that a stale extrinsic
        (wrong map origin, re-surveyed tags) is diagnosable by eye rather than
        by watching the robot drive into a wall.
        """
        t = transform.transform.translation
        q = transform.transform.rotation
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = (
            "# GENERATED by global_cam_calibrator on "
            f"{stamp} -- do not hand-edit.\n"
            "#\n"
            "# Pose of the overhead camera's optical frame in the `map` frame,\n"
            "# solved by cv2.solvePnP over floor tags 1/2/3. Load it with\n"
            "# use_saved_extrinsic:=true to bring the overhead chain up with no\n"
            "# AprilTags on the floor.\n"
            "#\n"
            "# ONLY VALID FOR the map origin and the surveyed tag poses this was\n"
            "# solved against. A fresh slam_toolbox session, or a re-run of\n"
            "# global_cam_map_align, silently invalidates it -- re-capture (see\n"
            "# README §6). Re-capture too if the camera is bumped.\n"
            "#\n"
            f"#   floor_tags:        {self.floor_tags_yaml}\n"
            f"#   tags used:         {sorted(seen_ids)}\n"
            f"#   rms reprojection:  {rms_error:.2f} px\n"
        )

        payload = {
            "solved_at": stamp,
            "parent_frame_id": transform.header.frame_id,
            "child_frame_id": transform.child_frame_id,
            "rms_reprojection_px": round(rms_error, 4),
            "floor_tags_yaml": self.floor_tags_yaml,
            "tags_used": sorted(int(i) for i in seen_ids),
            "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
            "rotation": {
                "x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w),
            },
        }

        tmp_path = f"{self.extrinsic_yaml}.tmp"
        try:
            with open(tmp_path, "w") as f:
                f.write(header)
                yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, self.extrinsic_yaml)
        except OSError as error:
            self.get_logger().warning(
                f"Could not save extrinsic to {self.extrinsic_yaml}: {error}",
                throttle_duration_sec=30.0)
            return

        self.get_logger().info(
            f"Saved extrinsic to {self.extrinsic_yaml} "
            f"(rms_reprojection={rms_error:.2f}px)")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id

    def _republish(self) -> None:
        """Re-send the last good extrinsic with a current stamp.

        Only fills gaps: a successful solve overwrites last_transform, so this
        never competes with live data. It does mean a moved camera or a moved
        floor tag keeps publishing a stale answer until tags are seen again --
        hence the warning, which is the operator's cue that the overhead chain
        is coasting rather than tracking.
        """
        if self.last_transform is None or self.republish_rate <= 0.0:
            return

        now = self.get_clock().now()
        age = (now - self.last_solve_time).nanoseconds * 1e-9
        if age < 1.0 / max(1e-3, self.republish_rate):
            return      # a live solve just went out; nothing to fill

        if age > self.stale_warn_sec:
            if self.extrinsic_from_file:
                # Expected steady state for a frozen run -- info, not warning,
                # or this fires forever and trains the operator to ignore it.
                self.get_logger().info(
                    "broadcasting the saved extrinsic; no floor tags seen "
                    f"({age:.0f}s). Expected with use_saved_extrinsic and the "
                    "tags lifted; re-capture if the camera has been moved.",
                    throttle_duration_sec=60.0)
            else:
                self.get_logger().warning(
                    f"no floor-tag solve for {age:.0f}s -- still publishing the "
                    "last extrinsic. Fine if the camera has not moved; check that "
                    f">={self.min_tags} of tags {sorted(FLOOR_TAG_IDS)} are still "
                    "visible (ros2 run risk_perception global_cam_tag_monitor).",
                    throttle_duration_sec=10.0)

        self.last_transform.header.stamp = now.to_msg()
        self.tf_broadcaster.sendTransform(self.last_transform)

    def _detections_cb(self, msg: AprilTagDetectionArray) -> None:
        if self.camera_info is None:
            return

        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        seen_ids = set()

        for detection in msg.detections:
            tag_id = int(detection.id)

            if tag_id not in FLOOR_TAG_IDS:
                continue

            pose = self.floor_tags.get(f"tag_{tag_id}")

            if pose is None:
                self.get_logger().warning(
                    f"Tag {tag_id} seen but not surveyed in floor_tags.yaml",
                    throttle_duration_sec=5.0)
                continue

            pixel_corners = np.array(
                [[c.x, c.y] for c in detection.corners], dtype=np.float64)

            if pixel_corners.shape != (4, 2):
                continue

            object_points.append(tag_corners_world(pose, self.tag_size))
            image_points.append(pixel_corners)
            seen_ids.add(tag_id)

        if len(seen_ids) < self.min_tags:
            return

        object_points_arr = np.concatenate(object_points, axis=0)
        image_points_arr = np.concatenate(image_points, axis=0)

        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        D = (
            np.array(self.camera_info.d, dtype=np.float64)
            if self.camera_info.d else np.zeros(5)
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points_arr, image_points_arr, K, D,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok:
            self.get_logger().warning("solvePnP failed", throttle_duration_sec=5.0)
            return

        reprojected, _ = cv2.projectPoints(object_points_arr, rvec, tvec, K, D)
        reprojected = reprojected.reshape(-1, 2)
        rms_error = float(np.sqrt(
            np.mean(np.sum((reprojected - image_points_arr) ** 2, axis=1))))
        self.last_rms_error = rms_error

        if rms_error > self.max_reproj_err:
            self.get_logger().warning(
                f"High reprojection error ({rms_error:.1f}px) solving extrinsic "
                f"from tags {sorted(seen_ids)} -- check tag survey / corner order "
                "in geometry_utils.tag_corners_world",
                throttle_duration_sec=5.0)

        R_cam_map, _ = cv2.Rodrigues(rvec)
        t_cam_map = tvec.reshape(3)

        # Invert: T_map_cam = T_cam_map^-1
        R_map_cam = R_cam_map.T
        t_map_cam = -R_map_cam @ t_cam_map

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.camera_frame_id or "global_cam_optical_frame"
        transform.transform.translation.x = float(t_map_cam[0])
        transform.transform.translation.y = float(t_map_cam[1])
        transform.transform.translation.z = float(t_map_cam[2])

        qx, qy, qz, qw = rotation_to_quaternion(R_map_cam)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(transform)
        self.last_transform = transform
        self.last_solve_time = self.get_clock().now()
        # Live data has taken over from whatever came off disk.
        self.extrinsic_from_file = False

        self.solve_count += 1
        if self.solve_count % 30 == 0:
            self.get_logger().info(
                f"map -> {transform.child_frame_id}: "
                f"t=({t_map_cam[0]:.3f}, {t_map_cam[1]:.3f}, {t_map_cam[2]:.3f}) "
                f"rms_reprojection={rms_error:.2f}px (tags {sorted(seen_ids)})")

        # Only ever persist solves we would trust ourselves -- a bad solve
        # written to disk is worse than no file at all, because the next frozen
        # run starts from it silently.
        if (self.save_extrinsic and self.extrinsic_yaml
                and rms_error <= self.max_reproj_err
                and self.solve_count % self.save_every_n == 0):
            self._save_extrinsic(transform, rms_error, seen_ids)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamCalibratorNode()
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
