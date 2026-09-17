#!/usr/bin/env python3
"""
global_cam_map_align_node.py  --  TRACK 1b-align  (run ONCE per saved map)

Solves the overhead camera's pose directly in the robot's SLAM `map` frame,
then reads the three floor tags' map-frame poses off it. Writes
floor_tags.yaml. This is the "closes the loop" step for global_cam_survey_node
-- that node gives the floor tags a complete, self-consistent pose relative to
EACH OTHER (tag 1 at the origin, tag 2 on +x), but that frame's origin is
arbitrary; it is not SLAM's `map`. Overhead detections written in that frame
would land on the RViz map rotated and offset from the RGB-D camera's, and
object_tracker_node would fuse the two cameras' views into separate,
duplicate tracks instead of one.

The fix needs exactly one thing measured in BOTH frames. Tag 0, riding on top
of the robot, is that thing: SLAM (or, after the fact, AMCL against a saved
map) reports the robot's pose in `map`; the overhead camera reports where it
sees tag 0. Park the robot at a few spots, and every tag-0 sighting gives one
(map-frame position, image-pixel position) correspondence for tag 0's four
corners -- computed from `map -> base_footprint` TF plus the fixed
`tag0_to_base_footprint` mechanical offset (config/floor_tags.yaml), the same
convention global_cam_localizer_node.py uses, just run forward instead of
inverted.

Method, reusing global_cam_calibrator_node's exact algorithm (stack every
correspondence, ONE joint cv2.solvePnP) rather than averaging independent
per-pose solves:
  1. Collect tag-0 corner correspondences across >= min_poses well-separated
     robot positions (weakly conditioned otherwise -- all of tag 0's corners
     sit in one plane at a constant height above the floor).
  2. ONE cv2.solvePnP over all of them -> T_cam_map -> invert -> T_map_cam.
     The solved camera HEIGHT is the strongest sanity check available: it was
     never an input, so if it matches a tape measurement the solve is sound.
  3. With T_map_cam known, a per-tag solvePnP on each floor tag's own corners
     (any frame it's visible in, tags don't move) gives T_cam_tag_i; compose
     T_map_tag_i = T_map_cam @ T_cam_tag_i and read off (x, y, yaw).
  4. Cross-check the solved inter-tag distances against config/floor_tags.yaml
     `distances_m` (the tape measurements) as a free accuracy check.

Run the robot base + SLAM (or AMCL against a saved map) FIRST, and this node
SECOND, in the SAME session -- the result is only valid against the exact map
origin live at solve time. See README §6 for the full runbook. Re-run only if
the camera moves, a floor tag moves, or a new map is built.

For a camera-only bring-up with no robot available at all, see
global_cam_survey_node.py -- it solves the tags' relative frame from tape
measurements alone, but leaves the map-frame alignment for this node.
"""

import datetime
import shutil
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
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
    tag_corners_world,
    tag_object_points,
    yaw_to_rotation_z,
)

ROBOT_TAG_ID = 0
FLOOR_TAG_IDS = (1, 2, 3)
FLOOR_TAG_SAMPLE_TARGET = 30  # per tag, matches global_cam_survey_node's num_samples


class GlobalCamMapAlignNode(Node):
    def __init__(self) -> None:
        super().__init__("global_cam_map_align")

        self.declare_parameter("floor_tags_yaml", "")
        self.declare_parameter("detections_topic", "/global_cam/apriltag/detections")
        self.declare_parameter("camera_info_topic", "/global_cam/camera_info")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("min_poses", 3)
        self.declare_parameter("samples_per_pose", 20)
        self.declare_parameter("min_pose_separation_m", 0.4)
        # Manual gating: wait for the operator to press ENTER before sampling,
        # instead of inferring "parked" from a >min_pose_separation_m jump.
        # The automatic mode keeps sampling while the robot is driven away
        # (anything under that threshold still counts as the current pose), so
        # in-motion frames get mixed into the solve -- and because the bridge
        # stamps each frame on ARRIVAL, not capture, a moving robot's TF
        # lookup resolves to a pose slightly ahead of where the photo was
        # actually taken. Both vanish if the robot is stationary.
        self.declare_parameter("manual_trigger", True)
        # A pose is only usable if the robot really is parked. Reject a
        # capture whose map->base_footprint drifts more than this while its
        # samples are being collected.
        self.declare_parameter("max_motion_during_pose_m", 0.02)
        self.declare_parameter("distance_tolerance_m", 0.02)
        self.declare_parameter("max_reprojection_error_px", 5.0)
        self.declare_parameter("transform_timeout_sec", 0.25)
        # Every input to the solve is banked here when it completes. The solve
        # itself is pure maths over those raw observations, so re-running it
        # under a corrected tag0_to_base_footprint, tag0_size_m or tag_size_m
        # needs no robot, no camera and no driving -- see solve_from. Empty =
        # alongside floor_tags.yaml as align_captures.npz.
        # "floor_plane" takes rotation from the three floor tags and asks tag 0
        # only for yaw + x/y; "tag0_pnp" is the original all-6-DOF-from-tag-0
        # solve, kept as an escape hatch. See _solve_camera_pose_floor_plane.
        self.declare_parameter("solver", "floor_plane")
        self.declare_parameter("capture_file", "")
        # Replay: load a previous capture_file, solve, write, exit. Nothing
        # else needs to be running.
        self.declare_parameter("solve_from", "")

        gp = self.get_parameter

        self.yaml_path = str(gp("floor_tags_yaml").value)
        if not self.yaml_path:
            raise FileNotFoundError(
                "floor_tags_yaml parameter is required -- point it at "
                "config/floor_tags.yaml (needs tag_size_m, distances_m, and "
                "tag0_to_base_footprint)."
            )

        with open(self.yaml_path) as f:
            self.floor_tags = yaml.safe_load(f)

        self.tag_size = float(self.floor_tags["tag_size_m"])
        # Tag 0 may be physically a different size from the floor tags -- it
        # is the one that has to stay readable from across the room, so it
        # often gets printed larger (or, if it was cut from a different sheet,
        # smaller). Modelling it at the floor tags' size puts every corner
        # off by (tag_size - true_size)/2 * sqrt(2), which is a pure scale
        # error on the single measurement the whole alignment rests on.
        self.tag0_size = float(self.floor_tags.get("tag0_size_m", self.tag_size))

        try:
            distances = self.floor_tags["distances_m"]
            self.d12 = float(distances["d_1_2"])
            self.d13 = float(distances["d_1_3"])
            self.d23 = float(distances["d_2_3"])
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "floor_tags.yaml needs a distances_m block with d_1_2, "
                "d_1_3 and d_2_3 (measured center-to-center, in metres) -- "
                "used here only as a post-solve accuracy check."
            ) from exc

        offset = self.floor_tags["tag0_to_base_footprint"]
        self.offset_xyz = np.array(
            [float(offset["x"]), float(offset["y"]), float(offset["z"])])
        self.offset_yaw = float(offset["yaw"])

        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.min_poses = max(1, int(gp("min_poses").value))
        self.samples_per_pose = max(1, int(gp("samples_per_pose").value))
        self.min_pose_separation = float(gp("min_pose_separation_m").value)
        self.manual_trigger = bool(gp("manual_trigger").value)
        self.max_motion_during_pose = float(gp("max_motion_during_pose_m").value)
        self.distance_tolerance = float(gp("distance_tolerance_m").value)
        self.max_reproj_err = float(gp("max_reprojection_error_px").value)
        self.transform_timeout = float(gp("transform_timeout_sec").value)

        self.camera_info: Optional[CameraInfo] = None
        self.done = False

        capture_file = str(gp("capture_file").value)
        self.capture_file = Path(capture_file) if capture_file else (
            Path(self.yaml_path).parent / "align_captures.npz")
        self.solve_from = str(gp("solve_from").value)
        self.solver = str(gp("solver").value)

        # Tag-0 correspondences, stacked across every accepted sample from
        # every pose -- fed straight into ONE joint solvePnP, exactly the
        # calibrator's approach.
        self.object_points: List[np.ndarray] = []  # each (4, 3), map frame
        self.image_points: List[np.ndarray] = []   # each (4, 2), pixels
        # The raw (base position, base yaw) behind each sample. Kept so the
        # solve can REBUILD tag 0's corners under a different
        # tag0_to_base_footprint and score it -- that is what turns a hand-
        # measured mounting offset from an assumption into something testable.
        self.base_poses: List[Tuple[np.ndarray, float]] = []

        # Floor tags don't move -- collect their own per-tag solvePnP
        # samples opportunistically from any frame they're visible in.
        self.floor_tag_samples: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
            i: [] for i in FLOOR_TAG_IDS
        }

        self.current_pose_anchor: Optional[np.ndarray] = None
        self.current_pose_count = 0
        self.poses_captured = 0

        # -- manual-trigger state. Samples for the pose being captured are
        # STAGED here rather than appended straight onto the solve arrays, so
        # a capture that turns out to have been taken while the robot crept
        # can be thrown away instead of quietly poisoning the solve.
        self.capture_armed = False
        self.solve_requested = False
        self.staged_object: List[np.ndarray] = []
        self.staged_image: List[np.ndarray] = []
        self.staged_base_xy: List[np.ndarray] = []
        self.staged_tf_stamp: List[float] = []
        self.staged_base_pose: List[Tuple[np.ndarray, float]] = []
        self.accepted_pose_xy: List[np.ndarray] = []

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, str(gp("camera_info_topic").value), self._camera_info_cb, 10)
        self.create_subscription(
            AprilTagDetectionArray, str(gp("detections_topic").value),
            self._detections_cb, 10)

        if self.solve_from:
            # Replay: no camera, no robot, no driving. main() calls replay()
            # directly and never spins -- every input is already in the file,
            # so there is nothing to wait for, and calling rclpy.shutdown()
            # from inside a spinning executor's callback deadlocks.
            self._load_captures(Path(self.solve_from))
            return

        if self.manual_trigger:
            self.get_logger().info(
                f"MANUAL mode: drive to a spot, STOP, then press ENTER to "
                f"capture. Need >= {self.min_poses} spots, >= "
                f"{self.min_pose_separation:.2f} m apart, varying the robot's "
                f"heading between them.")
            threading.Thread(target=self._stdin_loop, daemon=True).start()
            # Poll for the solve request on a timer rather than in the
            # detection callback: by the time the operator is done they may
            # have driven tag 0 out of the camera's view, and then no
            # detection callback would ever fire to notice.
            self.create_timer(0.2, self._check_solve_request)
        else:
            self.get_logger().info(
                f"Waiting for tag {ROBOT_TAG_ID} + {self.map_frame} -> "
                f"{self.base_frame} TF. Drive to >= {self.min_poses} well-spread "
                f"spots (>= {self.min_pose_separation:.2f} m apart); "
                f"{self.samples_per_pose} samples per spot.")

    # -- manual trigger ----------------------------------------------------
    def _prompt(self) -> None:
        ready = self.poses_captured >= self.min_poses
        extra = "  (or type 's' to solve now)" if ready else ""
        sys.stderr.write(
            f"\n>>> {self.poses_captured} pose(s) captured. Park the robot at "
            f"the next spot, then press ENTER to capture{extra}: ")
        sys.stderr.flush()

    def _stdin_loop(self) -> None:
        """Arm a capture on ENTER. Runs on its own thread; the detection
        callback does the actual work, so this only ever flips flags."""
        self._prompt()
        for line in sys.stdin:
            if self.done:
                return
            if line.strip().lower() in ("s", "solve"):
                if self.poses_captured >= self.min_poses:
                    self.solve_requested = True
                    return
                self.get_logger().warning(
                    f"Need at least {self.min_poses} poses before solving "
                    f"(have {self.poses_captured}).")
                self._prompt()
                continue
            if self.capture_armed:
                continue  # already collecting; ignore stray ENTERs
            self.staged_object.clear()
            self.staged_image.clear()
            self.staged_base_xy.clear()
            self.staged_tf_stamp.clear()
            self.staged_base_pose.clear()
            self.capture_armed = True
            self.get_logger().info(
                f"capturing {self.samples_per_pose} samples -- hold still ...")

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _detections_cb(self, msg: AprilTagDetectionArray) -> None:
        if self.done or self.camera_info is None:
            return

        by_id = {int(d.id): d for d in msg.detections}

        self._collect_floor_tags(by_id)

        tag0 = by_id.get(ROBOT_TAG_ID)
        if tag0 is None:
            # Say so, or an armed capture just hangs with no output at all --
            # tag 0 is the smallest tag and the only one that moves, so it is
            # the one that drops out (motion blur while the robot is still
            # settling, or simply too far from the camera).
            if self.capture_armed:
                self.get_logger().warning(
                    f"tag {ROBOT_TAG_ID} not visible -- capture is waiting. "
                    "Let the robot settle; if it stays missing, it is out of "
                    "view or too far. Watch `ros2 run risk_perception "
                    "global_cam_tag_monitor`.",
                    throttle_duration_sec=3.0)
            return

        try:
            if self.manual_trigger:
                # Latest available transform, NOT the one stamped to match this
                # image, and deliberately non-blocking.
                #
                # Time-matching buys nothing here: manual mode only samples
                # while the robot is parked, and the drift check below proves
                # it. Meanwhile the stamped form is actively harmful. It has to
                # wait for a transform at least as new as the image, and
                # `timeout=` makes that wait BLOCK this callback -- on the same
                # single-threaded executor that owns the /tf subscription
                # filling the buffer. So the wait can never be satisfied: the
                # buffer cannot advance while we sit inside it. Every detection
                # then burns the full timeout, starving /tf further, and the
                # buffer falls further behind -- measured here as a lag growing
                # past 40 s, which is what "extrapolation into the future" was
                # reporting. A dedicated TF thread (spin_thread=True) does not
                # fix it; not blocking does.
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, Time())
            else:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=self.transform_timeout))
        except TransformException as error:
            self.get_logger().warning(
                f"Cannot transform {self.base_frame} -> {self.map_frame}: "
                f"{error} (is SLAM/AMCL running?)", throttle_duration_sec=5.0)
            return

        image_points = np.array(
            [[c.x, c.y] for c in tag0.corners], dtype=np.float64)
        if image_points.shape != (4, 2):
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        base_xy = np.array([t.x, t.y])
        R_map_base = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        base_yaw = float(np.arctan2(R_map_base[1, 0], R_map_base[0, 0]))
        t_map_base = np.array([t.x, t.y, t.z])

        object_points = self._tag0_object_points(t_map_base, base_yaw)

        if self.manual_trigger:
            stamp = transform.header.stamp
            self._handle_manual(object_points, image_points, base_xy,
                                stamp.sec + stamp.nanosec * 1e-9,
                                (t_map_base, base_yaw))
            return

        # --- pose-cluster bookkeeping (automatic mode) ---
        if self.current_pose_anchor is None:
            self.current_pose_anchor = base_xy
            self.current_pose_count = 0
        elif np.linalg.norm(base_xy - self.current_pose_anchor) > self.min_pose_separation:
            if self.current_pose_count > 0:
                self.poses_captured += 1
                self.get_logger().info(
                    f"pose {self.poses_captured}/{self.min_poses} captured "
                    "-- move the robot")
            self.current_pose_anchor = base_xy
            self.current_pose_count = 0

        if self.current_pose_count >= self.samples_per_pose:
            # This spot already has enough samples -- wait for the operator
            # to move on rather than piling up more here.
            return

        self.object_points.append(object_points)
        self.image_points.append(image_points)
        self.base_poses.append((t_map_base, base_yaw))
        self.current_pose_count += 1

        if self.current_pose_count % 5 == 0:
            self.get_logger().info(
                f"  pose {self.poses_captured + 1}: "
                f"{self.current_pose_count}/{self.samples_per_pose} samples")

        # The pose the operator is currently parked at can complete the
        # requirement without ever "moving away" to trigger the branch above.
        if (self.poses_captured + 1 >= self.min_poses
                and self.current_pose_count >= self.samples_per_pose):
            self.poses_captured += 1
            self.get_logger().info(
                f"pose {self.poses_captured}/{self.min_poses} captured -- solving")
            self.done = True
            self._solve_and_write()

    def _tag0_object_points(self, t_map_base: np.ndarray, base_yaw: float) -> np.ndarray:
        """Tag 0's four corners in `map`, from the TF-observed base pose.

        tag0_to_base_footprint is base -> tag0 in the robot's own frame (see
        floor_tags.yaml); apply it forward from the TF-observed base pose to
        get tag 0's pose in `map` at this instant.
        """
        t_map_tag0 = t_map_base + yaw_to_rotation_z(base_yaw) @ self.offset_xyz
        return tag_corners_world(
            {"x": t_map_tag0[0], "y": t_map_tag0[1], "z": t_map_tag0[2],
             "yaw": base_yaw + self.offset_yaw},
            self.tag0_size)

    def _check_solve_request(self) -> None:
        if not self.solve_requested or self.done:
            return
        self.done = True
        self.get_logger().info(f"solving on {self.poses_captured} captured poses")
        self._solve_and_write()

    def _handle_manual(self, object_points: np.ndarray, image_points: np.ndarray,
                       base_xy: np.ndarray, tf_stamp: float,
                       base_pose: Tuple[np.ndarray, float]) -> None:
        if not self.capture_armed:
            return

        self.staged_object.append(object_points)
        self.staged_image.append(image_points)
        self.staged_base_xy.append(base_xy)
        self.staged_tf_stamp.append(tf_stamp)
        self.staged_base_pose.append(base_pose)

        n = len(self.staged_object)
        if n % 5 == 0:
            self.get_logger().info(f"  {n}/{self.samples_per_pose} samples")
        if n < self.samples_per_pose:
            return

        self.capture_armed = False

        # The drift check is only meaningful if TF actually moved on during
        # the capture -- a frozen buffer would report a rock-steady robot no
        # matter what it did.
        tf_span = self.staged_tf_stamp[-1] - self.staged_tf_stamp[0]
        if tf_span < 0.5:
            self.get_logger().warning(
                f"DISCARDED: {self.map_frame} -> {self.base_frame} advanced only "
                f"{tf_span:.2f} s across the capture, so 'the robot held still' "
                "cannot be confirmed. Check SLAM is still publishing, then "
                "press ENTER again.")
            self._prompt()
            return

        # Did the robot actually hold still? Peak-to-peak beats first-vs-last:
        # a creep out and back would cancel in the latter.
        xy = np.array(self.staged_base_xy)
        drift = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        if drift > self.max_motion_during_pose:
            self.get_logger().warning(
                f"DISCARDED: the robot moved {drift * 100:.1f} cm during "
                f"capture (limit {self.max_motion_during_pose * 100:.0f} cm). "
                "Let it settle -- teleop keys can leave it drifting -- and "
                "press ENTER again.")
            self._prompt()
            return

        if self.accepted_pose_xy:
            nearest = min(float(np.linalg.norm(base_xy - p))
                          for p in self.accepted_pose_xy)
            if nearest < self.min_pose_separation:
                self.get_logger().warning(
                    f"DISCARDED: only {nearest * 100:.0f} cm from an already "
                    f"captured pose (need {self.min_pose_separation * 100:.0f} "
                    "cm). Move further and press ENTER again.")
                self._prompt()
                return

        self.object_points.extend(self.staged_object)
        self.image_points.extend(self.staged_image)
        self.base_poses.extend(self.staged_base_pose)
        self.accepted_pose_xy.append(base_xy)
        self.poses_captured += 1
        self.get_logger().info(
            f"pose {self.poses_captured} accepted at "
            f"({base_xy[0]:.2f}, {base_xy[1]:.2f})  drift={drift * 100:.1f} cm")
        self._prompt()

    def _collect_floor_tags(self, by_id: dict) -> None:
        """Bank each floor tag's raw corner pixels.

        Deliberately stores OBSERVATIONS, not per-sample solvePnP poses. The
        tags are bolted to the floor and the camera is static, so every sample
        is a repeat measurement of one fixed pose; the right estimator is a
        single least-squares fit over all of them (see _solve_floor_tag),
        which is the same "stack every correspondence, ONE joint solvePnP"
        argument this node's docstring makes for tag 0. Averaging independent
        per-sample poses instead is not equivalent -- a pose average is not a
        least-squares fit, and on tags this small (~35 px) it collapsed the
        solved inter-tag distances to a third of their true value.
        """
        for tag_id in FLOOR_TAG_IDS:
            if len(self.floor_tag_samples[tag_id]) >= FLOOR_TAG_SAMPLE_TARGET:
                continue

            detection = by_id.get(tag_id)
            if detection is None:
                continue

            image_points = np.array(
                [[c.x, c.y] for c in detection.corners], dtype=np.float64)
            if image_points.shape != (4, 2):
                continue

            self.floor_tag_samples[tag_id].append(image_points)

    def _solve_floor_tag(self, tag_id: int, K: np.ndarray, D: np.ndarray):
        """One joint solvePnP over every corner observation of a static tag."""
        samples = self.floor_tag_samples[tag_id]
        image_points = np.concatenate(samples, axis=0)
        object_points = np.tile(tag_object_points(self.tag_size), (len(samples), 1))

        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        rms = float(np.sqrt(np.mean(np.sum(
            (reprojected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        return tvec.reshape(3), R_cam_tag, rms, len(samples)

    # -- capture persistence ------------------------------------------------
    def _save_captures(self) -> None:
        """Bank the raw observations behind this solve.

        Deliberately stores INPUTS -- corner pixels, base poses, intrinsics --
        and not the derived object points, because those bake in the very
        tag0_to_base_footprint and tag0_size_m values a re-solve wants to
        change.
        """
        try:
            payload = {
                "tag0_image_points": np.array(self.image_points),
                "base_positions": np.array([t for t, _ in self.base_poses]),
                "base_yaws": np.array([y for _, y in self.base_poses]),
                "K": np.array(self.camera_info.k).reshape(3, 3),
                "D": (np.array(self.camera_info.d) if self.camera_info.d
                      else np.zeros(5)),
            }
            for tag_id in FLOOR_TAG_IDS:
                payload[f"floor_{tag_id}"] = np.array(self.floor_tag_samples[tag_id])
            self.capture_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.capture_file, **payload)
            self.get_logger().info(
                f"captures saved -> {self.capture_file}\n"
                "    Re-solve with different offsets/sizes WITHOUT re-driving:\n"
                "      ros2 run risk_perception global_cam_map_align --ros-args \\\n"
                f"        -p floor_tags_yaml:={self.yaml_path} \\\n"
                f"        -p solve_from:={self.capture_file}")
        except Exception as error:   # never let bookkeeping lose a good solve
            self.get_logger().warning(f"could not save captures: {error}")

    def _load_captures(self, path: Path) -> None:
        data = np.load(path)
        self.image_points = [np.asarray(p, dtype=np.float64)
                             for p in data["tag0_image_points"]]
        self.base_poses = [
            (np.asarray(t, dtype=np.float64), float(y))
            for t, y in zip(data["base_positions"], data["base_yaws"])
        ]
        for tag_id in FLOOR_TAG_IDS:
            self.floor_tag_samples[tag_id] = [
                np.asarray(p, dtype=np.float64) for p in data[f"floor_{tag_id}"]
            ]
        info = CameraInfo()
        info.k = [float(v) for v in np.asarray(data["K"]).reshape(-1)]
        info.d = [float(v) for v in np.asarray(data["D"]).reshape(-1)]
        self.camera_info = info
        self.get_logger().info(
            f"replaying {path}: {len(self.image_points)} tag-0 samples, "
            f"{len(self.base_poses)} base poses. Using the CURRENT "
            f"floor_tags.yaml: tag_size_m={self.tag_size}, "
            f"tag0_size_m={self.tag0_size}, "
            f"offset=({self.offset_xyz[0]:+.4f}, {self.offset_xyz[1]:+.4f}, "
            f"yaw={self.offset_yaw:.5f})")

    def replay(self) -> None:
        """Solve straight from a loaded capture file. No spinning."""
        self.done = True
        self._solve_and_write()

    def _score_offset(self, offset_xyz: np.ndarray, offset_yaw: float,
                      K: np.ndarray, D: np.ndarray,
                      tag0_size: Optional[float] = None):
        """Rebuild tag 0's corners under a candidate mounting offset, re-solve,
        and return (rms_px, rvec, tvec). Everything else is held fixed, so the
        only thing being scored is the offset."""
        object_points = np.concatenate([
            tag_corners_world(
                {"x": (t + yaw_to_rotation_z(yaw) @ offset_xyz)[0],
                 "y": (t + yaw_to_rotation_z(yaw) @ offset_xyz)[1],
                 "z": (t + yaw_to_rotation_z(yaw) @ offset_xyz)[2],
                 "yaw": yaw + offset_yaw},
                self.tag0_size if tag0_size is None else tag0_size)
            for t, yaw in self.base_poses
        ], axis=0)
        image_points = np.concatenate(self.image_points, axis=0)

        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return float("inf"), None, None

        reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        rms = float(np.sqrt(np.mean(np.sum(
            (reprojected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        return rms, rvec, tvec

    def _scan_mounting_yaw(self, K: np.ndarray, D: np.ndarray):
        """Score tag 0's mounting yaw at each 90 deg step.

        Which way tag 0's own +x edge points relative to robot-forward is the
        one term in tag0_to_base_footprint that cannot be eyeballed reliably,
        and getting it wrong is not subtle: a 90 deg error slides every corner
        one position round the square -- a full tag width, ~38 px at these
        ranges, which swamps everything else. It is also a discrete choice, so
        scoring all four costs three extra solvePnP calls and removes the
        guesswork entirely.
        """
        results = []
        for step in range(4):
            candidate = self.offset_yaw + step * (np.pi / 2.0)
            rms, _, _ = self._score_offset(self.offset_xyz, candidate, K, D)
            results.append((rms, step, candidate))

        self.get_logger().info("mounting-yaw scan (tag0_to_base_footprint.yaw):")
        best = min(results)
        for rms, step, candidate in results:
            mark = "  <== best" if (rms, step) == (best[0], best[1]) else ""
            current = "  (configured)" if step == 0 else ""
            self.get_logger().info(
                f"  yaw {np.degrees(candidate):+7.1f} deg -> RMS {rms:7.2f} px"
                f"{current}{mark}")

        if best[1] != 0 and best[0] < results[0][0] * 0.5:
            self.get_logger().error(
                f"tag0_to_base_footprint.yaw looks WRONG. Set it to "
                f"{best[2]:.5f} ({np.degrees(best[2]):+.0f} deg) in "
                f"{self.yaml_path} and re-run -- RMS drops from "
                f"{results[0][0]:.1f} px to {best[0]:.1f} px. Nothing else "
                "here can be trusted until that is fixed.")
        return best

    def _refine_offset_xy(self, offset_yaw: float, K: np.ndarray, D: np.ndarray):
        """Coarse-to-fine search for the tag-0 mounting offset's x/y.

        Reported as a suggestion, never applied. This is identifiable only
        because the offset lives in the ROBOT's frame: it enters each sample
        as R(base_yaw) @ offset, so it rotates with the robot while a camera
        pose error does not. That is exactly why varying the heading between
        capture spots matters -- with identical headings the two are
        indistinguishable and this search would just relabel camera error as
        mounting offset.
        """
        centre = self.offset_xyz.copy()
        best = (float("inf"), centre)
        span, step = 0.15, 0.03

        for _ in range(3):
            grid = np.arange(-span, span + 1e-9, step)
            for dx in grid:
                for dy in grid:
                    candidate = np.array(
                        [centre[0] + dx, centre[1] + dy, centre[2]])
                    rms, _, _ = self._score_offset(candidate, offset_yaw, K, D)
                    if rms < best[0]:
                        best = (rms, candidate)
            centre = best[1].copy()
            span, step = step, step / 5.0

        return best

    def _scan_tag0_size(self, offset_yaw: float, K: np.ndarray, D: np.ndarray):
        """Scan tag 0's physical edge length, 40-250 mm, and report the best.

        A wrong tag-0 size is a pure scale error on its corners, which no
        camera pose can absorb -- it shows up as an RMS floor that resists
        every other adjustment. Reported, never applied: go measure the tag
        against this number rather than taking it on faith.
        """
        sizes = np.arange(0.040, 0.2501, 0.0005)
        scored = [(self._score_offset(self.offset_xyz, offset_yaw, K, D, float(s))[0],
                   float(s)) for s in sizes]
        return min(scored)

    @staticmethod
    def _floor_tilt_deg(entries: dict, solved_z: List[float]) -> float:
        """Angle of the solved floor plane from horizontal.

        The three tags really are on the floor, so any tilt here is error in
        the camera's solved rotation -- reported in degrees because that is
        far easier to judge than three separate z offsets.
        """
        A = np.array([[entries[i]["x"], entries[i]["y"], 1.0]
                      for i in FLOOR_TAG_IDS])
        try:
            gx, gy, _ = np.linalg.solve(A, np.array(solved_z))
        except np.linalg.LinAlgError:      # tags collinear -> plane undefined
            return float("nan")
        return float(np.degrees(np.arctan(np.hypot(gx, gy))))

    def _pose_spread_area(self) -> float:
        """Convex-hull area of the distinct capture positions, m^2.

        The single number that governs how well rotation is constrained: the
        tag-0 correspondences are coplanar, so only the footprint they span
        separates rotation from translation. Monotone chain, to avoid a scipy
        dependency this package treats as optional.
        """
        pts = sorted({(round(float(t[0]), 2), round(float(t[1]), 2))
                      for t, _ in self.base_poses})
        if len(pts) < 3:
            return 0.0

        def half(points):
            out: List[tuple] = []
            for p in points:
                while len(out) >= 2:
                    (ax, ay), (bx, by) = out[-2], out[-1]
                    if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                        break
                    out.pop()
                out.append(p)
            return out[:-1]

        hull = half(pts) + half(pts[::-1])
        return 0.5 * abs(sum(
            hull[i][0] * hull[(i + 1) % len(hull)][1]
            - hull[(i + 1) % len(hull)][0] * hull[i][1]
            for i in range(len(hull))))

    @staticmethod
    def _quad_centre(corners: np.ndarray) -> np.ndarray:
        """Intersection of a quad's diagonals.

        The projectively correct centre of a planar square: under perspective
        the centroid of the four projected corners is NOT the projection of
        the centre, though the diagonals still meet there.
        """
        p, r = corners[0], corners[2] - corners[0]
        q, s = corners[1], corners[3] - corners[1]
        denom = np.cross(r, s)
        if abs(denom) < 1e-9:
            return corners.mean(axis=0)
        return p + (np.cross(q - p, s) / denom) * r

    def _solve_camera_pose_floor_plane(self, K: np.ndarray, D: np.ndarray):
        """Camera pose with rotation taken from the FLOOR, not from tag 0.

        The tag-0-only solve asks four coplanar corners at one constant height
        to pin all 6 DOF, which is the degenerate case for PnP: rotation and
        translation trade off against each other, so a small capture footprint
        yields a low reprojection error and a badly tilted camera at the same
        time.

        But the three floor tags ARE the floor. Their camera-frame centres are
        recovered far more accurately than tag 0's pose (they are larger,
        static, and averaged over 30 samples), and the plane through them
        gives roll, pitch and camera height outright. That leaves tag 0
        responsible for only yaw and x/y -- a 2D rigid fit, well conditioned
        even from a thin footprint.

        Tag 0 contributes only its CENTRE here, found by ray/plane
        intersection at its known mounting height. Its own size, its tilt and
        its 90 deg mounting yaw therefore stop mattering entirely -- the three
        things that cost the most time to get right under the old solver.

        The trade: tag 0's mounting HEIGHT now matters (it sets the plane the
        rays are intersected with), where before it was absorbed. And the
        floor-tag z check becomes vacuous -- z is 0 by construction -- so the
        honest accuracy figure is the 2D fit residual returned here, in metres.
        """
        centres = {}
        for tag_id in FLOOR_TAG_IDS:
            solved = self._solve_floor_tag(tag_id, K, D)
            if solved is None:
                return None
            centres[tag_id] = solved[0]

        normal = np.cross(centres[2] - centres[1], centres[3] - centres[1])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            self.get_logger().error(
                "The three floor tags are collinear in the camera's view, so "
                "they define no plane. Move one of them.")
            return None
        normal /= norm
        if normal @ (-centres[1]) < 0:      # "up" = toward the camera at origin
            normal = -normal
        cam_height = float(normal @ (-centres[1]))

        # Tag 0 rides at a known height above that plane.
        plane_point = centres[1] + self.offset_xyz[2] * normal

        cam_pts, map_pts = [], []
        for corners, (t_map_base, base_yaw) in zip(self.image_points, self.base_poses):
            pixel = self._quad_centre(np.asarray(corners, dtype=np.float64))
            ray = cv2.undistortPoints(pixel.reshape(1, 1, 2), K, D).reshape(2)
            ray = np.array([ray[0], ray[1], 1.0])
            ray /= np.linalg.norm(ray)
            denom = normal @ ray
            if abs(denom) < 1e-9:
                continue
            cam_pts.append(ray * ((normal @ plane_point) / denom))
            map_pts.append(
                (t_map_base + yaw_to_rotation_z(base_yaw) @ self.offset_xyz)[:2])

        if len(cam_pts) < 2:
            self.get_logger().error("too few tag-0 centres to fit.")
            return None
        cam_pts, map_pts = np.array(cam_pts), np.array(map_pts)

        # In-plane basis, so the remaining fit is a plain 2D rigid alignment.
        e1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)

        uv = np.c_[(cam_pts - plane_point) @ e1, (cam_pts - plane_point) @ e2]
        uv_c, map_c = uv.mean(axis=0), map_pts.mean(axis=0)
        U, _, Vt = np.linalg.svd((uv - uv_c).T @ (map_pts - map_c))
        R2 = Vt.T @ np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t2 = map_c - R2 @ uv_c

        R_map_cam = np.array([
            R2[0, 0] * e1 + R2[0, 1] * e2,
            R2[1, 0] * e1 + R2[1, 1] * e2,
            normal,
        ])
        t_map_cam = np.array([
            t2[0] - R_map_cam[0] @ plane_point,
            t2[1] - R_map_cam[1] @ plane_point,
            -float(normal @ centres[1]),
        ])

        # Self-check through the assembled transform, not the 2D fit that
        # produced it -- a transposed rotation still satisfies the 2D residual
        # and still lands every floor tag at z=0, so neither would catch it.
        residuals = np.linalg.norm(
            (R_map_cam @ cam_pts.T).T + t_map_cam - np.c_[map_pts, np.full(len(map_pts), self.offset_xyz[2])],
            axis=1)
        return R_map_cam, t_map_cam, cam_height, residuals

    def _solve_and_write(self) -> None:
        missing = [i for i in FLOOR_TAG_IDS if not self.floor_tag_samples[i]]
        if missing:
            self.get_logger().error(
                f"Floor tags {missing} were never seen during collection -- "
                "cannot solve their map-frame pose. Re-run with all three "
                "floor tags in view at some point (they don't need to be "
                "visible at the same time as tag 0).")
            rclpy.shutdown()
            return

        # Rebuild tag 0's corners from the raw base poses rather than reusing
        # what was built during capture -- on a replay those were built under
        # a different tag0_to_base_footprint/tag0_size_m, and this guarantees
        # the solve always reflects the config as it stands right now.
        object_points_arr = np.concatenate([
            self._tag0_object_points(t, yaw) for t, yaw in self.base_poses
        ], axis=0) if self.base_poses else np.concatenate(self.object_points, axis=0)
        image_points_arr = np.concatenate(self.image_points, axis=0)

        K = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        D = (
            np.array(self.camera_info.d, dtype=np.float64)
            if self.camera_info.d else np.zeros(5)
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points_arr, image_points_arr, K, D,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.get_logger().error("solvePnP failed on the stacked tag-0 correspondences.")
            rclpy.shutdown()
            return

        reprojected, _ = cv2.projectPoints(object_points_arr, rvec, tvec, K, D)
        reprojected = reprojected.reshape(-1, 2)
        rms_error = float(np.sqrt(
            np.mean(np.sum((reprojected - image_points_arr) ** 2, axis=1))))

        floor_plane_result = None
        if self.solver == "floor_plane":
            floor_plane_result = self._solve_camera_pose_floor_plane(K, D)
            if floor_plane_result is None:
                self.get_logger().error(
                    "floor-plane solve failed; falling back to solver:=tag0_pnp.")

        area = self._pose_spread_area()
        n_spots = len({(round(float(t[0]), 2), round(float(t[1]), 2))
                       for t, _ in self.base_poses})
        self.get_logger().info(
            f"capture geometry: {n_spots} distinct spots spanning "
            f"{area:.2f} m^2")
        if area < 1.0 and self.solver != "floor_plane":
            self.get_logger().warning(
                f"Capture footprint is only {area:.2f} m^2 -- thin. Rotation "
                "is the weakly-determined part of this solve (all tag-0 "
                "corners lie in one horizontal plane), and footprint is what "
                "constrains it. A low RMS from a small footprint can still "
                "hide a badly tilted camera; watch the floor-tag z values "
                "below, not the RMS. Aim for >1.5 m^2 with poses at the "
                "corners of the camera's view. Or switch to the default "
                "solver:=floor_plane, which takes rotation from the floor "
                "tags and is largely insensitive to footprint.")

        self.get_logger().info(f"RMS reprojection error: {rms_error:.2f} px")
        if rms_error > self.max_reproj_err:
            self.get_logger().warning(
                f"High reprojection error ({rms_error:.1f}px) -- check "
                "tag0_to_base_footprint, tag_size_m, and that the robot was "
                "stationary while each pose's samples were collected.")
            if self.base_poses:
                best_rms, _, best_yaw = self._scan_mounting_yaw(K, D)
                # Only worth suggesting an x/y once the yaw is settled -- a
                # 90 deg yaw error dwarfs any centimetre-scale translation, and
                # the search would just chase it.
                if best_rms > self.max_reproj_err:
                    rms_size, size = self._scan_tag0_size(best_yaw, K, D)
                    self.get_logger().info(
                        f"tag-0 size scan: best fit {size * 1000:.1f} mm "
                        f"(configured {self.tag0_size * 1000:.1f} mm) -> RMS "
                        f"{best_rms:.2f} px becomes {rms_size:.2f} px")
                    if abs(size - self.tag0_size) > 0.003:
                        self.get_logger().error(
                            f"tag_0's modelled size looks WRONG. MEASURE the "
                            f"black square on tag 0; if it is ~{size * 1000:.0f} mm, "
                            f"add `tag0_size_m: {size:.4f}` to {self.yaml_path} "
                            "and re-run. A wrong tag size is a scale error no "
                            "camera pose can absorb.")

                    rms_xy, offset_xy = self._refine_offset_xy(best_yaw, K, D)
                    dx, dy = offset_xy[0] - self.offset_xyz[0], offset_xy[1] - self.offset_xyz[1]
                    self.get_logger().info(
                        "mounting-offset x/y refinement (best fit, NOT applied):")
                    self.get_logger().info(
                        f"  x: {self.offset_xyz[0]:+.4f} -> {offset_xy[0]:+.4f} m "
                        f"({dx * 100:+.1f} cm)")
                    self.get_logger().info(
                        f"  y: {self.offset_xyz[1]:+.4f} -> {offset_xy[1]:+.4f} m "
                        f"({dy * 100:+.1f} cm)")
                    self.get_logger().info(
                        f"  RMS {best_rms:.2f} px -> {rms_xy:.2f} px. This is a "
                        "FIT, not a measurement: check it against the hardware "
                        "before trusting it, and re-run with more/wider-spread "
                        "poses if it looks implausible.")

        if floor_plane_result is not None:
            R_map_cam, t_map_cam, cam_height, residuals = floor_plane_result
            self.get_logger().info(
                "solver=floor_plane: rotation + height taken from the three "
                "floor tags; tag 0 supplies only yaw and x/y.")
            self.get_logger().info(
                f"  tag-0 fit residual: mean {residuals.mean() * 100:.1f} cm, "
                f"max {residuals.max() * 100:.1f} cm over {len(residuals)} "
                "samples -- this is the honest accuracy figure, in metres: "
                "how far the camera's view of the robot sits from where SLAM "
                "put it. The floor-tag z check below is now vacuous (z=0 by "
                "construction); judge by this and the distance check.")
            if residuals.mean() > 0.05:
                self.get_logger().warning(
                    f"mean residual {residuals.mean() * 100:.1f} cm is large. "
                    "Most likely tag0_to_base_footprint's x/y or z (z now "
                    "matters -- it sets the plane the tag-0 rays are "
                    "intersected with), or SLAM pose error at the capture "
                    "spots.")
        else:
            R_cam_map, _ = cv2.Rodrigues(rvec)
            t_cam_map = tvec.reshape(3)
            R_map_cam = R_cam_map.T
            t_map_cam = -R_map_cam @ t_cam_map

        self.get_logger().info(
            f"camera position in map: "
            f"({t_map_cam[0]:.3f}, {t_map_cam[1]:.3f}, {t_map_cam[2]:.3f}) m "
            f"-- that z is the camera's height above the floor, "
            "sanity-check it against a tape measurement")

        # --- read the floor tags' map-frame poses off the solved camera pose ---
        entries = {}
        solved_z: List[float] = []
        for tag_id in FLOOR_TAG_IDS:
            solved = self._solve_floor_tag(tag_id, K, D)
            if solved is None:
                self.get_logger().error(
                    f"solvePnP failed for floor tag {tag_id} -- cannot write "
                    "a calibration.")
                rclpy.shutdown()
                return
            tvec, R_cam_tag, tag_rms, n_samples = solved

            R_map_tag = R_map_cam @ R_cam_tag
            t_map_tag = R_map_cam @ tvec + t_map_cam
            yaw = float(np.arctan2(R_map_tag[1, 0], R_map_tag[0, 0]))

            entries[tag_id] = {
                "x": float(t_map_tag[0]), "y": float(t_map_tag[1]),
                "z": 0.0, "yaw": yaw,
            }
            # z is forced to 0 above (the tags ARE on the floor), so keep the
            # solved value -- it is the sharpest check on the camera ROTATION
            # available. Nothing constrained it, so a floor tag landing well
            # off z=0 means R_map_cam is tilted, which then shrinks the
            # horizontal distance check below by cos(tilt) and makes it look
            # like a scale error.
            solved_z.append(float(t_map_tag[2]))
            self.get_logger().info(
                f"  tag_{tag_id}: x={entries[tag_id]['x']:.4f} "
                f"y={entries[tag_id]['y']:.4f} z={t_map_tag[2]:+.4f} "
                f"yaw={np.degrees(yaw):+.1f} deg "
                f"(range {np.linalg.norm(tvec):.2f} m, {n_samples} samples, "
                f"fit {tag_rms:.2f} px)")

        # --- consistency check: solved distances vs. the tape measurements ---
        observed = {
            "d_1_2": float(np.hypot(entries[1]["x"] - entries[2]["x"],
                                     entries[1]["y"] - entries[2]["y"])),
            "d_1_3": float(np.hypot(entries[1]["x"] - entries[3]["x"],
                                     entries[1]["y"] - entries[3]["y"])),
            "d_2_3": float(np.hypot(entries[2]["x"] - entries[3]["x"],
                                     entries[2]["y"] - entries[3]["y"])),
        }
        measured = {"d_1_2": self.d12, "d_1_3": self.d13, "d_2_3": self.d23}

        worst_z = max(abs(z) for z in solved_z)
        if worst_z > 0.05:
            self.get_logger().error(
                f"Floor tag solved {worst_z * 100:.0f} cm off the floor "
                "(z should be ~0). The camera ROTATION is wrong, so the "
                "distance check below is measuring a tilted plane's shadow "
                "and will read short (by cos(tilt) -- so treat any distance "
                "mismatch below as UNDIAGNOSTIC until this is fixed).")
            self.get_logger().error(
                f"  the solved floor plane is tilted "
                f"{self._floor_tilt_deg(entries, solved_z):.1f} deg. Every "
                "tag-0 corner sits in one horizontal plane at a constant "
                "height, which is the degenerate case for recovering rotation "
                "-- so rotation is pinned by how WIDELY the capture poses are "
                "spread, not by how still the robot was. If drift was ~0 cm "
                "at every pose, motion is not the problem: recapture with "
                "more poses (6-8) pushed out to the corners of the camera's "
                "view.")

        self.get_logger().info("distance check (solved vs. your tape measure):")
        worst = 0.0
        for key in ("d_1_2", "d_1_3", "d_2_3"):
            error = observed[key] - measured[key]
            worst = max(worst, abs(error))
            self.get_logger().info(
                f"  {key}: solved {observed[key]:.4f} m  vs  measured "
                f"{measured[key]:.4f} m   (off by {error * 100:+.1f} cm)")

        if worst > self.distance_tolerance:
            bad = [k for k in observed
                   if abs(observed[k] - measured[k]) > self.distance_tolerance]
            if len(bad) == 1:
                # The camera solves all three pairs through the same intrinsics
                # and the same plane, so a genuine calibration fault moves them
                # together. One pair out on its own is far more likely to be a
                # mis-measured tape than a bad solve.
                self.get_logger().warning(
                    f"{bad[0]} is off by "
                    f"{(observed[bad[0]] - measured[bad[0]]) * 100:+.1f} cm "
                    "while the other two agree to within "
                    f"{max(abs(observed[k] - measured[k]) for k in observed if k not in bad) * 100:.1f} cm. "
                    "A real scale or pose error would shift all three "
                    f"together, so RE-MEASURE {bad[0]} before believing this "
                    "-- update distances_m in floor_tags.yaml and re-solve "
                    "with solve_from (no re-driving).")
            else:
                self.get_logger().error(
                    f"MISMATCH of {worst * 100:.1f} cm exceeds the "
                    f"{self.distance_tolerance * 100:.0f} cm tolerance, on "
                    f"{len(bad)} of 3 pairs. Likely causes, in order: "
                    "(1) tag_size_m does not match the actual printed tag; "
                    "(2) tag0_to_base_footprint is wrong; (3) the robot moved "
                    "during a pose's sample collection. Writing the file "
                    "anyway, but the calibration will be off.")

        self._write_yaml(entries, observed)
        if not self.solve_from:
            self._save_captures()
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

        text = f"""# floor_tags.yaml -- GENERATED by global_cam_map_align on {stamp}
#
# Do not hand-edit the tag_N blocks; re-run the alignment instead:
#   ros2 run risk_perception global_cam_map_align --ros-args \\
#     -p floor_tags_yaml:=<this file>
#
# Unlike global_cam_survey_node's output, these poses are in the robot's
# ACTUAL SLAM `map` frame (solved from tag 0 + map -> base_footprint TF, see
# global_cam_map_align_node.py), not an arbitrary tag-defined frame. They are
# only valid as long as the map origin doesn't move -- i.e. as long as you
# keep reloading the SAME saved map this was solved against, rather than
# re-running SLAM from scratch. Re-run this alignment if the camera is
# bumped, a floor tag moves, or you build a new map.
#
# Solved camera-observed distances at align time were
# d_1_2={observed['d_1_2']:.4f}, d_1_3={observed['d_1_3']:.4f},
# d_2_3={observed['d_2_3']:.4f} m -- compare against distances_m below.

tag_family: "tag36h11"
tag_size_m: {self.tag_size}          # outer edge of the black square, metres
tag0_size_m: {self.tag0_size}         # tag 0 only, if it differs from the floor tags

# Measured center-to-center distances -- kept only as a standing accuracy
# check against the solved positions above (see the log from this run).
distances_m:
  d_1_2: {self.d12}
  d_1_3: {self.d13}
  d_2_3: {self.d23}

# Solved tag poses in the SLAM `map` frame.
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
            "Alignment complete. global_cam_calibrator's map -> camera TF "
            "now refers to the real SLAM map frame.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalCamMapAlignNode()
    try:
        if node.solve_from:
            node.replay()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
