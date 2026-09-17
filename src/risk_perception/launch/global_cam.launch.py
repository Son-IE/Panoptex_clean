#!/usr/bin/env python3
"""
global_cam.launch.py

Terminal C (`conda activate panoptex` -- same environment as
risk_perception.launch.py, since this also runs GroundingDINO/SAM2). Brings
up the overhead camera and its full chain:

  global_cam_bridge (TCP:5000 -> /global_cam/image_raw + /global_cam/camera_info)
    -> apriltag_ros (tags 0,1,2,3)
         -> global_cam_calibrator  (tags 1,2,3 -> TF map -> camera)
         -> global_cam_localizer   (tag 0      -> /global_cam/robot_pose)
    -> gdino_detector (2nd instance) -> sam2_segmenter (2nd instance)
         -> global_cam_projector    -> /risk_perception/detections_3d_map
                                        (same topic the RGB-D chain uses --
                                        object_tracker_node fuses both)

Run alongside robot_base.launch.py (Terminal A) and risk_perception.launch.py
(Terminal B):
  ros2 launch risk_perception global_cam.launch.py

BEFORE this is useful, run the one-time survey to solve the floor tags'
poses from your measured distances (robot not needed, see the node's
docstring):
  ros2 launch risk_perception global_cam.launch.py enable_perception:=false
  ros2 run risk_perception global_cam_survey --ros-args \\
    -p floor_tags_yaml:=<share>/config/floor_tags.yaml

`enable_perception:=false` brings up just the camera + AprilTag half, with
no GPU models loaded -- which is what you want for calibration, and for
checking which physical tag carries which ID.

FROZEN EXTRINSIC: once the calibrator has written config/global_cam_extrinsic.yaml
(it saves good solves automatically -- see the node's docstring), the floor
tags are no longer needed at runtime:

  ros2 launch risk_perception global_cam.launch.py \\
      use_saved_extrinsic:=true enable_apriltag:=false enable_localizer:=false

`enable_apriltag:=false` ONLY makes sense with `use_saved_extrinsic:=true` --
otherwise nothing ever feeds the calibrator and map -> <camera optical frame>
never appears, which takes the whole overhead chain down with it. The
localizer is the tag-0 robot pose, so it is dead weight whenever the robot is
not running.

ISAAC SIM: `global_cam_source:=sim` drops the TCP bridge and nothing else --
Isaac Sim's ROS 2 bridge publishes /global_cam/image_raw and
/global_cam/camera_info itself (see tools/isaac_sim/global_cam_isaac.py),
and every downstream node is agnostic to who produced those topics. The
sim camera has zero distortion and Isaac publishes its own CameraInfo, so
global_cam_intrinsics_yaml is simply unused in this mode. Typical sim run
(camera placed from the same cached extrinsic the calibrator broadcasts):

  ros2 launch risk_perception global_cam.launch.py global_cam_source:=sim \\
      use_saved_extrinsic:=true enable_apriltag:=false enable_localizer:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

HOME = os.path.expanduser("~")
# see risk_perception.launch.py -- the checkout is at ~/workspace/Panoptex on
# some machines and ~/workspaces/Panoptex on others; probe rather than guess.
REPO_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces", "Panoptex"),
                 os.path.join(HOME, "workspace", "Panoptex")) if os.path.isdir(p)),
    os.path.join(HOME, "workspace", "Panoptex"))
# GroundingDINO's code (incl. its bundled model config) is installed editable
# into the `panoptex` conda env by setup.sh -- resolve it via CONDA_PREFIX so
# this works regardless of where conda itself is installed, as long as
# `conda activate panoptex` was run before launching (see README §3/§5).
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX", os.path.join(HOME, "miniconda3", "envs", "panoptex"))
GDINO_CONFIG_DEFAULT = os.path.join(
    CONDA_PREFIX, "src", "groundingdino", "groundingdino", "config",
    "GroundingDINO_SwinT_OGC.py")
# Weights are binaries, not code -- setup.sh downloads them straight into
# this repo (weights/, gitignored) rather than some scattered clone under $HOME.
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")
    params = os.path.join(share, "config", "risk_perception.yaml")
    apriltag_params = os.path.join(share, "config", "apriltag.yaml")

    args = [
        DeclareLaunchArgument(
            "floor_tags_yaml",
            default_value=os.path.join(share, "config", "floor_tags.yaml"),
        ),
        DeclareLaunchArgument(
            "global_cam_intrinsics_yaml",
            default_value=os.path.join(share, "config", "global_cam_intrinsics.yaml"),
        ),
        DeclareLaunchArgument(
            "gdino_config",
            default_value=GDINO_CONFIG_DEFAULT,
        ),
        DeclareLaunchArgument(
            "gdino_weights",
            default_value=os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth"),
        ),
        DeclareLaunchArgument(
            "sam2_checkpoint",
            default_value=os.path.join(WEIGHTS_DIR, "sam2.1_hiera_small.pt"),
        ),
        # The overhead scene changes slowly -- run detection less often than
        # the robot cam's default stride (3) to leave GPU headroom for both
        # model instances running concurrently.
        DeclareLaunchArgument("global_inference_stride", default_value="6"),
        # "real" = Pi over TCP via global_cam_bridge; "sim" = Isaac Sim's ROS 2
        # bridge publishes /global_cam/image_raw + /global_cam/camera_info
        # itself, so the TCP bridge (the only hardware-facing node in this
        # chain) is the one thing that must not run. Everything downstream is
        # topic-driven and identical in both modes.
        DeclareLaunchArgument("global_cam_source", default_value="real"),
        DeclareLaunchArgument("enable_bridge", default_value="true"),
        # Off = camera + AprilTag only, no GPU models. Use this for
        # calibration/survey runs.
        DeclareLaunchArgument("enable_perception", default_value="true"),
        # Where the calibrator caches / reads back the solved camera pose.
        DeclareLaunchArgument(
            "extrinsic_yaml",
            default_value=os.path.join(share, "config", "global_cam_extrinsic.yaml"),
        ),
        # Seed the calibrator from that file at startup so the overhead chain
        # comes up with no floor tags visible. See the module docstring.
        DeclareLaunchArgument("use_saved_extrinsic", default_value="false"),
        # Only safe to turn off together with use_saved_extrinsic:=true.
        DeclareLaunchArgument("enable_apriltag", default_value="true"),
        # Tag-0 robot pose -- pointless with no robot on the floor.
        DeclareLaunchArgument("enable_localizer", default_value="true"),
    ]

    run_bridge = PythonExpression([
        "'", LaunchConfiguration("enable_bridge"), "' == 'true' and '",
        LaunchConfiguration("global_cam_source"), "' == 'real'"
    ])

    bridge = Node(
        package="risk_perception",
        executable="global_cam_bridge",
        name="global_cam_bridge",
        output="screen",
        parameters=[params, {
            "intrinsics_yaml": LaunchConfiguration("global_cam_intrinsics_yaml"),
        }],
        condition=IfCondition(run_bridge),
    )

    # Deliberately fed the RAW image, not a rectified one: our calibrator /
    # survey / localizer nodes all run solvePnP with the full distortion
    # coefficients, which is only valid on unrectified pixels. Rectifying
    # first and then applying those same coefficients would correct the
    # distortion twice (~15 px near the frame edges with this lens).
    # CameraInfo is not remapped -- image_transport's CameraSubscriber
    # derives it from the image topic's namespace, giving
    # /global_cam/camera_info automatically.
    apriltag = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag",
        output="screen",
        parameters=[apriltag_params],
        remappings=[
            ("image_rect", "/global_cam/image_raw"),
            ("detections", "/global_cam/apriltag/detections"),
        ],
        condition=IfCondition(LaunchConfiguration("enable_apriltag")),
    )

    calibrator = Node(
        package="risk_perception",
        executable="global_cam_calibrator",
        name="global_cam_calibrator",
        output="screen",
        parameters=[params, {
            "floor_tags_yaml": LaunchConfiguration("floor_tags_yaml"),
            "extrinsic_yaml": LaunchConfiguration("extrinsic_yaml"),
            "use_saved_extrinsic": LaunchConfiguration("use_saved_extrinsic"),
        }],
    )

    localizer = Node(
        package="risk_perception",
        executable="global_cam_localizer",
        name="global_cam_localizer",
        output="screen",
        parameters=[params, {
            "floor_tags_yaml": LaunchConfiguration("floor_tags_yaml"),
        }],
        condition=IfCondition(LaunchConfiguration("enable_localizer")),
    )

    gdino_global = Node(
        package="risk_perception",
        executable="gdino_detector",
        name="gdino_detector_global",
        output="screen",
        parameters=[params, {
            "image_topic": "/global_cam/image_raw",
            "config_path": LaunchConfiguration("gdino_config"),
            "checkpoint_path": LaunchConfiguration("gdino_weights"),
            "inference_stride": LaunchConfiguration("global_inference_stride"),
            # prompt / box_threshold / text_threshold come from the yaml's
            # `gdino_detector_global:` block -- do NOT hardcode them here: a
            # launch override shadows the yaml silently. The prompt keeps
            # robot vocabulary on purpose; the projector's self-exclusion
            # gate grants robot-labeled detections its looser radius, and
            # that branch only fires if the prompt can produce them.
        }],
        # gdino_detector_node.py hardcodes its two publisher topics -- these
        # remappings are the only way to keep the two camera chains apart.
        remappings=[
            ("/risk_perception/detections_2d", "/global_cam/detections_2d"),
            ("/risk_perception/detection_image", "/global_cam/detection_image"),
        ],
        condition=IfCondition(LaunchConfiguration("enable_perception")),
    )

    sam2_global = Node(
        package="risk_perception",
        executable="sam2_segmenter",
        name="sam2_segmenter_global",
        output="screen",
        parameters=[params, {
            "image_topic": "/global_cam/image_raw",
            "detections_topic": "/global_cam/detections_2d",
            "mask_topic": "/global_cam/instance_mask",
            "segmentation_image_topic": "/global_cam/segmentation_image",
            "checkpoint_path": LaunchConfiguration("sam2_checkpoint"),
        }],
        condition=IfCondition(LaunchConfiguration("enable_perception")),
    )

    projector = Node(
        package="risk_perception",
        executable="global_cam_projector",
        name="global_cam_projector",
        output="screen",
        parameters=[params],
        condition=IfCondition(LaunchConfiguration("enable_perception")),
    )

    return LaunchDescription(args + [
        bridge, apriltag, calibrator, localizer,
        gdino_global, sam2_global, projector,
    ])
