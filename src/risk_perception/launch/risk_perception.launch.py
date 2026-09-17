#!/usr/bin/env python3
"""
risk_perception.launch.py

Terminal B (ROS + venv). Brings up the camera and the whole perception chain:

  realsense -> gdino_detector -> sam2_segmenter -> rgbd_projector
            -> map_frame_projector -> object_tracker -> risk_costmap_node
                                                     -> predictive_risk_costmap_node
                                                     -> spatial_prior_node

risk_costmap_node (reactive, /risk_costmap) and predictive_risk_costmap_node
(Stage 2+3+4, /risk_costmap_predictive) both run: different topics, so they
never contend, and running them side by side is what makes the ablation a
matter of repointing risk_layer.topic rather than relaunching.

object_tracker consumes /risk_perception/detections_3d_map, the same topic
global_cam.launch.py's chain publishes onto -- this is the single point
where the robot camera's and overhead camera's views become one set of
fused tracks (see README).

Model paths are launch arguments (ROS yaml cannot expand ~), everything else
lives in config/risk_perception.yaml.

Run:
  ros2 launch risk_perception risk_perception.launch.py
Override a path:
  ros2 launch risk_perception risk_perception.launch.py gdino_weights:=/other/path.pth
Skip the costmap stage while still verifying detections:
  ros2 launch risk_perception risk_perception.launch.py enable_costmap:=false
Replay a bag through the world model alone (no camera, no GPU models):
  ros2 launch risk_perception risk_perception.launch.py \\
      enable_camera:=false enable_rviz:=true
Bring up RViz preloaded with the map/costmap/markers displays (rviz/panoptex.rviz):
  ros2 launch risk_perception risk_perception.launch.py enable_rviz:=true

`use_sim_time` (default false) is merged as a real bool parameter into every
node this file spawns, RViz included -- panoptex_sim.launch.py sets it true
for the whole Isaac Sim stack. `rviz_config` (default rviz/panoptex.rviz from
the installed share) lets the sim caller swap in rviz/panoptex_sim.rviz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

HOME = os.path.expanduser("~")


def _first_existing(candidates, fallback):
    """Pick the first path that is actually there, else the fallback.

    The checkout has been at both ~/workspace/Panoptex and
    ~/workspaces/Panoptex on different machines; hardcoding either one makes
    gdino_weights/sam2_checkpoint default to a path that does not exist, and
    the failure surfaces much later as a model-load error. Probe instead.
    """
    for c in candidates:
        if os.path.isdir(c):
            return c
    return fallback


REPO_ROOT = _first_existing(
    [os.path.join(HOME, "workspaces", "Panoptex"),
     os.path.join(HOME, "workspace", "Panoptex")],
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
    params = os.path.join(
        get_package_share_directory("risk_perception"),
        "config",
        "risk_perception.yaml",
    )

    args = [
        # Default matches config/risk_perception.yaml's gdino_detector.prompt
        # (the lab value) so this argument is a no-op unless a caller
        # overrides it -- panoptex_sim.launch.py does, to the SAME
        # warehouse-safe vocabulary gdino_detector_global already uses
        # (person . robot . forklift . cart .), because "chair . table .
        # monitor ." grounds on warehouse shelving here exactly the way it
        # was already found to for the overhead cameras -- see
        # global_cams_sim.yaml's box_threshold comment. Unlike
        # relation_prompt, empty is NOT a valid "off" value for this one
        # (GDINO would detect nothing), so the sentinel here is "matches the
        # yaml default", not "".
        DeclareLaunchArgument(
            "robot_cam_prompt",
            # Kept in step with risk_perception.yaml's gdino_detector.prompt,
            # per the "sentinel is 'matches the yaml default'" rule above --
            # this arg is passed unconditionally, so any other value here
            # would silently override the yaml. The 2026-09-11 merge kept
            # ours for that yaml prompt (user-a/sandbox's trim to
            # "person . robot . cart . chair . table . monitor . " was not in
            # the merge recipe), so this default follows it. Override with
            # robot_cam_prompt:= per launch/per ablation.
            default_value=(
                "person . ground mobile robot . cart . chair . table . "
                "monitor . camera . cable . "
            ),
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
        DeclareLaunchArgument("enable_camera", default_value="true"),
        # The ROBOT-camera half only (gdino/sam2/rgbd_projector/map_frame_projector).
        # Turning it off leaves object_tracker + risk_costmap + RViz running,
        # which is what the overhead camera needs when the robot is not on the
        # floor at all: object_tracker fuses whatever lands on
        # /risk_perception/detections_3d_map, and neither it nor
        # risk_costmap_node touches TF. See bench_global_cam.launch.py.
        DeclareLaunchArgument("enable_robot_cam_chain", default_value="true"),
        DeclareLaunchArgument("enable_world_model", default_value="true"),
        DeclareLaunchArgument("enable_costmap", default_value="true"),
        DeclareLaunchArgument("enable_predictive_costmap", default_value="true"),
        DeclareLaunchArgument("enable_spatial_prior", default_value="true"),
        DeclareLaunchArgument("enable_rviz", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        # Research logging -- a DIRECTORY handed to object_tracker /
        # predictive_risk_costmap / spatial_prior as `debug_log_dir`; each
        # writes its own <node>_<UTC>.csv there. "" = off (default).
        # tracker_log_unconfirmed also logs tracks below min_hits /
        # min_confidence so those gates' effect is visible.
        DeclareLaunchArgument("prior_log_dir", default_value=""),
        DeclareLaunchArgument("tracker_log_unconfirmed", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(
                get_package_share_directory("risk_perception"),
                "rviz", "panoptex.rviz"),
        ),
    ]

    # ParameterValue(..., value_type=bool): a LaunchConfiguration substitution
    # is always a string ("true"/"false"); merging that string straight into
    # a parameters dict hands every node the Python str "true", which rclpy
    # does not coerce to a bool -- see robot_base.launch.py.
    use_sim_time = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool)

    # Research-logging params, shared by the prior nodes below. debug_log_dir
    # is a plain string ("" = off); log_unconfirmed needs the same bool
    # coercion as use_sim_time.
    prior_log_dir = LaunchConfiguration("prior_log_dir")
    tracker_log_unconfirmed = ParameterValue(
        LaunchConfiguration("tracker_log_unconfirmed"), value_type=bool)

    # FindPackageShare, not get_package_share_directory: the latter runs while
    # the LaunchDescription is being *built*, so a missing realsense2_camera
    # aborts the whole launch even with enable_camera:=false (the camera is
    # often driven on the robot instead, by its own Astra driver). The
    # substitution is only resolved if the conditioned action actually runs.
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch", "rs_launch.py",
            ])
        ]),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "pointcloud.enable": "false",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_camera")),
    )

    run_robot_cam_chain = IfCondition(LaunchConfiguration("enable_robot_cam_chain"))

    gdino = Node(
        package="risk_perception",
        executable="gdino_detector",
        name="gdino_detector",
        output="screen",
        parameters=[params, {
            "config_path": LaunchConfiguration("gdino_config"),
            "checkpoint_path": LaunchConfiguration("gdino_weights"),
            "prompt": LaunchConfiguration("robot_cam_prompt"),
            "use_sim_time": use_sim_time,
            "debug_log_dir": prior_log_dir,
        }],
        condition=run_robot_cam_chain,
    )

    sam2 = Node(
        package="risk_perception",
        executable="sam2_segmenter",
        name="sam2_segmenter",
        output="screen",
        parameters=[params, {
            "checkpoint_path": LaunchConfiguration("sam2_checkpoint"),
            "use_sim_time": use_sim_time,
            "debug_log_dir": prior_log_dir,
        }],
        condition=run_robot_cam_chain,
    )

    rgbd = Node(
        package="risk_perception",
        executable="rgbd_projector",
        name="rgbd_projector",
        output="screen",
        parameters=[params, {"use_sim_time": use_sim_time}],
        condition=run_robot_cam_chain,
    )

    map_proj = Node(
        package="risk_perception",
        executable="map_frame_projector",
        name="map_frame_projector",
        output="screen",
        parameters=[params, {"use_sim_time": use_sim_time}],
        condition=run_robot_cam_chain,
    )

    object_tracker = Node(
        package="risk_perception",
        executable="object_tracker",
        name="object_tracker",
        output="screen",
        parameters=[params, {
            "use_sim_time": use_sim_time,
            "debug_log_dir": prior_log_dir,
            "log_unconfirmed": tracker_log_unconfirmed,
        }],
        condition=IfCondition(LaunchConfiguration("enable_world_model")),
    )

    costmap = Node(
        package="risk_perception",
        executable="risk_costmap",
        name="risk_costmap_node",
        output="screen",
        parameters=[params, {
            "use_sim_time": use_sim_time, "debug_log_dir": prior_log_dir}],
        condition=IfCondition(LaunchConfiguration("enable_costmap")),
    )

    # Publishes /risk_costmap_predictive, a different topic from the reactive
    # costmap above -- both run at once so they can be compared, and nothing
    # Nav2 sees changes until risk_layer.topic is repointed.
    predictive_costmap = Node(
        package="risk_perception",
        executable="predictive_risk_costmap",
        name="predictive_risk_costmap_node",
        output="screen",
        parameters=[params, {
            "use_sim_time": use_sim_time, "debug_log_dir": prior_log_dir}],
        condition=IfCondition(LaunchConfiguration("enable_predictive_costmap")),
    )

    spatial_prior = Node(
        package="risk_perception",
        executable="spatial_prior",
        name="spatial_prior_node",
        output="screen",
        parameters=[params, {
            "use_sim_time": use_sim_time, "debug_log_dir": prior_log_dir}],
        condition=IfCondition(LaunchConfiguration("enable_spatial_prior")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
    )

    return LaunchDescription(
        args + [camera, gdino, sam2, rgbd, map_proj, object_tracker,
                costmap, predictive_costmap, spatial_prior, rviz])
