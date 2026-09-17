#!/usr/bin/env python3
"""
bench_sim_multicam.launch.py

The Isaac Sim entry point: N overhead cameras -> fused risk map, no robot.

  sim_global_cams.launch.py      per-camera static TF + gdino + sam2 +
                                 projector for every camera in
                                 config/global_cams_sim.yaml
  risk_perception.launch.py      object_tracker + RViz only (robot-cam chain,
                                 predictive costmap and spatial prior off)
  risk_costmap_node              spawned HERE, not via risk_perception.launch,
                                 so its grid geometry can be overridden with
                                 the warehouse-floor extent from the config's
                                 `costmap:` block (the yaml defaults are the
                                 10x10 m lab)

Prereqs: tools/isaac_sim/global_cam_isaac.py already ran against the stage
(graphs + per-camera extrinsic yamls exist) and Isaac is Playing. Then:

  conda activate panoptex
  ros2 launch risk_perception bench_sim_multicam.launch.py

No map_server/amcl/Nav2 -- there is no robot and no serialized warehouse map
yet; the `map` frame exists purely as the root of the static camera TFs
(= the stage's /Root origin). RViz's Map display erroring on a missing
/map is expected bench noise, same as bench_global_cam.launch.py's missing
RobotModel. When the nav phase adds a robot, this file is where map_server /
Nav2 get included -- the costmap extent and TRANSIENT_LOCAL /risk_costmap
are already Nav2-shaped.

Verify (per camera, then fused):
  ros2 topic hz /global_cam_0/image_raw
  ros2 run tf2_ros tf2_echo map global_cam_0_optical_frame
  RViz: per-camera /global_cam_<i>/map_markers land on the same floor spot
        for an object seen by several cameras; /risk_perception/world_markers
        shows ONE track for it; /risk_costmap paints a blob under it.
"""

import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Same src-tree-first resolution as sim_global_cams.launch.py (edit config,
# relaunch, no rebuild).
HOME = os.path.expanduser("~")
REPO_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces", "Panoptex"),
                 os.path.join(HOME, "workspace", "Panoptex")) if os.path.isdir(p)),
    os.path.join(HOME, "workspace", "Panoptex"))
SRC_CONFIG_DIR = os.path.join(REPO_ROOT, "src", "risk_perception", "config")


def _costmap_node(context, *args, **kwargs):
    config_path = LaunchConfiguration("cams_config").perform(context)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geometry = cfg.get("costmap") or {}
    allowed = {"resolution", "width_m", "height_m", "origin_x", "origin_y"}
    overrides = {k: float(v) for k, v in geometry.items() if k in allowed}

    # Already inside an OpaqueFunction with `context` -- see
    # sim_global_cams.launch.py's _setup for why perform()+str-compare
    # instead of ParameterValue here.
    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        == "true"
    )
    overrides["use_sim_time"] = use_sim_time

    share = get_package_share_directory("risk_perception")
    params = os.path.join(share, "config", "risk_perception.yaml")
    return [Node(
        package="risk_perception",
        executable="risk_costmap",
        name="risk_costmap_node",
        output="screen",
        # yaml block first (alpha/min_risk/falloff tuning), then the
        # warehouse-floor geometry from global_cams_sim.yaml on top.
        parameters=[params, overrides],
    )]


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")

    args = [
        DeclareLaunchArgument(
            "cams_config",
            default_value=(
                os.path.join(SRC_CONFIG_DIR, "global_cams_sim.yaml")
                if os.path.isfile(os.path.join(SRC_CONFIG_DIR, "global_cams_sim.yaml"))
                else os.path.join(share, "config", "global_cams_sim.yaml")),
        ),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        # Isaac stamps every camera topic from /clock -- see
        # sim_global_cams.launch.py. True by default: this file is the Isaac
        # Sim entry point, there is no real-hardware caller.
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]

    cams = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "sim_global_cams.launch.py")),
        launch_arguments={
            "cams_config": LaunchConfiguration("cams_config"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # Tracker + RViz. Costmap is spawned by _costmap_node above instead
    # (geometry overrides); predictive costmap / spatial prior keep their
    # 10x10 m lab grids in risk_perception.yaml, so they stay off until the
    # nav phase gives them the same extent treatment.
    core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "risk_perception.launch.py")),
        launch_arguments={
            "enable_camera": "false",
            "enable_robot_cam_chain": "false",
            "enable_world_model": "true",
            "enable_costmap": "false",
            "enable_predictive_costmap": "false",
            "enable_spatial_prior": "false",
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    return LaunchDescription(args + [cams, core, OpaqueFunction(function=_costmap_node)])
