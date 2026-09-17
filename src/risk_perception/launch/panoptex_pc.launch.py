#!/usr/bin/env python3
"""
panoptex_pc.launch.py  --  everything the PC runs, in one command.

The robot side stays manual and unchanged (on the Jetson, over SSH):

    ros2 launch yahboomcar_nav laser_bringup_launch.py
    ros2 launch astra_camera astro_pro_plus.launch.xml depth_registration:=true

Then here:

    ros2 launch risk_perception panoptex_pc.launch.py

which brings up, in dependency order:

  localization   robot_base.launch.py with enable_base:=false -- map_server +
                 amcl against the SAVED map. Never slam_toolbox: the overhead
                 calibration is solved against one fixed map origin and a
                 fresh SLAM session silently invalidates it. Skipped when
                 enable_nav2:=true, because Nav2's own bringup already
                 starts map_server + amcl (see run_localization below).
  overhead cam   global_cam.launch.py -- bridge, apriltag, calibrator,
                 localizer, and the overhead GDINO/SAM2/projector chain.
  initial pose   global_cam_initialpose -- seeds AMCL from tag 0, so no
                 "2D Pose Estimate" click.
  world model    risk_perception.launch.py with enable_camera:=false, because
                 the RGB-D camera is driven on the robot by the Astra driver.
                 This include hosts BOTH the robot-camera chain (gated on
                 enable_robot_cam) and the camera-independent half -- tracker,
                 risk costmap, RViz -- which stay up either way. object_tracker
                 fuses whatever reaches /risk_perception/detections_3d_map, so
                 with enable_robot_cam:=false the overhead camera alone still
                 produces a risk costmap. Neither it nor risk_costmap_node
                 touches TF, which is what makes bench_global_cam.launch.py
                 (no robot at all) possible.
  navigation     yahboomcar_nav's navigation_dwa_launch.py, OFF by default.

WHY NAV2 RUNS HERE AND NOT ON THE JETSON: dwa_nav_params.yaml in THIS
workspace wires nav2_risk_layer::RiskLayer into the global costmap; the
Jetson's copy does not, and libnav2_risk_layer.so is built only in this
workspace's install space. Nav2 loads costmap plugins in-process, so the
Jetson would fail to load it and the global costmap would never activate.

Turn pieces off individually, e.g. to calibrate:

    ros2 launch risk_perception panoptex_pc.launch.py \\
        enable_perception:=false enable_robot_cam:=false enable_nav2:=false

(the GPU models compete with apriltag_node for CPU and slow the overhead
stream badly -- see README §6).

To exercise the overhead camera's risk map with no robot at all, use
bench_global_cam.launch.py, which is this file with enable_amcl:=false,
enable_robot_cam:=false, enable_initialpose:=false and use_saved_extrinsic:=true.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

HOME = os.path.expanduser("~")
# Empty until a map has actually been saved (README Step 2) -- robot_base.launch.py
# treats "" as "run SLAM fresh" and a non-empty path as "load this saved map via
# AMCL" (its own default is already ""; this file used to override that with a
# hardcoded ~/maps/lab.yaml regardless of whether it existed, which failed
# map_server outright on any machine that hadn't saved a map yet, and there is
# no clean way to override it back to "" from the ros2 launch CLI -- it rejects
# `map_yaml:=""` as a malformed argument, since the shell collapses it to no
# value at all). Probe instead of assume.
_maybe_map = os.path.join(HOME, "maps", "lab.yaml")
DEFAULT_MAP = _maybe_map if os.path.isfile(_maybe_map) else ""
# ~/workspace vs ~/workspaces differs per machine -- see risk_perception.launch.py
_WS_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces"), os.path.join(HOME, "workspace"))
     if os.path.isdir(p)),
    os.path.join(HOME, "workspace"))
DEFAULT_NAV2_PARAMS = os.path.join(
    _WS_ROOT, "yahboomcar_ws", "src", "yahboomcar_nav",
    "params", "dwa_nav_params.yaml")


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")

    args = [
        DeclareLaunchArgument("map_yaml", default_value=DEFAULT_MAP),
        DeclareLaunchArgument("enable_localization", default_value="true"),
        # Off = serve the saved map without amcl. Needs the lidar + odom frame,
        # i.e. the robot; see robot_base.launch.py.
        DeclareLaunchArgument("enable_amcl", default_value="true"),
        DeclareLaunchArgument("enable_global_cam", default_value="true"),
        # "real" (Pi over TCP) or "sim" (Isaac Sim publishes the /global_cam
        # topics itself -- see global_cam.launch.py and
        # tools/isaac_sim/global_cam_isaac.py).
        DeclareLaunchArgument("global_cam_source", default_value="real"),
        # GPU models for the OVERHEAD camera. Off = camera + AprilTags only,
        # which is what calibration wants.
        DeclareLaunchArgument("enable_perception", default_value="true"),
        # Bring the overhead camera's extrinsic up from the cached solve rather
        # than waiting on the floor tags. See global_cam.launch.py.
        DeclareLaunchArgument("use_saved_extrinsic", default_value="false"),
        # Which cached solve. Default = the installed real-lab capture; for
        # Isaac Sim pass the sim export by absolute src/ path (no rebuild
        # needed that way), e.g. config/global_cam_extrinsic_sim.yaml.
        DeclareLaunchArgument(
            "extrinsic_yaml",
            default_value=os.path.join(
                get_package_share_directory("risk_perception"),
                "config", "global_cam_extrinsic.yaml"),
        ),
        # Only safe to turn off together with use_saved_extrinsic:=true, or
        # nothing feeds the calibrator and map -> camera never appears.
        DeclareLaunchArgument("enable_apriltag", default_value="true"),
        # Tag-0 robot pose -- pointless with no robot on the floor.
        DeclareLaunchArgument("enable_localizer", default_value="true"),
        DeclareLaunchArgument("enable_initialpose", default_value="true"),
        # The robot-camera chain ONLY (GDINO/SAM2/rgbd_projector/
        # map_frame_projector). The tracker, risk costmap and RViz are
        # independent of it -- see enable_world_model / enable_costmap below.
        DeclareLaunchArgument("enable_robot_cam", default_value="true"),
        DeclareLaunchArgument("enable_world_model", default_value="true"),
        DeclareLaunchArgument("enable_costmap", default_value="true"),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        # Off by default: send a goal only once localization has converged and
        # you have checked the fused overlay, or the robot plans against its
        # own phantom detection. See README.
        DeclareLaunchArgument("enable_nav2", default_value="false"),
        DeclareLaunchArgument("nav2_params_file", default_value=DEFAULT_NAV2_PARAMS),
    ]

    # Nav2's bringup_launch.py ALWAYS includes localization_launch.py when
    # slam:=false (its default) -- map_server + amcl + a lifecycle manager
    # named, identically to ours, lifecycle_manager_localization. So bringing
    # up our own localization alongside it means two AMCLs both broadcasting
    # map->odom and fighting over it. Nav2's amcl block in dwa_nav_params.yaml
    # is byte-identical to config/amcl.yaml (the latter was copied from it),
    # so deferring to Nav2's copy costs nothing.
    run_localization = PythonExpression([
        "'", LaunchConfiguration("enable_localization"), "' == 'true' and '",
        LaunchConfiguration("enable_nav2"), "' != 'true'"
    ])

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "robot_base.launch.py")),
        launch_arguments={
            "enable_base": "false",          # the base driver lives on the robot
            "map_yaml": LaunchConfiguration("map_yaml"),
            "enable_amcl": LaunchConfiguration("enable_amcl"),
        }.items(),
        condition=IfCondition(run_localization),
    )

    global_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "global_cam.launch.py")),
        launch_arguments={
            "global_cam_source": LaunchConfiguration("global_cam_source"),
            "enable_perception": LaunchConfiguration("enable_perception"),
            "use_saved_extrinsic": LaunchConfiguration("use_saved_extrinsic"),
            "extrinsic_yaml": LaunchConfiguration("extrinsic_yaml"),
            "enable_apriltag": LaunchConfiguration("enable_apriltag"),
            "enable_localizer": LaunchConfiguration("enable_localizer"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_global_cam")),
    )

    initialpose = Node(
        package="risk_perception",
        executable="global_cam_initialpose",
        name="global_cam_initialpose",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_initialpose")),
    )

    # NOT conditioned on enable_robot_cam: this include also hosts
    # object_tracker, risk_costmap and RViz, which are camera-agnostic. Gating
    # the whole include on the robot camera used to take all three down with
    # it, which meant the overhead camera could never be tested on its own.
    world_model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "risk_perception.launch.py")),
        launch_arguments={
            # The Astra on the robot publishes the RGB-D topics; the RealSense
            # driver this would otherwise start is not present here.
            "enable_camera": "false",
            "enable_robot_cam_chain": LaunchConfiguration("enable_robot_cam"),
            "enable_world_model": LaunchConfiguration("enable_world_model"),
            "enable_costmap": LaunchConfiguration("enable_costmap"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
        }.items(),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("yahboomcar_nav"),
                         "launch", "navigation_dwa_launch.py")),
        launch_arguments={
            "map": LaunchConfiguration("map_yaml"),
            "params_file": LaunchConfiguration("nav2_params_file"),
            "use_sim_time": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_nav2")),
    )

    return LaunchDescription(
        args + [localization, global_cam, initialpose, world_model, nav2])
