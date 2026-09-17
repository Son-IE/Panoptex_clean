#!/usr/bin/env python3
"""
robot_base.launch.py

Terminal A (ROS only, NO venv -- robot_localization and the yahboom python nodes
live in system python).

Brings up:
  * yahboomcar_bringup X3   -> URDF tf (incl. base_link->camera_link), driver,
                               imu filter, EKF (odom -> base_footprint)
  * slam_toolbox async      -> map -> odom   (use_sim_time:=false baked in),
                               UNLESS map_yaml is set (see below)
  * map_server + amcl + lifecycle_manager (only when `map_yaml` is set)
                             -> map -> odom, loading a SAVED map instead of
    building a new one. Use this after the overhead-camera calibration
    (README §6, global_cam_map_align): that calibration is solved once
    against a fixed map origin and silently goes stale if the origin moves,
    which is exactly what happens every time slam_toolbox starts a fresh map.
    Setting `map_yaml` forces slam off -- only one node may publish
    map->odom (CLAUDE.md), so the two localization paths are mutually
    exclusive, not layered. `enable_amcl:=false` serves the map WITHOUT
    amcl: no map->odom at all, just /map. That is what you want with no
    robot on the floor (bench_global_cam.launch.py) -- amcl has no /scan
    and no odom frame to work with there, and would hang the lifecycle
    manager, taking /map down with it.
  * ekf_global (optional, OFF by default) -> map -> odom, REPLACING
    whichever of the above owns that transform. Fuses the overhead camera's
    tag-0 robot pose (global_cam_localizer_node, from global_cam.launch.py)
    on top of the odom-frame EKF's continuous estimate. See
    config/ekf_global.yaml and the plan's Workstream B before enabling this
    on hardware.

`use_sim_time` (default false) reaches map_server, amcl and both lifecycle
managers only -- the slam_toolbox include always hardcodes false (a
hardware-only path here; see its own launch_arguments below) and the
yahboomcar_bringup include has no use_sim_time of its own to thread.
`amcl_params` (default config/amcl.yaml, the lab/hardware tuning) lets a sim
caller swap in config/amcl_sim.yaml (Omni motion model, baked initial pose --
see panoptex_sim.launch.py) without touching this file.

Run:
  ros2 launch risk_perception robot_base.launch.py
  ros2 launch risk_perception robot_base.launch.py map_yaml:=$HOME/maps/lab.yaml
  ros2 launch risk_perception robot_base.launch.py enable_ekf_global:=true enable_slam:=false
  ros2 launch risk_perception robot_base.launch.py \
      map_yaml:=$HOME/maps/lab.yaml enable_base:=false enable_amcl:=false
  ros2 launch risk_perception robot_base.launch.py \
      enable_base:=false enable_slam:=false use_sim_time:=true \
      map_yaml:=$HOME/workspace/warehouse/maps/warehouse_x3_nav.yaml \
      amcl_params:=$HOME/workspace/Panoptex/src/risk_perception/config/amcl_sim.yaml

Do NOT also launch cartographer / rtabmap / gmapping -- they compete for
map->odom. Do NOT launch a static_transform_publisher for the camera: the URDF
already provides base_link -> camera_link.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("enable_slam", default_value="true"),
        # The base driver talks to the robot's serial hardware, so it only
        # runs on the robot. Set false to bring up map_server + amcl alone --
        # which is what you want when the base and lidar are already running
        # on the Jetson (laser_bringup_launch.py) and the saved map lives on
        # the PC. Localization is pure computation over /scan and TF, so it
        # can run on either machine; the PC is the better host here because
        # the map and the overhead-camera nodes are already there.
        DeclareLaunchArgument("enable_base", default_value="true"),
        DeclareLaunchArgument("enable_ekf_global", default_value="false"),
        # Non-empty -> load this saved map via map_server+amcl instead of
        # mapping fresh with slam_toolbox (see module docstring).
        DeclareLaunchArgument("map_yaml", default_value=""),
        # AMCL localizes /scan against the saved map, so it needs the lidar and
        # the odom frame -- i.e. the robot. Set false to serve the map ALONE,
        # which is what an overhead-camera-only bench test wants: without it,
        # amcl never reaches the active state and the lifecycle manager stalls
        # waiting on it, so map_server never publishes /map either.
        DeclareLaunchArgument("enable_amcl", default_value="true"),
        # Isaac Sim's overhead-camera + X3 stack runs entirely on sim time
        # (see panoptex_sim.launch.py / global_cams_sim.yaml); on hardware
        # this stays false. ParameterValue(..., value_type=bool) below turns
        # the launch-arg STRING "true"/"false" into an actual bool parameter
        # -- passing the LaunchConfiguration straight into a parameters dict
        # would hand nav2_amcl/map_server the Python str "true", which they
        # do not coerce.
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "amcl_params",
            default_value=os.path.join(
                get_package_share_directory("risk_perception"),
                "config", "amcl.yaml"),
        ),
    ]

    use_sim_time = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool)

    have_map = PythonExpression(["'", LaunchConfiguration("map_yaml"), "' != ''"])
    # slam runs only if explicitly enabled AND no saved map was given --
    # only one node may publish map->odom (CLAUDE.md).
    run_slam = PythonExpression([
        "'", LaunchConfiguration("enable_slam"), "' == 'true' and '",
        LaunchConfiguration("map_yaml"), "' == ''"
    ])
    run_amcl = PythonExpression([
        "'", LaunchConfiguration("map_yaml"), "' != '' and '",
        LaunchConfiguration("enable_amcl"), "' == 'true'"
    ])
    map_only = PythonExpression([
        "'", LaunchConfiguration("map_yaml"), "' != '' and '",
        LaunchConfiguration("enable_amcl"), "' != 'true'"
    ])

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("yahboomcar_bringup"),
                         "launch", "yahboomcar_bringup_X3_launch.py")
        ]),
        # pub_odom_tf stays false: the EKF owns odom->base_footprint.
        launch_arguments={"pub_odom_tf": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("enable_base")),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("slam_toolbox"),
                         "launch", "online_async_launch.py")
        ]),
        # The stock launch defaults this to TRUE. On hardware it must be false,
        # or slam silently never publishes map->odom.
        launch_arguments={"use_sim_time": "false"}.items(),
        condition=IfCondition(run_slam),
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "yaml_filename": LaunchConfiguration("map_yaml"),
            "use_sim_time": use_sim_time,
        }],
        condition=IfCondition(have_map),
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            LaunchConfiguration("amcl_params"),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(run_amcl),
    )

    # map_server and amcl are lifecycle nodes -- they stay UNCONFIGURED and
    # publish nothing without a manager to configure+activate them. The manager
    # brings its node_names up in order and blocks on each, so the list must
    # match exactly what was launched: naming amcl when it is not running
    # wedges map_server too. node_names cannot be built from a substitution,
    # hence two mutually exclusive managers rather than one parameterized node.
    localization_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
        condition=IfCondition(run_amcl),
    )

    map_only_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_server"],
        }],
        condition=IfCondition(map_only),
    )

    ekf_global_params = os.path.join(
        get_package_share_directory("risk_perception"), "config", "ekf_global.yaml")

    ekf_global = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_global_filter_node",
        output="screen",
        parameters=[ekf_global_params],
        condition=IfCondition(LaunchConfiguration("enable_ekf_global")),
    )

    return LaunchDescription(args + [
        base, slam, map_server, amcl,
        localization_lifecycle_manager, map_only_lifecycle_manager, ekf_global,
    ])
