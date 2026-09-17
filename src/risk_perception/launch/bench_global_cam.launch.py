#!/usr/bin/env python3
"""
bench_global_cam.launch.py  --  test the overhead camera's risk map with NO ROBOT.

Answers "does the global camera actually produce a sane risk map?" without
needing the Rosmaster on the floor. Put a real object under the overhead
camera and watch it become a marker and a risk blob on the saved map in RViz.

This works because the overhead path never needed the robot in the first
place: global_cam_projector_node looks up only map -> <camera optical frame>
(with Time(), i.e. latest -- correct for a bolted-down camera), and
object_tracker_node and risk_costmap_node touch no TF at all. What used to
make this awkward was launch-file coupling, not physics.

Brings up:

  map_server (+ lifecycle manager)  -> /map, the saved ~/maps/lab.yaml
  global_cam.launch.py              -> bridge, calibrator, GDINO/SAM2, projector
  object_tracker                    -> /risk_perception/world_objects + markers
  risk_costmap                      -> /risk_costmap
  RViz                              -> rviz/panoptex.rviz

Deliberately ABSENT, and why:

  robot base    it is not here.
  amcl          localizes /scan against the map, so it needs the lidar and the
                odom frame. With neither, it never activates and stalls the
                lifecycle manager, taking /map down with it. Hence
                enable_amcl:=false -- map_server alone.
  odom, /scan   consequences of the above. There is no map->odom transform in
                this mode and nothing that needs one.
  initialpose   seeds amcl from tag 0. No amcl, no robot, nothing to seed.
  robot cam     no Astra stream to consume, and a second GDINO+SAM2 pair would
                only compete for the GPU with the overhead pair.
  apriltag      NOT disabled here -- see PREREQUISITE below.

Because there is no robot, RViz's RobotModel display (/robot_description) and
LaserScan display (/scan) will show red errors. That is expected in this mode;
everything else on the map is live.

PREREQUISITE -- capture the extrinsic once, with the floor tags still down:

    ros2 launch risk_perception global_cam.launch.py enable_perception:=false \\
        extrinsic_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/global_cam_extrinsic.yaml

(the src/ path, not the installed share copy, so the result is committable --
same reason README §6 Step 3 does it). Wait for rms_reprojection under 5 px and
for "Saved extrinsic to ...", then Ctrl-C and rebuild.

Run:
    ros2 launch risk_perception bench_global_cam.launch.py

The floor tags can now be lifted: use_saved_extrinsic:=true is set below, so
the calibrator broadcasts the cached pose from startup. apriltag_node is still
launched, so putting the tags back re-solves live and overrides the cached
value -- pass enable_apriltag:=false to drop it and reclaim the CPU it competes
with the GPU models for (README §6).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

HOME = os.path.expanduser("~")
DEFAULT_MAP = os.path.join(HOME, "maps", "lab.yaml")


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")

    args = [
        # Must be the SAME map the extrinsic was solved against -- floor_tags.yaml
        # and the cached extrinsic are both tied to one map origin (README §6).
        DeclareLaunchArgument("map_yaml", default_value=DEFAULT_MAP),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        # Leave the live solve available by default; set false once you trust
        # the cached extrinsic and want the CPU back.
        DeclareLaunchArgument("enable_apriltag", default_value="true"),
        # "sim" swaps the Pi's TCP bridge for Isaac Sim's own /global_cam
        # publishers (see global_cam.launch.py). Bench mode is the natural
        # first target for sim -- no robot either way.
        DeclareLaunchArgument("global_cam_source", default_value="real"),
        # Which cached camera pose the calibrator loads (bench always runs
        # use_saved_extrinsic:=true, see below). Default = the installed
        # real-lab capture; for Isaac Sim pass the exported sim pose by
        # absolute src/ path, e.g.
        #   extrinsic_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/global_cam_extrinsic_sim.yaml
        DeclareLaunchArgument(
            "extrinsic_yaml",
            default_value=os.path.join(
                share, "config", "global_cam_extrinsic.yaml"),
        ),
    ]

    bench = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "panoptex_pc.launch.py")),
        launch_arguments={
            "map_yaml": LaunchConfiguration("map_yaml"),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "enable_apriltag": LaunchConfiguration("enable_apriltag"),
            "global_cam_source": LaunchConfiguration("global_cam_source"),
            "extrinsic_yaml": LaunchConfiguration("extrinsic_yaml"),
            # Tag-0 robot pose -- no robot, no tag 0.
            "enable_localizer": "false",
            # Serve the map, but do not try to localize a robot that is not here.
            "enable_amcl": "false",
            # Cached camera pose -> no floor tags required at startup.
            "use_saved_extrinsic": "true",
            # Nothing to seed, nothing to drive, no RGB-D stream to consume.
            "enable_initialpose": "false",
            "enable_robot_cam": "false",
            "enable_nav2": "false",
            # The point of the exercise.
            "enable_world_model": "true",
            "enable_costmap": "true",
        }.items(),
    )

    return LaunchDescription(args + [bench])
