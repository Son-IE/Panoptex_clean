#!/usr/bin/env python3
"""
evaluation_reference.launch.py

Launches a SECOND predictive_risk_costmap_node instance, forced to
full-system settings (every mechanism on), publishing to
/risk_costmap_reference -- for SCORING ONLY, never wired to Nav2.

Why this exists: risk_perception/evaluation_metrics.py's risk_exposure()
must be computed against a risk field that is IDENTICAL across every
condition in an ablation study. Sampling exposure from whichever costmap
actually drove Nav2 that run makes an ablated ("weaker") condition
self-report lower exposure merely because it computed less risk -- not
because the robot was actually any safer. Running this fixed reference
node ALONGSIDE whatever ablated configuration is under test, on the same
tracked world state, gives every run a common yardstick.

Run this alongside risk_perception.launch.py (whatever ablation flags that
launch is using), then point tools/evaluate_run.py's --risk-topic at
/risk_costmap_reference when recording or replaying the evaluation bag:

  ros2 launch risk_perception risk_perception.launch.py <ablation flags>
  ros2 launch risk_perception evaluation_reference.launch.py

spatial_prior_weight / flow_blend_weight default to 0.0, matching
config/risk_perception.yaml -- override them once Spatial-Flow has actually
accumulated real data (see the Spatial-Flow prior's persist_path), or the
"full system" reference will silently exclude that prior's contribution.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params = os.path.join(
        get_package_share_directory("risk_perception"),
        "config",
        "risk_perception.yaml",
    )

    args = [
        DeclareLaunchArgument("reference_spatial_prior_weight", default_value="0.0"),
        DeclareLaunchArgument("reference_flow_blend_weight", default_value="0.0"),
        DeclareLaunchArgument(
            "enable_evaluation", default_value="true",
            description="also start evaluation_node, the live trial collector "
                        "tools/run_trials.py drives over /evaluation/trial_control"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="REQUIRED whenever a /clock source (Isaac Sim) is driving "
                        "the run: evaluation_node mixes get_clock().now() for "
                        "sampling with message header stamps for latency/realtime "
                        "factor, so wall vs sim time would corrupt both"),
    ]

    reference_costmap = Node(
        package="risk_perception",
        executable="predictive_risk_costmap",
        # Distinct node name -- this is a SECOND instance of the same
        # executable, running alongside whatever instance
        # risk_perception.launch.py already started; ROS 2 requires unique
        # node names on the graph.
        name="predictive_risk_costmap_node_reference",
        output="screen",
        parameters=[params, {
            # Load the same base config, then force every mechanism on
            # regardless of what the ablation under test disabled.
            "costmap_topic": "/risk_costmap_reference",
            "marker_topic": "/risk_perception/prediction_markers_reference",
            "use_class_consequence": True,
            "enable_relative_motion": True,
            "use_motion_mixture": True,
            "spatial_prior_weight": LaunchConfiguration("reference_spatial_prior_weight"),
            "flow_blend_weight": LaunchConfiguration("reference_flow_blend_weight"),
            # Same clock as the run being scored -- this node stamps the grids
            # evaluation_node samples and measures latency against.
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    evaluation = Node(
        package="risk_perception",
        executable="evaluation_node",
        name="evaluation_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_evaluation")),
        # params carries the evaluation_node block from risk_perception.yaml;
        # risk_topic is pinned again here so this launch file is self-documenting
        # about which grid gets scored.
        parameters=[params, {
            "risk_topic": "/risk_costmap_reference",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription(args + [reference_costmap, evaluation])
