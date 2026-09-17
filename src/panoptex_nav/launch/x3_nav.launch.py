#!/usr/bin/env python3
"""
x3_nav.launch.py -- WP3.4/WP-C: the two-arm nav2 study launch for the sim X3.

Perception source, selected by `perception` (independent of `learn_lanes`
below -- both modes below use whichever `perception` picks):

  perception:=panoptex (default) -- the full sim camera chain (overhead
  cams + GDINO/SAM2 + object_tracker), as every study run to date.
  perception:=oracle -- skip cameras/GDINO/SAM2/object_tracker entirely and
  publish `/risk_perception/world_objects` straight off `/gt_tf` (Isaac's
  ground truth) via gt_tracks (WP-C, panoptex_nav/gt_tracks_node.py) --
  for bench-validating prediction/planning (the risk costmap, the DWB/MPPI
  critics, mission_supervisor) without perception noise/dropouts in the
  loop. In this mode: enable_global_cams/enable_robot_cam/enable_world_model
  are forced false (no cameras, no GDINO/SAM2, no object_tracker) and
  enable_spatial_prior is forced false regardless of `learn_lanes` (nothing
  populates S without the camera-derived tracker in the loop). The
  costmap/predictive-costmap nodes stay on in both modes -- gt_tracks feeds
  them exactly the topic object_tracker_node would.

Three modes, selected by `learn_lanes`:

  learn_lanes:=false (default) -- the study launch. Brings up, for one of the
  four study arms, against the same Isaac Sim warehouse scenario:

    baseline       pure lidar nav2, DWB
    panoptex       DWB + risk costmap layers + PredictedRisk DWB critic
    baseline_mppi  pure lidar nav2, MPPI (WP2)
    panoptex_mppi  MPPI + risk costmap layers + PredictedRiskMppiCritic on
                   /risk_stack (WP2 -- the arm the plan is actually about)

  The arm selects config/nav2_x3_<arm>.yaml; the risk_speed_governor and the
  mission_supervisor's yield behaviour are on for BOTH panoptex arms and off
  for both baselines. The steps:

    1. risk_perception's panoptex_sim.launch.py, with its own AMCL/map_server
       OFF (enable_localization:=false) -- nav2_bringup below owns
       localization instead. Started identically in BOTH arms: the overhead
       cameras + predictive risk costmap + spatial prior run regardless of
       arm, so the two arms carry the same compute load and only differ in
       what nav2 DOES with the risk signal (baseline: nothing, it isn't
       wired into the costmap or the controller; panoptex: costmap layers +
       DWB critic).
    2. nav2_bringup's bringup_launch.py (map_server, amcl, controller_server,
       planner_server, behavior_server, bt_navigator, waypoint_follower,
       velocity_smoother, both costmaps) with the arm's params file.
    3. risk_speed_governor (both panoptex arms) -- publishes nav2_msgs/SpeedLimit
       onto controller_server's speed_limit_topic.
    4. mission_supervisor (panoptex_nav), which drives the waypoint loop
       (x3_sim_waypoints.yaml) itself and internally delays `start_delay_sec`
       before its first goal (nav2's lifecycle needs time to activate).

  learn_lanes:=true -- lane warm-up mode (WP-C, see
  tools/warmup_lanes.sh): starts ONLY panoptex_sim.launch.py, forced
  enable_spatial_prior:=true regardless of the enable_spatial_prior arg --
  no nav2_bringup, no risk_speed_governor, no mission_supervisor. nav2_collision_
  monitor (WP5) is the one exception: it still starts (unless
  enable_collision_monitor:=false), so warm-up drives behind the same
  cmd_vel -> cmd_vel_safe safety chain a study run does. Intended to run
  against Carter patrol traffic with no X3 nav stack competing for the
  domain, so spatial_prior_node's S channel accumulates a clean read of the
  AMR lanes into spatial_prior_path.

Usage:
    ros2 launch panoptex_nav x3_nav.launch.py arm:=baseline
    ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex
    ros2 launch panoptex_nav x3_nav.launch.py arm:=baseline_mppi
    ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex_mppi
    ros2 launch panoptex_nav x3_nav.launch.py learn_lanes:=true \\
        spatial_prior_path:=/some/path/spatial_prior_sim.npz

Domain guard: this launch file is part of the risk-aware study and expects
ROS_DOMAIN_ID=44 (warehouse/env_study.sh) -- NOT 55 (env_sim.sh, general sim
work) or 77 (the real X3 on the LAN). Bench-testing the launch graph itself
on another domain: set PANOPTEX_NAV_ANY_DOMAIN=1.

Launch-file gotcha (see risk_perception/launch/panoptex_sim.launch.py's own
comment on this): IncludeLaunchDescription.execute() sets launch
configurations directly on the shared context with no push/pop scoping, so
an include's launch_arguments overwrite the same-named configuration for
the REST of this launch, not just inside that include. The OpaqueFunction
below (arm validation, domain guard, params_file resolution) therefore runs
FIRST, before any include, and every include below passes its arguments
explicitly rather than relying on ambient LaunchConfiguration values.
"""
import os

from ament_index_python.packages import (get_package_prefix,
                                         get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, SetLaunchConfiguration)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

DEFAULT_MAP = os.path.join(
    os.path.expanduser("~"), "workspace", "warehouse", "maps", "warehouse_x3_nav.yaml")

# The four study arms, each with its own config/nav2_x3_<arm>.yaml. The
# `*_mppi` pair (WP2) swaps DWB for nav2_mppi_controller::MPPIController; the
# `panoptex*` pair is the risk-aware side (risk costmap layers + the
# PredictedRisk critic + the speed governor + supervisor yielding).
ARMS = ("baseline", "panoptex", "baseline_mppi", "panoptex_mppi")

#: Arms that carry the Panoptex risk behaviour (governor + supervisor yield).
PANOPTEX_ARMS = ("panoptex", "panoptex_mppi")


def _resolve(context, *args, **kwargs):
    """Runs before every include (see module docstring). Validates `arm`,
    enforces the domain guard, and -- since `params_file`'s real default
    depends on `arm`, which isn't known until launch time -- resolves an
    unset params_file to config/nav2_x3_<arm>.yaml. An explicit
    params_file:= always wins over this derivation. Unaffected by
    learn_lanes -- params_file is still derived even when nav2 isn't
    actually started, since it's a cheap string and keeps this function's
    logic in one place.
    """
    arm = LaunchConfiguration("arm").perform(context)
    if arm not in ARMS:
        raise RuntimeError(
            "x3_nav.launch.py: arm must be one of {}, got {!r}".format(
                "|".join(ARMS), arm))

    domain = os.environ.get("ROS_DOMAIN_ID")
    escape_hatch = os.environ.get("PANOPTEX_NAV_ANY_DOMAIN") == "1"
    if domain != "44" and not escape_hatch:
        raise RuntimeError(
            "x3_nav.launch.py: ROS_DOMAIN_ID is {!r}, expected '44' (the "
            "risk-aware study domain). Source warehouse/env_study.sh in "
            "every terminal for this launch, or set "
            "PANOPTEX_NAV_ANY_DOMAIN=1 to bypass this guard for a bench "
            "test on another domain.".format(domain))

    actions = []
    params_file = LaunchConfiguration("params_file").perform(context)
    if not params_file:
        share = get_package_share_directory("panoptex_nav")
        params_file = os.path.join(share, "config", "nav2_x3_{}.yaml".format(arm))
        actions.append(SetLaunchConfiguration("params_file", params_file))
        derived = " (derived from arm)"
    else:
        derived = " (explicit override)"

    actions.append(LogInfo(
        msg="x3_nav.launch.py: arm={} params_file={}{} learn_lanes={}".format(
            arm, params_file, derived,
            LaunchConfiguration("learn_lanes").perform(context))))
    return actions


def generate_launch_description() -> LaunchDescription:
    risk_perception_share = get_package_share_directory("risk_perception")
    nav2_bringup_share = get_package_share_directory("nav2_bringup")
    panoptex_nav_share = get_package_share_directory("panoptex_nav")

    # mission_supervisor's own params file -- optional (agent-B deliverable;
    # may not exist yet, e.g. at launch-test time before it lands). Checked
    # once here, at launch-description-generation time, same as any other
    # static default in this file (see DEFAULT_MAP above) -- this is a
    # plain os.path.exists on a share-dir path, not something that depends
    # on a LaunchConfiguration, so it needs no OpaqueFunction.
    mission_supervisor_yaml = os.path.join(
        panoptex_nav_share, "config", "mission_supervisor.yaml")
    mission_supervisor_params = (
        [mission_supervisor_yaml] if os.path.isfile(mission_supervisor_yaml) else [])

    declare_args = [
        DeclareLaunchArgument(
            "arm", default_value="panoptex",
            choices=list(ARMS),
            description="Study arm: 'baseline'/'panoptex' (DWB) or "
                        "'baseline_mppi'/'panoptex_mppi' (MPPI, WP2). The "
                        "'panoptex*' arms are the risk-aware ones."),
        DeclareLaunchArgument(
            "map", default_value=DEFAULT_MAP,
            description="Warehouse map yaml, passed to both panoptex_sim.launch.py "
                        "(map_yaml) and nav2_bringup (map)."),
        DeclareLaunchArgument(
            "waypoints_file",
            default_value=os.path.join(panoptex_nav_share, "config", "x3_sim_waypoints.yaml"),
            description="Waypoint loop for mission_supervisor."),
        DeclareLaunchArgument("loop", default_value="true",
            description="Restart the waypoint loop after the last waypoint."),
        DeclareLaunchArgument("laps", default_value="0",
            description="mission_supervisor laps parameter -- 0 means unbounded "
                        "(subject to `loop`); a positive count stops the mission "
                        "after that many laps regardless of `loop`."),
        DeclareLaunchArgument("runner_delay", default_value="12.0",
            description="Seconds mission_supervisor waits (its own "
                        "start_delay_sec parameter) before its first goal, so "
                        "nav2's lifecycle has time to activate first."),
        DeclareLaunchArgument("use_sim_time", default_value="true",
            description="Use Isaac Sim's /clock."),
        # WP-C: default flipped true<-false. Both arms now carry the same
        # spatial-prior compute load (this file's own rationale for why
        # panoptex_sim.launch.py is started identically in both arms), and
        # sim traffic no longer risks contaminating the lab statistic --
        # spatial_prior_path below always overrides persist_path to a
        # sim-only file (see risk_perception/launch/panoptex_sim.launch.py).
        # The panoptex arm needs S populated for lane_layer to do anything
        # at all; keeping it on for baseline too is what keeps the two
        # arms' compute identical.
        DeclareLaunchArgument("enable_spatial_prior", default_value="true",
            description="Forwarded to panoptex_sim.launch.py. Forced 'true' "
                        "regardless of this value when learn_lanes:=true."),
        DeclareLaunchArgument(
            "spatial_prior_path",
            default_value=os.path.join(
                os.path.expanduser("~"), ".panoptex", "spatial_prior_sim.npz"),
            description="Forwarded to panoptex_sim.launch.py's spatial_prior_path "
                        "arg -- spatial_prior_node's persist_path override for "
                        "sim (WP-C); never the lab's ~/.panoptex/spatial_prior.npz."),
        DeclareLaunchArgument("enable_rviz", default_value="false",
            description="Forwarded to panoptex_sim.launch.py."),
        DeclareLaunchArgument(
            "params_file", default_value="",
            description="Nav2 params file. Empty (default) derives "
                        "config/nav2_x3_<arm>.yaml at launch time; set explicitly "
                        "to override."),
        DeclareLaunchArgument(
            "learn_lanes", default_value="false",
            description="Lane warm-up mode (WP-C, see tools/warmup_lanes.sh): "
                        "when 'true', start ONLY panoptex_sim.launch.py (with "
                        "enable_spatial_prior forced 'true') -- no nav2_bringup, "
                        "no risk_speed_governor, no mission_supervisor."),
        DeclareLaunchArgument(
            "enable_collision_monitor", default_value="true",
            description="WP5: spawn nav2_collision_monitor (+ its own "
                        "lifecycle_manager_collision) reading config/"
                        "collision_monitor_x3.yaml, in ALL FOUR arms and in "
                        "BOTH learn_lanes modes -- the warm-up drive should use "
                        "the same cmd_vel -> cmd_vel_safe chain a study run "
                        "does. The sim base must be started with "
                        "X3_CMD_VEL_TOPIC=cmd_vel_safe for this to do anything "
                        "(see results/wiring_check/run_arm.sh); set 'false' "
                        "together with PANOPTEX_NO_CM=1 (which leaves the base "
                        "on plain cmd_vel) to bench a run with the monitor out "
                        "of the loop entirely."),
        DeclareLaunchArgument(
            "perception", default_value="panoptex",
            choices=["panoptex", "oracle"],
            description="'panoptex' (default): the sim's camera perception "
                        "chain (overhead cams + GDINO/SAM2 + object_tracker), "
                        "as every study run to date. 'oracle': skip perception "
                        "entirely and publish /risk_perception/world_objects "
                        "straight off /gt_tf via gt_tracks (WP-C) -- for "
                        "testing prediction/planning without perception in "
                        "the loop. See module docstring."),
        DeclareLaunchArgument(
            "yield_enabled",
            # Same PANOPTEX_ARMS check the mission_supervisor Node below used
            # to compute inline -- now an explicit arg (still defaulting to
            # the same value) so a bench/unit run (results/avoidance/
            # run_unit.sh) can force it off regardless of arm, e.g. to
            # exercise the bare nav2 + risk-costmap/critic path without
            # mission_supervisor's corridor-yield state machine.
            default_value=PythonExpression(
                ["'", LaunchConfiguration("arm"), "' in ", str(PANOPTEX_ARMS)]),
            description="mission_supervisor's corridor-yield layer. Defaults "
                        "to True for the panoptex/panoptex_mppi arms and "
                        "False for the baselines; set explicitly to override "
                        "(e.g. yield_enabled:=false on a panoptex arm)."),
    ]

    resolve_first = OpaqueFunction(function=_resolve)

    learn_lanes = LaunchConfiguration("learn_lanes")
    not_learn_lanes = UnlessCondition(learn_lanes)
    perception = LaunchConfiguration("perception")
    is_oracle = PythonExpression(["'", perception, "' == 'oracle'"])

    # enable_spatial_prior forwarded to panoptex_sim.launch.py: forced
    # "false" when perception:=oracle (nothing populates S without the
    # camera-derived tracker in the loop -- see module docstring), else
    # "true" when learn_lanes:=true (the whole point of that mode),
    # otherwise this file's own enable_spatial_prior arg. oracle wins over
    # learn_lanes since the two are not meant to be combined.
    effective_enable_spatial_prior = PythonExpression([
        "'false' if '", perception, "' == 'oracle' else ('true' if '",
        learn_lanes, "' == 'true' else '",
        LaunchConfiguration("enable_spatial_prior"), "')"])

    # perception:=oracle forwards enable_global_cams/enable_robot_cam/
    # enable_world_model:=false to panoptex_sim.launch.py -- no overhead
    # cams, no GDINO/SAM2, no object_tracker_node (see module docstring).
    # perception:=panoptex (default) leaves them at panoptex_sim.launch.py's
    # own defaults (all true) by forwarding "true" explicitly, same effect
    # as omitting them, but keeps both branches of this expression visible
    # at the call site below rather than only conditionally present.
    effective_enable_global_cams = PythonExpression([
        "'false' if '", perception, "' == 'oracle' else 'true'"])
    effective_enable_robot_cam = PythonExpression([
        "'false' if '", perception, "' == 'oracle' else 'true'"])
    effective_enable_world_model = PythonExpression([
        "'false' if '", perception, "' == 'oracle' else 'true'"])

    # 1. Overhead cams + world model + predictive risk costmap + spatial
    # prior. Identical in both arms, and started in BOTH learn_lanes modes
    # (learn_lanes needs exactly this and nothing else) -- localization is
    # OFF here (nav2_bringup owns it below, only when not learn_lanes;
    # running AMCL/map_server in both places double-publishes map->odom).
    # The camera-chain/spatial-prior enables above additionally depend on
    # `perception` (oracle mode replaces the camera chain with gt_tracks,
    # started below).
    risk_perception_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(risk_perception_share, "launch", "panoptex_sim.launch.py")),
        launch_arguments={
            "enable_localization": "false",
            "enable_global_cams": effective_enable_global_cams,
            "enable_robot_cam": effective_enable_robot_cam,
            "enable_world_model": effective_enable_world_model,
            "enable_predictive_costmap": "true",
            "enable_costmap": "true",
            "enable_spatial_prior": effective_enable_spatial_prior,
            "spatial_prior_path": LaunchConfiguration("spatial_prior_path"),
            # Freeze the lane prior during study runs (learn_lanes:=false):
            # learning stays on only in warm-up mode. See panoptex_sim.launch.py.
            # Irrelevant when effective_enable_spatial_prior is "false"
            # (perception:=oracle) -- spatial_prior_node doesn't spawn at
            # all in that case, so these two are simply unused.
            "spatial_prior_learn_rate": PythonExpression([
                "'0.20' if '", LaunchConfiguration("learn_lanes"), "' == 'true' else '0.0'"]),
            "spatial_prior_autosave_sec": PythonExpression([
                "'60.0' if '", LaunchConfiguration("learn_lanes"), "' == 'true' else '1.0e9'"]),
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "map_yaml": LaunchConfiguration("map"),
        }.items(),
    )

    # 1b. WP-C oracle perception: ground-truth Detection3DArray tracks off
    # /gt_tf, publishing onto the SAME topic object_tracker_node would
    # (/risk_perception/world_objects) -- see gt_tracks_node.py's module
    # docstring for the packed class_id convention and the gt_centre_offset_m
    # derivation. Only spawned when perception:=oracle; started in BOTH
    # learn_lanes modes like risk_perception_sim above (harmless either way
    # since learn_lanes+oracle isn't a combination this study uses).
    # `robots` is the two Carters -- both patrol in every scenario this
    # launch file is used against (the full study loop and run_unit.sh's
    # shorter carter_shuttle.yaml).
    gt_tracks_node = Node(
        package="panoptex_nav",
        executable="gt_tracks",
        name="gt_tracks_node",
        output="screen",
        parameters=[{
            "robots": ["carter1", "carter2"],
            "output_topic": "/risk_perception/world_objects",
            "frame_id": "map",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
        condition=IfCondition(is_oracle),
    )

    # 2. nav2 bringup: map_server, amcl, controller_server, planner_server,
    # behavior_server, bt_navigator, waypoint_follower, velocity_smoother,
    # both costmaps -- all from the arm's params file. Skipped in
    # learn_lanes mode (see module docstring).
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, "launch", "bringup_launch.py")),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "params_file": LaunchConfiguration("params_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": "true",
        }.items(),
        condition=not_learn_lanes,
    )

    # 3. Speed governor -- both panoptex arms, and only outside learn_lanes
    # mode. (Executable may not exist yet if WP3.2 hasn't landed; the node
    # is left in regardless -- it simply fails to spawn until it does.)
    # 2026-09-09: config/risk_speed_governor.yaml was never passed here, so every
    # run to date used the node's code defaults (robot_closing 30 %, static_slow
    # 50 %) while README/yaml claimed the 60 % retune. Prepend the yaml (same
    # pattern as mission_supervisor below) so the file actually governs.
    governor_yaml = os.path.join(panoptex_nav_share, "config", "risk_speed_governor.yaml")
    risk_speed_governor = Node(
        package="panoptex_nav",
        executable="risk_speed_governor",
        name="risk_speed_governor",
        output="screen",
        parameters=([governor_yaml] if os.path.isfile(governor_yaml) else [])
        + [{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration("arm"), "' in ", str(PANOPTEX_ARMS), " and '",
             learn_lanes, "' == 'false'"])),
    )

    # 4. mission_supervisor (agent-B deliverable; executable/package may not
    # exist yet). Replaces the old yahboomcar_nav waypoint_follower_launch.py
    # TimerAction include: mission_supervisor owns the delay itself
    # (start_delay_sec) instead of the launch file delaying the include.
    # mission_supervisor_params (mission_supervisor.yaml) is prepended when
    # that file exists on disk at generate-time; the explicit dict after it
    # always wins on any overlapping key (later entries in a Node
    # `parameters` list override earlier ones). yield_enabled is normally a
    # panoptex-only behaviour (yielding to AMR traffic needs lane_layer,
    # which only the panoptex arms' costmaps carry) -- the `yield_enabled`
    # arg declared above defaults to exactly that same PANOPTEX_ARMS check,
    # but can be overridden explicitly (e.g. run_unit.sh forces it false
    # regardless of arm). Skipped entirely in learn_lanes mode along with
    # nav2/governor.
    #
    # UNLIKE risk_speed_governor above, a genuinely missing executable is
    # NOT safe to "leave in regardless": launch_ros.Node.execute() raises
    # synchronously when it can't find the executable on the libexec
    # directory, and that exception is NOT contained to this one action --
    # it aborts the WHOLE LaunchService, tearing down every node already
    # started by risk_perception_sim/nav2_bringup within about a second
    # (verified empirically, 2026-09-09 bench: the entire graph -- overhead
    # cams, costmaps, nav2 -- came up and was then killed almost immediately
    # because of this one missing executable). So this checks for the
    # installed executable at generate-time (same os.path.isfile pattern as
    # mission_supervisor_params above) and only builds the Node action if it
    # is actually there; otherwise a LogInfo stands in, and the rest of the
    # graph (nav2, risk_speed_governor, the overhead cams) still comes up
    # and stays up.
    mission_supervisor_exe = os.path.join(
        get_package_prefix("panoptex_nav"), "lib", "panoptex_nav", "mission_supervisor")
    if os.path.isfile(mission_supervisor_exe):
        mission_supervisor_actions = [Node(
            package="panoptex_nav",
            executable="mission_supervisor",
            name="mission_supervisor",
            output="screen",
            parameters=mission_supervisor_params + [{
                "waypoints_file": LaunchConfiguration("waypoints_file"),
                "loop": LaunchConfiguration("loop"),
                "laps": LaunchConfiguration("laps"),
                "start_index": 0,
                "yield_enabled": LaunchConfiguration("yield_enabled"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "start_delay_sec": LaunchConfiguration("runner_delay"),
            }],
            condition=not_learn_lanes,
        )]
    else:
        mission_supervisor_actions = [LogInfo(
            msg="x3_nav.launch.py: panoptex_nav/mission_supervisor executable "
                "not found (not built yet) -- skipping the mission_supervisor "
                "node so the rest of the graph still comes up.",
            condition=not_learn_lanes)]

    # 5. nav2_collision_monitor (WP5) -- the last-resort reactive layer,
    # AFTER velocity_smoother in the command chain: controller_server's
    # cmd_vel_nav -> velocity_smoother -> cmd_vel -> collision_monitor ->
    # cmd_vel_safe -> base. All four arms, identically (sensing is not a
    # Panoptex-arm difference, same rationale as the depth costmap source --
    # see the nav2_x3_*.yaml comments), and in BOTH learn_lanes modes (no
    # not_learn_lanes condition below) so lane warm-up drives behind the same
    # safety chain a study run does. Its own lifecycle_manager_collision is
    # deliberately separate from nav2_bringup's internal lifecycle manager
    # (different node_names list) so this node's lifecycle is independent of
    # the rest of the nav2 stack coming up cleanly.
    enable_collision_monitor = LaunchConfiguration("enable_collision_monitor")
    collision_monitor_params_file = os.path.join(
        panoptex_nav_share, "config", "collision_monitor_x3.yaml")
    collision_monitor_node = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[
            collision_monitor_params_file,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        condition=IfCondition(enable_collision_monitor),
    )
    lifecycle_manager_collision_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_collision",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": True,
            "node_names": ["collision_monitor"],
        }],
        condition=IfCondition(enable_collision_monitor),
    )

    return LaunchDescription(declare_args + [
        resolve_first,
        risk_perception_sim,
        gt_tracks_node,
        nav2_bringup,
        risk_speed_governor,
        collision_monitor_node,
        lifecycle_manager_collision_node,
    ] + mission_supervisor_actions)
