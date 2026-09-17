#!/usr/bin/env python3
"""
panoptex_sim.launch.py  --  everything the PC runs, in Isaac Sim, in one command.

The sim twin of panoptex_pc.launch.py: localization + the three overhead
camera chains + the robot RGB-D chain + the fused world model + costmaps +
RViz, all toggleable, all on sim time.

Bring-up order (every terminal has sourced
~/workspace/warehouse/env_sim.sh, domain 55):

    A  Isaac: open Baseline_scenario_metric.usd, Script Editor -> run
       x3_sim/x3_base_controller.py, Play
       (headless equivalent: ./x3_sim/isaac.sh x3_sim/run_headless.py)
    B  ros2 launch yahboomcar_nav x3_sim_bringup_launch.py
    C  (optional) ros2 run yahboomcar_ctrl yahboom_keyboard
    D  (optional) ros2 launch warehouse/launch/carters_patrol.launch.py \\
           map:=.../warehouse_gt_carter.yaml
    E  conda activate panoptex && \\
           ros2 launch risk_perception panoptex_sim.launch.py

Brings up, in dependency order:

  localization    robot_base.launch.py with enable_base:=false,
                  enable_slam:=false -- map_server + amcl against the
                  warehouse's SAVED map (map_yaml), seeded from
                  config/amcl_sim.yaml's baked initial pose (the X3's spawn
                  point in the stage, which is also the map origin). Never
                  slam_toolbox: a fresh SLAM session would move the map
                  origin and invalidate the overhead cameras' extrinsics,
                  same rule as the lab (robot_base.launch.py's docstring).
  overhead cams   sim_global_cams.launch.py -- one static TF (ground truth,
                  no calibrator) + gdino + sam2 + projector chain per camera
                  in cams_config (global_cams_sim.yaml), all publishing onto
                  the shared /risk_perception/detections_3d_map topic.
  world model     risk_perception.launch.py with enable_camera:=false (the
                  Astra/RealSense driver isn't relevant here -- Isaac
                  publishes the robot's RGB-D topics itself),
                  enable_robot_cam_chain:=enable_robot_cam. Its own
                  costmap/predictive-costmap/spatial-prior nodes stay OFF
                  (enable_costmap:=false etc.) -- this file spawns its own
                  copies of those three below (plus the WP4 coverage mask),
                  geometry-overridden from cams_config's `costmap:` block
                  (the warehouse floor, not risk_perception.yaml's 10x10 m
                  lab default). Same split as bench_sim_multicam.launch.py,
                  extended from one grid node to all four (see _grid_nodes
                  below).
  navigation      yahboomcar_nav's navigation_dwa_launch.py with
                  config/nav2_sim.yaml, OFF by default (enable_nav2). That
                  params file is dwa_nav_params.yaml carrying THIS repo's
                  nav2_risk_layer::RiskLayer in global_costmap.plugins, on
                  sim time, with config/amcl_sim.yaml's amcl block.
                  risk_layer_enabled/risk_topic are the two ablation knobs
                  (see _nav2 below) -- this is what tools/run_trials.py
                  drives goals into, and what publishes the /plan that
                  evaluation_node.py counts replans on.

No global_cam_initialpose, no localizer, no bridge, no apriltag -- the
overhead cameras' poses are ground truth exported off the USD stage (static
TF), not solved from floor tags.

If the X3 was driven before this launches, config/amcl_sim.yaml's baked
initial pose is stale -- use RViz's "2D Pose Estimate" once instead of
restarting anything.

NAV2 NEEDS THIS WORKSPACE'S OVERLAY SOURCED. Costmap plugins load
in-process, so install/setup.bash from this repo must be on AMENT_PREFIX_PATH
in the launching terminal or pluginlib cannot find nav2_risk_layer::RiskLayer,
global_costmap never activates, planner_server never activates, and the whole
Nav2 lifecycle stalls with no error that looks like a costmap problem.
~/.bashrc already sources it; warehouse/env_sim.sh does NOT add it (ROS +
yahboomcar_ws + carter_ws only), it just inherits whatever is already there.

Run:
  ros2 launch risk_perception panoptex_sim.launch.py
Same, with risk-aware Nav2 (send goals from RViz's 2D Goal Pose):
  ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true
The three ablation arms run_trials.py scores (--condition-prefix):
  ... enable_nav2:=true risk_layer_enabled:=false          # baseline
  ... enable_nav2:=true risk_topic:=/risk_costmap          # reactive
  ... enable_nav2:=true risk_topic:=/risk_costmap_predictive   # predictive
Bench-test localization/costmap plumbing with no GPU models loaded:
  ros2 launch risk_perception panoptex_sim.launch.py \\
      enable_global_cams:=false enable_robot_cam:=false enable_rviz:=false
Robot-cam + overhead cams, no localization (e.g. VRAM headroom check):
  ros2 launch risk_perception panoptex_sim.launch.py enable_localization:=false
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml

HOME = os.path.expanduser("~")
# ~/workspace vs ~/workspaces differs per machine -- see risk_perception.launch.py
_WS_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces"), os.path.join(HOME, "workspace"))
     if os.path.isdir(p)),
    os.path.join(HOME, "workspace"))
# The warehouse checkout is NOT always a sibling of this repo: it sits under
# ~/Digital-Twin-Project on at least one machine, where the old
# _WS_ROOT/warehouse guess pointed at a directory that does not exist and
# map_server died with "Failed to load map yaml file" -- which with
# enable_nav2:=true takes AMCL and the whole Nav2 localization half with it.
# Probe for the map itself instead of assuming a layout (same idiom as
# panoptex_pc.launch.py's DEFAULT_MAP). Pass map_yaml:= to override.
_MAP_REL = os.path.join("maps", "warehouse_x3_nav.yaml")
DEFAULT_MAP_YAML = next(
    (os.path.join(w, _MAP_REL) for w in (
        os.path.join(_WS_ROOT, "warehouse"),
        os.path.join(HOME, "workspaces", "warehouse"),
        os.path.join(HOME, "workspace", "warehouse"),
        os.path.join(HOME, "Digital-Twin-Project", "warehouse"),
    ) if os.path.isfile(os.path.join(w, _MAP_REL))),
    os.path.join(_WS_ROOT, "warehouse", _MAP_REL))
# Same src-tree-first resolution as sim_global_cams.launch.py /
# bench_sim_multicam.launch.py: edit the camera list or the costmap: block
# and relaunch, no rebuild needed.
REPO_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces", "Panoptex"),
                 os.path.join(HOME, "workspace", "Panoptex")) if os.path.isdir(p)),
    os.path.join(HOME, "workspace", "Panoptex"))
SRC_CONFIG_DIR = os.path.join(REPO_ROOT, "src", "risk_perception", "config")


def _config(name):
    """Src-tree-first path to a config/ file, same resolution as cams_config
    below: edit the yaml and relaunch, no rebuild needed."""
    src = os.path.join(SRC_CONFIG_DIR, name)
    if os.path.isfile(src):
        return src
    return os.path.join(
        get_package_share_directory("risk_perception"), "config", name)


# The four grid nodes' geometry parameters -- identical set on
# risk_costmap_node.py, predictive_risk_costmap_node.py,
# spatial_prior_node.py and coverage_mask_node.py (see risk_perception.yaml),
# so one allow-list covers all four.
_GEOMETRY_KEYS = {"resolution", "width_m", "height_m", "origin_x", "origin_y"}

# --- Prior / mechanism ablation knobs (2026-09-13) --------------------------
# One entry per INDEPENDENTLY switchable mechanism of the risk representation,
# each mapped onto a predictive_risk_costmap_node parameter. Exposed as launch
# args so tools/run_ablation_repeats.sh can run a leave-one-out arm without
# hand-editing risk_perception.yaml between runs -- the same reasoning that
# already made risk_layer_enabled/risk_topic launch args rather than yaml
# edits, and the same failure it avoids (a hand-edited yaml that silently
# stays edited into the NEXT arm's runs).
#
# Default "" means "do not override" -- the yaml value stands. Every arm is
# therefore expressible as a delta from the shipped configuration, and an arm
# that passes nothing is exactly the full system.
#
# NOTE these all apply to the PREDICTIVE node only. risk_costmap_node (the
# reactive arm) consumes none of them -- no rollout, no encounter term, no
# relation bonus, no flow -- so crossing these knobs with the reactive or
# baseline arm would produce identical runs under different labels.
_ABLATION_ARGS = (
    # (launch arg / node param, caster, help)
    ("use_motion_mixture", "bool",
     "Behavioral prior: false forces pmot=0, so every track paints its "
     "stationary hypothesis only and the two-hypothesis mixture collapses."),
    ("semantic_modifier_enabled", "bool",
     "Semantic prior magnitude: false discards the class consequence table "
     "and every track paints the class-agnostic value."),
    ("use_class_consequence", "bool",
     "false makes every class weigh 1.0 before the agnostic override."),
    ("use_relation_bonus", "bool",
     "Relation prior: false zeroes relbonus at the point it enters severity, "
     "leaving the evidence pipeline and its CSV column intact."),
    ("enable_relative_motion", "bool",
     "Encounter term: false pins enc=1.0, reproducing the un-amplified grid."),
    ("two_hypothesis_cpa", "bool",
     "false scores both hypotheses with ONE CPA reading (historical)."),
    ("stack_moving_pmov_min", "float",
     "pmov floor on the moving hypothesis; 0.0 disables the gate."),
    ("spatial_prior_weight", "float",
     "Spatial-Flow additive floor weight; 0.0 disables the floor."),
    ("flow_blend_weight", "float",
     "Spatial-Flow rollout blend weight; 0.0 disables the blend."),
)


def _ablation_overrides(context):
    """Resolve _ABLATION_ARGS into a predictive-node parameter dict, skipping
    any argument left at its "" default. Returns {} for a full-system run."""
    resolved = {}
    for name, kind, _help in _ABLATION_ARGS:
        raw = LaunchConfiguration(name).perform(context).strip()
        if not raw:
            continue
        if kind == "bool":
            low = raw.lower()
            if low not in ("true", "false"):
                raise RuntimeError(
                    f"{name}:={raw} is not a bool -- use true or false")
            resolved[name] = (low == "true")
        else:
            resolved[name] = float(raw)
    return resolved


def _grid_nodes(context, *args, **kwargs):
    """Spawn risk_costmap_node / predictive_risk_costmap_node /
    spatial_prior_node / coverage_mask_node here (not via
    risk_perception.launch.py, which keeps its own copies of the first
    three OFF -- enable_costmap:=false etc. below) so their grid geometry
    can be overridden with the warehouse floor's extent from cams_config's
    `costmap:` block. Hand-generalises
    bench_sim_multicam.launch.py's `_costmap_node()` OpaqueFunction (that
    file overrides only risk_costmap_node; this one applies the same
    overrides to all four grid nodes, since the predictive costmap, the
    spatial prior and the WP4 coverage mask are also on by default in the sim
    stack). Keep the two in sync if the geometry key set ever changes.
    """
    config_path = LaunchConfiguration("cams_config").perform(context)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geometry = cfg.get("costmap") or {}
    overrides = {k: float(v) for k, v in geometry.items() if k in _GEOMETRY_KEYS}

    # Already inside an OpaqueFunction with `context` -- perform()+str-compare
    # rather than ParameterValue, same reasoning as sim_global_cams.launch.py.
    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        == "true"
    )
    overrides["use_sim_time"] = use_sim_time

    params = os.path.join(
        get_package_share_directory("risk_perception"),
        "config", "risk_perception.yaml")

    # Research logging (from user-a/sandbox) -- kept OUT of the shared
    # `overrides` dict: coverage_mask does not declare debug_log_dir and
    # would reject an undeclared param at startup. risk_costmap now does
    # too (2026-09 latency instrumentation, see debug_log.open_latency_csv)
    # so it's spread into its own parameters below like predictive_costmap
    # already was.
    log_dir = LaunchConfiguration("prior_log_dir").perform(context)
    log_extra = {"debug_log_dir": log_dir} if log_dir else {}

    # WP-C: spatial_prior_node's own persist_path override for sim, so sim
    # traffic (Carter patrol warm-up runs, learn_lanes:=true bench sessions
    # -- see panoptex_nav/tools/warmup_lanes.sh) accumulates into its own
    # file rather than contaminating the lab's ~/.panoptex/spatial_prior.npz
    # (risk_perception.yaml's spatial_prior_node.persist_path default).
    # os.path.expanduser here, at launch time, since Node parameter dicts
    # are not shell-expanded the way a bash arg would be.
    spatial_prior_path = os.path.expanduser(
        LaunchConfiguration("spatial_prior_path").perform(context))
    spatial_prior_overrides = dict(overrides)
    spatial_prior_overrides["persist_path"] = spatial_prior_path
    # Learning/autosave knobs (2026-09-09): study runs must NOT keep learning --
    # the X3's own tracks and detector junk saturated the lane prior in one
    # 700 s run (64 % of the map became "lane") and autosave overwrote the
    # warm-up file. learn_rate 0 freezes deposits; a huge autosave period
    # keeps the file untouched.
    spatial_prior_overrides["learn_rate"] = float(
        LaunchConfiguration("spatial_prior_learn_rate").perform(context))
    spatial_prior_overrides["autosave_period_sec"] = float(
        LaunchConfiguration("spatial_prior_autosave_sec").perform(context))
    spatial_prior_overrides.update(log_extra)

    costmap = Node(
        package="risk_perception",
        executable="risk_costmap",
        name="risk_costmap_node",
        output="screen",
        parameters=[params, {**overrides, **log_extra}],
        condition=IfCondition(LaunchConfiguration("enable_costmap")),
    )

    # Ablation overrides go LAST so they win over both the yaml and the
    # geometry/log overrides, and are applied to this node ONLY -- see
    # _ABLATION_ARGS' note on why crossing them with the reactive arm is
    # meaningless.
    predictive_costmap = Node(
        package="risk_perception",
        executable="predictive_risk_costmap",
        name="predictive_risk_costmap_node",
        output="screen",
        parameters=[params,
                    {**overrides, **log_extra, **_ablation_overrides(context)}],
        condition=IfCondition(LaunchConfiguration("enable_predictive_costmap")),
    )

    spatial_prior = Node(
        package="risk_perception",
        executable="spatial_prior",
        name="spatial_prior_node",
        output="screen",
        parameters=[params, spatial_prior_overrides],
        condition=IfCondition(LaunchConfiguration("enable_spatial_prior")),
    )

    # WP4 -- the analytic exposure map (/risk_perception/coverage). Spawned
    # HERE, with the other three, purely so it inherits the same `costmap:`
    # geometry override: a coverage grid on a different origin/resolution
    # than the spatial prior would make mission_supervisor's zone lookups
    # (which sample both by world coordinate) disagree about which cells an
    # approach zone even contains. Its own camera list is
    # risk_perception.yaml's coverage_mask_node block, hand-mirrored from
    # global_cams_sim.yaml's `cameras:` -- see that node's docstring.
    coverage_mask = Node(
        package="risk_perception",
        executable="coverage_mask",
        name="coverage_mask_node",
        output="screen",
        parameters=[params, overrides],
        condition=IfCondition(LaunchConfiguration("enable_coverage_mask")),
    )

    return [costmap, predictive_costmap, spatial_prior, coverage_mask]


def _nav2(context, *args, **kwargs):
    """Include yahboomcar_nav's navigation_dwa_launch.py against a copy of
    nav2_params_file with the two ablation knobs resolved into it:
    risk_layer.enabled <- risk_layer_enabled, risk_layer.topic <- risk_topic.
    Grafted from user-a/sandbox (2026-09-11 merge).

    Why the copy rather than a substitution: nav2_bringup rewrites the params
    file through RewrittenYaml, but its param_rewrites match on the LEAF key
    name only -- rewriting "topic" would also hit obstacle_layer's scan.topic
    (/scan) in both costmaps, and "enabled" would hit every layer in both.
    So the values we vary are resolved here instead, the same "template ->
    /tmp -> hand nav2 the resolved file" shape warehouse/launch/
    carters_nav.launch.py uses for its per-robot namespaces. use_sim_time and
    yaml_filename are left to nav2's own RewrittenYaml, which applies them on
    top of whatever this writes.
    """
    if LaunchConfiguration("enable_nav2").perform(context).strip().lower() != "true":
        return []

    params_file = LaunchConfiguration("nav2_params_file").perform(context)
    risk_topic = LaunchConfiguration("risk_topic").perform(context).strip()
    risk_enabled = (
        LaunchConfiguration("risk_layer_enabled").perform(context).strip().lower()
        == "true")

    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Fail loudly rather than silently launching a Nav2 with no risk layer:
    # a params file without this block is the lab/exploration copy, and the
    # whole point of enable_nav2 here is the layer.
    try:
        risk_layer = params["global_costmap"]["global_costmap"]["ros__parameters"]["risk_layer"]
    except (KeyError, TypeError):
        raise RuntimeError(
            f"{params_file} has no global_costmap.risk_layer block -- it is not "
            "a risk-layer params file. Use config/nav2_sim.yaml (or add the "
            "block, and 'risk_layer' to global_costmap.plugins).")
    risk_layer["topic"] = risk_topic
    risk_layer["enabled"] = risk_enabled

    # Scoring yardstick, never a planning input -- see nav2_sim.yaml's
    # risk_layer comment and evaluation_reference.launch.py's docstring.
    if risk_enabled and risk_topic == "/risk_costmap_reference":
        raise RuntimeError(
            "risk_topic:=/risk_costmap_reference would make every ablation arm "
            "plan against the same fixed full-system field, collapsing the "
            "study. That grid is evaluation_reference.launch.py's scoring "
            "yardstick; plan against /risk_costmap or /risk_costmap_predictive.")

    resolved = os.path.join(
        tempfile.mkdtemp(prefix="panoptex_nav2_"), "nav2_sim.yaml")
    with open(resolved, "w") as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)

    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("yahboomcar_nav"),
                         "launch", "navigation_dwa_launch.py")),
        launch_arguments={
            "map": LaunchConfiguration("map_yaml"),
            "params_file": resolved,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )]


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")

    args = [
        DeclareLaunchArgument("map_yaml", default_value=DEFAULT_MAP_YAML),
        DeclareLaunchArgument(
            "cams_config",
            default_value=(
                os.path.join(SRC_CONFIG_DIR, "global_cams_sim.yaml")
                if os.path.isfile(os.path.join(SRC_CONFIG_DIR, "global_cams_sim.yaml"))
                else os.path.join(share, "config", "global_cams_sim.yaml")),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("enable_localization", default_value="true"),
        DeclareLaunchArgument("enable_global_cams", default_value="true"),
        DeclareLaunchArgument("enable_robot_cam", default_value="true"),
        DeclareLaunchArgument("enable_world_model", default_value="true"),
        DeclareLaunchArgument("enable_costmap", default_value="true"),
        DeclareLaunchArgument("enable_predictive_costmap", default_value="true"),
        # Off by default here (x3_nav.launch.py's own enable_spatial_prior
        # arg turns it on for the two-arm study -- see that file). Even when
        # on, sim traffic never touches the lab's ~/.panoptex/spatial_prior.npz:
        # spatial_prior_path below overrides spatial_prior_node's
        # persist_path to a sim-only file (WP-C).
        DeclareLaunchArgument("enable_spatial_prior", default_value="false"),
        DeclareLaunchArgument(
            "spatial_prior_path",
            default_value=os.path.join(HOME, ".panoptex", "spatial_prior_sim.npz"),
            description="spatial_prior_node's persist_path for sim (WP-C) -- "
                        "expanded with os.path.expanduser at launch time and "
                        "applied ONLY to the spatial_prior_node instance "
                        "spawned by _grid_nodes below, so sim runs (Carter "
                        "patrol warm-up, learn_lanes:=true) never read from "
                        "or write to the lab's real "
                        "~/.panoptex/spatial_prior.npz."),
        DeclareLaunchArgument(
            "spatial_prior_learn_rate", default_value="0.20",
            description="spatial_prior_node learn_rate override (0.0 freezes learning "
                        "for study runs; the lane warm-up keeps the yaml value)."),
        DeclareLaunchArgument(
            "spatial_prior_autosave_sec", default_value="60.0",
            description="spatial_prior_node autosave_period_sec override (set huge "
                        "together with learn_rate 0 so a study run never rewrites the prior)."),
        # WP1 -- lidar dynamic-cluster detector (scan_cluster_detector_node,
        # config/risk_perception.yaml's scan_cluster_detector block).
        # Publishes onto the shared /risk_perception/detections_3d_map topic
        # ("lidar_cluster" class_id); object_tracker's label-agnostic pass
        # fuses it in. Depends only on /scan + /map (from `localization`
        # above), so it stays on even with the camera chains disabled for a
        # bench probe.
        DeclareLaunchArgument("enable_lidar_detector", default_value="true"),
        # WP4 -- analytic exposure map (coverage_mask_node), spawned inside
        # _grid_nodes above so it inherits the sim grid geometry. Cheap
        # (camera footprints are computed once and cached; the lidar
        # ray-cast is one vectorised numpy pass at 2 Hz) and read only by
        # mission_supervisor's crossing policy, which degrades to its
        # blind-crossing branch if this is off.
        DeclareLaunchArgument("enable_coverage_mask", default_value="true"),
        # --- Nav2 ablation arms (grafted from user-a/sandbox 2026-09-11) ---
        # Off by default, same as panoptex_pc.launch.py: send a goal only once
        # AMCL has converged and the fused overlay looks right, or the X3 plans
        # against phantom risk. Turning this ON also turns our own localization
        # OFF -- see run_localization below.
        DeclareLaunchArgument("enable_nav2", default_value="false"),
        DeclareLaunchArgument(
            "nav2_params_file", default_value=_config("nav2_sim.yaml")),
        # The two ablation knobs, resolved into the params file by _nav2().
        # risk_layer_enabled:=false is the BASELINE arm (stock Nav2, layer
        # loaded but writing nothing); risk_topic picks reactive vs predictive.
        # Kept as launch args so tools/run_trials.py's conditions never require
        # hand-editing yaml between runs.
        DeclareLaunchArgument("risk_layer_enabled", default_value="true"),
        DeclareLaunchArgument("risk_topic", default_value="/risk_costmap"),
        # Research logging: a DIRECTORY for the per-tick prior CSVs
        # (object_tracker / predictive_risk_costmap / spatial_prior each write
        # their own <node>_<UTC>.csv there). "" = off. Analyse with
        # tools/prior_report.py. tracker_log_unconfirmed also records tracks
        # below min_hits / min_confidence.
        # --- Sec. IV mechanism ablation knobs (see _ABLATION_ARGS) ---
        DeclareLaunchArgument(
            "use_motion_mixture", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Behavioral prior off -> stationary-only painting."),
        DeclareLaunchArgument(
            "semantic_modifier_enabled", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Semantic consequence off -> class-agnostic value only."),
        DeclareLaunchArgument(
            "use_class_consequence", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Class severity table off."),
        DeclareLaunchArgument(
            "use_relation_bonus", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Relation prior off (evidence still logged)."),
        DeclareLaunchArgument(
            "enable_relative_motion", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Encounter CPA/TTC term off."),
        DeclareLaunchArgument(
            "two_hypothesis_cpa", default_value="",
            description="Ablation (bool); \"\" keeps the yaml value. Single shared CPA reading for both hypotheses."),
        DeclareLaunchArgument(
            "stack_moving_pmov_min", default_value="",
            description="Ablation (float); \"\" keeps the yaml value. pmov gate on the moving hypothesis."),
        DeclareLaunchArgument(
            "spatial_prior_weight", default_value="",
            description="Ablation (float); \"\" keeps the yaml value. Spatial-Flow additive floor."),
        DeclareLaunchArgument(
            "flow_blend_weight", default_value="",
            description="Ablation (float); \"\" keeps the yaml value. Spatial-Flow rollout blend."),
        DeclareLaunchArgument("prior_log_dir", default_value=""),
        DeclareLaunchArgument("tracker_log_unconfirmed", default_value="false"),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(share, "rviz", "panoptex_sim.rviz"),
        ),
    ]

    # Nav2's bringup_launch.py ALWAYS includes localization_launch.py when
    # slam:=false (its default) -- map_server + amcl + a lifecycle manager
    # named, identically to robot_base.launch.py's, lifecycle_manager_
    # localization. Running ours alongside it means two AMCLs both
    # broadcasting map->odom and fighting over it. config/nav2_sim.yaml's
    # amcl block is config/amcl_sim.yaml's (Omni model, baked spawn pose),
    # so deferring to Nav2's copy costs nothing -- same trade, and same
    # comment, as panoptex_pc.launch.py. Grafted from user-a/sandbox.
    run_localization = PythonExpression([
        "'", LaunchConfiguration("enable_localization"), "' == 'true' and '",
        LaunchConfiguration("enable_nav2"), "' != 'true'"
    ])

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "robot_base.launch.py")),
        launch_arguments={
            "enable_base": "false",       # the base driver runs inside Isaac
            "enable_slam": "false",       # saved map only -- see module docstring
            "map_yaml": LaunchConfiguration("map_yaml"),
            "enable_amcl": "true",
            "amcl_params": os.path.join(share, "config", "amcl_sim.yaml"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
        condition=IfCondition(run_localization),
    )

    global_cams = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "sim_global_cams.launch.py")),
        launch_arguments={
            "cams_config": LaunchConfiguration("cams_config"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "prior_log_dir": LaunchConfiguration("prior_log_dir"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_global_cams")),
    )

    # enable_costmap/enable_predictive_costmap/enable_spatial_prior are all
    # false HERE regardless of this file's own args of the same name --
    # _grid_nodes() above spawns the geometry-overridden copies instead.
    # NOT conditioned on enable_robot_cam: this include also hosts
    # object_tracker and RViz, which are camera-agnostic (same reasoning as
    # panoptex_pc.launch.py's world_model include).
    world_model = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "risk_perception.launch.py")),
        launch_arguments={
            "enable_camera": "false",
            "enable_robot_cam_chain": LaunchConfiguration("enable_robot_cam"),
            "enable_world_model": LaunchConfiguration("enable_world_model"),
            "enable_costmap": "false",
            "enable_predictive_costmap": "false",
            "enable_spatial_prior": "false",
            "enable_rviz": LaunchConfiguration("enable_rviz"),
            "rviz_config": LaunchConfiguration("rviz_config"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "prior_log_dir": LaunchConfiguration("prior_log_dir"),
            "tracker_log_unconfirmed": LaunchConfiguration(
                "tracker_log_unconfirmed"),
        }.items(),
    )

    # WP1 -- lidar dynamic-cluster detector. Plain Node (not inside
    # _grid_nodes' OpaqueFunction): it does not need the costmap: block's
    # geometry override, only the static map + risk_perception.yaml's own
    # scan_cluster_detector block. use_sim_time is coerced the same way
    # risk_perception.launch.py does (ParameterValue(..., value_type=bool):
    # a bare LaunchConfiguration substitution is always a string, and rclpy
    # does not coerce "true"/"false" strings to bool).
    lidar_detector = Node(
        package="risk_perception",
        executable="scan_cluster_detector",
        name="scan_cluster_detector",
        output="screen",
        parameters=[
            os.path.join(share, "config", "risk_perception.yaml"),
            {"use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"), value_type=bool)},
        ],
        condition=IfCondition(LaunchConfiguration("enable_lidar_detector")),
    )

    # OpaqueFunction(_grid_nodes) MUST be visited before `world_model`:
    # IncludeLaunchDescription.execute() calls SetLaunchConfiguration()
    # directly on the shared launch context with no push/pop scoping (see
    # launch/actions/include_launch_description.py) -- launch_arguments
    # passed into an include are NOT scoped to that include, they overwrite
    # the launch configuration for the rest of THIS launch too. world_model
    # passes enable_costmap/enable_predictive_costmap/enable_spatial_prior
    # as "false" to risk_perception.launch.py (see above); if _grid_nodes ran
    # after it, LaunchConfiguration("enable_costmap") would already read
    # "false" there too and none of the three grid nodes would ever spawn
    # (found by the enable_global_cams:=false smoke test -- confirmed empty
    # `ros2 node list` for all three, no error printed anywhere).
    # OpaqueFunction(_nav2) goes LAST for the same reason _grid_nodes goes
    # first: an include's launch_arguments are written straight onto the
    # shared launch context, unscoped (see the note above).
    # navigation_dwa_launch.py and nav2_bringup between them set
    # map/params_file/slam/autostart/namespace/..., so nothing of ours may be
    # read after it runs.
    return LaunchDescription(
        args + [OpaqueFunction(function=_grid_nodes),
                localization, global_cams, world_model, lidar_detector,
                OpaqueFunction(function=_nav2)])
