#!/usr/bin/env python3
"""
sim_global_cams.launch.py

N Isaac-Sim overhead cameras -> N detection chains -> one fused topic.
Spawns, PER CAMERA listed in config/global_cams_sim.yaml (the same file
tools/isaac_sim/global_cam_isaac.py wires the sim from):

  static_transform_publisher      map -> <name>_optical_frame   (/tf_static,
                                  from config/<name>_extrinsic_sim.yaml --
                                  ground-truth pose exported off the USD prim)
  gdino_detector_<name>           /<name>/image_raw -> /<name>/detections_2d
  sam2_segmenter_<name>           -> /<name>/instance_mask
  global_cam_projector_<name>     -> /risk_perception/detections_3d_map

Every projector publishes onto the SAME output topic, so object_tracker fuses
all cameras for free -- the same one-topic contract that fuses the robot
RGB-D and overhead cams in the real stack. No bridge, no AprilTag, no
calibrator, no localizer: Isaac publishes images/CameraInfo itself, and the
extrinsics are ground truth (static TF, not a solve).

Not usually launched directly -- bench_sim_multicam.launch.py wraps this plus
the tracker/costmap/RViz half. Direct use:

  ros2 launch risk_perception sim_global_cams.launch.py
      [cams_config:=/path/to/global_cams_sim.yaml]

WHY AN OpaqueFunction: the camera list is data, so the launch description is
built in Python at launch time. ROS 2 matches yaml param blocks by node NAME,
which cannot work for generated names like gdino_detector_global_cam_1 --
instead the relevant blocks of risk_perception.yaml (gdino_detector_global,
sam2_segmenter, global_cam_projector) are parsed here with PyYAML and merged
into each node's inline parameters. Editing those blocks still works; adding
a per-camera block in risk_perception.yaml does nothing.

CONFIG AND EXTRINSICS COME FROM THE SRC TREE (repo config/, not the install
share) by default, matching where global_cam_isaac.py writes the extrinsics:
edit the camera list or re-export poses and relaunch, no rebuild needed. The
share copy is the fallback for machines without the checkout.
"""

import os

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

HOME = os.path.expanduser("~")
# see risk_perception.launch.py -- the checkout is at ~/workspace/Panoptex on
# some machines and ~/workspaces/Panoptex on others; probe rather than guess.
REPO_ROOT = next(
    (p for p in (os.path.join(HOME, "workspaces", "Panoptex"),
                 os.path.join(HOME, "workspace", "Panoptex")) if os.path.isdir(p)),
    os.path.join(HOME, "workspace", "Panoptex"))
SRC_CONFIG_DIR = os.path.join(REPO_ROOT, "src", "risk_perception", "config")
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX", os.path.join(HOME, "miniconda3", "envs", "panoptex"))
GDINO_CONFIG_DEFAULT = os.path.join(
    CONDA_PREFIX, "src", "groundingdino", "groundingdino", "config",
    "GroundingDINO_SwinT_OGC.py")
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")

# Isaac renders ~30-60 fps vs the Pi's ~10 -- stride 30 keeps the same
# inferences-per-second budget as the real chain's stride 6, per camera.
DEFAULT_SIM_STRIDE = 30


def _first_existing(*candidates):
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[-1]


def load_cam_specs(config_path):
    """Per-camera spec dicts with every optional field derived from `name`.
    Mirrors (by hand) _load_cam_specs in tools/isaac_sim/global_cam_isaac.py,
    which cannot be imported here -- it executes Isaac bootstrap at module
    scope. Keep the derivations in sync.

    Returns (cfg, specs): the caller (_setup below) also reads two optional
    top-level keys straight off `cfg` that have no per-camera shape --
    `prompt` (open-vocabulary string, applied to every gdino_detector_<name>
    instance, overriding risk_perception.yaml's gdino_detector_global block)
    and `projector_max_range_m` (applied as `max_range_m` to every
    global_cam_projector_<name> instance) -- see global_cams_sim.yaml's
    header comment for why these live at the top level, not per-camera."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cameras = cfg.get("cameras") or []
    if not cameras:
        raise RuntimeError(f"no cameras listed in {config_path}")

    specs, seen = [], set()
    for entry in cameras:
        name = entry.get("name")
        prim_path = entry.get("prim_path")
        if not name or not prim_path:
            raise RuntimeError(
                f"every cameras[] entry needs `name` and `prim_path` "
                f"(offending entry: {entry!r})")
        if name in seen:
            raise RuntimeError(f"duplicate camera name {name!r} in config")
        seen.add(name)
        specs.append({
            "name": name,
            "frame_id": entry.get("frame_id", f"{name}_optical_frame"),
            "image_topic": entry.get("image_topic", f"/{name}/image_raw"),
            "info_topic": entry.get("info_topic", f"/{name}/camera_info"),
            "extrinsic_yaml": os.path.expanduser(entry.get(
                "extrinsic_yaml",
                os.path.join(SRC_CONFIG_DIR, f"{name}_extrinsic_sim.yaml"))),
            "inference_stride": int(entry.get("inference_stride",
                                              DEFAULT_SIM_STRIDE)),
        })
    return cfg, specs


def _yaml_block(params_path, node_name):
    """One node's ros__parameters dict out of risk_perception.yaml."""
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    return dict((all_params.get(node_name) or {}).get("ros__parameters") or {})


def _load_extrinsic(spec, config_path):
    """Translation/rotation for the static TF, from the exported yaml."""
    path = spec["extrinsic_yaml"]
    if not os.path.isfile(path):
        raise RuntimeError(
            f"extrinsic for camera {spec['name']} not found: {path}\n"
            "Export the ground-truth poses off the USD stage first:\n"
            "  cd ~/isaacsim && ./python.sh "
            f"{REPO_ROOT}/tools/isaac_sim/global_cam_isaac.py --save\n"
            f"(cameras listed in {config_path})")
    with open(path) as f:
        ext = yaml.safe_load(f)
    child = str(ext.get("child_frame_id", ""))
    if child != spec["frame_id"]:
        # The Isaac graphs stamp CameraInfo with the config-derived frame_id;
        # a mismatched file would put the TF under a frame no projector looks
        # up. Stale export (camera renamed since?) -- refuse rather than warn.
        raise RuntimeError(
            f"{path}: child_frame_id {child!r} != expected "
            f"{spec['frame_id']!r} for camera {spec['name']} -- stale export? "
            "Re-run global_cam_isaac.py.")
    t, r = ext["translation"], ext["rotation"]
    return t, r


def _setup(context, *args, **kwargs):
    config_path = LaunchConfiguration("cams_config").perform(context)
    cfg, specs = load_cam_specs(config_path)

    share = get_package_share_directory("risk_perception")
    params_path = os.path.join(share, "config", "risk_perception.yaml")
    gdino_defaults = _yaml_block(params_path, "gdino_detector_global")
    sam2_defaults = _yaml_block(params_path, "sam2_segmenter")
    projector_defaults = _yaml_block(params_path, "global_cam_projector")

    gdino_config = LaunchConfiguration("gdino_config").perform(context)
    gdino_weights = LaunchConfiguration("gdino_weights").perform(context)
    sam2_checkpoint = LaunchConfiguration("sam2_checkpoint").perform(context)
    # Per-node compute-time breakdown (off unless set) -- see
    # gdino_detector_node.py's own comment on debug_log_dir. Forwarded here
    # from panoptex_sim.launch.py's prior_log_dir the same way
    # risk_perception.launch.py already does for the robot-cam chain.
    prior_log_dir = LaunchConfiguration("prior_log_dir").perform(context)

    # Already inside an OpaqueFunction with `context` -- perform() the
    # substitution to a plain string here rather than pull in
    # ParameterValue, and convert to a real bool ourselves (a bare "true"
    # string in a parameters dict is not a bool to rclpy).
    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        == "true"
    )

    # Top-level overrides from global_cams_sim.yaml -- see load_cam_specs()'s
    # docstring. Sim vocabulary/range guard, not per-camera.
    sim_prompt = cfg.get("prompt")
    sim_max_range_m = cfg.get("projector_max_range_m")
    # Optional per-key overrides for every global_cam_projector instance,
    # applied AFTER risk_perception.yaml's block (sim-specific tuning such as
    # the self-exclusion radii lives in global_cams_sim.yaml, not in the lab
    # config). Plain mapping: {param_name: value}.
    projector_overrides = cfg.get("projector_overrides") or {}
    if not isinstance(projector_overrides, dict):
        raise RuntimeError("global_cams_sim.yaml: projector_overrides must be a mapping")

    nodes = []
    for spec in specs:
        name = spec["name"]
        t, r = _load_extrinsic(spec, config_path)

        # Ground-truth constant -> latched /tf_static (unlike the real
        # calibrator's re-broadcast dynamic TF, which exists to let new
        # solves supersede old ones -- nothing supersedes ground truth).
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"static_tf_{name}",
            output="screen",
            arguments=[
                "--x", str(t["x"]), "--y", str(t["y"]), "--z", str(t["z"]),
                "--qx", str(r["x"]), "--qy", str(r["y"]),
                "--qz", str(r["z"]), "--qw", str(r["w"]),
                "--frame-id", "map", "--child-frame-id", spec["frame_id"],
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ))

        gdino_params = {**gdino_defaults,
                        "image_topic": spec["image_topic"],
                        "config_path": gdino_config,
                        "checkpoint_path": gdino_weights,
                        "inference_stride": spec["inference_stride"],
                        "use_sim_time": use_sim_time,
                        "debug_log_dir": prior_log_dir}
        if sim_prompt is not None:
            # Overrides risk_perception.yaml's gdino_detector_global block --
            # the sim scene's vocabulary (adds "forklift") differs from the
            # real lab overhead camera's.
            gdino_params["prompt"] = str(sim_prompt)

        nodes.append(Node(
            package="risk_perception",
            executable="gdino_detector",
            name=f"gdino_detector_{name}",
            output="screen",
            parameters=[gdino_params],
            # gdino_detector_node.py hardcodes its two publisher topics --
            # remapping is the only way to keep the camera chains apart
            # (same workaround as global_cam.launch.py).
            remappings=[
                ("/risk_perception/detections_2d", f"/{name}/detections_2d"),
                ("/risk_perception/detection_image", f"/{name}/detection_image"),
            ],
        ))

        nodes.append(Node(
            package="risk_perception",
            executable="sam2_segmenter",
            name=f"sam2_segmenter_{name}",
            output="screen",
            parameters=[{**sam2_defaults,
                         "image_topic": spec["image_topic"],
                         "detections_topic": f"/{name}/detections_2d",
                         "mask_topic": f"/{name}/instance_mask",
                         "segmentation_image_topic": f"/{name}/segmentation_image",
                         "checkpoint_path": sam2_checkpoint,
                         "use_sim_time": use_sim_time,
                         "debug_log_dir": prior_log_dir}],
        ))

        projector_params = {**projector_defaults,
                             "camera_info_topic": spec["info_topic"],
                             "detections_topic": f"/{name}/detections_2d",
                             "mask_topic": f"/{name}/instance_mask",
                             "marker_topic": f"/{name}/map_markers",
                             "use_sim_time": use_sim_time}
        if sim_max_range_m is not None:
            projector_params["max_range_m"] = float(sim_max_range_m)
        projector_params.update(projector_overrides)

        # output_topic stays the shared /risk_perception/detections_3d_map
        # from the yaml block -- that single topic IS the fusion mechanism.
        nodes.append(Node(
            package="risk_perception",
            executable="global_cam_projector",
            name=f"global_cam_projector_{name}",
            output="screen",
            parameters=[projector_params],
        ))

    return nodes


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("risk_perception")
    return LaunchDescription([
        DeclareLaunchArgument(
            "cams_config",
            default_value=_first_existing(
                os.path.join(SRC_CONFIG_DIR, "global_cams_sim.yaml"),
                os.path.join(share, "config", "global_cams_sim.yaml")),
        ),
        DeclareLaunchArgument("gdino_config", default_value=GDINO_CONFIG_DEFAULT),
        DeclareLaunchArgument(
            "gdino_weights",
            default_value=os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")),
        DeclareLaunchArgument(
            "sam2_checkpoint",
            default_value=os.path.join(WEIGHTS_DIR, "sam2.1_hiera_small.pt")),
        # Isaac stamps every /global_cam_*/... topic from /clock -- see
        # global_cams_sim.yaml's use_system_time comment. Default true here
        # (unlike risk_perception.launch.py's false) because this file is
        # sim-only; there is no real-hardware caller to protect.
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("prior_log_dir", default_value=""),
        OpaqueFunction(function=_setup),
    ])
