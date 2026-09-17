#!/usr/bin/env python3
"""
global_cam_isaac.py -- wire EXISTING overhead camera prims in an Isaac Sim
stage into Panoptex's per-camera /global_cam_<i> topic contract.

Reads the camera list from src/risk_perception/config/global_cams_sim.yaml
(the single source of truth shared with launch/sim_global_cams.launch.py) and,
for EACH camera entry, WITHOUT touching its pose:

  1. sets its optics to the real Pi Camera Module 1's calibrated field of
     view (63 deg horizontal, 4:3), read from
     src/risk_perception/config/global_cam_intrinsics.yaml;
  2. builds an ActionGraph publishing, via Isaac's ROS 2 bridge, at
     `render_width` x `render_height` (default: the Pi's native 1296x972):
         <image_topic>   sensor_msgs/Image (rgb8)     (/<name>/image_raw)
         <info_topic>    sensor_msgs/CameraInfo       (/<name>/camera_info)
     both with frame_id <name>_optical_frame -- per-camera copies of the
     contract global_cam_bridge_node.py provides from the real Pi. Each
     graph's ROS2Context takes its domain from ROS_DOMAIN_ID
     (useDomainIDEnvVar, like every graph build_x3_graphs.py /
     build_carter_graphs.py author), so the graphs land on domain 55 when Kit
     is started from a shell that sourced warehouse/env_sim.sh. On Isaac Sim
     6.0+, CameraInfo is published by a dedicated ROS2CameraInfoHelper node
     (older versions fall back to ROS2CameraHelper's "camera_info" type, which
     6.0 removed). Re-running this script deletes and rebuilds every graph it
     owns -- including the legacy single-camera /Root/GlobalCamGraph, so a
     stage wired by the old version stops publishing /global_cam/* -- rather
     than accumulating duplicates;
  3. exports each camera's pose as a calibrator-format extrinsic yaml
     (config/<name>_extrinsic_sim.yaml), converting from the USD camera
     convention (looks down local -Z, +Y up) to the ROS optical frame (+Z out
     of the lens, +Y down) -- a 180-degree rotation about local X. This
     replaces the floor-AprilTag solve in sim: the pose is simply read off
     the prim. (To just LOOK at a prim's pose without wiring anything, use
     tools/isaac_sim/print_prim_pose.py.)

A FOURTH thing it does NOT do: publish /clock. Before building its own clock
graph it scans the stage for any ROS2PublishClock node; the warehouse stage
already carries one (/ClockGraph/Clock, authored by
warehouse/x3_sim/build_x3_graphs.py) and a SECOND /clock publisher makes the
two timelines interleave, which silently invalidates every sim-time consumer
(TF lookups, AMCL, the projector's age gate). The clock graph is only built
for a stage that has none.

TWO WAYS TO RUN
  * Headless, from a normal terminal (edits and saves the USD in place --
    snapshot it first, e.g.
    `cp Baseline_scenario_metric.usd Baseline_scenario_metric_pre_globalcams.usd`):
        cd ~/workspace/warehouse && source env_sim.sh
        ./x3_sim/isaac.sh \
            ~/workspace/Panoptex/tools/isaac_sim/global_cam_isaac.py --save
        # then, because Kit re-saved the stage, scrub the stray OmniGraph
        # overrides the Carter asset references pick up:
        python3 -c "import sys; sys.path.insert(0,'carters'); \
            import build_carter_graphs as b; \
            print(b.scrub_asset_overrides('Baseline_scenario_metric.usd'))"
    isaac.sh is the warehouse repo's retry wrapper for Kit's flaky start-up
    segfault; it appends `--/crashreporter/enabled=false`, which is why this
    script parses argv with parse_known_args.
    (--config <yaml> to use another camera list; --stage <usd> overrides the
    config's `stage:` entry.)
  * Inside Isaac Sim's Script Editor with the stage already open: paste/run
    this file, then Ctrl+S to persist the graphs into the USD. This mode has
    no argv -- it reads CONFIG_YAML below.

Then press Play and, on the PC (the launch reads this config and the
extrinsic yamls straight from the src tree -- no rebuild needed):

  ros2 launch risk_perception panoptex_sim.launch.py     # full stack
  ros2 launch risk_perception bench_sim_multicam.launch.py   # cameras only

THE EXTRINSICS' ONE ASSUMPTION: the map the PC stack serves has its origin at
the stage's /Root origin (same axes, floor at z=0). There is no AprilTag
cross-check in this mode -- if the map and the stage disagree about where the
origin is, every detection is silently offset by exactly that disagreement.

WHY AN IDEAL PINHOLE IS CORRECT HERE (not a shortcut): every consumer takes
K and D from camera_info, and Isaac derives that message from the camera prim
-- so the rendered pixels and the published model agree exactly, which is the
property the ground-plane projection depends on. Replicating the real lens's
distortion would make sim images cosmetically closer to the Pi but adds
nothing downstream.

TIME: the config's `use_system_time` (default FALSE) decides what the
publishers stamp with. False -- the setting the warehouse stage wants -- means
sim time from /clock, which is what /tf, /odom, /scan and the Carters in that
stage already use; the PC stack must then run use_sim_time:=true. Wall-clock
stamps (`use_system_time: true`) are only correct while the simulator holds
~1x real time, and this scene sits at 0.6-0.9x, so mixed clocks would skew TF
lookups, AMCL and the projector's self-exclusion age check. The flag is
applied through the graph's SET_VALUES on Isaac 6.0+ (authoring USD, so it
survives the save) and through the best-effort _set_optional path on older
bridges.

RATE and SIZE: the graph publishes every rendered frame (~30-60 fps vs the
Pi's ~10). The per-camera GDINO strides frames (inference_stride in the
config, default 30), so the rate only costs bandwidth. What costs FRAME TIME
is the render product, which renders on every frame no matter how often its
publisher fires -- so `render_width` / `render_height` in the config (default:
the intrinsics yaml's 1296x972) is the real FPS lever; halve them to 648x486
if the measured cost is too high. The FoV does not move with them:
_set_camera_optics derives the focal length from the intrinsics yaml's fx and
ITS width, and Isaac scales CameraInfo with the render product, so the
projection stays exact at any render size.
"""

import argparse
import math
import os
import sys

import yaml

# ---------------------------------------------------------------- config ---

PANOPTEX = os.path.join(os.path.expanduser("~"), "workspace", "Panoptex")
CONFIG_DIR = os.path.join(PANOPTEX, "src", "risk_perception", "config")
INTRINSICS_YAML = os.path.join(CONFIG_DIR, "global_cam_intrinsics.yaml")
# The camera list. Script Editor mode has no argv, so this constant IS the
# config selection there; headless mode can override with --config.
CONFIG_YAML = os.path.join(CONFIG_DIR, "global_cams_sim.yaml")

CLOCK_GRAPH_PATH = "/Root/ClockGraph"
# The pre-multicam version of this script built a single graph here publishing
# /global_cam/*. Always deleted on re-run so old stages stop publishing it.
LEGACY_GRAPH_PATH = "/Root/GlobalCamGraph"
# Fallback only -- the real value is the config's `use_system_time` (default
# below). See TIME in the module docstring.
DEFAULT_USE_SYSTEM_TIME = False

# Any value works for the aperture -- only focal/aperture RATIOS reach the
# projection. 20.955 mm is the USD default, kept for readability in the GUI.
HORIZONTAL_APERTURE_MM = 20.955


# --------------------------------------------------------------- helpers ---

def _load_intrinsics(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    k = [float(v) for v in d["camera_matrix"]["data"]]
    return int(d["image_width"]), int(d["image_height"]), k[0], k[4]


def _load_cam_specs(config_path):
    """Parse global_cams_sim.yaml into per-camera spec dicts, deriving every
    optional field from `name`. Returns (cfg, stage_path_or_None, [spec, ...]).

    Mirrored (by hand) by load_cam_specs() in
    launch/sim_global_cams.launch.py, which cannot import this module -- it
    executes Isaac bootstrap at module scope. Keep the derivations in sync;
    the top-level keys each side owns are documented in the config itself."""
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
            "prim_path": prim_path,
            "frame_id": entry.get("frame_id", f"{name}_optical_frame"),
            "image_topic": entry.get("image_topic", f"/{name}/image_raw"),
            "info_topic": entry.get("info_topic", f"/{name}/camera_info"),
            "graph_path": entry.get("graph_path", f"/Root/GlobalCamGraph_{name}"),
            "extrinsic_yaml": os.path.expanduser(entry.get(
                "extrinsic_yaml",
                os.path.join(CONFIG_DIR, f"{name}_extrinsic_sim.yaml"))),
        })

    stage_path = cfg.get("stage")
    if stage_path:
        stage_path = os.path.abspath(os.path.expanduser(str(stage_path)))
    return cfg, stage_path, specs


def _render_size(cfg, width, height):
    """Render-product size: the config's render_width/height, defaulting to
    the intrinsics yaml's image size. Only the pixel count changes -- the FoV
    comes from fx/width in the intrinsics yaml (see _set_camera_optics) and
    Isaac scales CameraInfo with the render product."""
    rw = int(cfg.get("render_width") or width)
    rh = int(cfg.get("render_height") or height)
    if (rw, rh) != (width, height):
        print(f"[global_cam_isaac] render size {rw}x{rh} (intrinsics yaml is "
              f"{width}x{height}; FoV unchanged, CameraInfo scales with it)")
    return rw, rh


def _q_optical_to_usd():
    """180 deg about local X: ROS optical frame <-> USD camera frame (its own
    inverse as a rotation, so the same quaternion converts either way)."""
    from pxr import Gf
    return Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))


def _set_camera_optics(stage, spec, width, height, fx):
    """Match a sim camera's FoV/aspect to the Pi calibration. Pose untouched."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(spec["prim_path"])
    if not prim.IsValid():
        raise RuntimeError(
            f"{spec['prim_path']} not found in stage -- place the camera prim "
            f"first (or fix `prim_path` for {spec['name']} in the config)")

    focal_mm = fx * HORIZONTAL_APERTURE_MM / width
    # USD/Isaac render square pixels: the vertical aperture that preserves
    # them at this resolution is A_H * H/W (which implies fy_sim == fx; the
    # real camera's fx/fy differ by 0.2% -- below the calibration's noise).
    vertical_aperture_mm = HORIZONTAL_APERTURE_MM * height / width
    cam = UsdGeom.Camera(prim)
    cam.GetFocalLengthAttr().Set(focal_mm)
    cam.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE_MM)
    cam.GetVerticalApertureAttr().Set(vertical_aperture_mm)
    cam.GetHorizontalApertureOffsetAttr().Set(0.0)
    cam.GetVerticalApertureOffsetAttr().Set(0.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))

    hfov = math.degrees(2 * math.atan(width / (2 * fx)))
    print(f"[global_cam_isaac] {spec['prim_path']}: focal={focal_mm:.3f}mm, "
          f"apertures={HORIZONTAL_APERTURE_MM}x{vertical_aperture_mm:.3f}mm "
          f"(hfov={hfov:.1f}deg, {width}x{height})")


def _node_type(short_name):
    """Resolve a bridge/core node type across the 4.5+ extension rename."""
    import omni.graph.core as og
    candidates = {
        "ROS2CameraHelper": ["isaacsim.ros2.bridge.ROS2CameraHelper",
                             "omni.isaac.ros2_bridge.ROS2CameraHelper"],
        "ROS2CameraInfoHelper": ["isaacsim.ros2.bridge.ROS2CameraInfoHelper"],
        "ROS2Context": ["isaacsim.ros2.bridge.ROS2Context",
                        "omni.isaac.ros2_bridge.ROS2Context"],
        "ROS2PublishClock": ["isaacsim.ros2.bridge.ROS2PublishClock",
                             "omni.isaac.ros2_bridge.ROS2PublishClock"],
        "IsaacCreateRenderProduct": [
            "isaacsim.core.nodes.IsaacCreateRenderProduct",
            "omni.isaac.core_nodes.IsaacCreateRenderProduct"],
        "IsaacReadSimulationTime": [
            "isaacsim.core.nodes.IsaacReadSimulationTime",
            "omni.isaac.core_nodes.IsaacReadSimulationTime"],
    }[short_name]
    for name in candidates:
        if og.get_node_type(name) is not None:
            return name
    raise RuntimeError(
        f"none of {candidates} registered -- is the ROS 2 Bridge extension "
        "enabled (Window > Extensions > 'ros2 bridge')?")


def _set_optional(node_path, attr, value):
    """Set an input only some bridge versions expose (e.g. useSystemTime).
    update_usd=True is load-bearing: the classmethod form of Controller.set
    defaults to Fabric-only, which a later stage save silently drops."""
    import omni.graph.core as og
    try:
        target = og.Controller.attribute(f"{node_path}.inputs:{attr}")
        try:
            og.Controller.set(target, value, update_usd=True)
        except TypeError:
            og.Controller.set(target, value)
    except Exception:
        print(f"[global_cam_isaac] note: {attr} not available on this bridge "
              "version, leaving default")


def _make_camera_graph(spec, width, height, use_system_time):
    import omni.graph.core as og
    keys = og.Controller.Keys

    try:
        info_type = _node_type("ROS2CameraInfoHelper")  # Isaac 6.0+
        legacy_info = False
    except RuntimeError:
        # pre-6.0: CameraInfo came from the generic camera helper, selected
        # via its "type" input rather than being its own node type.
        info_type = _node_type("ROS2CameraHelper")
        legacy_info = True
    print(f"[global_cam_isaac] camera_info node type: {info_type} "
          f"({'legacy' if legacy_info else '6.0+ dedicated'})")

    set_values = [
        # Take the ROS domain from ROS_DOMAIN_ID of the shell that started Kit
        # (55 via warehouse/env_sim.sh), exactly like every graph
        # build_x3_graphs.py and build_carter_graphs.py author. Without it the
        # graph pins domain 0 and nothing in this stage can see the cameras.
        ("context.inputs:useDomainIDEnvVar", True),
        ("render.inputs:cameraPrim", spec["prim_path"]),
        ("render.inputs:width", width),
        ("render.inputs:height", height),
        ("rgb.inputs:type", "rgb"),
        ("rgb.inputs:topicName", spec["image_topic"]),
        ("rgb.inputs:frameId", spec["frame_id"]),
        ("info.inputs:topicName", spec["info_topic"]),
        ("info.inputs:frameId", spec["frame_id"]),
    ]
    if legacy_info:
        # 6.0+'s dedicated node has no "type" input -- setting it would fail.
        set_values.append(("info.inputs:type", "camera_info"))
    if not legacy_info:
        # SET_VALUES authors USD (unlike the post-edit Controller.set path,
        # which is Fabric-only by default and vanishes on save). Both 6.0
        # nodes are confirmed to expose this input; the legacy path keeps the
        # best-effort _set_optional below instead. Written even when False so
        # the stamp source is explicit in the saved USD rather than a default.
        set_values += [("rgb.inputs:useSystemTime", bool(use_system_time)),
                       ("info.inputs:useSystemTime", bool(use_system_time))]

    og.Controller.edit(
        {"graph_path": spec["graph_path"], "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("context", _node_type("ROS2Context")),
                ("render", _node_type("IsaacCreateRenderProduct")),
                ("rgb", _node_type("ROS2CameraHelper")),
                ("info", info_type),
            ],
            keys.SET_VALUES: set_values,
            keys.CONNECT: [
                ("tick.outputs:tick", "render.inputs:execIn"),
                ("render.outputs:execOut", "rgb.inputs:execIn"),
                ("render.outputs:execOut", "info.inputs:execIn"),
                ("render.outputs:renderProductPath",
                 "rgb.inputs:renderProductPath"),
                ("render.outputs:renderProductPath",
                 "info.inputs:renderProductPath"),
                ("context.outputs:context", "rgb.inputs:context"),
                ("context.outputs:context", "info.inputs:context"),
            ],
        },
    )
    if legacy_info:
        _set_optional(f"{spec['graph_path']}/rgb", "useSystemTime",
                      bool(use_system_time))
        _set_optional(f"{spec['graph_path']}/info", "useSystemTime",
                      bool(use_system_time))
    print(f"[global_cam_isaac] {spec['graph_path']}: publishes "
          f"{spec['image_topic']} + {spec['info_topic']} at {width}x{height} "
          f"(frame_id={spec['frame_id']}, "
          f"stamps={'wall clock' if use_system_time else 'sim time'})")


def _find_clock_publishers(stage, ignore_paths=()):
    """Every prim in the stage that IS a ROS2PublishClock OmniGraph node.

    OmniGraph stores a node's type in the `node:type` string attribute, and
    the bridge was renamed across versions (omni.isaac.ros2_bridge.* ->
    isaacsim.ros2.bridge.*), so match on the suffix. `ignore_paths` skips
    prims this script is about to delete and rebuild."""
    hits = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(path == p or path.startswith(p + "/") for p in ignore_paths):
            continue
        attr = prim.GetAttribute("node:type")
        if not attr:
            continue
        val = attr.Get()
        if isinstance(val, str) and val.endswith("ROS2PublishClock"):
            hits.append((path, val))
    return hits


def _clean_render_settings(stage):
    """Undo what a Kit save does to the stage's render settings.

    Mirrors clean_render_settings() in warehouse/x3_sim/build_x3_graphs.py --
    every script that saves this stage has to do it, because Kit re-authors
    both of these on the way out:

      * `omni:rtx:rendermode` goes back to RealTimePathTracing, which makes
        the NEXT headless start spend minutes compiling the path-tracing
        shader set (and hit a corrupt entry in the shipped cache); the RTX
        lidar, the D455 and these cameras only need RaytracedLighting;
      * ~40 `/Render/OmniverseKit/HydraTextures/Replicator_*` render products
        are left behind, and Kit recreates each one (a 1280x720 render
        target) on every load.

    Measured: without this, one --save run flipped the warehouse stage's
    rendermode from RaytracedLighting to RealTimePathTracing."""
    textures = stage.GetPrimAtPath("/Render/OmniverseKit/HydraTextures")
    removed = 0
    if textures and textures.IsValid():
        for child in list(textures.GetChildren()):
            if child.GetName().startswith("Replicator"):
                stage.RemovePrim(child.GetPath())
                removed += 1
    if removed:
        print(f"[global_cam_isaac] removed {removed} stale Replicator render "
              "products")
    for prim in stage.Traverse():
        attr = prim.GetAttribute("omni:rtx:rendermode")
        if attr and attr.Get() not in (None, "RaytracedLighting"):
            print(f"[global_cam_isaac] {prim.GetPath()}: rendermode "
                  f"{attr.Get()} -> RaytracedLighting")
            attr.Set("RaytracedLighting")


def _make_clock_graph():
    import omni.graph.core as og
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": CLOCK_GRAPH_PATH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("context", _node_type("ROS2Context")),
                ("simtime", _node_type("IsaacReadSimulationTime")),
                ("clock", _node_type("ROS2PublishClock")),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("simtime.outputs:simulationTime", "clock.inputs:timeStamp"),
                ("context.outputs:context", "clock.inputs:context"),
            ],
        },
    )
    print(f"[global_cam_isaac] {CLOCK_GRAPH_PATH}: publishes /clock")


def export_extrinsic_yaml(stage, spec):
    """Write a camera prim's CURRENT world pose as a calibrator-format
    extrinsic yaml, in the ROS optical-frame convention. This is the sim
    replacement for the floor-AprilTag solve. Only valid while the map the PC
    stack serves has its origin at the stage's /Root origin.

    Conversion mirrored (self-contained) in print_prim_pose.py."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(spec["prim_path"])
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    # Gf.Transform factors out any scale/shear before handing back the
    # rotation -- ExtractRotationQuat on a scaled matrix would not.
    xf = Gf.Transform(m)
    q_usd = xf.GetRotation().GetQuat()
    t = xf.GetTranslation()
    q_opt = q_usd * _q_optical_to_usd()
    qi, qw = q_opt.GetImaginary(), q_opt.GetReal()

    out_path = spec["extrinsic_yaml"]
    with open(out_path, "w") as f:
        f.write(
            "# GENERATED by tools/isaac_sim/global_cam_isaac.py -- pose of the\n"
            f"# SIM overhead camera ({spec['prim_path']}) read straight off the\n"
            "# USD prim, converted to the ROS optical-frame convention.\n"
            "# Replaces the floor-AprilTag solve in sim. ONLY VALID while the\n"
            "# map the PC stack serves has its origin at the stage's /Root\n"
            "# origin. Consumed per-camera by sim_global_cams.launch.py (or\n"
            "# load a single camera with use_saved_extrinsic:=true\n"
            "# extrinsic_yaml:=<this file>).\n")
        yaml.safe_dump({
            "parent_frame_id": "map",
            "child_frame_id": spec["frame_id"],
            "translation": {"x": float(t[0]), "y": float(t[1]),
                            "z": float(t[2])},
            "rotation": {"x": float(qi[0]), "y": float(qi[1]),
                         "z": float(qi[2]), "w": float(qw)},
        }, f, sort_keys=False)

    # The optical +Z axis is the view direction -- print it as a sanity check
    # (should point into the scene, z-component negative for a camera looking
    # anywhere below horizontal).
    view = Gf.Rotation(q_opt).TransformDir(Gf.Vec3d(0, 0, 1))
    print(f"[global_cam_isaac] wrote {out_path}")
    print(f"[global_cam_isaac]   translation: ({t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f})")
    print(f"[global_cam_isaac]   view direction (world): "
          f"({view[0]:.3f}, {view[1]:.3f}, {view[2]:.3f})")


# ------------------------------------------------------------------ main ---

def main(config_path=CONFIG_YAML):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cfg, _, specs = _load_cam_specs(config_path)
    width, height, fx, fy = _load_intrinsics(INTRINSICS_YAML)
    render_w, render_h = _render_size(cfg, width, height)
    use_system_time = bool(cfg.get("use_system_time", DEFAULT_USE_SYSTEM_TIME))

    # Fail on a missing prim BEFORE deleting/rebuilding anything, so a typo'd
    # prim_path leaves the stage exactly as it was.
    for spec in specs:
        if not stage.GetPrimAtPath(spec["prim_path"]).IsValid():
            raise RuntimeError(
                f"{spec['prim_path']} (camera {spec['name']}) not found in "
                "stage -- place the prim first or fix `prim_path` in "
                f"{config_path}")

    # Re-running this script must not accumulate duplicate/stale graphs (e.g.
    # after switching Isaac versions, renaming a camera, or coming from the
    # single-camera version whose graph lived at LEGACY_GRAPH_PATH) -- delete
    # every graph prim this script may have created before rebuilding.
    import omni.kit.commands
    owned = ([s["graph_path"] for s in specs]
             + [CLOCK_GRAPH_PATH, LEGACY_GRAPH_PATH])
    stale = [p for p in owned if stage.GetPrimAtPath(p).IsValid()]
    if stale:
        omni.kit.commands.execute("DeletePrims", paths=stale)
        print(f"[global_cam_isaac] deleted stale graph(s): {stale}")

    for spec in specs:
        # FoV from the intrinsics yaml's own (width, fx); pixels at whatever
        # render size the config asked for.
        _set_camera_optics(stage, spec, width, height, fx)
        _make_camera_graph(spec, render_w, render_h, use_system_time)
        export_extrinsic_yaml(stage, spec)

    # NEVER add a second /clock publisher: the warehouse stage already has
    # /ClockGraph/Clock, and two publishers interleave into a meaningless
    # timeline for every sim-time consumer. (The graphs this script owns were
    # deleted just above, so they cannot match here.)
    existing_clocks = _find_clock_publishers(stage)
    if existing_clocks:
        for path, node_type in existing_clocks:
            print(f"[global_cam_isaac] /clock already published by {path} "
                  f"({node_type}) -- NOT building {CLOCK_GRAPH_PATH}")
    else:
        _make_clock_graph()

    _clean_render_settings(stage)

    names = ", ".join(s["name"] for s in specs)
    print(f"[global_cam_isaac] done ({len(specs)} camera(s): {names}). "
          "Press Play, then verify from the PC with:\n"
          + "".join(f"  ros2 topic hz {s['image_topic']}\n" for s in specs)
          + f"  ros2 topic echo {specs[0]['info_topic']} --once")


def _kit_already_running():
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except Exception:
        return False


if _kit_already_running():
    # Script Editor path: stage is already open in the GUI; just do the work.
    # (Ctrl+S afterwards to persist the graphs into the USD.)
    main()
else:
    # Headless path: boot our own SimulationApp, open the stage, save.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=CONFIG_YAML,
                        help="camera-list yaml (default: %(default)s)")
    parser.add_argument("--stage", default=None,
                        help="USD stage to edit (default: `stage:` from the config)")
    parser.add_argument("--save", action="store_true",
                        help="save the stage in place after adding the graphs")
    # parse_known_args, not parse_args: warehouse/x3_sim/isaac.sh (the retry
    # wrapper for Kit's start-up segfault) appends Kit's own
    # `--/crashreporter/enabled=false` to whatever it runs.
    args, _unknown = parser.parse_known_args()

    config_path = os.path.abspath(os.path.expanduser(args.config))
    _, stage_from_cfg, _ = _load_cam_specs(config_path)
    stage_path = (os.path.abspath(os.path.expanduser(args.stage))
                  if args.stage else stage_from_cfg)
    if not stage_path:
        print("[global_cam_isaac] no --stage given and no `stage:` in the config")
        sys.exit(1)

    try:
        from isaacsim import SimulationApp  # 4.5+/5.x
    except ImportError:
        from omni.isaac.kit import SimulationApp  # older 4.x
    # RaytracedLighting, not the default path tracer: a headless start under
    # RealTimePathTracing spends minutes compiling the PT shader set (and hits
    # a corrupt entry in the shipped cache) -- see warehouse/README.md
    # Troubleshooting. Every warehouse script passes the same.
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except ImportError:
            from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("isaacsim.ros2.bridge")
        app.update()

        import omni.usd
        usd_ctx = omni.usd.get_context()
        res = usd_ctx.open_stage(stage_path)
        ok = res[0] if isinstance(res, tuple) else bool(res)
        if not ok:
            print(f"[global_cam_isaac] FAILED to open {stage_path}")
            sys.exit(1)
        app.update()

        main(config_path)

        if args.save:
            usd_ctx.save_stage()
            print(f"[global_cam_isaac] saved {stage_path}")
        else:
            print("[global_cam_isaac] dry run -- stage NOT saved "
                  "(pass --save to persist)")
    finally:
        app.close()
