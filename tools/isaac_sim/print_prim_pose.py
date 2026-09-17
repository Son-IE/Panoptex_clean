#!/usr/bin/env python3
"""
print_prim_pose.py -- print a prim's exact world pose from an Isaac Sim /
USD stage, in both the USD and ROS-optical conventions. Read-only: never
modifies or saves anything.

For each prim it prints:
  * world translation (x, y, z) -- meters, stage /Root frame;
  * world rotation as a USD quaternion (w, x, y, z);
  * the same rotation converted to the ROS OPTICAL frame as (x, y, z, w) --
    exactly the quaternion global_cam_isaac.py writes into the per-camera
    extrinsic yaml, so the two outputs must match value-for-value;
  * the world view direction (optical +Z, i.e. where the lens points --
    z-component negative for a camera looking below horizontal);
  * roll/pitch/yaw of the optical rotation in degrees, for eyeballing;
  * a ready-to-paste calibrator-format yaml snippet.

TWO WAYS TO RUN
  * Headless, from a normal terminal:
        cd ~/isaacsim && ./python.sh \
            ~/workspace/Panoptex/tools/isaac_sim/print_prim_pose.py \
            --stage ~/Documents/DT-Project/warehouse_vlm_bag1.usd \
            /Root/GlobalCamera [/Root/GlobalCamera_01 ...]
    With no prim arguments, falls back to every cameras[].prim_path in
    src/risk_perception/config/global_cams_sim.yaml (which also supplies the
    default --stage). python.sh does not put pxr on the path by itself, so
    the script bootstraps it: first by re-execing itself with kit's bundled
    USD libs on PYTHONPATH/LD_LIBRARY_PATH (takes seconds), and only if that
    fails by booting a headless SimulationApp (~2 min). Read-only either way.
  * Inside Isaac Sim's Script Editor with the stage already open: paste/run
    this file. No argv there -- edit PRIM_PATHS below (empty list = the
    config fallback again).

Deliberately self-contained: the Script Editor paste path has no reliable
__file__/sys.path, so the ~10 lines of pose extraction + convention flip are
DUPLICATED from global_cam_isaac.py (export_extrinsic_yaml / _q_optical_to_usd)
rather than imported. Keep the two in sync.
"""

import argparse
import math
import os
import sys

# Script Editor mode: put prim paths here, e.g.
#   PRIM_PATHS = ["/Root/GlobalCamera", "/Root/GlobalCamera_01"]
# Empty list -> read every cameras[].prim_path from CONFIG_YAML.
PRIM_PATHS = []

PANOPTEX = os.path.join(os.path.expanduser("~"), "workspace", "Panoptex")
CONFIG_YAML = os.path.join(
    PANOPTEX, "src", "risk_perception", "config", "global_cams_sim.yaml")


def _config_fallback():
    """(stage_path or None, [prim_path, ...]) from global_cams_sim.yaml."""
    import yaml
    try:
        with open(CONFIG_YAML) as f:
            cfg = yaml.safe_load(f)
    except OSError as e:
        print(f"[print_prim_pose] no prims given and cannot read {CONFIG_YAML}: {e}")
        sys.exit(1)
    prims = [c["prim_path"] for c in cfg.get("cameras") or [] if c.get("prim_path")]
    stage = cfg.get("stage")
    return (os.path.abspath(os.path.expanduser(str(stage))) if stage else None,
            prims)


def print_prim_pose(stage, prim_path):
    # Same math as global_cam_isaac.py's export_extrinsic_yaml -- kept in sync
    # by hand (see module docstring).
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"\n{prim_path}: NOT FOUND in stage")
        return False

    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    # Gf.Transform factors out any scale/shear before handing back the
    # rotation -- ExtractRotationQuat on a scaled matrix would not.
    xf = Gf.Transform(m)
    q_usd = xf.GetRotation().GetQuat()
    t = xf.GetTranslation()
    # 180 deg about local X: USD camera (looks down -Z, +Y up) -> ROS optical
    # (+Z out of the lens, +Y down).
    q_opt = q_usd * Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
    qi, qw = q_opt.GetImaginary(), q_opt.GetReal()
    ui, uw = q_usd.GetImaginary(), q_usd.GetReal()

    view = Gf.Rotation(q_opt).TransformDir(Gf.Vec3d(0, 0, 1))

    # ZYX (yaw-pitch-roll) angles of the optical rotation, from the standard
    # quaternion formulas -- eyeball aid only, the quaternion is the truth.
    x, y, z, w = float(qi[0]), float(qi[1]), float(qi[2]), float(qw)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    print(f"\n{prim_path}")
    print(f"  translation (world):     ({t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f})")
    print(f"  rotation USD (w,x,y,z):  ({uw:.6f}, {ui[0]:.6f}, {ui[1]:.6f}, {ui[2]:.6f})")
    print(f"  rotation ROS-optical (x,y,z,w): ({x:.6f}, {y:.6f}, {z:.6f}, {w:.6f})")
    print(f"  view direction (world, optical +Z): "
          f"({view[0]:.4f}, {view[1]:.4f}, {view[2]:.4f})"
          + ("" if view[2] < 0 else "   <-- NOT looking downward"))
    print(f"  optical RPY (deg):       roll={math.degrees(roll):.2f}, "
          f"pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}")
    print("  extrinsic yaml snippet:")
    print("    parent_frame_id: map")
    print(f"    translation: {{x: {float(t[0])!r}, y: {float(t[1])!r}, z: {float(t[2])!r}}}")
    print(f"    rotation: {{x: {x!r}, y: {y!r}, z: {z!r}, w: {w!r}}}")
    return True


def run(stage, prim_paths):
    ok = True
    for p in prim_paths:
        ok = print_prim_pose(stage, p) and ok
    return ok


def _kit_already_running():
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except Exception:
        return False


if _kit_already_running():
    # Script Editor path: use the stage already open in the GUI.
    import omni.usd
    _stage = omni.usd.get_context().get_stage()
    _prims = PRIM_PATHS or _config_fallback()[1]
    if not _prims:
        print("[print_prim_pose] no prims: edit PRIM_PATHS or fill "
              f"cameras[] in {CONFIG_YAML}")
    else:
        run(_stage, _prims)
else:
    # Headless path: plain pxr stage open, no SimulationApp, read-only.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("prims", nargs="*",
                        help="prim paths (default: cameras[].prim_path from "
                             "global_cams_sim.yaml)")
    parser.add_argument("--stage", default=None,
                        help="USD stage to read (default: `stage:` from "
                             "global_cams_sim.yaml)")
    args = parser.parse_args()

    cfg_stage, cfg_prims = (None, [])
    if not args.prims or not args.stage:
        cfg_stage, cfg_prims = _config_fallback()
    prims = args.prims or cfg_prims
    stage_path = (os.path.abspath(os.path.expanduser(args.stage))
                  if args.stage else cfg_stage)
    if not stage_path:
        print("[print_prim_pose] no --stage given and no `stage:` in the config")
        sys.exit(1)
    if not prims:
        print("[print_prim_pose] no prim paths given and none in the config")
        sys.exit(1)

    app = None
    try:
        from pxr import Usd
    except ModuleNotFoundError:
        # python.sh alone does not expose pxr -- it lives in kit's extscache.
        # Tier 1: re-exec ourselves with the bundled USD libs on the path
        # (seconds). LD_LIBRARY_PATH must be set before the process starts,
        # hence exec rather than sys.path surgery.
        import glob
        if os.environ.get("_PRINT_PRIM_POSE_REEXEC") != "1":
            # sys.executable is <isaacsim>/kit/python/bin/python3*
            isaac = os.path.abspath(
                os.path.join(os.path.dirname(sys.executable), "..", "..", ".."))
            usd_libs = sorted(glob.glob(
                os.path.join(isaac, "extscache", "omni.usd.libs-*")))
            pip_pre = sorted(glob.glob(os.path.join(
                isaac, "extscache", "omni.services.pip_archive-*",
                "pip_prebundle")))  # provides yaml
            if usd_libs:
                env = dict(os.environ, _PRINT_PRIM_POSE_REEXEC="1")
                env["PYTHONPATH"] = os.pathsep.join(
                    [usd_libs[-1]] + pip_pre[-1:]
                    + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
                env["LD_LIBRARY_PATH"] = os.pathsep.join(
                    [os.path.join(usd_libs[-1], "bin"),
                     os.path.join(isaac, "kit"),
                     env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
                os.execve(sys.executable,
                          [sys.executable, os.path.abspath(__file__)]
                          + sys.argv[1:], env)
        # Tier 2: full headless SimulationApp boot (slow but always works).
        print("[print_prim_pose] pxr not importable directly -- booting "
              "headless SimulationApp (takes a couple of minutes)")
        try:
            from isaacsim import SimulationApp  # 4.5+/5.x
        except ImportError:
            from omni.isaac.kit import SimulationApp  # older 4.x
        app = SimulationApp({"headless": True})
        from pxr import Usd

    try:
        stage = Usd.Stage.Open(stage_path)
        if stage is None:
            print(f"[print_prim_pose] FAILED to open {stage_path}")
            sys.exit(1)
        print(f"[print_prim_pose] stage: {stage_path}")
        code = 0 if run(stage, prims) else 2
    finally:
        if app is not None:
            del stage
            app.close()
    sys.exit(code)
