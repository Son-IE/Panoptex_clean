# Panoptex

### risk_perception — risk-aware perception & navigation

Real-time semantic **risk map** for autonomous mobile robot (AMR) navigation.
A robot-mounted Intel RealSense (RGB-D) and a static overhead camera each
detect and localize objects into a shared `map` frame; the result is fused
into a temporally persistent world model that Nav2 consumes as a graded
costmap layer.

```
RealSense RGB-D ─► GroundingDINO ─► SAM2 ─► RGB-D projector ──────► map projector ─┐
   (robot cam)       (2D boxes)    (masks)   (3D, cam frame)        (3D, map)      │
                                                                                    ├─► object_tracker ─► risk_costmap ─► Nav2
Overhead RGB    ─► GroundingDINO ─► SAM2 ─► ground-plane projector ───────────────┘   (fused world model)  (OccupancyGrid)
   (global cam)      (2D boxes)    (masks)   (3D via floor AprilTags, map)
```

Both cameras publish onto the same `/risk_perception/detections_3d_map`
topic, so `object_tracker_node` fuses them for free — the same chair seen by
both becomes one track, not two. The overhead camera has no depth sensor, so
it locates objects by intersecting a camera ray with the floor plane,
anchored to `map` via three tag36h11 AprilTags surveyed onto the floor (see
§6). A fourth tag riding on the robot gives an independent, overhead-camera
estimate of the robot's own pose (`global_cam_localizer_node`) — used to seed
AMCL automatically at startup (`global_cam_initialpose_node`), so no manual
"2D Pose Estimate" is needed.

Day to day the stack is **two commands on the robot and one on the PC**
(§5.1) — `panoptex_pc.launch.py` brings up localization, both camera chains,
the tracker, the risk costmap, RViz and (optionally) Nav2 together. The
individual launch files behind it are in §5.4, for when you want one stage in
its own terminal. Before any of it is useful, the overhead camera needs its
one-time calibration (§6).

---

## 1. Prerequisites

| Component | Version / note |
|---|---|
| OS / ROS | Ubuntu 22.04 · ROS 2 **Humble** |
| GPU stack | CUDA **12.1**, torch **2.5.1+cu121** (in the `panoptex` conda env) |
| Python env | conda env `panoptex` (Python 3.10), created by `./setup.sh` — see §3 |
| Robot base | Yahboom X3 (`~/workspace/yahboomcar_ws`), publishes odom + EKF + URDF tf |
| Robot camera | Orbbec Astra Pro Plus RGB-D (or RealSense), `realsense2_camera`/Astra driver |
| SLAM | `slam_toolbox` (provides `map → odom`) |
| Models | GroundingDINO + SAM2 (Grounded-SAM-2), weights downloaded locally |
| Overhead camera | Raspberry Pi Camera v2, TCP:5000 JPEG stream → `global_cam_bridge` |
| Simulation (optional) | Isaac Sim 6.0.1 warehouse in `~/workspace/warehouse` (its README is the run-book); three overhead cameras + the X3, all on `ROS_DOMAIN_ID=55` — §5.3b/§5.3c |
| Overhead cam calibration | `camera_calibration` (`cameracalibrator`) for intrinsics; 3× tag36h11 floor AprilTags for extrinsics |
| AprilTags | `apriltag_ros` (tag36h11, IDs 0-3: 0 on the robot, 1/2/3 on the floor) |

`~/.bashrc` already exports the hardware config every terminal needs:
```bash
ROS_DOMAIN_ID=77
ROBOT_TYPE=x3           # r2, x3
RPLIDAR_TYPE=a1          # a1, s2, 4ROS
CAMERA_TYPE=astraplus     # astrapro, astraplus
```
Change these there (not per-terminal) if the hardware changes.

Model files, and where they come from — `./setup.sh` (§3.1) sets up all of
this, nothing here needs manual cloning:

```
GroundingDINO code + config: $CONDA_PREFIX/src/groundingdino/  (editable install, from requirements-models.txt)
GroundingDINO weights      : weights/groundingdino_swint_ogc.pth       (downloaded by setup.sh)
SAM2 code + config         : $CONDA_PREFIX/src/sam-2/                 (editable install, from requirements-models.txt)
SAM2 checkpoint            : weights/sam2.1_hiera_small.pt             (downloaded by setup.sh)
```

`weights/` is a directory in this repo (gitignored — binaries don't belong
in git); `risk_perception.launch.py` and `global_cam.launch.py` default
their `gdino_weights`/`sam2_checkpoint` launch args there, and their
`gdino_config` arg to the conda env's editable GroundingDINO checkout (via
`$CONDA_PREFIX`, so it resolves correctly regardless of where conda itself
is installed — just make sure `conda activate panoptex` happened before you
launch). Verify the compiled CUDA op built correctly:
`python -c "import torch; from groundingdino import _C; print('GPU ops OK')"`
(if that fails, GroundingDINO fell back to CPU — see §9).

---

## 2. Directory structure

Everything lives directly in this repo — there is no separate overlay
workspace.

```
~/workspace/Panoptex/                   # this repo == the colcon workspace root
├── setup.sh                            # one-time: creates the `panoptex` conda env
├── requirements.txt, requirements-models.txt
├── tools/
│   ├── make_apriltag.py                # printable tag36h11 SVGs at an exact mm size (§6 Step 0)
│   ├── scenario_publisher.py           # synthetic world model — drive Stage 2/3/4 with no camera/robot/bag (§5.5)
│   ├── run_trials.py                   # automates waypoint-tour trials per ablation condition (§5.5)
│   ├── evaluate_run.py                 # post-hoc Efficiency/Safety/Latency/Clearance scoring of a recorded bag (§5.5)
│   ├── prior_report.py                 # turns the prior nodes' research CSVs into percentile tables + plots (§5.5)
│   ├── spatial_flow_heatmap.py         # renders spatial_prior_node's persisted .npz as a figure (§5.5)
│   └── isaac_sim/
│       ├── global_cam_isaac.py         # wires the stage's camera prims into ROS 2 + exports extrinsics (§5.3b)
│       ├── print_prim_pose.py          # read-only pose printer for any prim
│       └── check_sim_tracks.py         # tracks vs the stage's ground-truth object positions (§5.3c)
└── src/
    ├── risk_perception/                # main perception package (ament_python)
    │   ├── launch/
    │   │   ├── panoptex_pc.launch.py       # everything the PC runs, one command (§5.1)
    │   │   ├── robot_base.launch.py        # base + slam, or AMCL w/ enable_base:=false (Terminal A)
    │   │   ├── risk_perception.launch.py   # robot camera + perception (Terminal B)
    │   │   ├── global_cam.launch.py        # overhead camera + perception (Terminal C)
    │   │   ├── bench_global_cam.launch.py  # overhead camera only, no robot (§5.3)
    │   │   ├── sim_global_cams.launch.py   # Isaac Sim: one chain per overhead camera (§5.3b)
    │   │   ├── bench_sim_multicam.launch.py # Isaac Sim: N overhead cameras, no robot (§5.3b)
    │   │   ├── panoptex_sim.launch.py      # Isaac Sim: the full stack with the X3 (§5.3c)
    │   │   └── evaluation_reference.launch.py # fixed full-system scoring costmap, for ablation studies (§5.5)
    │   ├── config/
    │   │   ├── risk_perception.yaml        # all node params
    │   │   ├── floor_tags.yaml             # surveyed/aligned tag positions (see §6)
    │   │   ├── global_cam_extrinsic.yaml   # cached map→camera pose (§6 Step 8, generated)
    │   │   ├── apriltag.yaml               # apriltag_ros config
    │   │   ├── amcl.yaml                   # saved-map localization (robot_base.launch.py map_yaml:=...)
    │   │   ├── amcl_sim.yaml               # the same for the Isaac X3 (omni model, baked spawn pose, sim time)
    │   │   ├── global_cams_sim.yaml        # Isaac Sim camera list + costmap extent (§5.3b), read by tool AND launch
    │   │   ├── <name>_extrinsic_sim.yaml   # generated: ground-truth map→camera pose per sim camera
    │   │   ├── nav2_sim.yaml               # dwa_nav_params.yaml + nav2_risk_layer + amcl_sim.yaml, sim time (§5.3c)
    │   │   ├── ekf_global.yaml             # Workstream B2, off by default
    │   │   └── global_cam_intrinsics.yaml  # from `camera_calibration cameracalibrator`
    │   ├── rviz/
    │   │   ├── panoptex.rviz               # map/costmap/markers preset (risk_perception.launch.py enable_rviz:=true)
    │   │   └── panoptex_sim.rviz           # same, with the three sim cameras' raw markers + feeds (§5.3c)
    │   └── risk_perception/
    │       ├── model_adapters/
    │       │   ├── grounding_dino_adapter.py
    │       │   └── sam2_adapter.py
    │       ├── geometry_utils.py             # shared rotation/quaternion helpers
    │       ├── gdino_detector_node.py        # RGB → 2D detections (run twice: robot + overhead)
    │       ├── sam2_segmenter_node.py        # boxes → instance masks (run twice: robot + overhead)
    │       ├── rgbd_projector_node.py        # mask+depth → 3D (camera frame, robot cam only)
    │       ├── map_frame_projector_node.py   # 3D → map frame + markers (robot cam only)
    │       ├── global_cam_bridge_node.py     # TCP:5000 JPEG → /global_cam/image_raw
    │       ├── global_cam_calibrator_node.py # floor tags 1/2/3 → TF map->camera (solvePnP, continuous)
    │       ├── global_cam_survey_node.py     # run-once: 3 measured distances → floor_tags.yaml (tag-defined frame)
    │       ├── global_cam_map_align_node.py  # run-once: tag 0 + SLAM → floor_tags.yaml (SLAM map frame, see §6)
    │       ├── global_cam_projector_node.py  # ground-plane projection → detections_3d_map + map_markers
    │       ├── global_cam_localizer_node.py  # tag 0 on robot → /global_cam/robot_pose
    │       ├── global_cam_initialpose_node.py # tag 0 → /initialpose, seeds AMCL (no 2D Pose Estimate)
    │       ├── global_cam_tag_monitor_node.py # live "are all 4 tags visible" readout + overlay (§6)
    │       ├── global_cam_align_check_node.py # calibration error in cm: overhead pose vs SLAM (§6 Step 6)
    │       ├── object_tracker_node.py        # persistent world model, fuses both cameras
    │       ├── risk_costmap_node.py          # world model → /risk_costmap OccupancyGrid
    │       ├── predictive_risk_costmap_node.py # Stage 2/3/4: predicted occupancy + CPA/TTC → /risk_costmap_predictive
    │       ├── spatial_prior_node.py         # Spatial-Flow prior: persistent map-aligned "observed moving here" grid
    │       ├── evaluation_node.py            # live Efficiency/Safety/Latency/Clearance scoring, trial_control protocol
    │       ├── evaluation_metrics.py         # pure metric computation, shared by evaluation_node + tools/evaluate_run.py
    │       ├── encounter_geometry.py         # pure geometry: Stage 4 consequence + CPA/TTC (no rclpy)
    │       ├── mask_relation.py              # pure geometry: relation prior, GDINO-free
    │       ├── relation_matching.py          # pure geometry: relation prior (proposed), no ROS/torch
    │       ├── nms.py                        # pure non-max suppression for GDINO's raw 2D detections
    │       ├── debug_log.py                  # shared CSV logger for the prior research nodes (§5.5)
    │       ├── calibrate_intrinsics.py       # standalone CLI: chessboard capture + calibration
    │       └── risk_visualization.py         # risk_score_from_label, risk_to_bgr
    ├── risk_map_publisher/         # synthetic /risk_map test publisher (not the real pipeline)
    └── nav2_risk_layer/            # custom Nav2 costmap layer (C++ plugin)
```

Registered executables (`ros2 pkg executables risk_perception`):
`gdino_detector`, `sam2_segmenter`, `rgbd_projector`, `map_frame_projector`,
`risk_costmap`, `predictive_risk_costmap`, `spatial_prior`,
`global_cam_bridge`, `global_cam_calibrator`, `global_cam_projector`,
`global_cam_localizer`, `global_cam_survey`, `global_cam_map_align`,
`global_cam_tag_monitor`, `global_cam_align_check`, `global_cam_initialpose`,
`object_tracker`, `evaluation_node`.

---

## 3. Environment

Every new terminal already has ROS + this workspace sourced automatically
(`~/.bashrc` runs `source /opt/ros/humble/setup.bash`, then the
`yahboomcar_ws` and `Panoptex` installs, in that order). You do not need to
source anything by hand for plain ROS work.

The only thing you still activate manually is the **`panoptex` conda env**,
and only in terminals that run a perception node (torch / GroundingDINO /
SAM2 — GDINO/SAM2 nodes, i.e. Terminal B and Terminal C below):

```bash
conda activate panoptex
```

Terminal A (robot base + SLAM) doesn't need it — `robot_localization` and
the Yahboom driver nodes are plain system-Python ROS nodes, and CUDA_HOME
is already exported globally in `~/.bashrc` for the terminals that do need it.

**`panoptex_pc.launch.py` (§5.1) does need it**, since it starts the
perception nodes alongside localization. Mixing the two in one environment is
safe: `nav2_map_server` and `robot_localization`'s `ekf_node` were both
verified to start normally with the env active. §5.1 has the two-terminal
fallback if you ever suspect otherwise.

### 3.1 One-time setup: the `panoptex` conda env

```bash
./setup.sh
```

Safe to re-run any time. It:
- Creates the `panoptex` conda env (Python 3.10, matching ROS 2 Humble's
  system Python) if missing, and reuses it otherwise.
- Sets `PYTHONNOUSERSITE=1` so the env can't silently pick up stray packages
  from `~/.local/lib/python3.10/site-packages`.
- Auto-detects your installed CUDA toolkit (via `nvcc`) and installs the
  matching PyTorch/torchvision build — nothing is hardcoded to one CUDA
  version. If no GPU or usable CUDA toolkit is found, it stops and tells you
  to re-run with `--cpu-only` rather than silently falling back.
- Installs `requirements.txt`, then GroundingDINO + SAM2 from
  `requirements-models.txt` (editable checkouts land under
  `$CONDA_PREFIX/src`, not in this repo's own `src/`).

Flags: `--recreate` (drop and rebuild the env from scratch), `--cpu-only`
(force a CPU-only torch build), `--src-dir PATH` (override the editable
checkout location). `./setup.sh --help` for the summary.

---

## 4. Build

```bash
cd ~/workspace/Panoptex
source /opt/ros/humble/setup.bash
conda activate panoptex                     # required — see below
colcon build --symlink-install --packages-select risk_perception
ros2 pkg executables risk_perception        # confirm nodes are listed
```

**Build with `panoptex` active.** `colcon` is installed *into* the env
(`requirements.txt`), so with the env active `which colcon` resolves to
`$CONDA_PREFIX/bin/colcon`. That matters because `ament_python` packages get
their `console_scripts` shebang from whichever python is running colcon: the
apt-installed `/usr/bin/colcon` stamps `#!/usr/bin/python3`, and the nodes
then start *outside* the env and die on `import torch`. Check with:

```bash
which colcon                                             # -> .../envs/panoptex/bin/colcon
head -1 install/risk_perception/lib/risk_perception/gdino_detector
#   -> #!/home/<you>/miniconda3/envs/panoptex/bin/python3.10
```

If the shebang is wrong, activate the env and rebuild - a re-source won't fix
an already-generated shebang.

**The launch files only work if `setup.py` installs `launch/` and
`config/`** — already the case in this repo's `setup.py` (`data_files`
includes both `glob('launch/*.launch.py')` and `glob('config/*.yaml')`); if
you add a *new* launch or config file, a rebuild is required to pick it up

Other packages, when needed:
```bash
colcon build --packages-select nav2_risk_layer risk_map_publisher
```

---

## 5. Run

Kill stale processes first (a leftover `odom` publisher silently corrupts tf):
```bash
pkill -f slam_toolbox; pkill -f ekf; pkill -f base_node; pkill -f gdino; pkill -f sam2
ros2 daemon stop && ros2 daemon start
ros2 node list        # should be empty
```

### 5.1 Normal operation — Jetson + one PC command

This is the everyday path once §6's calibration is done. The robot side stays
manual; everything on the PC comes up together.

**On the Jetson** (over SSH, two terminals):
```bash
ros2 launch yahboomcar_nav laser_bringup_launch.py
ros2 launch astra_camera astro_pro_plus.launch.xml depth_registration:=true
```
Do **not** start `slam_toolbox` here. It builds a fresh map with a new origin
every time, which silently invalidates the overhead-camera calibration (§6).
The PC localizes against the saved map instead.

**On the PC:**
```bash
conda activate panoptex
ros2 launch risk_perception panoptex_pc.launch.py
```
Brings up, in one shot: AMCL against `~/maps/lab.yaml` (base driver off — it
lives on the robot), the overhead-camera chain, `global_cam_initialpose`
(seeds AMCL from tag 0, so no "2D Pose Estimate" click), the robot-camera
chain, object tracker, risk costmap, and RViz. Nav2 is included but **off**;
add `enable_nav2:=true` when you want it.

| Argument | Default | |
|---|---|---|
| `map_yaml` | `~/maps/lab.yaml` | saved map to localize against |
| `enable_localization` | `true` | map_server + amcl — **automatically skipped when `enable_nav2:=true`**, since Nav2's own bringup starts an identically-configured map_server + amcl; running both means two AMCLs fighting over `map → odom` |
| `enable_amcl` | `true` | set false to serve `/map` **without** amcl. Amcl needs `/scan` and the `odom` frame, i.e. the robot; without them it never activates and stalls the lifecycle manager, taking `/map` down with it |
| `enable_global_cam` | `true` | overhead bridge, apriltag, calibrator, localizer |
| `global_cam_source` | `real` | `real` (Pi over TCP) or `sim` (Isaac Sim publishes `/global_cam` topics itself — see `global_cam.launch.py`, `tools/isaac_sim/global_cam_isaac.py`) |
| `enable_perception` | `true` | overhead GDINO/SAM2 — set false for calibration |
| `use_saved_extrinsic` | `false` | start the overhead calibrator from the cached `config/global_cam_extrinsic.yaml` instead of waiting on the floor tags (§6) |
| `enable_apriltag` | `true` | only safe to disable together with `use_saved_extrinsic:=true` |
| `enable_localizer` | `true` | tag-0 robot pose — pointless with no robot |
| `enable_initialpose` | `true` | seed AMCL from tag 0 |
| `enable_robot_cam` | `true` | the RGB-D chain **only** (GDINO/SAM2/rgbd_projector/map_frame_projector) |
| `enable_world_model` | `true` | `object_tracker` — independent of either camera |
| `enable_costmap` | `true` | `risk_costmap` — independent of either camera |
| `enable_rviz` | `true` | |
| `enable_nav2` | `false` | `navigation_dwa_launch.py` |
| `nav2_params_file` | yahboomcar_ws `dwa_nav_params.yaml` | the copy with the risk layer wired in |

`enable_robot_cam` gates the RGB-D chain **only**. The tracker, risk costmap
and RViz have their own toggles, because they depend on neither camera:
`object_tracker` fuses whatever reaches `/risk_perception/detections_3d_map`,
and neither it nor `risk_costmap_node` touches TF. That is what makes
`enable_robot_cam:=false` a valid overhead-camera-only run — and what makes
§5.3 possible at all.

**Nav2 must run on the PC, not the Jetson.** Only this workspace's
`dwa_nav_params.yaml` wires `nav2_risk_layer::RiskLayer` into the global
costmap, and `libnav2_risk_layer.so` is built only in this workspace's
install space. Costmap plugins load in-process, so the Jetson would fail to
load it and the global costmap would never activate.

**Activate `panoptex` for this launch.** `global_cam.launch.py` resolves
GroundingDINO's config from `$CONDA_PREFIX` (falling back to a hardcoded
path), and any rebuild from that terminal needs the env to stamp the right
console-script shebang — see §4. `nav2_map_server` and `robot_localization`
were both verified to start normally with the env active.

If localization or the EKF ever misbehaves in a way that smells
environmental, split back into the two-environment layout CLAUDE.md
describes — the toggles make this a one-liner each:
```bash
# terminal 1, NO conda: localization (+ nav2)
ros2 launch risk_perception panoptex_pc.launch.py \
  enable_global_cam:=false enable_robot_cam:=false enable_initialpose:=false \
  enable_world_model:=false enable_costmap:=false enable_rviz:=false

# terminal 2, conda active: cameras, models, tracker, costmap, rviz
ros2 launch risk_perception panoptex_pc.launch.py \
  enable_localization:=false enable_nav2:=false
```

Terminal 1 must turn off `enable_world_model`, `enable_costmap` and
`enable_rviz` explicitly: they no longer ride on `enable_robot_cam`, so
leaving them at their defaults starts a second copy of each alongside
terminal 2's.

### 5.3 Bench test — the overhead camera with no robot

Exercises the overhead camera's whole path — detection, ground-plane
projection, tracking, risk costmap — with the Rosmaster switched off. Put a
real object under the camera and watch it become a marker and a risk blob on
the saved map in RViz.

**Once**, with the floor tags still down and the camera streaming, cache the
extrinsic:
```bash
ros2 launch risk_perception global_cam.launch.py enable_perception:=false \
  extrinsic_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/global_cam_extrinsic.yaml
```
Point it at `src/`, not the installed share copy, so the result is
committable — same reason as §6 Step 3. Wait for `rms_reprojection` under 5 px
and a `Saved extrinsic to ...` line, then Ctrl-C and rebuild (§4).

**Then**, tags optional, robot off:
```bash
conda activate panoptex
ros2 launch risk_perception bench_global_cam.launch.py
```

This is `panoptex_pc.launch.py` with `enable_amcl:=false`,
`enable_robot_cam:=false`, `enable_initialpose:=false`,
`enable_localizer:=false` and `use_saved_extrinsic:=true`. There is no robot,
so there is no `odom` frame, no `map → odom` transform and no `/scan` — RViz's
RobotModel and LaserScan displays will show red errors. Expected; everything
else on the map is live.

The headline check is that this resolves with the floor tags lifted off the
floor:
```bash
ros2 run tf2_ros tf2_echo map global_cam_optical_frame
```
`apriltag_node` still runs by default, so putting the tags back re-solves live
and overrides the cached pose. Add `enable_apriltag:=false` to drop it and
reclaim the CPU it competes with the GPU models for (§6).

### 5.3b Isaac Sim — N overhead cameras, no robot

The sim equivalent of §5.3, against the warehouse stage
`~/workspace/warehouse/Baseline_scenario_metric.usd` (see that repo's README for
the scene itself). As many overhead cameras as
`src/risk_perception/config/global_cams_sim.yaml` lists — three today,
`global_cam_mid` / `global_cam_entry` / `global_cam_exit`, one per
`/GlobalCameraMid|Entry|Exit` prim; adding one is a single config entry. No
AprilTags, no calibrator: each camera's exact pose is read off its USD prim and
broadcast as a static `map → <name>_optical_frame` transform. The `map` frame is
the stage's `/Root` origin, which is also the frame of the warehouse's exporter
map (`warehouse/maps/warehouse_x3_nav.yaml`) — that identity is the one
assumption everything here rests on, so never serve a SLAM map against these
extrinsics. Each camera gets its own GroundingDINO + SAM2 +
ground-plane-projector chain; all of them publish onto
`/risk_perception/detections_3d_map`, so the tracker fuses every view into one
set of tracks exactly the way it fuses the two real cameras.

**Once per camera-layout change** (cameras placed/moved in the stage, or the
config's camera list / render size edited) — builds one ROS 2 bridge graph per
camera at the stage root (`/GlobalCamGraph_<suffix>`), sets the prims' optics
to the real Pi camera's 63° FOV, and exports the ground-truth extrinsics to
`config/<name>_extrinsic_sim.yaml`. Isaac writes the USD, so snapshot first
(the warehouse repo's `_pre_*` convention), and scrub afterwards — a Kit save
of this stage leaves stray OmniGraph override specs under the Nova Carters
(documented in the warehouse README under *Idempotency, and the OmniGraph
override trap*):
```bash
cd ~/workspace/warehouse && source env_sim.sh          # domain 55, retry wrapper below
cp Baseline_scenario_metric.usd Baseline_scenario_metric_pre_globalcams.usd
./x3_sim/isaac.sh ~/workspace/Panoptex/tools/isaac_sim/global_cam_isaac.py --save
PXR=$HOME/isaacsim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312
PHYSX=$(ls -d $HOME/isaacsim/extscache/omni.usd.schema.physx-*/ | head -1)
PYTHONPATH=$PXR:$PHYSX LD_LIBRARY_PATH=$PXR/bin:$PXR/lib:$PHYSX/bin:$PHYSX/lib \
  $HOME/isaacsim/kit/python/bin/python3 -c "import sys; sys.path.insert(0,'carters'); \
  import build_carter_graphs as b; print(b.scrub_asset_overrides('Baseline_scenario_metric.usd'))"
```
The script never creates a second `/clock` publisher (the stage already has
`/ClockGraph`; it detects any `ROS2PublishClock` and skips its own), forces
every context onto the shell's `ROS_DOMAIN_ID`, and restores
`RaytracedLighting` after the save (Kit flips the render mode back to path
tracing, which hangs the next headless start). Inside the GUI's Script Editor
the same file edits the open stage; Ctrl+S, then run the scrub line.

**Sim time.** The camera graphs stamp from `/clock` (`use_system_time: false`
in the config), like every other topic in the stage, and the simulator runs at
0.4–0.6x wall — so every launch in this section defaults `use_sim_time:=true`
and RViz is started on sim time too. Mixing wall-clock nodes in makes RViz drop
TF lookups and AMCL time out; there is no wall-clock mode for the sim any more.

**Render size.** A render product costs the same per frame whether or not its
publisher fires, so resolution is the FPS lever (`render_width/height` in the
config; Isaac scales `CameraInfo` with it, so projection stays exact and the
FOV still comes from the intrinsics yaml). Measured with
`x3_sim/run_headless.py --seconds 30` on the RTX 3090, X3 + both Carters active:

| overhead cameras | FPS | sim / wall |
|---|---|---|
| none | 35.0 | 0.58x |
| 3 × 1296×972 (the Pi's native size) | 19.2 | 0.32x |
| **3 × 648×486 (shipped)** | **28.4** | **0.48x** |

GDINO resizes to 800 px anyway; the next lever would be quartering again or
dropping a camera.

**Then**, with Isaac playing (GUI or `./x3_sim/isaac.sh x3_sim/run_headless.py`):
```bash
conda activate panoptex
ros2 launch risk_perception bench_sim_multicam.launch.py
```

Both read `global_cams_sim.yaml` and the exported extrinsics from `src/`, not
the installed share, so layout changes need no rebuild. To eyeball any prim's
pose (world translation, USD + ROS-optical quaternions, view direction)
without touching the stage:
```bash
cd ~/workspace/warehouse && ./x3_sim/isaac.sh \
  ~/workspace/Panoptex/tools/isaac_sim/print_prim_pose.py --stage Baseline_scenario_metric.usd /GlobalCameraMid
```

Per-camera sanity checks, then the fused one:
```bash
ros2 topic info /clock -v                                   # exactly ONE publisher
ros2 topic hz /global_cam_mid/image_raw                     # ~22 Hz wall at 0.48x
ros2 run tf2_ros tf2_echo map global_cam_mid_optical_frame  # == extrinsic yaml == prim
```
Put one object where two cameras see it: its raw per-camera markers
(`/global_cam_<name>/map_markers`) must land on the same floor spot (an offset
means that camera's extrinsic is stale — re-run the export), the tracker must
show ONE track for it, and `/risk_costmap` paints one blob under it. The
costmap covers the warehouse extent from the config's `costmap:` block
(20 × 30.4 m from (-10.4, -12.3), the exporter map's own footprint) instead of
the 10×10 m lab default. As in §5.3 there is no robot, so RViz's
Map/RobotModel/LaserScan displays erroring is expected noise.

Two sim-only knobs live in the same config: `prompt:` (the overhead GDINO
vocabulary — the warehouse one adds `forklift`) and `projector_max_range_m:`
(15 m; the cameras sit ~3.5 m up tilted only ~24° below horizontal, so a mask
whose bottom pixel lands in the top rows projects to a floor point tens of
metres away — this drops those instead of trusting the 4 m² covariance cap).

### 5.3c Isaac Sim — the full stack with the X3

`panoptex_sim.launch.py` is the sim twin of `panoptex_pc.launch.py` (§5.1):
localization against the warehouse map, the three overhead chains, the X3's
RGB-D chain, tracker, costmaps and RViz, in one command, all on sim time.
The robot side is the warehouse repo's X3 sim (the Isaac stage publishes
`/scan`, `/odom`, `/tf odom→base_footprint`, the RGB-D topics and consumes
`/cmd_vel`; `yahboomcar_nav`'s `x3_sim_bringup_launch.py` adds the URDF and
static TFs). Bring-up order — every terminal sources
`~/workspace/warehouse/env_sim.sh` first (domain 55; `~/.bashrc` already
sourced this workspace):

| # | Terminal | Command |
|---|---|---|
| A | Isaac | open `Baseline_scenario_metric.usd`, Script Editor → Run `x3_sim/x3_base_controller.py`, **Play** (headless: `./x3_sim/isaac.sh x3_sim/run_headless.py`) |
| B | X3 ROS side | `ros2 launch yahboomcar_nav x3_sim_bringup_launch.py` |
| C | optional | `ros2 run yahboomcar_ctrl yahboom_keyboard` |
| D | optional | `ros2 launch ~/workspace/warehouse/launch/carters_patrol.launch.py map:=$HOME/workspace/warehouse/maps/warehouse_gt_carter.yaml` — the Carters are the moving "robots" the overhead cameras track; they cost ~14 FPS |
| E | Panoptex | `conda activate panoptex && ros2 launch risk_perception panoptex_sim.launch.py` |

| Argument | Default | |
|---|---|---|
| `map_yaml` | `~/workspace/warehouse/maps/warehouse_x3_nav.yaml` | the Isaac occupancy-exporter map (0.02–0.30 m band, stage frame). **Not** a SLAM map |
| `cams_config` | `src/.../config/global_cams_sim.yaml` | camera list + costmap extent + sim prompt |
| `use_sim_time` | `true` | everything, RViz included |
| `enable_localization` | `true` | map_server + AMCL via `robot_base.launch.py enable_base:=false`, params `config/amcl_sim.yaml` |
| `enable_global_cams` | `true` | `sim_global_cams.launch.py` |
| `enable_robot_cam` | `true` | the RGB-D chain (GDINO/SAM2/rgbd_projector/map_frame_projector) on Isaac's D455 topics |
| `enable_world_model` / `enable_costmap` / `enable_predictive_costmap` | `true` | grid nodes spawned here with the warehouse extent |
| `enable_spatial_prior` | `false` | `~/.panoptex/spatial_prior.npz` is a lab statistic; do not mix sim into it |
| `enable_rviz` / `rviz_config` | `true` / `rviz/panoptex_sim.rviz` | three per-camera raw-marker displays + segmentation feeds |
| `enable_nav2` | `false` | risk-aware Nav2 via `yahboomcar_nav`'s `navigation_dwa_launch.py`. **Turning it on turns `enable_localization` off** — Nav2's own bringup starts an identically-named `map_server` + `amcl` + `lifecycle_manager_localization`, and two AMCLs fight over `map → odom` |
| `nav2_params_file` | `config/nav2_sim.yaml` | `dwa_nav_params.yaml` + `nav2_risk_layer::RiskLayer` + `amcl_sim.yaml`'s amcl block, on sim time |
| `risk_layer_enabled` | `true` | `false` = the **baseline** ablation arm: stock Nav2, layer still loaded but writing no cost (so pluginlib, layer count and update cycle stay identical across arms) |
| `risk_topic` | `/risk_costmap` | reactive vs `/risk_costmap_predictive`. Pointing it at `/risk_costmap_reference` is refused — that grid is `evaluation_reference.launch.py`'s fixed scoring yardstick, and planning against it would collapse the study |

Those last two are the ablation knobs `tools/run_trials.py` varies per
`--condition-prefix`, so a condition never means hand-editing yaml:

```bash
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_layer_enabled:=false        # baseline
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_topic:=/risk_costmap        # reactive
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_topic:=/risk_costmap_predictive  # predictive
```

What differs from the lab stack, and why:

- **Localization** is `config/amcl_sim.yaml`: the warehouse repo's tuned block
  (`OmniMotionModel` — the sim X3 really strafes — 180 beams, 12 m A1 range,
  `sigma_hit 0.1`; 0.08 m mean / 0.23 m max over its 90 s scripted drive) with
  the initial pose **baked in** at the X3's spawn (0.38, 0.07, yaw 0), since
  there is no tag 0 to seed from. If the X3 was driven before terminal E
  starts, click "2D Pose Estimate" once.
- **Depth format.** Isaac publishes depth at 320×240 `32FC1` metres while
  colour and masks are 640×480; `rgbd_projector` now nearest-neighbour
  upsamples depth onto the mask grid when the ratio is uniform (logged once),
  instead of rejecting every frame. The colour `CameraInfo` still applies:
  same camera, same FOV, only the sampling grid differed.
- **Self-exclusion** works through TF (`map → base_footprint` from AMCL) —
  no localizer, no tag 0 — so the X3 never paints risk around itself, while
  the Nova Carters are tracked as `ground mobile robot`.
- **Nav2.** Off by default, wired by `enable_nav2:=true` against
  `config/nav2_sim.yaml` rather than the lab's `dwa_nav_params.yaml` (sim AMCL
  tuning, sim time, and the same `risk_layer` block); `risk_layer_enabled:=`
  and `risk_topic:=` are the ablation knobs `tools/run_trials.py` varies, and
  this is the Nav2 whose `/plan` `evaluation_node.py` counts replans on.
  `enable_nav2:=true` also turns this file's own AMCL off — Nav2's bringup
  starts its own, and two AMCLs would fight over `map → odom`. The grids are
  `TRANSIENT_LOCAL`, in `map`, at the warehouse extent, which is what let
  `nav2_risk_layer` drop in without rework (`risk_layer.cpp` resamples by
  world coordinates). Only the **global** costmap carries the layer: the local
  costmap's `global_frame` is `odom`, and `risk_layer.cpp` rejects any grid
  whose `frame_id` differs from the costmap's global frame.
  `panoptex_nav`'s `x3_nav.launch.py` (§8) is the other way in, and builds on
  exactly this launch file — `enable_localization:=false`, since that launch's
  own AMCL bringup owns `map → odom` instead — to run the two-arm
  baseline-vs-Panoptex Nav2 study.

What you see, and where: in **RViz** (fixed frame `map`) the static map, the X3
with its `/scan` on the walls, green spheres for what the RGB-D chain found,
orange spheres per overhead camera, the fused tracks (one marker per object
however many cameras see it) and the graded `/risk_costmap` blob under each.
The Isaac viewport shows the robots moving; Panoptex draws nothing back into
Isaac — run the two windows side by side.

Checks, headless or GUI:
```bash
ros2 run tf2_ros tf2_echo map base_footprint        # AMCL: ≈ (0.38, 0.07) at spawn
ros2 topic hz /risk_perception/detections_3d_map    # both sources land here
ros2 topic info /risk_costmap -v                    # TRANSIENT_LOCAL, 200×304 @ 0.10 m
# ground truth without driving: the five standing people + the parked Carters
python3 tools/isaac_sim/check_sim_tracks.py --seconds 60 \
  --extra AMR_01:1.265:1.4315 --extra AMR_02:6.5625:5.0 --robot-xy 0.38:0.07 \
  --ros-args -p use_sim_time:=true
```
`check_sim_tracks.py` reads the people's stage positions from
`warehouse/carters/tests/people_positions.json` and reports, per object, the
nearest track's label and distance; the pass bar is one track per object within
0.5 m (the projector's own ~0.3 m σ) and no track at the robot. Then drive the
X3 toward a person and the same track should gain a green (RGB-D) marker.

**Verified 2026-09-05, headless** (Isaac `run_headless.py` + terminals B and E,
`enable_rviz:=false`, Carters parked, RTX 3090; full log set in the session
scratchpad). What passed: one `/clock` publisher; AMCL at (0.380, 0.071, 0.000 rad)
on the baked pose and **0.083 m / 0.026 rad** from sim odometry after a closed-loop
90° turn + 1.5 m drive; all three `map → global_cam_*_optical_frame` TFs equal the
exported yamls; `CameraInfo` 648×486 with fx 528.15; all eight model instances
loaded; `/risk_costmap` RELIABLE + TRANSIENT_LOCAL, 200×304 @ 0.10 m from
(-10.4, -12.3); the depth resize logged exactly once and the RGB-D chain then
projected `person: 6.06 m` after the drive; zero tracebacks, zero
`Cannot transform`. Ground truth (`check_sim_tracks.py`, 0.5 m tolerance):

| object | static run (60 s) | after the drive (40 s) | why |
|---|---|---|---|
| doorman | person, **0.03 m** | 0.02 m | in `entry`'s view |
| packageman | person, **0.06 m** | 0.07 m | 15.1 m from `entry` — right on the 15 m range gate |
| person1a | person, **0.18 m** | 0.04 m | column 643 of 648 in `mid` — one pixel from off-frame |
| person2a, person3a | no track | person, **0.10 / 0.18 m** | outside every overhead view (off `mid`'s right edge, 26 m from entry/exit); found by the RGB-D chain only |
| AMR_02 | cart, **0.10 m** | 0.22 m | in `exit`'s view |
| AMR_01 | **no track** | no track | 14.4 m from `mid`, >15 m from the others — invisible with the current placement |
| X3 | no track within 0.6 m | same | expected — but self-exclusion never fired either (nothing projected near the X3), so the gate is still unexercised |

Every matched object is within 0.02–0.22 m of its stage position, inside the
projector's ~0.3 m σ. Two things to know before trusting the fused layer:

- **Overhead and RGB-D tracks of the same person do not merge** (person1a: two
  tracks 0.14 / 0.18 m from truth; person2a: three). `object_tracker` associates
  only *same-label* detections (`tr.label != d.label` → new track) and GDINO's
  open-vocabulary output concatenates phrases (`person` vs `person ground mobile
  robot` vs `person ground`), so the split is a label-normalisation gap in the
  tracker, not a projection error. Same cause behind the 20–40 extra
  racking/shelf tracks per run (`table` / `ground mobile robot forklift` pairs at
  identical x,y). Next tuning item; the plan's gate ladder does not fix a label
  mismatch.
- **Rates with everything on one GPU**: sim/wall **0.30x** (0.48x without the
  models), VRAM **17.2 GB / 24.6**, `/risk_costmap` 1.5 Hz, GDINO ~0.5 Hz per
  camera. Budget ~3.3 s wall per sim second for any scripted drive.

Not verified: RViz and the `panoptex_sim.rviz` displays, the GUI session, the
Carter patrol (nothing moved, so velocity/predictive behaviour is untested),
the `/risk_costmap` cell contents, `/risk_costmap_predictive`, long runs.
Re-aim or move `GlobalCameraMid` (person2a/3a, AMR_01 are just outside it) in
the GUI, then re-run the wiring command above — the extrinsics follow the prim.


### 5.4 The individual launch files

`panoptex_pc.launch.py` is a thin wrapper over these three; run them directly
when you want one piece in its own terminal.

**Terminal A — robot base + SLAM** (mapping only; see §6 Step 2)
```bash
ros2 launch risk_perception robot_base.launch.py
```
Brings up the Yahboom X3 base (URDF tf, driver, IMU, EKF → `odom → base_footprint`)
and slam_toolbox (`map → odom`, with `use_sim_time:=false` baked in). Wait until
`map → odom` is live before starting Terminal B/C.

**Terminal B — robot camera + perception**
```bash
conda activate panoptex
ros2 launch risk_perception risk_perception.launch.py
```
Brings up the RealSense/Astra RGB-D and the full chain: gdino_detector →
sam2_segmenter → rgbd_projector → map_frame_projector → object_tracker →
risk_costmap. Add `enable_rviz:=true` to also open RViz preloaded with
`rviz/panoptex.rviz` (map, costmap, fused/raw markers, both segmentation
image feeds).

**Terminal C — overhead camera + perception (needs a completed calibration, §6)**
```bash
conda activate panoptex
ros2 launch risk_perception global_cam.launch.py
```
Brings up global_cam_bridge (TCP:5000 → `/global_cam/image_raw`), apriltag_ros,
global_cam_calibrator (floor tags → TF `map → global_cam_optical_frame`),
global_cam_localizer (tag 0 → `/global_cam/robot_pose`), and a second
gdino_detector/sam2_segmenter/global_cam_projector chain publishing onto the
same `/risk_perception/detections_3d_map` topic Terminal B uses —
object_tracker fuses both cameras automatically. Images are fed to
apriltag_ros RAW, not rectified — the calibration/localization nodes run
`solvePnP` with the full distortion coefficients themselves, which is only
valid on unrectified pixels. **§6 must be completed first** — the committed
`config/floor_tags.yaml` is a placeholder and produces detections offset
from the real map.

Useful launch overrides:
```bash
# point at a different weights/config location than setup.sh's defaults
ros2 launch risk_perception risk_perception.launch.py \
  gdino_config:=/path/to/GroundingDINO_SwinT_OGC.py \
  gdino_weights:=/path/to/groundingdino_swint_ogc.pth

# verify detections without the costmap stage
ros2 launch risk_perception risk_perception.launch.py enable_costmap:=false

# base + AMCL against a saved map, instead of building a fresh one with SLAM
# (forces slam off automatically -- see §6)
ros2 launch risk_perception robot_base.launch.py map_yaml:=$HOME/maps/lab.yaml

# AMCL only, no base driver -- for running localization on the PC while the
# base and lidar run on the Jetson (what panoptex_pc.launch.py does)
ros2 launch risk_perception robot_base.launch.py \
  enable_base:=false map_yaml:=$HOME/maps/lab.yaml
```

> With `output="screen"`, all the perception nodes in a launch file interleave
> in that terminal's log. Filter when debugging one:
> `ros2 launch ... 2>&1 | grep -i gdino`. A single node crashing leaves the
> rest running — look for the Python traceback rather than assuming the whole
> stack died.

### Manual / debugging (one node per terminal)

When you need to isolate a stage, run it directly instead of via the launch
file. Each command below is the ground truth for what that node needs
(`conda activate panoptex` first, for any node importing torch):

```bash
# camera
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true enable_sync:=true \
  align_depth.enable:=true pointcloud.enable:=false

# 1. GroundingDINO detector
ros2 run risk_perception gdino_detector --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p config_path:=$CONDA_PREFIX/src/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p checkpoint_path:=$HOME/workspace/Panoptex/weights/groundingdino_swint_ogc.pth \
  -p prompt:="person . ground mobile robot . cart . chair . table . monitor . camera . cable . " \
  -p box_threshold:=0.35 -p text_threshold:=0.25 -p inference_stride:=3 -p device:=cuda

# 2. SAM2 segmenter
ros2 run risk_perception sam2_segmenter --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p detections_topic:=/risk_perception/detections_2d \
  -p model_config:="configs/sam2.1/sam2.1_hiera_s.yaml" \
  -p checkpoint_path:=$HOME/workspace/Panoptex/weights/sam2.1_hiera_small.pt \
  -p device:=cuda -p image_cache_size:=120 -p overlay_alpha:=0.45

# 3. RGB-D projector
ros2 run risk_perception rgbd_projector --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p detections_topic:=/risk_perception/detections_2d \
  -p mask_topic:=/risk_perception/instance_mask \
  -p output_topic:=/risk_perception/detections_3d \
  -p depth_scale:=0.001 -p sync_tolerance_sec:=0.05 -p cache_size:=180

# 4. Map-frame projector  (needs tf: map → camera)
ros2 run risk_perception map_frame_projector --ros-args \
  -p input_topic:=/risk_perception/detections_3d \
  -p output_topic:=/risk_perception/detections_3d_map \
  -p marker_topic:=/risk_perception/map_markers \
  -p target_frame:=map -p transform_timeout_sec:=0.25

# 5. Object tracker (world model, fuses both cameras)
ros2 run risk_perception object_tracker

# 6. Risk costmap
ros2 run risk_perception risk_costmap
```

### 5.5 Research tools — synthetic scenarios, prior logging, evaluation trials

Everything here is dev/research tooling around the predictive risk stack
(Stage 2/3/4) and the Spatial-Flow prior — none of it is needed for §5.1's
day-to-day operation.

**Synthetic scenarios, no camera/robot/bag needed.** `scenario_publisher.py`
publishes exactly what `object_tracker_node` publishes
(`/risk_perception/world_objects` + `/odom` + `map → base_footprint`), so
`predictive_risk_costmap_node`/`spatial_prior_node` can be exercised against
known geometry while the perception chain — the part most likely broken on
any given day — is left out of the loop entirely:
```bash
python3 tools/scenario_publisher.py --scenario head_on
```
Other scenarios: `crossing`, `parallel`, `robot_forward`, `corridor`,
`circular`, `zigzag`, `crowd`, `relation_test`, `multi`, `twins`, `mutual`
(`--help` documents each). Run alongside `risk_perception.launch.py
enable_world_model:=false enable_robot_cam:=false enable_global_cam:=false`
so the tracker's real output doesn't fight the synthetic one.

**Prior research logging.** `panoptex_sim.launch.py` and
`risk_perception.launch.py` both take `prior_log_dir` (a directory; each of
`object_tracker`, `spatial_prior` and `predictive_risk_costmap` writes its
own `<node>_<UTC>.csv` there, never clobbering) and `tracker_log_unconfirmed`
(also logs tracks below `min_hits`/`min_confidence`):
```bash
ros2 launch risk_perception panoptex_sim.launch.py \
  enable_predictive_costmap:=true enable_spatial_prior:=true \
  prior_log_dir:=~/.panoptex/logs tracker_log_unconfirmed:=true
```
Then read what landed in that directory — per-prior percentile tables and
(if matplotlib works) histograms/timelines:
```bash
python3 tools/prior_report.py --log-dir ~/.panoptex/logs \
  --static-labels chair,table,pallet
```
`spatial_prior_node` autosaves its own persisted `.npz` (`autosave_period_sec`
in `risk_perception.yaml`, default 60 s, or a clean Ctrl-C — not `kill`) —
render it any time, without stopping the live system:
```bash
python3 tools/spatial_flow_heatmap.py --npz ~/.panoptex/spatial_prior.npz \
  --out out/spatial_flow
```

**Evaluation trials (ablation studies).** Risk exposure must be scored
against a costmap that is identical across every condition, so
`evaluation_reference.launch.py` runs a second, fixed full-system
`predictive_risk_costmap_node` publishing `/risk_costmap_reference` —
alongside, never wired to Nav2, never a valid `risk_topic:=` planning input.
The three arms and their knobs (`risk_layer_enabled`, `risk_topic`) are
`panoptex_sim.launch.py`'s (§5.3c) — this whole workflow assumes the sim
stack; on the lab stack the same two knobs live in `panoptex_pc.launch.py`'s
own `nav2_params_file` instead.

*Terminals, in order* (A–E are §5.3c's bring-up, unchanged and left running
across every arm):

| # | Terminal | Command |
|---|---|---|
| A–D | Isaac + X3 + (optional) Carters | see §5.3c |
| E | Nav2 for the arm under test | `ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true <arm's flags>` |
| F | fixed scoring reference — **same every arm, never restart between arms** | `ros2 launch risk_perception evaluation_reference.launch.py` |
| G | bag recording — start **before** G, stop **after** the trial finishes | `ros2 bag record -o results/<label>_bag <topics below>` |
| H | drives the waypoint tour, only after G is confirmed recording | `python3 tools/run_trials.py ...` |

```bash
# Terminal E, one line per arm (baseline first -- see "start here" below)
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_layer_enabled:=false        # baseline
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_topic:=/risk_costmap        # reactive
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true risk_topic:=/risk_costmap_predictive  # predictive
```

Record a run and score it against the Efficiency/Safety/Latency/Clearance
metrics in `evaluation_metrics.py`. Minimum topic set for scoring, plus two
optional ones worth adding whenever you're also debugging *why* a trial
failed rather than just scoring one that worked (`/cmd_vel` shows whether the
controller ever issued a nonzero command; `/local_costmap/costmap` shows
whether the robot's own footprint is boxed in by lethal cost):
```bash
ros2 bag record -o results/<label>_bag /tf /tf_static /odom \
  /risk_perception/world_objects /risk_perception/detections_2d \
  /risk_costmap_reference /plan /cmd_vel /local_costmap/costmap
python3 tools/evaluate_run.py results/<label>_bag --label <label> \
  --out results/<label>.json
```
`-o` refuses to reuse an existing folder — pick a fresh `<label>` per attempt
(or `rm -rf` the old one) rather than re-running the same name.

**Sim-time gotcha, already handled by `evaluate_run.py`:** `ros2 bag record`
always timestamps messages by wall-clock arrival time, regardless of
`use_sim_time` — but every message's own `header.stamp` (what the tf2 buffer
is keyed by) is sim time. `evaluate_run.py` reads each message's
`header.stamp` rather than the bag's arrival stamp for exactly this reason;
if a future edit ever reintroduces the bag's own timestamp for sampling,
every `lookup_transform()` call will fail outright ("no valid map->base_frame
tf found for any sample") since the two time bases differ by ~8 orders of
magnitude. No special flag is needed on `ros2 bag record` itself.

**Nav2's own console output is easy to miss:** `navigation_dwa_launch.py`
does not set `output="screen"` for `controller_server` / `planner_server` /
`bt_navigator`, so their logs never reach terminal E — only per-node log
files under the directory `ros2 launch` prints at startup
(`All log files can be found below ...`):
```bash
tail -f ~/.ros/log/<that run's dir>/controller_server-*.log
tail -f ~/.ros/log/<that run's dir>/planner_server-*.log
tail -f ~/.ros/log/<that run's dir>/bt_navigator-*.log
```
This is where an actual navigation failure (stuck-replanning loop, lethal
inflation at the robot's own footprint, controller rejecting every plan)
shows up — a `TIMEOUT` leg outcome from `run_trials.py` with no console
errors almost always means the error is sitting in one of these files, not
that nothing went wrong.

**Start here: one baseline smoke test before the real trials.** A single
short trial (1-2 waypoints) on the **baseline** arm validates the whole
pipeline — localization, bag recording, `evaluation_reference`, the
`trial_control` protocol — without needing either risk costmap to be
correct, since baseline's layer writes no cost. Confirm the result isn't
degenerate before trusting a longer run: `path_length_m` should be well
above zero (near-zero means the robot never actually moved — check the Nav2
log files above), and `stopped_time_s` should be well under the trial's
duration. Only once that looks sane, move on to the real trial config and
all three arms; re-run baseline for real (a smoke test's bag is for
pipeline-checking, not for the actual results table).

`run_trials.py` automates the waypoint-tour loop end to end — repeated once
per condition label in a JSON trial config, paired with `evaluation_node`'s
live `/evaluation/trial_control` → `/evaluation/trial_result` protocol,
optionally owning the Isaac Sim process lifecycle itself
(`--isaac-launch-cmd`, relaunched fresh per trial). It does **not** record a
bag itself — Terminal G above must already be recording before it starts:
```bash
python3 tools/run_trials.py trials.json --condition-prefix baseline \
  --out-dir results/baseline --ros-domain-id 55
```
Every process in the pipeline — Isaac's ROS 2 bridge, `risk_perception`,
`evaluation_reference`, `evaluation_node`, and this script — must share one
`ROS_DOMAIN_ID`; a node on the wrong domain sees nothing and raises no error.

---

## 6. Overhead camera calibration — one-time

### What this does, and why it only happens once

The overhead camera must express its detections in the robot's `map` frame.
The three floor tags give it a rigid frame of its own — but that frame's
origin is tag 1, not SLAM's map origin (wherever the robot booted). Tag 0 on
the robot bridges them: SLAM reports the robot's pose in `map`, the camera
sees tag 0. One observation in both frames closes the gap.

It stays valid as long as the map origin doesn't move — which is why you
reload a **saved** map from here on rather than re-running SLAM.

### Step 0 — print the tags

Skip if they're already on the floor.

```bash
# floor tags 1/2/3, and a LARGER tag 0 for the robot
python3 tools/make_apriltag.py --ids 1 2 3 --size-mm 115 --out tags/
python3 tools/make_apriltag.py --ids 0     --size-mm 150 --out tags/
```

Writes one SVG per tag. **Print at 100% / "Actual size", never "Fit to
page"** — that silently rescales and there is no way to tell from the image.
`--size-mm` is the outer edge of the black square, the same quantity as
`tag_size_m`/`tag0_size_m`; each tag prints its intended size as a caption so
it can be re-checked later.

Make **tag 0 bigger than the floor tags**. It is the only tag that moves, so
it is the one that fails: it needs ~40 px of edge in the overhead image to be
reliable, it shrinks with range as the robot drives away, and it is the only
tag that suffers motion blur. At ~1050 px focal length and 4 m range, 150 mm
gives ~39 px where 115 mm gives only ~30 px.

Then **measure what actually came out of the printer** and put that number in
`config/floor_tags.yaml`. If tag 0 differs from the floor tags — it should —
record it separately as `tag0_size_m`.

### Before you start

| Check | How |
|---|---|
| Tag size matches config | Measure the outer edge of the black square. The three floor tags must equal `tag_size_m` in `config/floor_tags.yaml` (0.115). **If tag 0 is a different size, set `tag0_size_m` alongside it** — it is often printed larger to stay readable across the room. Printers rescale by a few % too. A wrong size is a scale error no camera pose can absorb, and is the #1 cause of a bad solve. Note `sizes:` in `apriltag.yaml` does *not* cover this: it only feeds apriltag_ros's own pose estimate, which nothing in this pipeline consumes. |
| 3 floor tags flat, spread, all visible | They must form a clear triangle (not near-collinear) in the overhead view. Tape them flat — a curled corner shifts the PnP. |
| Tag 0 flat and level on the robot | Put its x/y offset from the robot's centre and its yaw vs. robot-forward into `tag0_to_base_footprint`. Height doesn't matter (solvePnP recovers full 3D). Centred ⇒ x=0, y=0. |
| Intrinsics match stream resolution | `config/global_cam_intrinsics.yaml` is 1296×972; the Pi must stream 1296×972. |
| Stream alive | `ros2 topic hz /global_cam/image_raw` |

### Step 1 — bring up the robot (on the Jetson)

All three of these run **on the robot**, over SSH. Every terminal on both
machines needs `ROS_DOMAIN_ID=77` — if the PC can't see the robot's topics,
check this first.

```bash
# J1 — base + lidar
ros2 launch yahboomcar_nav laser_bringup_launch.py

# J2 — SLAM (this is what owns map -> odom)
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/home/jetson/yahboomcar_ros2_ws/yahboomcar_ws/src/yahboomcar_nav/params/slam_toolbox_params.yaml

# J3 — RGB-D camera. NOT needed for calibration; start it later, for §7.
ros2 launch astra_camera astro_pro_plus.launch.xml depth_registration:=true
```

Confirm from the PC before going further — this must print advancing stamps:
```bash
ros2 run tf2_ros tf2_echo map base_footprint
```

### Step 2 — build and save the map

```bash
# P1 — drive
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Drive the whole area, including under the overhead camera. Then, **without
stopping SLAM**:
```bash
mkdir -p ~/maps
ros2 run nav2_map_server map_saver_cli -f ~/maps/lab
```

### Step 3 — calibrate, in the SAME session

Do not restart SLAM between steps 2 and 3. The alignment is only meaningful
against the exact map origin baked into `~/maps/lab.yaml`.

```bash
# P2 — overhead camera + AprilTags only
conda activate panoptex
ros2 launch risk_perception global_cam.launch.py enable_perception:=false
```

> **Do not run `risk_perception.launch.py` during calibration.** Its
> `gdino_detector`, `sam2_segmenter`, `rgbd_projector` and
> `map_frame_projector` nodes are *unconditional* — `enable_camera:=false`
> only skips the RealSense driver, not the GPU models. They compete for CPU
> with `apriltag_node` and slow the overhead stream from ~11 fps to ~3.5,
> which is what makes each capture take 6 s instead of 2. For a map view
> during calibration, run RViz on its own:
> ```bash
> ros2 run rviz2 rviz2 -d $(ros2 pkg prefix risk_perception)/share/risk_perception/rviz/panoptex.rviz
> ```
Now confirm the camera can see all four tags. Rather than repeatedly running
`ros2 topic echo /global_cam/apriltag/detections --once` and scrolling for
tag 0, use the live readout:
```bash
# P3 — rewrites one line in place; leave it up for the whole calibration
ros2 run risk_perception global_cam_tag_monitor
```
```
tag0 OK  28px | tag1 OK  49px | tag2 OK  35px | tag3 OK  36px  ->  ALL 4 (steady 4.2s)
```
`steady` is how long *all four* have been continuously visible — the number
that matters, since a capture needs tag 0 held for its full ~6 s. It also
publishes `/global_cam/apriltag/overlay`, drawn from apriltag_ros's own
detections (so a tag missing there is genuinely missing from the pipeline),
colour-coded by edge size: green ≥40 px, amber ≥20 px (flaky), red below.
```bash
ros2 run rqt_image_view rqt_image_view /global_cam/apriltag/overlay
```

**Tag 0 is the one that drops out.** It is the smallest tag and the only one
that moves, so it fails first — under motion blur whenever the robot is
driving, and by range at the far end of the workspace. Park within reach of
the camera: prefer spots where the monitor shows tag 0 at ≥40 px.
```bash
# P4 — the alignment tool.
# Note the src/ path, NOT the installed share copy, so the result is committable.
# Do NOT add solve_from here -- that is replay-only, see Step 5.
conda activate panoptex
ros2 run risk_perception global_cam_map_align --ros-args \
  -p floor_tags_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/floor_tags.yaml
```
The node runs in **manual mode** by default: it captures only when you tell
it to, so no in-motion frames reach the solve.

```
>>> 0 pose(s) captured. Park the robot at the next spot, then press ENTER to capture:
```

Drive to a spot, **stop**, press ENTER, and wait for `pose N accepted`. Repeat
for **at least 3 well-separated spots** forming a wide triangle — not a
straight line — varying the robot's *heading* as well as its position. All of
tag 0's corners lie in a single plane at constant height, a weakly-conditioned
geometry; spreading the poses is what makes the solve stable. 5-6 spots costs
nothing and helps. Once you have `min_poses`, type `s` + ENTER to solve.

Each capture takes ~6 s (20 samples at the overhead camera's ~3.5 Hz). A
capture is **discarded**, with a reason, if the robot drifted more than
`max_motion_during_pose_m` (2 cm) or if the spot is within
`min_pose_separation_m` of one already captured — just re-park and press
ENTER again. If your SLAM pose jitters enough to reject good captures, raise
`-p max_motion_during_pose_m:=0.03`.

> `-p manual_trigger:=false` restores the old behaviour, where the node
> infers "parked" from a >`min_pose_separation_m` jump. That mode keeps
> sampling while you drive away (anything under the threshold still counts as
> the current pose), which mixes moving frames into the solve — the overhead
> bridge stamps each frame on *arrival*, not capture, so a moving robot's TF
> lookup resolves slightly ahead of where the photo was actually taken. Both
> problems disappear when the robot is stationary.

### Step 4 — read the output

```
RMS reprojection error: 1.8 px            <- want < 5
camera position in map: (2.10, 1.35, 2.41) m   <- z is height above the floor, compare against a tape measure
  tag_1: x=0.3329 y=0.5334 z=+0.0031 yaw=+154.5 deg (range 2.67 m, 30 samples, fit 0.31 px)
  ...
distance check (solved vs. your tape):
  d_1_2: solved 1.1287 m  vs  measured 1.1350 m   (off by -0.6 cm)
  ...
wrote .../config/floor_tags.yaml  (backup -> floor_tags.yaml.bak)
```
Read the checks in this order — they fail in a cascade, and only the first one
tells you anything:

1. **RMS reprojection error** (< 5 px). This is the tag-0 solve itself. ~28 px
   ⇒ `tag0_to_base_footprint`'s `yaw` is 90° out. Tens of px ⇒ in-motion
   frames got in, or the offset's x/y is wrong.
2. **Floor-tag `z`** (≈ 0). Nothing constrains it — the tags really are on the
   floor — so it is the sharpest check on the camera's solved *rotation*. A
   tag tens of cm off the floor means R is tilted, and everything below is
   meaningless.
3. **Camera height** (`z` of the camera position). Also never an input, so if
   it matches a tape measure the solve is sound.
4. **Distance check** (within 2 cm). Weakest and most easily misread: it
   compares *horizontal* distances, so a tilted solve shrinks every pair by
   `cos(tilt)` and reads like a scale error. Only trust it once 1-3 pass. If
   they do and this still fails, re-check `tag_size_m` against the actual
   printed tag, then `tag0_to_base_footprint`.

Note a constant error in `tag0_to_base_footprint` shifts every solved position
equally, which leaves the distance check *passing* — so a green distance check
is not evidence the mounting offset is right.

When RMS exceeds the threshold the node also prints three diagnostics that
turn each guess into a measurement — a **mounting-yaw scan** (all four 90°
options, scored), a **tag-0 size scan**, and an **x/y refinement**. Each is
reported, never applied: change the number in `floor_tags.yaml` yourself, then
re-solve with Step 5 — no re-driving.

### Step 5 — re-solve without driving again

The solve is pure maths over the raw observations, so the node banks them to
`config/align_captures.npz` on every successful run. Changed
`tag0_to_base_footprint`, `tag0_size_m` or `tag_size_m`? Re-solve in about a
second, with no robot, no camera and no SLAM running:

```bash
ros2 run risk_perception global_cam_map_align --ros-args \
  -p floor_tags_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/floor_tags.yaml \
  -p solve_from:=$HOME/workspace/Panoptex/src/risk_perception/config/align_captures.npz
```

Only re-drive if the camera moved, a floor tag moved, or you built a new map —
the captures are tied to that map's origin.

### Step 6 — verify the frame

```bash
ros2 run tf2_ros tf2_echo map global_cam_optical_frame
```
Should be stable, with a translation matching where the camera physically sits
relative to the map origin. In RViz (fixed frame `map`) the `tag1`/`tag2`/
`tag3` frames should land where the tags physically are.

Then check the calibration in **metres**, which is the unit that actually
matters — pixels of reprojection error don't translate directly into map
error:

```bash
ros2 run risk_perception global_cam_align_check
```
It compares the overhead camera's own estimate of the robot's pose against
SLAM's `map -> base_footprint` and prints the gap live. Drive the whole
working area and judge the **worst** value: under 5 cm is good, under 15 cm is
usable (still well inside `object_tracker`'s 0.6 m association gate, so both
cameras fuse into single tracks), above that re-run with wider-spread poses.
For scale, `global_cam_projector` already declares `base_cov_m2: 0.10` for its
own object positions — a ~32 cm standard deviation — so a few cm of extrinsic
error is not this pipeline's limiting term.

### Step 7 — commit

`config/floor_tags.yaml` now holds map-frame poses. Commit it alongside the
saved map. Delete `floor_tags.yaml.bak`.

### Step 8 — cache the extrinsic (optional, but it frees the floor tags)

Everything above solves where the **tags** are. `global_cam_calibrator` then
re-solves where the **camera** is on every frame, which is why the tags have
to stay taped down and visible, and why `apriltag_node` keeps burning CPU the
GPU models want. Since the camera is bolted in place, that answer is a
constant and can be cached:

```bash
ros2 launch risk_perception global_cam.launch.py enable_perception:=false \
  extrinsic_yaml:=$HOME/workspace/Panoptex/src/risk_perception/config/global_cam_extrinsic.yaml
```

The calibrator writes `config/global_cam_extrinsic.yaml` every 30 solves, but
**only** for solves at or under `max_reprojection_error_px` (5 px) — a bad
solve on disk is worse than no file, because the next run starts from it
silently. Wait for `Saved extrinsic to ...`, Ctrl-C, rebuild, commit the file.

From then on `use_saved_extrinsic:=true` broadcasts `map →
global_cam_optical_frame` from startup with no tags in view, and
`enable_apriltag:=false` drops the detector entirely. A live solve still wins
whenever tags are visible, so this is a cold-start seed rather than a mode.
`bench_global_cam.launch.py` (§5.3) is built on exactly this.

⚠️ The cached extrinsic is tied to the same map origin and the same
`floor_tags.yaml` as everything else in this section — its header records
both. **Re-capture it whenever you re-run Steps 2–7, or bump the camera.**

### Every session after this

⚠️ **`slam_toolbox online_async_launch.py` builds a NEW map every time it
starts, with a new origin — which silently invalidates this calibration.**
The floor-tag poses in `floor_tags.yaml` are expressed in one specific map's
frame. From here on, localize against the **saved** map instead of re-mapping.

Two ways to do that; run exactly one, because only one node may publish
`map → odom`:

```bash
# either — this repo's wrapper, on the robot. Setting map_yaml forces slam
# off and brings up map_server + amcl + lifecycle_manager in its place.
ros2 launch risk_perception robot_base.launch.py map_yaml:=$HOME/maps/lab.yaml

# or — the yahboomcar navigation launch, in place of J2 above, pointed at
# the saved map. (Check the arg name against your yahboomcar_nav version.)
```

Then set the initial pose (RViz "2D Pose Estimate", or drive a little to let
AMCL converge) and confirm `map → odom` is live before starting anything else.

Bring-up order for a normal (non-calibration) session:

| # | Where | Command |
|---|---|---|
| 1 | Jetson | base + lidar, then localization against the saved map |
| 2 | Jetson | `ros2 launch astra_camera astro_pro_plus.launch.xml depth_registration:=true` |
| 3 | PC | `ros2 launch risk_perception global_cam.launch.py` (no `enable_perception:=false` — you want the models now) |
| 4 | PC | `ros2 launch risk_perception risk_perception.launch.py enable_camera:=false enable_rviz:=true` |

`enable_camera:=false` in step 4 because the RGB-D camera is driven on the
robot by the Astra driver, not by RealSense on the PC.

### Re-run the calibration only if

the camera is moved or bumped, a floor tag moves, or you build a new map. Any
of those also invalidates the cached extrinsic from Step 8 — re-capture it too.

> No robot available yet? `global_cam_survey_node` (`ros2 run risk_perception
> global_cam_survey`) solves the same three floor tags from tape-measured
> distances alone — enough to check tag IDs and geometry, but its output frame
> is defined by the tags, not SLAM's `map`, so it still needs this alignment
> before the overhead camera's detections will land correctly on the RViz map.

---

## 7. Verify

### What success looks like

The goal is risk-aware navigation: a path that routes **around** a person it
would otherwise cut through. Lidar already avoids collisions — the point here
is that a *person* costs more to pass close to than a *chair*, and that the
overhead camera contributes obstacles the robot **cannot see itself** (around
a corner, behind furniture, outside its FOV). That is the whole reason for a
second camera with no depth.

Build up to it in this order; each step is meaningless if the one above it
fails.

| # | Check | Looks like |
|---|---|---|
| 1 | Localization | `seeded AMCL at (x, y) yaw θ` in the log; the RobotModel where the robot really is; the red laser scan lying **on** the black map walls, not rotated or offset |
| 2 | Overhead extrinsic | `global_cam_optical_frame` steady in the map; `tag1`/`tag2`/`tag3` frames land where the tags physically are |
| 3 | Overhead objects | Put a chair in the overhead view → a marker in `Fused Overlay` at its true floor position |
| 4 | **Dual-camera fusion** | An object both cameras can see makes **one** marker, not two. This is the payoff of §6's calibration |
| 5 | Risk field | Tick `Risk Costmap` → a graded blob under each object, hotter for people than chairs |
| 6 | Risk-aware path | Send a goal past a person → the green plan bows around them |

Steps 4-6 have not yet been demonstrated end-to-end on hardware. Known
blockers, in the order they will bite:

- **The robot detects itself.** The GDINO prompt includes `ground mobile
  robot` and nothing filters detections at the robot's own pose, so the
  overhead camera tracks the robot as an obstacle and Nav2 plans around it.
  Looks like broken navigation; isn't.
- **RGB-D timestamp matching.** `No timestamp-matched aligned depth frame`
  plus `extrapolation into the past` means the robot camera is producing no
  detections — so step 4 cannot pass, only overhead markers appear.
- **Risk-layer plugin load.** If pluginlib cannot find
  `nav2_risk_layer::RiskLayer`, the global costmap never activates and Nav2
  does not come up at all. Check the `controller_server`/`planner_server`
  startup log before blaming a goal that does nothing.

If a 2D Goal Pose does nothing, check first that anything is listening:
```bash
ros2 topic info /goal_pose      # Subscription count must be >= 1
```
Zero subscribers means Nav2 is not running — RViz published into the void.

### Commands

```bash
# tf — map must be a single root with advancing (not frozen) stamps
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_tools view_frames            # writes frames.pdf

# camera ~30 Hz
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw

# pipeline (put a prompt-matching object, e.g. a chair, in view)
ros2 run rqt_image_view rqt_image_view /risk_perception/detection_image  # what GDINO sees
ros2 topic echo /risk_perception/detections_3d_map --once                # map-frame output
```

RViz: `ros2 launch risk_perception risk_perception.launch.py enable_rviz:=true`
opens `rviz/panoptex.rviz` preloaded (fixed frame `map`) with TF, `/map`,
`/risk_costmap`, the fused `/risk_perception/world_markers` overlay, and (off
by default, toggle in the Displays panel) the two cameras' raw markers
(`/risk_perception/map_markers`, `/global_cam/map_markers`) and segmentation
image feeds. Detection markers use `DELETEALL` + 1 s lifetime, so they clear
within a second when nothing is detected — expected, not a bug.

### tf tree (healthy = single root)
```
map                       ← slam_toolbox, or amcl if reloading a saved map (or ekf_global, if enabled -- see §10)
├── odom                  ← ekf_filter_node
│   └── base_footprint
│       └── base_link     ← robot_state_publisher (URDF)
│           ├── camera_link → camera_color_frame → camera_color_optical_frame  ← robot-cam detections born here
│           ├── laser / laser_link / imu_link
│           └── wheels ×4
└── global_cam_optical_frame  ← global_cam_calibrator (solved from floor tags 1/2/3) -- overhead-cam detections born here
```

### Topics
| topic | type | from |
|---|---|---|
| `/risk_perception/detections_2d` | Detection2DArray | gdino_detector (robot cam) |
| `/risk_perception/instance_mask` | Image (mono16) | sam2_segmenter (robot cam) |
| `/risk_perception/detections_3d` | Detection3DArray (cam frame) | rgbd_projector |
| `/risk_perception/detections_3d_map` | Detection3DArray (map) | map_frame_projector **and** global_cam_projector (shared -- this is where the two cameras fuse) |
| `/risk_perception/map_markers` | MarkerArray | map_frame_projector |
| `/risk_perception/world_objects` | Detection3DArray (map) | object_tracker (confirmed, fused tracks) |
| `/risk_perception/world_markers` | MarkerArray | object_tracker |
| `/risk_costmap` | OccupancyGrid | risk_costmap_node |
| `/global_cam/image_raw`, `/global_cam/camera_info` | Image, CameraInfo | global_cam_bridge |
| `/global_cam/apriltag/detections` | AprilTagDetectionArray | apriltag_ros |
| `/global_cam/detections_2d`, `/global_cam/instance_mask` | Detection2DArray, Image | gdino_detector_global, sam2_segmenter_global |
| `/global_cam/map_markers` | MarkerArray | global_cam_projector (raw, pre-fusion) |
| `/global_cam/robot_pose` | PoseWithCovarianceStamped (map) | global_cam_localizer |

---

## 8. Nav2 predictive dynamic-obstacle stack (sim)

Sim-only, `ROS_DOMAIN_ID=44` (`warehouse/env_study.sh`), not yet ported to
the real robot. Where §5.3c wires the overhead cameras + predictive costmap
into RViz for inspection, this section wires the same predictive signal into
Nav2 itself — a DWB critic that scores trajectories against *future* risk,
not just present risk, plus a hard speed cap — and runs a study (pure-lidar
baseline vs. risk-aware) against the same Isaac Sim warehouse scenario to
see whether it actually helps. As of 2026-09-09 this is a **four-arm**
study, not two — `baseline`/`panoptex` (DWB) plus `baseline_mppi`/
`panoptex_mppi` (MPPI) — see **Dynamic-avoidance stack (2026-09-09)** below
for the current architecture; the narrative and results tables further down
this section that predate that date describe the original two-DWB-arm
study and are left as historical record.

### Perception prerequisites the nav stack relies on

Everything below this point — the moving-hypothesis layers in `/risk_stack`,
the CPA/TTC amplification in the collapsed grid, `risk_speed_governor`'s
`robot_closing` cap — is conditioned on `object_tracker_node.py` actually
setting `p_motion` above 0 for a real mover. If it doesn't, the system does
not error or warn loudly: it just quietly collapses onto the stationary
hypothesis everywhere, and everything downstream that exists specifically
to reason about a *moving* obstacle goes dark while the reactive lidar
layers keep working normally, making the failure easy to miss in a log
scroll. This is exactly what happened in the `panoptex_1` run (see
**Verified results** below): carter1 was tracked accurately the whole time,
but every one of its tracks had `p_motion = 0.00`, so the predictive layers
never carried anything but a stationary blob for it and `robot_closing`
never had a chance to fire.

Two `object_tracker_node.py` behaviours matter here, and **both are general
tracker bug fixes, not sim-only tuning** — they change the lab tracker's
behaviour on the real robot too:

- **Association.** `association_key` (default `"category"`) matches
  detections to tracks by `risk_visualization.label_category` (person /
  robot / wheeled / furniture) instead of requiring GroundingDINO's exact
  phrase to repeat frame to frame — open-vocabulary detection legitimately
  calls the same object `"mobile robot"`, `"ground mobile robot"`, or
  `"cart"` from one frame to the next, and under the old exact-match rule
  each phrase change spawned a brand-new track. Set `association_key:
  label` to restore the old exact-match behaviour if a regression shows up.
- **The association gate itself grows with time.** `gate_speed_mps`
  (default `1.0`) widens the association distance gate by
  `gate_speed_mps × (seconds since that track was last updated)` on top of
  the fixed `gate_distance_m` (0.6 m) — needed because a 0.6 m/s mover seen
  at only ~1 Hz per camera can travel farther than a fixed 0.6 m gate
  between two sightings, which was re-spawning it as a fresh, zero-velocity
  track every time instead of updating one continuous track's velocity
  estimate.
- **The motion test gained a second, absolute path.** `motion_threshold`
  (Mahalanobis sigmas, default `3.0`) alone never fires for a sparsely
  re-observed mover, because the velocity covariance stays wide (0.5-0.9 m/s
  σ) under sparse updates and a genuine ~0.6 m/s reads as only ~1-2 sigma.
  `motion_speed_mps` (0.3) + `motion_speed_score` (0.5) add an OR branch:
  moving if Mahalanobis-score > `motion_threshold` **or** (speed ≥
  `motion_speed_mps` **and** Mahalanobis-score ≥ `motion_speed_score`).
  `motion_speed_mps: 0.0` disables this branch and restores the old
  statistical-only test.
- Category-associated tracks adopt whichever matched detection's label is
  currently more confident (`ObjectTrackerNode._adopt_label`), so a track
  that started as `"cart"` and later gets more confident `"mobile robot"`
  detections reports the latter downstream — consequence weighting in the
  predictive costmap follows the evidence instead of freezing at whatever
  label happened to spawn the track.
- **Measurement-time correction.** `measurement_time_correction` (default
  `True`) and `max_measurement_age_sec` (3.0 s) — `object_tracker_node`
  logs "measurement age (capture → tracker)" (`_measurement_age`) so a
  growing capture-to-tracker gap is visible directly rather than inferred
  from downstream symptoms. Measured mean gap in `panoptex_3`: 0.10 s — i.e.
  perception latency is *not* what puts a moving track 0.4-1.1 m behind its
  true position (see **Verified results** below); that offset is
  projection/association error, not latency.

Even with `p_motion` firing correctly, two more gaps affect whether the
predictive layers reflect reality (both in `predictive_risk_costmap_node.py`,
config in `config/risk_perception.yaml`'s STAGE 5 block):

- **Stack magnitude.** `stack_consequence_mode` (`"class"`, default, vs.
  `"scaled"`) — `"scaled"` multiplies `CLASS_BASE_RISK[label]` by the
  track's own decayed detection confidence, which at ~1 Hz sightings idles
  around 0.2-0.3 and painted a correctly-moving Carter's cell at only
  0.14-0.17 in `panoptex_2` — below anything a `lethal_threshold` around
  0.45 can catch. `"class"` instead paints the full class severity (0.75
  for a mobile robot) for any track above `stack_min_track_score` (0.1):
  *where* an object is is the Kalman covariance's job, not the detector
  score's. `stack_categories` (default `[person, robot, wheeled]`) then
  limits which categories the **stack** (not the collapsed grid) paints at
  that magnitude at all — static furniture and unknown-label junk
  ("ground", floor-patch mislabels) painted at class magnitude made whole
  aisles lethal and froze the robot in `panoptex_3` (3 missed waypoints,
  no collision only because it never moved).
- **Object extent.** `use_object_extent` (default `True`) adds the track's
  own bbox half-extent (squared, as variance) to its positional spread
  before painting — capped per category by `extent_cap_{person,robot,
  wheeled,other}_m` (0.30 / 0.45 / 0.50 / 0.25 m by default) rather than
  trusting the raw floor-projected bbox, which is unreliable for large or
  flat objects. Affects both the collapsed grid and the stack (it widens
  the shared covariance both draw from), not just the stack.

Covered by `test/test_object_tracker_motion.py` (8 tests: the Mahalanobis
test's insensitivity to a sparse mover, the speed test firing for one and
staying quiet for a static object, category association absorbing phrase
churn, `association_key: label` reproducing the legacy exact match, the
gate widening with time, `_adopt_label` following confidence, and
measurement-age logging) and `test/test_risk_stack.py`'s
`test_stack_consequence_overrides_magnitude_but_not_grid`.

### Architecture

```
object_tracker (risk_perception)
      │  /risk_perception/world_objects  (Detection3DArray, map frame,
      │  vx/vy + pmot/pmov packed into class_id)
      ▼
predictive_risk_costmap_node  (risk_perception, STAGE 2-5)
      │                                        │
      │ collapsed grid: CPA/TTC-amplified,     │ time-layered stack:
      │ gamma-discounted, one snapshot         │ EGO-INDEPENDENT (no CPA
      │                                        │ factor, no gamma discount)
      ▼                                        ▼
/risk_costmap_predictive                 /risk_stack
  (nav_msgs/OccupancyGrid)                 (panoptex_msgs/RiskStack)
      │                                        │
      ▼                                        ▼
global_costmap.risk_layer                controller_server.FollowPath
  (nav2_risk_layer::RiskLayer,              .PredictedRisk
   max_cost 120 -- SUB-LETHAL, so             (panoptex_nav::PredictedRiskCritic
   NavFn's Dijkstra prefers routing            -- scores each DWB candidate
   around it but is not forced off it)         pose against the LAYER
      │                                         matching that pose's own
      ▼                                         time offset, not a single
planner_server (global plan)                    snapshot)
      │                                              │
      └───────────────────► controller_server (DWB) ◄┘
                                    │      ▲
                                    │      │ nav2_msgs/SpeedLimit (percentage,
                                    ▼      │  0.0 == "no limit", never "stop")
                                 /cmd_vel   │
                                       risk_speed_governor (panoptex_nav)
                                       -- reads /risk_perception/world_objects
                                          DIRECTLY (not the stack, not the
                                          costmap); person/robot/wheeled
                                          distance+CPA policy, independent
                                          of the critic above
```

The split that matters: **the planner sees a collapsed, sub-lethal
predictive grid** (one 2D snapshot, time already discounted away, via
`RiskLayer`) so it can still route through a risk cell when there's no
better option; **the controller sees the full time-indexed stack** (via
`PredictedRiskCritic`), so it can legally choose a candidate trajectory that
passes through a cell that is risky *now* as long as it arrives after that
layer's hazard has moved on — something a static costmap critic cannot
express. The **governor** is a third, independent path: a hard percentage
cap on top speed, sitting entirely outside both the costmap and the critic,
so a slow-moving hazard the critic would merely nudge trajectories away
from can still force the whole robot to slow down.

### Dynamic-avoidance stack (2026-09-09)

WP0–WP6 of the plan at
`~/.claude/plans/can-panoptex-s-priors-get-enumerated-oasis.md` landed the
work the **Results — avoidance runs** and **What to do next** sections
below identify as missing: lidar fused into the tracker, an MPPI arm with a
time-indexed risk critic, a farther-seeing planner-grid route layer, and a
coverage/headway-aware crossing policy. This subsection documents what
landed; the study runs that validate it are pending (see **Results**
below).

**Five timescales, one world model** (topic names as actually published,
not the plan's placeholders):

```
 sensors ──► world model ──► prediction ──► planning stack (4 arms) ──► base

 3 overhead cams (class, coverage)           ┐
 RGB-D (class, metric depth, front)          ├─► object_tracker_node
 lidar clusters (scan_cluster_detector_node, ┘    /risk_perception/detections_3d_map
   NEW, 10 Hz, cm-precision)                            (shared topic, fused in one pass)
                                                         │
                                                         ▼
                                       /risk_perception/world_objects
                                       (one KF/track: p_mot, p_mov, v, Σ)
                                                         │
                                                         ▼
                              predictive_risk_costmap_node (CV rollout, + flow blend)
                                    │                     │                    │
                                    ▼                     ▼                    ▼
                            /risk_stack           /risk_costmap_planner  /risk_costmap_predictive
                          (0–6 s, 21 layers,       (swept lane, NEW,      (collapsed, gamma 0.9,
                           ego-independent)         planner_gamma 0.97)    CPA/TTC-amplified)

  10–100 s  mission_supervisor: waypoints, corridor yield, coverage+headway crossing policy (NEW)
   5–30 s   NavFn global plan: static+obstacle+risk_layer(→ planner grid in *_mppi arms, →
                                predictive grid in DWB arms)+lane_layer+inflation
   0–6 s    controller_server.FollowPath: DWB+PredictedRiskCritic (max_horizon_s 3.3, on
                                /risk_stack) in the DWB arms, OR MPPI(Omni)+PredictedRiskMppiCritic
                                (scores all 60 steps, time_discount 0.97, on /risk_stack_srm — see
                                **Spatiotemporal risk maps (SRM) and MPPI in (x, y, t)** below,
                                2026-09-10) in the MPPI arms, + ObstaclesCritic
                                (lidar+RGB-D local costmap) + SpeedLimit (risk_speed_governor,
                                panoptex arms only)
   0–0.5 s  nav2_collision_monitor on /scan + /camera/camera/depth/points: slowdown/stop
                                polygons, cmd_vel → cmd_vel_safe (last resort, all four arms)
```

**What feeds what:**

| Source | Topic | Consumer(s) |
|---|---|---|
| 3 overhead cams (GDINO/SAM2 + AprilTag ground-plane) | `/risk_perception/detections_3d_map` | `object_tracker_node` |
| RGB-D (GDINO/SAM2 + depth back-projection) | `/risk_perception/detections_3d_map` | `object_tracker_node` |
| lidar clusters (`scan_cluster_detector_node`, NEW) | `/risk_perception/detections_3d_map` | `object_tracker_node` (label-agnostic pass) |
| `object_tracker_node` | `/risk_perception/world_objects` | `predictive_risk_costmap_node`, `risk_speed_governor`, `mission_supervisor` |
| `predictive_risk_costmap_node` | `/risk_stack` | `PredictedRiskCritic` (DWB arms) |
| `predictive_risk_costmap_node` (NEW, 2026-09-10) | `/risk_stack_srm` (SRM conversion of `/risk_stack`, windowed) | `PredictedRiskMppiCritic` (MPPI arms — see **SRM and MPPI in (x, y, t)** below) |
| `predictive_risk_costmap_node` | `/risk_costmap_predictive` | `global_costmap.risk_layer` (DWB `panoptex` arm) |
| `predictive_risk_costmap_node` (NEW) | `/risk_costmap_planner` | `global_costmap.risk_layer` (`panoptex_mppi` arm only) |
| `spatial_prior_node` | `/risk_perception/spatial_prior` | `global_costmap.lane_layer`, `mission_supervisor` lane band, `crossing_choice` ranking |
| `spatial_prior_node` | `/risk_perception/spatial_flow/<category>` | planner-grid flow blend, `corridor.py` heading snap |
| `spatial_prior_node` (NEW headway channels) | `/risk_perception/spatial_headway/<category>` | `mission_supervisor` blind-crossing branch |
| `coverage_mask_node` (NEW) | `/risk_perception/coverage` | `mission_supervisor` `zone_coverage` gate |
| `coverage_mask_node` (NEW) | `/risk_perception/coverage_count` | none (RViz/analysis only) |
| Sim X3 RGB-D | `/camera/camera/depth/points` | `local_costmap.obstacle_layer` (all arms), `nav2_collision_monitor` |
| A1 lidar | `/scan` | `scan_cluster_detector_node`, `local_costmap.obstacle_layer`, `nav2_collision_monitor` |
| `risk_speed_governor` | `/speed_limit` | `controller_server.speed_limit_topic` (panoptex arms only) |
| `controller_server`/`velocity_smoother` | `cmd_vel_nav`/`cmd_vel` | `nav2_collision_monitor` |
| `nav2_collision_monitor` | `cmd_vel_safe` | sim base (`X3_CMD_VEL_TOPIC`) |
| `mission_supervisor` | `/mission/state` | `tools/analyze_run.py`, the experiment harness |

**Lidar cluster → tracker fusion contract (WP1).** `scan_cluster_detector_node.py`
(pure math in `scan_clustering.py`) static-map-subtracts and range-adaptively
clusters `/scan` (frame `laser`), then publishes each surviving cluster onto
the **same** `/risk_perception/detections_3d_map` topic the two camera
chains use — `class_id="lidar_cluster"`, score 0.5. `object_tracker_node`'s
`label_agnostic_labels` param (default `["lidar_cluster"]`,
`config/risk_perception.yaml`) routes any detection carrying such a label
through a *second* association pass, after the labeled pass: nearest
existing track of any category, sequential KF update — it never calls
`_adopt_label`, never revives a graveyard track, and never changes a
track's category; an unmatched cluster spawns a new track with category
`unknown`. Measurement covariance is ordered deliberately so the Kalman
update trusts whichever source is actually more precise at that range —
lidar `cov_base_m2: 0.02` + `cov_per_m: 0.002·r` (i.e. `0.02 + 0.002r`) <
the RGB-D path's implicit `default_cov_m2: 0.04` < the overhead cameras'
ground-plane projection, `0.10·(r/3)²` — the paper's covariance-weighted
fusion rule, now with a third, tighter source in the mix.

**RiskStack: 21 layers / 6 s; the DWB critic's horizon is unchanged.**
`predictive_risk_costmap_node`'s STAGE 5 stack grew from `horizon_steps: 10`
(3 s, 11 layers) to `horizon_steps: 20` (`config/risk_perception.yaml`) —
0–6 s over 21 layers (`k = 0..20`), matching the MPPI arms' `time_steps: 60`
× `model_dt: 0.1` = 6.0 s horizon. `panoptex_nav::PredictedRiskCritic`
(the DWB critic) still caps at `max_horizon_s: 3.3` — DWB only ever samples
~3 s constant-velocity arcs, so scoring layers past that horizon would just
read cells no DWB candidate trajectory can reach. Only
`PredictedRiskMppiCritic` (no such cap — it scores every one of the 60
rollout steps) uses the stack's full 6 s — as of 2026-09-10 it scores a
*second*, derived stream (`/risk_stack_srm`, same wire type, same 21×6 s
layering) rather than `/risk_stack` itself; see **Spatiotemporal risk maps
(SRM) and MPPI in (x, y, t)** below.

**Planner grid: which arm reads which grid.** `/risk_costmap_planner` (NEW,
STAGE 6, same node) is a second collapsed grid, independently parameterised
from the STAGE 3/4 grid: `planner_horizon_steps: 20`, `planner_gamma: 0.97`
(vs. the collapsed grid's `gamma: 0.9` — far-future barely fades, so a
mover's whole swept lane stays visible to NavFn), `planner_flow_blend_weight:
0.5` (vs. `0.0` elsewhere — leans on the learned Spatial-Flow heading for
its far-horizon steps), ego-independent (CPA/TTC factor pinned to `1.0`).
Which grid feeds `global_costmap.risk_layer` is now arm-dependent: the DWB
`panoptex` arm still points it at `/risk_costmap_predictive` (`max_cost:
120`); the `panoptex_mppi` arm points it at `/risk_costmap_planner`
(`max_cost: 140`, still sub-lethal) instead — `config/nav2_x3_panoptex_mppi.yaml`
is the only params file that reads the planner grid. `baseline`/
`baseline_mppi` carry no `risk_layer` at all.

**Two-hypothesis CPA (`two_hypothesis_cpa`, ablation knob, `false` default).**
The STAGE 3/4 collapsed grid (`/risk_costmap_predictive`) used to compute the
CPA/TTC `factor` **once**, from a track's raw `(vx, vy)`, and reuse that same
reading for both the `(1 - pmot)` stationary blob and the `pmot * gamma^k`
moving rollout — so a track the behavioral prior calls uncertain (`pmot`
near 0.5) has its stationary and moving hypotheses share one encounter
geometry, even though they mean opposite things kinematically (`v_obj = 0`
vs. `v_obj = v̂_obj`). Turning `two_hypothesis_cpa` on computes `factor`
**twice** — `factor_stat` with `v_obj = 0`, `factor_mov` with the track's own
believed velocity — and fuses each hypothesis with its own reading
(`C_stat = combine_severity(consequence, factor_stat, relbonus)`, `C_mov`
likewise) *before* the `(1 - pmot)`/`pmot` split, so a track that reads safe
under "it's moving away" but would be a head-on collision under "it's
actually parked" no longer has that risk diluted by sharing the safe
reading. `false` reproduces the historical single-`factor` grid bit for
bit (`paint_track`'s `factor_stat=None` default just reuses `factor`) — see
`predictive_risk_costmap_node.py`'s module docstring (STAGE 4) and
`test_risk_stack.py`'s two-hypothesis-CPA tests. STAGE 5's stack and
STAGE 6's planner grid are unaffected either way — both already pin `factor`
to `1.0` for the unrelated ego-independence reason described above and in
the RiskStack contract below, so there is no second CPA to split there.

**Coverage mask + headway + the crossing policy (WP4).** `coverage_mask_node`
(NEW) publishes `/risk_perception/coverage` (0–100 = analytic union of the
3 overhead cameras' floor-projected frustums, capped at `max_range_m: 15.0`,
and a lidar line-of-sight ray-cast against the static map out to
`lidar_range_m: 12.0` — a *visibility* argument, not a live `/scan`).
`spatial_prior_node` (learn mode only) gained per-category headway channels
(`/risk_perception/spatial_headway/<category>`: `headway_mean`,
`headway_count`, `last_pass_time`) recording the time between successive
`robot`/`wheeled` passes at each lane cell band (`headway_min_s: 2.0` drops
sub-2 s "gaps" that are really re-association churn). `corridor.py`'s
`crossing_policy()` combines both, in this exact order (verbatim from its
own docstring and `mission_supervisor._hold_candidate`/`_crossing_decision`):

1. Run the classic gap-acceptance test (`decide_crossing`: hold iff
   `0 < tta < t_clear + t_margin_sec`).
2. If `coverage_mask_node` isn't publishing at all, stop here — decision =
   the classic test, mode `"gap"`. This is the pre-WP4 behaviour exactly,
   which is what makes the coverage node optional rather than load-bearing.
3. Otherwise compute the `approach_zone` (the corridor strip upstream of
   the crossing, `v_lane · (t_clear + t_margin_sec)` long) and its coverage
   (`zone_coverage`, cells at/above `coverage_cell_min: 50.0` count as
   seen).
4. If the zone is `≥ coverage_min` (`0.8`) covered: mode `"seen"` and
   decision `"go"` if nothing tracked is inside it; mode `"gap"` and
   decision = the classic test if something is (the branch every run before
   WP4 always took).
5. If the zone is **not** covered (`"blind"`): decision `"go"` iff
   `blind_crossing_ok` — the *visible* part of the zone is empty **and**
   either the time since the last observed pass is `≥ headway_mean` (needs
   `headway_min_count: 3.0` samples of evidence) or, with no such
   statistic, `blind_wait_max_s: 15.0` s have elapsed — otherwise `"hold"`.
6. **Invariant, enforced in one place**: if step 1's classic test said
   `"hold"`, the final decision is `"hold"` regardless of the above — the
   coverage/headway branches can only ever *add* a hold, never overrule the
   timing test into a `"go"`.

Two limitations, stated plainly (both from `corridor.py`'s own docstrings):

- **The blind rule only applies to a tracked corridor's approach zone.** A
  corridor only exists once a `robot`/`wheeled` track with `pmot ≥
  pmot_min` is actually being tracked (`make_corridor`); a mover with *no*
  track at all still produces no corridor and is invisible to this policy
  exactly as before — the coverage gate only helps once something is at
  least intermittently tracked, it does not conjure a corridor out of pure
  geometry.
- **`crossing_choice()` is advisory only.** It scores each plan crossing by
  mean learned occupancy `S` and reports the lowest-`S` one in
  `/mission/state`'s `crossing.pref_xy` field, but nothing in this stack can
  re-route Nav2's global plan — it is telemetry ("would a route layer have
  picked somewhere else"), not a control input.

**The four arms:**

| Arm | Controller | `global_costmap.risk_layer` → | RiskStack critic | `risk_speed_governor` | `mission_supervisor` yield | `nav2_collision_monitor` |
|---|---|---|---|---|---|---|
| `baseline` | DWB | none | none | off | off | on |
| `panoptex` | DWB | `/risk_costmap_predictive` (`max_cost 120`) | `PredictedRiskCritic` (`max_horizon_s 3.3`) | on | on | on |
| `baseline_mppi` | MPPI (Omni) | none | none | off | off | on |
| `panoptex_mppi` | MPPI (Omni) | `/risk_costmap_planner` (`max_cost 140`) | `PredictedRiskMppiCritic` (6 s, all 60 steps, on `/risk_stack_srm` since 2026-09-10) | on | on | on |

Sensing (lidar cluster detector, RGB-D depth into the local costmap and the
collision monitor) is identical in all four arms — it is not a Panoptex-arm
difference, only what the risk layers/critic/yield logic do with the world
model is. See `panoptex_nav/README.md`'s **`panoptex_nav::PredictedRiskMppiCritic`**,
**RGB-D depth + collision monitor**, and **The sim study arms** sections for
the full parameter tables, pluginlib registration details, and
`check_arm_params`/`check_arm_params_mppi` diff-guard contract — not
repeated here.

### Results (2026-09-09 evening, headless Isaac, domain 44, scenario v3 triangle, 5 laps / 700 s cap)

Per-run detail: `results/avoidance/SUMMARY.md` (MPPI table) and `~/panoptex_runs/runs/<run>/analysis.md`
(`tools/analyze_run.py`). Contact episode = GT centre gap < 0.45 m; collision criterion < 0.60 m.

| run | arm | laps | activity | carter1 contacts / time | carter1 min gap | carter2 contacts | carter1 tracked % / offset p50 / pmot % / ids |
|---|---|---|---|---|---|---|---|
| mppi_baseline_1 | baseline_mppi | 5 in 296 s | 0.95 | 4 / 20.6 s | 0.24 m | 0 | 91 / 0.33 m / 93 / 35 |
| mppi_panoptex_2 | panoptex_mppi | 5 in 626 s | 0.89 | 7 / 26.7 s | 0.20 m | 0 | 99 / 0.23 m / 95 / 26 |
| mppi_panoptex_3 | panoptex_mppi | 3 (cap) | 0.89 | 6 / 78.4 s | 0.19 m | 2 | 90 / 0.30 m / 94 / 33 |
| mppi_panoptex_4 | panoptex_mppi | 3 (cap) | 0.95 | 5 / 51.4 s | 0.15 m | 0 | 83 / 0.31 m / 95 / 37 |

Reading: the **perception** criteria of the plan are met (lidar fusion: carter1 tracked ≥ 90 %, median
offset 0.23–0.30 m, motion flag ≥ 94 %); the **bench** proved the MPPI critic yields 0.6 m clearance
where the baseline drives to 0.15/0.008 m; the **sim study is not passed**: every risk-aware run still
made contact with carter1, for tactical reasons found by per-episode forensics — refuge poses chosen on
the Carter's centreline (run 2: 10/25 refuges within 0.6 m of it), the omnidirectional stop polygon
freezing the X3 while carter1 pushed it (run 2, since removed), then, in run 3, the new lane-line memory
fragmenting into 32 lanes so refuges were only found 3.5–4 m away and were killed by nav2 as collateral of
the supervisor's own cancel (50× "staying put" in the lane). Run 4 carried the goal-sequencing and lane-hygiene fixes (0 aborted goals, 0 missed waypoints, 8 lanes,
refuges at the 2.5 m stage) and still touched carter1 five times: beside waypoint C the 0.6 m strip between
the wall and the lane holds no cell 1.0 m clear of the lane, so the supervisor picked refuges on the far
side and crossed in front of the Carter (9×). The fix — a hard no-lane-crossing guard when the Carter
arrives before the crossing completes, a derived 0.75 m clearance (Carter half-width + robot radius +
margin), and a same-side wall-hug fallback — is in the tree with tests but has not run in sim. Two invalid runs are kept for the record
(`mppi_panoptex_1_INVALID_amcl_seed`, `mppi_panoptex_3_INVALID_supervisor_crash`).

### `panoptex_msgs/RiskStack` contract

```
std_msgs/Header header          # frame_id = the risk grid frame (map)
nav_msgs/MapMetaData info       # resolution / width / height / origin, all layers
float32 dt                      # seconds between consecutive layers
uint8 steps                     # number of layers (>= 1)
float32 horizon_start           # seconds from header.stamp to layer 0 (usually 0.0)
int8[] data                     # steps * info.height * info.width
```

Layer `k` covers the instant `header.stamp + horizon_start + k*dt`; layer 0
is "now". `data` is row-major per layer (`data[k*H*W + row*W + col]`), the
same convention as `nav_msgs/OccupancyGrid.data`, so any single layer slices
straight into an `OccupancyGrid` with no reshaping. Values are 0-100 risk,
`-1` = unknown, exactly like `OccupancyGrid`.

The field is **ego-independent by construction**: per track,
`predictive_risk_costmap_node`'s STAGE 5 splats the stationary hypothesis
into *every* layer with weight `(1 - pmot) * C_stack`, and the moving
hypothesis's step-`k` rollout position into layer `k` *only*, with weight
`pmot * C_stack`, where `C_stack = combine_severity(consequence, 1.0,
relbonus)` — the same consequence/relation-bonus fusion the collapsed grid
uses, but with the CPA/TTC `factor` pinned to `1.0` and no `gamma^step`
discount for either hypothesis. Both of those exist in the collapsed grid
specifically to fold ego motion and "prefer near-future risk" into one
scalar cost; once time is its own axis in the stack, a controller critic
that already knows its own candidate trajectory's speed supplies the ego
motion itself, and discounting near vs. far future would just be double
work. Published `RELIABLE + TRANSIENT_LOCAL`, `KeepLast(1)`, matching
`/risk_costmap_predictive` — a late-joining critic or RViz gets the last
stack for free, and a subscriber with mismatched durability gets nothing at
all (this bit `PredictedRiskCritic`'s subscription during development —
it now subscribes with the matching QoS explicitly).

### `panoptex_nav::PredictedRiskCritic`

A `dwb_core::TrajectoryCritic` plugin (`src/panoptex_nav/`). For every pose
`i` of a candidate trajectory, at time offset `t`:

1. `t > max_horizon_s` → pose skipped (too far out to trust).
2. the pose is transformed from the local costmap's global frame (`odom` on
   the X3) into `risk_frame` (`map`) via a TF cached once per `prepare()`
   call, not looked up per pose;
3. converted to a grid cell using `info.origin` (position **and** yaw) and
   `info.resolution`; outside the grid → skipped;
4. layer `k = clamp(lround((t + time_shift_s - horizon_start) / dt), 0, steps - 1)`;
5. `r = data[k*H*W + row*W + col] / 100` (`< 0` counts as `0`);
6. `r >= lethal_threshold` → the trajectory is rejected outright
   (`dwb_core::IllegalTrajectoryException`);
7. otherwise `score += time_discount^t * r^cost_power`.

`time_shift_s` is **not** a YAML parameter — it is computed by `prepare()`
every control cycle as `max(0, now - stack.header.stamp)`, i.e. the
`RiskStack`'s own age at scoring time. A trajectory pose's `t` is relative
to *this* control cycle, but the stack it is being scored against was
published up to `stale_timeout_s` seconds ago; adding the stack's age to `t`
before picking a layer is what keeps "the layer for 1.2 s from now" actually
meaning 1.2 s from now, not 1.2 s from whenever the stack happened to be
published.

**Fail-soft by design.** `prepare()` never returns `false` and never blocks
DWB: no `RiskStack` received yet, a stale stack, an unknown costmap frame,
or a failed TF lookup all leave the critic *inactive* — it scores every
trajectory `0.0` and throttle-logs a warning (`warn_period_s`) instead of
stalling navigation. Staleness is judged against `header.stamp` (so it
behaves correctly under `use_sim_time`), falling back to receive time if the
publisher left the stamp at zero.

Parameters, declared under `<dwb_plugin_name>.<critic_name>.`:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `topic` | string | `/risk_stack` | `panoptex_msgs/RiskStack` topic, subscribed `RELIABLE + TRANSIENT_LOCAL + KeepLast(1)` to match the publisher. |
| `cost_power` | double | `1.0` | Exponent on the 0-1 risk value; `>1` tolerates low risk and punishes high risk harder. |
| `time_discount` | double | `0.9` | Per-second discount: a pose at time `t` is weighted `time_discount^t`. |
| `lethal_threshold` | double | `0.85` | Risk (0-1) at or above which the trajectory is rejected outright. |
| `stale_timeout_s` | double | `2.0` | Stack older than this ⇒ inactive, score 0. |
| `max_horizon_s` | double | `3.3` | Poses with a larger time offset are ignored. |
| `risk_frame` | string | `map` | Frame of the risk grid; identity fast-path when it equals the costmap global frame. |
| `warn_period_s` | double | `5.0` | Throttle period for the "no stack / stale / no TF" warnings. |
| `scale` | double | `1.0` | Declared by `dwb_core::TrajectoryCritic` itself; DWB multiplies the returned raw score by it. |
| *(`time_shift_s`)* | — | — | **Not a parameter** — computed internally every cycle from the stack's age; see above. |

Listing it in `controller_server`'s DWB config (the exact block used by
`config/nav2_x3_panoptex.yaml`, as of the `panoptex_1`→`panoptex_2` retune
below):

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      critics:
        - "RotateToGoal"
        - "Oscillation"
        - "BaseObstacle"
        - "GoalAlign"
        - "PathAlign"
        - "PathDist"
        - "GoalDist"
        - "PredictedRisk"
      PredictedRisk.class: "panoptex_nav::PredictedRiskCritic"
      PredictedRisk.scale: 40.0
      PredictedRisk.topic: "/risk_stack"
      PredictedRisk.cost_power: 1.0
      PredictedRisk.time_discount: 0.9
      PredictedRisk.lethal_threshold: 0.45
      PredictedRisk.stale_timeout_s: 2.0
      PredictedRisk.max_horizon_s: 3.3
      PredictedRisk.risk_frame: "map"
      PredictedRisk.warn_period_s: 5.0
```

`PredictedRisk` (the entry in `critics:`) is an arbitrary short name; `.class`
is what pluginlib actually loads. `scale` needs tuning against the other
critics — the raw score is a discounted sum of 0-1 risk values over the
trajectory's poses, so it's typically well under 1 for a mildly risky path.

**`scale` 20.0→40.0 and `lethal_threshold` 0.85→0.45, changed after
`panoptex_1`** (see the yaml comment in `config/nav2_x3_panoptex.yaml`):
`risk_visualization.CLASS_BASE_RISK` caps a mobile robot's consequence at
`0.75 × confidence`, so a `RiskStack` cell for a Carter can never reach
`0.85` even at full confidence and correctly-firing motion — `panoptex_1`'s
stack peaked at `0.37` on carter1's cell. `0.45` is reachable by that same
class cap; `scale` was doubled alongside it since a lower `lethal_threshold`
alone would otherwise leave the *sub*-lethal scoring term (the `score +=
...` path, not the `IllegalTrajectoryException` path) too weak to move DWB's
trajectory choice.

### `risk_speed_governor`

A plain rclpy node (`panoptex_nav/risk_speed_governor.py`), publishing
`nav2_msgs/SpeedLimit` on `speed_limit_topic` (`speed_limit` — nav2's own
default, so `controller_server.speed_limit_topic` is a documented no-op key
rather than a functional rewire). It is the *only* publisher on that topic
and complements `PredictedRisk` rather than duplicating it: the critic
reshapes trajectory **scoring**; this node puts a hard ceiling on **how fast
any trajectory is allowed to go**, straight off
`/risk_perception/world_objects` (no costmap, no `RiskStack`). Per
`nav2_msgs/SpeedLimit`'s own contract, `speed_limit == 0.0` always means "no
limit", never "stop" — a hard stop stays lidar/costmap territory.

| Category | Gate | Cap |
|---|---|---|
| person | distance only (people change direction too fast for CPA) — `d <= person_crawl_radius_m` (1.0 m) | `person_crawl_pct` = 15% |
| person | `d <= person_slow_radius_m` (2.0 m) | `person_slow_pct` = 40% |
| robot / wheeled | moving (`pmot >= moving_pmot_min` = 0.5) **and** closing (`0 < t_cpa <= robot_ttc_s` = 3.0 s **and** `d_cpa <= robot_cpa_m` = 1.0 m) | `robot_closing_pct` = **60%** (was 30%) |
| robot / wheeled | not moving, `d <= static_obstacle_slow_radius_m` (1.0 m) | `static_slow_pct` = **60%** (was 50%) |
| furniture / unknown | ignored entirely — Nav2's own lidar costmap layers already handle static clutter | — |

`robot_closing_pct`/`static_slow_pct` raised to 60% after `panoptex_2`
(config comment, `config/risk_speed_governor.yaml`): a closing AMR the
Carters cannot see (their scan plane passes over the X3, per the probes)
has to be *evaded*, and a robot crawling at 30% cannot get out of the way —
`panoptex_2` capped at 30% from 2.5 m and still collided at 0.17 m. 60% is
a hypothesis carried into `panoptex_3`/`panoptex_4`, not itself validated
in isolation — see **Verified results** below for what actually happened
with it in place.

The most-restrictive applicable cap wins (minimum percentage across every
qualifying track), floored at `min_pct` (15%) so the robot is never capped
below a crawl. Hysteresis: a cap is applied the instant a track qualifies,
but only released back to "no limit" after `release_hold_s` (0.5 s) with
*no* qualifying track at all — otherwise a track flickering in and out of a
threshold radius would chatter the speed limit every control cycle. Tracks
below `min_track_score` or older than `max_track_age_sec` are dropped before
any policy check.

### Perception accuracy fixes (2026-09-09)

Three fixes landed alongside the corridor-yielding work below, all as
lab+sim defaults (User B's decision — they change the real robot's pipeline
too, not just sim tuning), because the corridor/lane machinery is only as
good as the track it yields against:

- **`grounding_dino_adapter.py`: `remove_combined=True`.** The adapter
  calls GDINO's module-level `predict()` directly (replicating
  `Model.predict_with_caption()`'s remaining steps) with `remove_combined=
  True`, instead of going through `predict_with_caption()`, which hardcodes
  `remove_combined=False` and takes no kwarg for it. The library default
  reconstructs each box's phrase from *every* caption token that scores
  above `text_threshold` anywhere in the image, not just the tokens
  belonging to the " . "-separated prompt entry that box actually matched —
  a two-object frame ("person . ground mobile robot .") came back labelled
  "person ground mobile robot" for **both** boxes instead of one "person"
  and one "ground mobile robot" (see `docs/probes_2026-09.md` for how this
  fragmented the tracker's old exact-`class_id` association and fed
  `CLASS_BASE_RISK` the wrong entry — `forklift`/`person` instead of
  `cart` — in `panoptex_3`). An empty phrase (possible when the thresholded
  tokens don't reduce cleanly to one entry) maps to the sentinel `"object"`
  rather than an empty string. `GroundingDinoAdapter` logs a running count
  of labels that still contain more than one prompt entry every 100
  frames — it should read 0; a nonzero, growing count means
  `remove_combined=True` stopped taking effect (e.g. a library upgrade
  changed the kwarg) silently. Verified before/after on
  `test_data/d455_lab.jpg` at thresholds 0.20/0.12: "ground mobile robot
  cable" / "ground mobile robot chair monitor" → "ground mobile robot";
  boxes and scores are bit-identical, only the label text changed.
- **`global_cam_projector_node.py`: near-edge correction
  (`apply_extent_offset()`).** The overhead mask's bottom-band pixel is the
  object's floor-contact point **nearest the camera**, not its centre — a
  person or Carter standing with their near edge toward the camera projects
  half a body-width short of where they actually are. `apply_extent_offset`
  is a pure function (`test/test_projector_extent.py`) that pushes the
  floor point further along the camera→point horizontal ray by a
  per-category half-extent: `extent_offset_person_m` 0.20, `_robot_m` /
  `_wheeled_m` 0.35, `_other_m` 0.0 (furniture/unknown labels get no
  correction — their bottom-band pixel is usually already close to their
  footprint centre, and an unknown label has no reliable extent to assume).
  `extent=0.0` is exact identity; the degenerate camera-on-point case
  returns the input unchanged rather than dividing by zero.
- **`object_tracker_node.py`: `dynamic_half_life_s` 2.0 → 4.0.** Moving
  tracks (`p_movable`/`p_motion > 0.5`) coast on prediction between
  sightings; the old 2.0 s half-life decayed a Carter's confidence too fast
  across the camera hand-over gaps the corridor/lane layer now depends on
  for a continuous track through a crossing. `max_unseen_sec_dynamic` was
  already 20 s (unchanged). Regression: a 0.6 m/s track with a 2.5 s
  observation gap keeps `p_motion > 0.5` and its id
  (`test_object_tracker_motion.py`).

`risk_perception`'s full pytest suite: 113 collected and passing as of this
pass (was 110 before this pass; re-verify with `python3 -m pytest
--collect-only -q` rather than trusting either number blindly).

### Corridor yielding (mission supervisor)

DWB's only response to a predicted-lethal cell is *stop, in the lane* (see
**Verified results** below) — nothing above it ever moves the robot out of
another agent's way before that. `mission_supervisor` (package
`panoptex_nav`, executable `mission_supervisor`; pure geometry in
`corridor.py`) is a mission executor that sits above Nav2 and, when
`yield_enabled:=true` (panoptex arm only), watches the Panoptex world model
and gets out of the way *before* DWB has to react at all.

**Why one `NavigateToPose` goal per waypoint, not `FollowWaypoints`.** The
yield layer has to interrupt the current leg — send the robot to a hold
point short of a crossing, or to a refuge off the lane — and then resume the
leg it interrupted. Nav2's `waypoint_follower`/`FollowWaypoints` owns the
whole list internally and gives no way to divert and return; a raw
`rclpy.action.ActionClient` per waypoint (the `yahboomcar_nav
waypoint_runner.py` pattern — including its `_StopRunner` sentinel-exception
unwind, since `rclpy.shutdown()` from inside a spinning callback deadlocks
on this machine) keeps that control in this node, at the cost of it doing
its own list/lap bookkeeping. **Both study arms run the same executor**;
`yield_enabled:=false` (baseline) is a plain waypoint loop with every yield
branch compiled out behind one flag, so the baseline arm is bit-for-bit the
panoptex arm's code path minus the corridor logic, not a second
implementation.

**Corridor model** (shared with `lane_layer` below — see `corridor.py`'s
module docstring for the full geometry). A *corridor* is the swept lane a
moving `robot`/`wheeled` track (`pmot ≥ pmot_min`, `|v| ≥
min_user_speed_mps`) is about to occupy: a rectangle starting `back_margin_m`
behind the track, running `|v| × t_corridor_sec` ahead of it along its
heading, `half_width` either side of that centre line. People are
deliberately excluded (they stop and turn on the spot, so a 10 s swept
rectangle is fiction — `risk_speed_governor`'s distance-only branch already
covers them). The heading is the track's raw velocity, **snapped to the
learned lane heading** (see **Learned lanes** below) when the flow field has
evidence at the track's own cell (`flow_min_conf` 0.3) and broadly agrees
with the observed velocity (within `flow_snap_deg` 30°) — the tracker's
instantaneous velocity jitters 10-20° frame to frame, which swings the far
end of a 10 s corridor by metres, while the learned flow is an EMA over the
whole run of what actually travels through that cell. Two different widths
answer two different questions (`corridor_half_width_m` 0.55 = 0.25 m Carter
+ 0.15 m X3 + 0.15 m margin for **gap acceptance** — "can I cross before it
arrives", where a crossing is transient and `t_margin_sec` slack absorbs
error; `danger_half_width_m` 0.90 for **containment** — "am I in the way",
where the tracked position's own 0.1-0.4 m error means the answer decides
whether the robot moves at all).

**Gap acceptance** (before a crossing). Intersect the current `/plan` with
each corridor; for the first crossing ahead:

```
t_clear = (d_entry + 2*half_width + clear_margin_m) / v_cross_mps
```

— the time needed to drive the remaining `d_entry` metres to the corridor
edge plus the full lane width plus a margin, at the deliberately
*pessimistic* `v_cross_mps` (0.20 m/s — Nav2 decelerates into and
accelerates out of a crossing, so this has to be a lower bound on the
robot's own speed or the decision is optimistic exactly where optimism costs
a collision). `tta` is the corridor user's own distance to the crossing over
its speed. **Hold** iff `0 < tta < t_clear + t_margin_sec` (2.0 s slack); a
non-positive or infinite `tta` (user already past, or stationary) is always
"go". Holding sends a `NavigateToPose` goal to a stand-off point
`hold_back_m` (0.7 m) short of the corridor entry — or, if that point is
already behind the robot or within `hold_cancel_radius_m` (0.3 m), just
cancels the current goal and stands still, **unless standing still would
itself leave the robot inside the danger corridor or the static lane band**
(see escalation below).

**Refuge** (already inside a corridor when its user starts approaching,
`0 < tta < t_yield_sec` = 8.0 s — typically because Nav2 routed the X3 up
the middle of a Carter's own lane). There is no gap to accept; the robot has
to leave the lane. `find_refuge` picks the nearest free, obstacle-clear cell
(`refuge_radius_m` 2.5, `refuge_clearance_m` 0.45 from the map) outside
*every* corridor **and** the static lane band (see **Learned lanes**),
preferring the side the robot is already on (crossing the lane to reach the
far side is a worse encounter than the one being avoided) and directions
perpendicular to the lane. When the lane band swallows every legal cell, the
search falls back to the least-bad in-band candidate rather than returning
`None` — logged at WARN so a run's analysis can count how often it fired.

**Escalation (hold → refuge).** A hold whose stand-off degenerates to
"stand still" is only a real yield if standing still is actually out of the
way — in `avoid_panoptex_2`, 8 of 21 yields logged "the stand-off is already
behind us" and one of those was the run's closest approach (0.15 m ground
truth): the robot parked 0.65 m off the lane centre while a Carter came down
it. `hold_in_place_is_unsafe()` checks the robot's position against the
danger corridor and the static lane band before accepting "stand still" as
the answer; when it would be unsafe, the supervisor takes a refuge instead.

**Release.** While holding/refuge: released once the point yielded for is
`half_width + release_margin_m` (0.3 m) **behind** the user (not simply
`along < 0` — the user's body still occupies the crossing the instant its
centre passes it), or the track is lost for `lost_timeout_sec` (3.0 s). A
lost track is **not** treated as evidence the lane is clear — one Carter
carried six different ids in a single run — so `loss_release_guard` asks the
geometry instead: is any *currently* tracked corridor user still sweeping
the point being yielded for within `t_yield_sec`? If so the yield re-binds
to that user (regardless of id) and continues; only if nobody is does the
all-clear have to hold for `confirm_ticks` consecutive ticks before
resuming. A resume caused by a loss-release is weak evidence, so the next
yield after one needs only a single confirming tick for
`post_loss_cooldown_sec` (2.0 s) rather than the usual `confirm_ticks` — the
robot is, by definition, sitting right next to a lane that was busy a moment
ago, and re-identification (not departure) is the likelier explanation.
**Never re-enter**: `_release()` checks whether the robot is still inside an
approaching user's corridor before resuming and takes a refuge instead if
so; `_eval_navigating` re-runs the instant the mission is `NAVIGATING`
again, so a lane that becomes a conflict mid-resume is caught the same as
any other tick.

**Hysteresis.** A yield needs `confirm_ticks` (3) consecutive agreeing
evaluations — 0.6 s at the supervisor's 5 Hz `publish_rate_hz` — before it
starts (a single noisy frame must not stop the mission), and is never
released before `min_hold_sec` (1.0 s) after committing (otherwise a
flickering velocity estimate can pass the release test the very next tick,
producing stop-go-stop stutter).

**State machine and `/mission/state`.**

```
paused  --(start_delay, waypoint_pause)-->  navigating
navigating --(plan crosses a lane, gap too small)--> holding
navigating --(robot already IN a lane, user approaching)--> refuge
holding | refuge --(user passed, or track lost + all-clear)--> resuming
resuming --(interrupted waypoint's goal accepted)--> navigating
navigating --(last waypoint of the last lap succeeded)--> complete
```

Published as `std_msgs/String` JSON on `/mission/state` (`state_topic`), both
on every transition and at `publish_rate_hz` regardless, so a recorder never
has to infer state from gaps between transitions:

```json
{"t": 1234.5, "waypoint": 2, "name": "wp_003", "lap": 0,
 "state": "holding", "yield_count": 1,
 "yield": {"user": "7", "kind": "hold", "tta": 10.2, "t_clear": 13.0,
           "xy": [1.32, -3.0]},
 "goal_xy": [0.62, -3.0]}
```

`goal_xy` is the goal **currently** being pursued (the hold/refuge point
during a yield, not the waypoint); `waypoint`/`name` always name the leg
being executed, i.e. what an interrupted goal resumes to.

**Params** (`config/mission_supervisor.yaml`; the corridor block is also
`corridor.CORRIDOR_DEFAULTS`, so the module, the ROS declarations and the
yaml comments cannot drift apart):

| Group | Param | Default | Meaning |
|---|---|---|---|
| mission | `loop` / `laps` / `start_index` | `true` / `0` / `0` | `laps=0` = unbounded while `loop:=true`; a positive count stops the mission after that many laps regardless. |
| mission | `waypoint_pause_sec` | `0.2` | Pause between waypoints. |
| mission | `yield_enabled` | `false` | The A/B switch — `true` only in the panoptex arm's launch. |
| gating | `self_exclusion_radius_m` | `0.7` | Tracks this close to the robot ARE the robot (the cameras see the X3 as just another "mobile robot"). |
| gating | `pmot_min` / `min_user_speed_mps` | `0.5` / `0.2` | Minimum motion confidence / speed to own a corridor. |
| shape | `t_corridor_sec` / `back_margin_m` | `10.0` / `1.0` | Corridor length ahead of the user / margin behind it. |
| shape | `corridor_half_width_m` | `0.55` | Gap-acceptance width (0.25 Carter + 0.15 X3 + 0.15 margin; 0.65 put the west-strip leg permanently inside lane 1's corridor). |
| shape | `danger_half_width_m` | `0.90` | Containment width (0.55 + 0.35 m of track-position error). |
| flow | `flow_min_conf` / `flow_snap_deg` | `0.3` / `30.0` | Snap corridor heading to the learned lane when confident and within this angle. |
| crossing | `clear_margin_m` / `v_cross_mps` / `t_margin_sec` | `0.3` / `0.20` / `2.0` | Gap-acceptance formula (see above). |
| crossing | `hold_back_m` / `hold_cancel_radius_m` | `0.7` / `0.3` | Stand-off distance / "close enough, just stop" radius. |
| refuge | `t_yield_sec` | `8.0` | TTA threshold that turns "already in a lane" into a refuge move. |
| refuge | `refuge_radius_m` / `refuge_clearance_m` | `2.5` / `0.45` | Search disc / minimum clearance from anything occupied or unmapped. |
| lane band | `lane_band_min_value` | `5.0` | Spatial-prior value at/above which a cell counts as "somebody's lane" (low — one pass is already evidence). |
| release | `release_margin_m` / `lost_timeout_sec` | `0.3` / `3.0` | Release-test margin / track-loss timeout. |
| release | `confirm_ticks` / `min_hold_sec` | `3` / `1.0` | Hysteresis (see above). |
| release | `post_loss_cooldown_sec` | `2.0` | Fast-confirm window after a loss-release. |

**Bench evidence.** `test/test_corridor.py` (25 pytest cases on the pure
`corridor.py` module: geometry sign conventions, user selection dropping
self/slow/wrong-category tracks, flow snapping and its refusal to snap on
disagreement, crossing-index/entry-distance extraction, the gap-acceptance
decision at its boundary, the worked `t_clear` example, the release margin,
refuge search — own-side preference, "everything blocked", "everything is
in a corridor", the second-lane-band rejection, the least-bad band fallback,
line-clearance preference — hold-point walk-back including past the lane
band, the loss-release guard's three behaviours, and containment differing
by width) plus a scripted-track bench on domain 199 (fake `/plan`, `/map`
via `nav2_map_server` on `warehouse_x3_nav.yaml`, a moving "mobile robot"
track) confirming `/mission/state` transitions `navigating → holding →
navigating`.

### Learned lanes (lane_layer, warm-up, frozen prior)

A live corridor only exists while a specific track is actively being
watched driving down it — not enough to pick a *refuge* by: in the
warehouse aisle the two AMR lanes are 0.8 m apart, and stepping out of the
one with a live track into the one whose Carter happens to be round the
corner isn't stepping aside at all. `avoid_panoptex_1` (2026-09-09) chose
four refuges inside the aisle this way and then thrashed through six
recomputes as the second lane's Carter came back into view.

`spatial_prior_node`'s learned occupancy (the S channel, an EMA of where
traffic actually goes, see that node's own docstring) fixes this two ways:

- **Static lane band** (`corridor.make_lane_band`, used by
  `mission_supervisor`'s refuge/hold-point search): cells at or above
  `lane_band_min_value` (5.0, deliberately low — one pass of a Carter is
  already evidence), dilated by `danger_half_width_m` so the static keep-out
  covers a body-width either side of the learned lane centre, exactly like a
  live corridor would. Holds whether or not anyone is currently driving down
  it — the fix for the `avoid_panoptex_1` thrash above.
- **`lane_layer`** (`config/nav2_x3_panoptex.yaml`'s global costmap, panoptex
  arm only): a **second** `nav2_risk_layer::RiskLayer` instance, separate
  from `risk_layer` (the instantaneous predictive costmap), reading
  `/risk_perception/spatial_prior` directly at `max_cost 90` (sub-lethal,
  under `risk_layer`'s 120 and well under lethal 254) and `min_risk_value
  5`. Sub-lethal on purpose: NavFn's Dijkstra search stops **routing along**
  a learned lane (parallel travel touches many cells at that cost) while a
  **perpendicular crossing** of the same lane stays cheap (it touches one
  cell's width) — the same per-cell cost penalizes the two cases very
  differently without an explicit direction term. `tools/check_arm_params.py`'s
  allow-list covers `lane_layer` alongside `risk_layer` (panoptex-only,
  baseline never gets it).

**Warm-up.** `x3_nav.launch.py learn_lanes:=true` starts *only*
`panoptex_sim.launch.py` (overhead cams, world model, predictive costmap,
`spatial_prior_node` with `enable_spatial_prior` forced `true` regardless of
that arg's own value) — no `nav2_bringup`, no `risk_speed_governor`, no
`mission_supervisor` — so the domain has no X3 nav stack competing while
Carter patrol traffic teaches the prior. `tools/warmup_lanes.sh
[sim_seconds=600] [spatial_prior_path]` drives one full session (Isaac
headless → `/clock` → X3 bringup → Carter patrol → this launch mode → wait
`sim_seconds` of **sim** time, not wall time) and then reports S-channel
coverage along both lanes from the saved `.npz` (max S within ±0.30 m of
each lane's line, every 0.5 m of `y`) — reusing `tools/spatial_flow_heatmap.py`'s
npz-loading code. **Execute it, never `source` it**: it uses `set -o
pipefail` and traps `EXIT` to tear down everything it started, and a
blanket `pkill -f run_headless.py` in an earlier version of this script
killed a *different*, concurrently-running session's Isaac — the current
version only reaps an Isaac it can prove (via its own pidfile) that it
started. Clean 600 s warm-up result (`results/avoidance/spatial_prior_sim_warmup.npz`):
carter1's lane at S ≥ 0.05 for 11 of 23 sampled points along it (max 0.40),
carter2's west leg for 10 of 23 (max 0.27) — patchy, but enough for
`lane_band_min_value` (5.0, on the published 0-100 scale) to treat both as
lanes end to end once dilated.

**Frozen prior during study runs.** `spatial_prior_learn_rate` and
`spatial_prior_autosave_sec` are forwarded from `x3_nav.launch.py` through to
`panoptex_sim.launch.py`'s `spatial_prior_node` overrides, and pinned to
`0.0` / `1.0e9` whenever `learn_lanes:=false` (i.e. every study run) —
learning and autosave stay on (`0.20` / `60.0`) only in `learn_lanes:=true`
warm-up mode. **Why this matters**: in `avoid_panoptex_2` the prior kept
learning *during* the run — the X3 itself is tracked as a "mobile robot" and
its own path got deposited into S alongside genuine Carter traffic, plus
static-clutter junk — and saturated to 38,828 cells ≥ 0.05 (64% of the map)
by the end, against ~1,500 cells after a clean warm-up; the run's own
autosave then overwrote the warm-up `.npz` on disk. A prior that has learned
"everywhere the X3 has ever driven" is not a lane model, it's a diary, and
`lane_layer`/the static lane band built from it become useless (everything
looks like a lane, or the file that would have been useful is gone). Always
pass `spatial_prior_path` explicitly to the same file the warm-up wrote, and
never launch a study run with `learn_lanes:=true`.

### Scenario v3

`config/x3_sim_waypoints.yaml` (frame `map`) is now a 3-waypoint triangle,
not the earlier 4-point rectangle: spawn (0.38, 0.07) → **A** (3.40, 2.50) →
**B** (3.40, 0.50) → **C** (0.65, 0.50) → (loop back to A). A→B runs the
east column outside both Carter lanes; **B→C and C→A each cross both
lanes** (carter1 at x=1.32, carter2's west leg at x=2.15) — exactly twice
per lap, as the plan requires. The crossing sits at y = 0.5, not the v1/v2
scenarios' y = -3.0: at y = -3 the crossing was 1 m from carter1's U-turn
point (`aisle_south`, y = -4.0), so the Carter reversed direction right
beside the X3, the constant-velocity corridor model pointed the wrong way
(`tta` 0.8 s against a Carter that was about to turn around and come back),
and one such contact turned into the Carter pushing the X3 8 m up the lane
(see `avoid_panoptex_3`'s own waypoint-file comment and **Results** below).
At y = 0.5 both Carters run at constant speed (carter1 turns at y = -4/+7,
carter2's west leg turns at y = -1) and overhead coverage there is
65-78% — better than the U-turn zone, though still short of what the yield
logic actually needs (see **Results** — camera coverage, not the corridor
model, is now the binding constraint).

Two earlier revisions are documented in the yaml's own header history: v2
(a 4-point rectangle A→B→C→D, replacing the original loop that started
directly at the X3's spawn pose) and v2.1 (the west-strip leg C→D moved to
a touch-and-return, because at x=0.60/0.65 it sat close enough to lane 1's
edge that the supervisor spent 118 s of one run in refuge on that leg
alone). Clearance was re-checked against `warehouse/maps/warehouse_gt_x3.
{yaml,pgm}` for every point/segment of v3 (≥0.30 m obstacle clearance,
≥1.5 m person distance from `warehouse/carters/tests/people_positions.json`
— v3 clears both with ≥0.35 m / ≥4.5 m margin; see the yaml header for the
full per-point/per-segment numbers).

### Build

```bash
cd ~/workspace/Panoptex
conda activate panoptex                # required for every build in this repo, §4
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select panoptex_msgs risk_perception panoptex_nav
colcon test --packages-select panoptex_nav && colcon test-result --verbose
```

`colcon test` runs three independent suites: the C++ gtest
(`test_predicted_risk_critic`, covering the scoring algorithm, the
time-shift layer selection, staleness, the frame transform, and pluginlib
loadability), the pytest suite for `compute_cap()`
(`test_risk_speed_governor`, no ROS graph needed), and `check_arm_params` — a
plain ctest wrapping `tools/check_arm_params.py`, which deep-diffs
`config/nav2_x3_baseline.yaml` against `config/nav2_x3_panoptex.yaml` and
fails if anything other than the sanctioned risk-layer/critic/speed-topic
keys has drifted between the two study arms. `colcon test --packages-select
panoptex_nav` passes green as of the `panoptex_2`-`panoptex_4` retuning
above. `risk_perception`'s own pytest suite (`cd src/risk_perception &&
python3 -m pytest --collect-only -q`, workspace + conda env sourced) collects
**110** tests as of this pass — not the 108 that had been reported
elsewhere; re-verify with the same command rather than trusting either
number blindly if it matters to you.

### Running each arm end-to-end

**Warm up the learned lanes first, once** (see **Learned lanes** below) —
`lane_layer` has nothing to route around until `spatial_prior_node` has
actually seen Carter traffic, and a study run must NOT be the thing that
teaches it (see **Frozen prior during study runs**):

```bash
source ~/workspace/warehouse/env_study.sh        # ROS_DOMAIN_ID=44
conda activate panoptex
cd ~/workspace/Panoptex/src/panoptex_nav
./tools/warmup_lanes.sh 600 ~/.panoptex/spatial_prior_sim.npz
# EXECUTE this (./tools/warmup_lanes.sh ...), never `source` it -- it sets
# -o pipefail and traps EXIT to tear down Isaac/Carters/the launch graph;
# sourcing it into an interactive shell applies those to that shell instead
# and a later `exit` (or the trap firing) closes the terminal.
```

Then, every terminal below sources the study environment first — **not**
`env_sim.sh`, which is domain 55 and will make `x3_nav.launch.py` refuse to
start:

```bash
source ~/workspace/warehouse/env_study.sh        # ROS_DOMAIN_ID=44
conda activate panoptex
```

One terminal each:

```bash
# A -- Isaac, headless
# X3_CMD_VEL_TOPIC=cmd_vel_safe: point the sim base at the collision monitor's
# gated output instead of plain cmd_vel (WP5; enable_collision_monitor:=true
# by default in x3_nav.launch.py below -- see panoptex_nav/README.md's "RGB-D
# depth + collision monitor"). Leave unset and add PANOPTEX_NO_CM=1 plus
# enable_collision_monitor:=false on terminal D's launch below to run with the
# monitor fully out of the loop.
export X3_CMD_VEL_TOPIC=cmd_vel_safe
~/isaacsim/python.sh ~/workspace/warehouse/x3_sim/run_headless.py \
    --scene ~/workspace/warehouse/Baseline_scenario_metric.usd
# (or GUI: open the stage, Script Editor -> Run x3_sim/x3_base_controller.py, Play)

# B -- X3 ROS-side bringup (URDF tf, laser/camera static TFs)
ros2 launch yahboomcar_nav x3_sim_bringup_launch.py

# C -- Nova Carter patrol (the moving obstacles the study measures against)
ros2 launch ~/workspace/warehouse/launch/carters_patrol.launch.py \
    map:=~/workspace/warehouse/maps/warehouse_gt_carter.yaml

# D -- the study arm itself
ros2 launch panoptex_nav x3_nav.launch.py arm:=baseline laps:=5    # pure-lidar nav2
ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex laps:=5    # risk-aware nav2, yielding on
```

Terminal D brings up, in order: `risk_perception`'s `panoptex_sim.launch.py`
with `enable_localization:=false` (identical compute load in both arms —
the overhead cams + predictive costmap + spatial prior always run; only what
Nav2 *does* with the signal differs) and, for a study run
(`learn_lanes:=false`, the default), `spatial_prior_learn_rate:=0.0` /
`spatial_prior_autosave_sec:=1.0e9` forwarded to it so the warmed-up prior is
read-only for the rest of the run; `nav2_bringup`'s `bringup_launch.py` with
the arm's params file (AMCL seeded from the baked spawn pose, so no manual
"2D Pose Estimate"); `risk_speed_governor` (panoptex arm only — the node
simply isn't spawned in baseline); and `mission_supervisor` (**both arms**,
replacing the old `yahboomcar_nav` waypoint-follower include), which owns its
own `start_delay_sec` (the `runner_delay` arg, 12 s by default, so nav2's
lifecycle has activated first) and then drives `config/x3_sim_waypoints.yaml`
— Scenario v3, a 3-waypoint triangle starting from the X3's spawn pose,
crossing both Carters' patrol lanes twice per lap (see **Scenario v3**
below) — as a `NavigateToPose` goal per waypoint, `laps` (0 = unbounded
while `loop:=true`, the default) stopping it after that many laps
regardless. Only the panoptex arm's supervisor actually yields
(`yield_enabled` = `arm == panoptex`); the baseline arm runs the identical
executor with the corridor logic switched off, so the two arms differ only
in what the yield layer + `lane_layer` do, not in how the waypoint list
itself is driven. The launch refuses to start unless `ROS_DOMAIN_ID=44`,
escape hatch `PANOPTEX_NAV_ANY_DOMAIN=1` for bench-testing the launch graph
on another domain (no Isaac, no `/clock`).

### Verified results (2026-09-08, headless, 240 s sim window each, domain 44)

Raw evidence (`monitor.json`, `x3_nav_excerpt.log`, `nodes.txt`,
`risk_stack_info.txt` per run) is under `results/wiring_check/<run>/`;
`results/wiring_check/SUMMARY.md` has the one-line-per-run table this
section expands on. Collision criterion throughout: ground-truth
X3↔carter1 centre-to-centre gap < 0.6 m (the plan's own definition, not the
Carter's visual/collision footprint).

| Metric | `baseline_1` | `panoptex_1` (before tracker fix) | `panoptex_2` (tracker fix + critic retune) | `panoptex_3` (stack-magnitude fix) | `panoptex_4` (per-category extent caps) |
|---|---|---|---|---|---|
| RTF | 0.296 | 0.300 | 0.291 | 0.292 | 0.297 |
| Min X3↔carter1 gap | **0.342 m** | **0.141 m** | **0.173 m** | **0.709 m** | **0.232 m** |
| Collision (<0.6 m) | **Yes** (t≈237 s) | **Yes** | **Yes** | **No** | **Yes** |
| cmd_vel messages (mobility) | 14495 | 14297 | 14747 | **6500** | 11213 |
| carter1 `p_motion` | N/A (no critic/governor in this arm) | **never > 0** (44 track ids, median 12 s lifetime) | **> 0.5** in 47% of tracked samples | — (categories other than person/robot/wheeled still dominated the stack) | **> 0.5** in 63% of tracked samples |
| `risk_speed_governor` cap fired | N/A | `static_slow` only; `robot_closing` never | `closing` fired 28× | — | governor active, X3 stopped in-lane (see below) |
| Headline problem | lidar reacts at 2.9-3.5 m, too late at closing speed | `p_motion` stuck at 0 — predictive layers never saw a mover | stack value at carter1's true cell only 0.14-0.17 at every horizon — too weak against a 0.45 threshold | 25% of layer-0 (~150 m²) ≥ lethal from mislabeled static clutter — X3 nearly frozen | X3 stopped **in carter1's lane** and wasn't run over only because carter1 also stopped nearby |

**`baseline_1`**: the X3 drove its full waypoint loop and collided with
carter1 (0.342 m min gap, sim t≈237 s) despite the lidar-only stack
reacting — carter1 only became lethal in the local costmap at 2.9-3.5 m
true range, too late at the closing speed involved. A probe separately
found carter1's own front 2D lidar does **not** see the X3 at all at 0.93 m
(it returns ≈11.6 m at the X3's bearing — the X3 sits below the Carter's
scan plane), so in every run below all collision-avoidance load is on the
X3's side; the Carters never react to it.

**`panoptex_1`** also collided (0.141 m — worse than baseline), for a
reason entirely upstream of the critic/governor: `object_tracker_node.py`
tracked carter1 accurately (within 1 m of ground truth 58% of the run,
including during the approach) but every one of its 44 fragmented track ids
had `p_motion = 0.00` (see **Perception prerequisites** above for the two
tracker bugs behind this). The predictive layers never carried anything but
a stationary blob for it, and `robot_closing` never had a chance to fire.

**`panoptex_2`** (tracker fix landed: category association, time-widened
gate, absolute speed test; critic retuned to `lethal_threshold: 0.45`,
`scale: 40.0`): `p_motion` now correctly exceeded 0.5 in 47% of tracked
samples, and the governor's `closing` cap fired 28 times — real progress
over `panoptex_1` — but the run still collided (0.173 m). The `RiskStack`
value at carter1's *true* cell stayed only 0.14-0.17 at every horizon step,
for three compounding reasons: (a) `stack_consequence_mode: scaled`
multiplied class severity by the track's decayed detection confidence
(idling at ~0.25 with ~1 Hz sightings), (b) carter1 was painted as a point
with no extent, and (c) the track itself trailed carter1's true position by
0.4-1.1 m. The X3 was capped to 30% from 2.5 m and a Carter blind to it
still drove into it — the collision that motivated raising
`robot_closing_pct`/`static_slow_pct` to 60% (see the governor table
above).

**`panoptex_3`** (`stack_consequence_mode: class`, bbox extent widening,
`measurement_time_correction`, 60% governor caps): **no collision by the
<0.6 m criterion (0.709 m min gap) — but only because the X3 barely
moved.** 6500 cmd_vel messages against a baseline/`panoptex_1`/`panoptex_2`
range of ~14-15k, and 3 missed waypoints. 25% of the layer-0 stack (≈150 m²
of a ~608 m² grid) sat at or above the lethal threshold, with peaks of
0.90-0.95 coming from GroundingDINO's concatenated phrases ("ground mobile
robot forklift", "person ground mobile robot" → matched against the
`forklift`/`person` entries in `CLASS_BASE_RISK`, both far higher than
`cart`) on bboxes up to 1.2 m, across roughly 33 tracks per message —
i.e. static clutter and phrase-concatenation mislabels, painted at full
class magnitude with generous extent, made large stretches of the floor
read as lethal and froze the robot far more often than any real hazard
justified. Measured mean capture→tracker latency this run: 0.10 s — ruling
latency out as the cause of the 0.4-1.1 m track-position offset seen in
`panoptex_2`; that offset is projection/association error, not a stale
pipeline.

**`panoptex_4`** (`stack_categories: [person, robot, wheeled]` — drops
furniture/unknown from the stack entirely — plus per-category extent caps
`extent_cap_{person,robot,wheeled,other}_m` = 0.30/0.45/0.50/0.25):
lethal area down to a median ≈2.4% of the grid (~15 m², from `panoptex_3`'s
25%), `p_motion > 0.5` in 63% of tracked samples, and the stack at
carter1's future true cell finally lights up ahead of it where it actually
goes (p90 0.60, median 0.21, at every horizon step) — the predictive
mechanism is working as designed at this point. **Still a collision by the
<0.6 m criterion (0.232 m), and 2 waypoints missed.** In the
closest-approach window the X3 was fully stationary (cmd_vel = 0) for 15 s
while carter1 approached slowly to 0.24 m, sat there roughly 10 s, then
left. Read plainly: **the stack correctly made "continue forward" illegal,
so DWB stopped — but it stopped *in carter1's lane*.** With the Carter
blind to the X3, stopping in its path is not sufficient; the X3 needed to
vacate the lane, which needs anticipation beyond DWB's ~3 s constant-
velocity trajectory horizon (a planner-level lane-avoidance or yield-aside
behaviour), plus better track position accuracy (`panoptex_4`'s tracks
still carried 0.3-0.6 m offsets, and only 63% of samples had `p_motion`
firing at all).

**This is the state of the art as of 2026-09-08, not a solved problem.**
Five runs, five collisions or near-freezes, in this order: no perception of
motion at all (`panoptex_1`) → motion perceived but too weak to matter
(`panoptex_2`) → risk over-painted everywhere, robot nearly can't move
(`panoptex_3`) → risk painted correctly, robot stops but in the hazard's
own lane (`panoptex_4`). Each run fixed the failure mode the previous run
exposed and exposed a new one underneath it.

**What to do next**, in the order these runs surfaced them:

1. ~~**Planner-level lane/yield behaviour.**~~ **Addressed (2026-09-09
   plan, WP2/WP3/WP4).** DWB's short constant-velocity horizon (~3 s)
   couldn't express "step aside and let it pass" — only "go slower on
   roughly the same path." The fix landed as three pieces, not one: the
   MPPI arms (`panoptex_mppi`) optimise a whole 6 s velocity sequence
   instead of sampling short arcs; the new planner grid
   (`/risk_costmap_planner`) gives NavFn a farther-seeing, slower-fading
   view of a mover's swept lane to route around, independent of the
   near-term stack; and the coverage/headway-aware crossing policy decides
   *whether* to enter a lane at all before the controller ever has to react.
   See **Dynamic-avoidance stack (2026-09-09)** above for what landed and
   **Results** there for whether it actually works — the study runs that
   validate this are pending.
2. ~~**Track position accuracy.**~~ **Partially addressed, pending the
   probe.** `panoptex_4` still carried 0.3-0.6 m position offsets and
   `p_motion` firing on only 63% of samples. WP1 fuses lidar clusters
   (10 Hz, centimetre-range, deliberately the smallest measurement
   covariance of the three sources — see **Lidar cluster → tracker fusion
   contract** above) into the tracker as a second, label-agnostic
   association pass, which should collapse the offset within lidar's 12 m
   line of sight and tighten the velocity covariance enough for the motion
   test to fire reliably. Not yet measured against ground truth — that is
   exactly what the plan's lidar-fusion probe (`docs/probes_2026-09.md`
   Addendum 4) is for; this item stays open until that probe's numbers
   land.
3. **GDINO phrase-concatenation cleanup.** `panoptex_3`'s worst lethal
   spikes (0.90-0.95) came from compound phrases like "ground mobile robot
   forklift" or "person ground mobile robot" matching the wrong
   `CLASS_BASE_RISK` entry. `stack_categories` papers over this for the
   sim scenario by dropping furniture/unknown from the stack outright;
   the underlying open-vocabulary label-splitting problem (already flagged
   in §5.3c) is unaddressed.
4. **RTF.** All five runs sit at 0.29-0.30x regardless of arm or how much
   the robot actually moved — a real-time (or even near-real-time) study
   would need this addressed independently of any of the above.

### Results — avoidance runs (2026-09-09)

Item 1 above ("planner-level lane/yield behaviour") is now built —
**Corridor yielding (mission supervisor)** and **Learned lanes** further
up this section — and re-run against Scenario v3
(`arm:=panoptex`, `mission_supervisor` with `yield_enabled:=true`, headless
Isaac, domain 44). It is not the fix: the yield logic fires and behaves as
designed, and the runs still show repeated close contact with carter1.
Raw evidence (`analysis.txt`, `x3_nav_excerpt.log`, `monitor.json` for
runs 1-3) is under `results/avoidance/<run>/`;
`results/avoidance/SUMMARY.md` has the one-line-per-run table below.
"Samples with gap < 0.6 m" (the `panoptex_1`-`4` collision criterion above)
turned out not to be a usable metric here — see `avoid_panoptex_3` below —
so these runs report **contact episodes** instead: contiguous windows
where the ground-truth X3↔Carter centre gap drops below 0.45 m.

| run | sim window (s) | RTF | laps done | yields | contact episodes: carter1 (gap<0.45 m) | carter2 | min gap c1 (m) | min gap c2 (m) | what changed |
|---|---|---|---|---|---|---|---|---|---|
| `avoid_panoptex_1` | 700.0 | 0.297 | 3 | 26 | 6 | 1 | 0.18 | 0.15 | v2 rectangle loop (C→D west strip), corridor half-width 0.65, prior learning ON during the run |
| `avoid_panoptex_2` | 700.0 | 0.298 | 4 | 22 | 7 | 1 | 0.14 | 0.19 | critic escape rule added, corridor half-width 0.55, triangle loop, lane-band refuges (prior saturated by in-run learning) |
| `avoid_panoptex_3` | 700.0 | 0.298 | 4 | 22 | 6 | 1 | 0.14 | 0.44 | prior frozen + re-warmed, `danger_half_width_m` 0.90, hold→refuge escalation (crossing still at y=-3, carter1's U-turn zone) |
| `avoid_panoptex_4` | ~400 (interrupted) | ~0.30 | 2 | 12 | 2 | 0 | 0.14 | >0.45 | crossing moved to y=+0.5 (constant-speed zone); **interrupted at ~400 s by a session restart, not a crash** |

No baseline run exists yet on the v3 loop (User B stopped the sim runs after
`avoid_panoptex_4` — see **Diagnosis** below); the baseline-vs-panoptex
comparison on the *v1* loop is `results/wiring_check` above.

**`avoid_panoptex_1`→`avoid_panoptex_2`: the critic escape rule.**
`avoid_panoptex_1` logged "No valid trajectories out of 2295" 263 times
(`predicted_risk_critic.hpp`'s own change-log comment) — DWB's first
trajectory pose is `t=0`, the robot's own current cell, and if the stack
already reads lethal there (because a Carter is adjacent), rejecting on it
makes *every* candidate illegal at once, freezing the robot and aborting
goals. Fixed by `skip_first_s` (0.5 s — poses that early aren't scored at
all, since no control choice changes them) and `escape_radius_m` (0.3 m —
poses within this distance of the trajectory's own start pose add graded
cost but never throw `IllegalTrajectoryException`, so leaving a lethal cell
always stays a legal option). 0 occurrences of "No valid trajectories" in
runs 2-4.

**`avoid_panoptex_3`: the push artefact, and why episodes replaced a gap
threshold.** One `avoid_panoptex_3` contact (t≈909 s sim, min gap 0.15 m)
lasted 50.7 s — not a single graze but the X3 held at ~0.1 m/s commanded
speed while carter1 pushed it roughly 8 m up the lane (see
`config/x3_sim_waypoints.yaml`'s own header comment, written from this
run's bag forensics, for why the crossing subsequently moved off carter1's
U-turn point). A single "closest approach" number or a raw count of
samples under a gap threshold both collapse this 50 s event to the same
weight as a single-frame flicker; **contact episodes** (contiguous
sub-0.45 m windows, with duration and the X3's position at entry) is the
metric that actually distinguishes a push from a graze, which is why the
table above reports it that way for every run.

**`avoid_panoptex_4`: the crossing moved, contacts dropped but did not
stop.** Moving the crossing off the U-turn zone to a constant-speed stretch
of both lanes cut carter1 contact episodes from 6-7 to 2 (a 1.6 s brush at
0.33 m; a 50 s push starting at (1.25, 0.72), with the X3 already inside
lane 1's corridor while carter1 had been 5.3 m away at the corridor's own
entry point) and carter2 contacts to 0. Better, not solved, and the run was
cut short by a session restart before a full 5-lap comparison could be
made.

**Diagnosis.** Gap acceptance needs `tta ≥ t_clear + t_margin_sec` ≈ 14 s at
this scenario's numbers — i.e. the Carter has to be roughly 9 m away for the
supervisor to accept a crossing — but the overhead cameras only track
carter1 within 1.5 m of ground truth 65-78% of the time in the crossing
band, and coverage is strongly asymmetric by position along the lane
(tracked-within-0.7 m samples from `avoid_panoptex_3`'s own bag: **89%** for
`y < -3.5`, dropping to **33%** for `y ≥ 3.5`, the poorly-covered north half
— consistent with the wider 25-36% range seen in other analysis of this
scenario). The practical failure mode: the X3 starts a crossing when no
Carter is currently tracked in the danger zone, a Carter then re-appears
mid-crossing, and a 0.2 m/s robot cannot clear a lane against a Carter that
the cameras themselves cannot resolve continuously. **The yield logic itself
now behaves as designed** — holds and refuges fire on the geometry described
above, and refuges land outside the learned lanes (bench evidence: the
supervisor's own docstrings and `test/test_corridor.py`) — the bottleneck
that remains is perception continuity and camera coverage of the crossing
band, not the corridor/refuge policy.

**What the priors can and cannot do (User B asked for this framed honestly).**
The spatial prior + class consequence shape the *global route* (`lane_layer`)
and the refuge search: they bias the planner off learned lanes and bias
refuges away from them, which reduces exposure to traffic over a whole
mission. They cannot substitute for a continuous 5-10 s track of the
*specific* agent about to cross — a static, learned "this is usually a
lane" fact says nothing about whether a Carter is in it *right now*, which
is exactly the information gap in the coverage numbers above. **Next
steps**: a prior-statistical crossing policy (wait in a covered pocket for
a gap the historical traffic pattern predicts, and only actually cross when
the lane's approach zone is within camera view — rather than assuming
"not currently tracked" means "clear") and extending camera coverage to the
north half of the aisle. User B stopped the sim runs here — there is no
baseline run on the v3 loop yet (baseline on the v1 loop is in
`results/wiring_check`, above).

### Limitations

- **"Wait then go" reads as "go slow now."** DWB samples candidate
  trajectories as short constant-velocity arcs; it has no candidate that
  stops, waits for a hazard to clear, and then proceeds. Both `PredictedRisk`
  and `risk_speed_governor` can only express "prefer/require a slower
  trajectory right now" — a scenario whose correct behaviour is genuinely
  "hold position 2 s, then go" gets approximated as continuous slowing
  instead.
- **Stack latency vs. perception latency.** `time_shift_s` compensates for
  how old the `RiskStack` message itself is by the time a trajectory is
  scored, but that is not the same thing as end-to-end perception latency.
  Track "age" as measured in `docs/probes_2026-09.md` (mean ≈0 s at 10 Hz
  sim on `/risk_perception/world_objects`) is close to zero because
  `object_tracker` stamps its output at *publish* time, not at detection
  time — so a near-zero age is a property of the timestamping convention,
  not evidence that detection-to-decision latency is actually near zero.
- **The Carters are blind to the X3.** Confirmed by probe: carter1's front
  2D lidar returns no hit at the X3's true bearing/range even at 0.93 m
  separation. Nothing in this stack changes that — Panoptex only makes the
  X3 more cautious; it does nothing for what the Carters themselves avoid.
- **People are static in this scenario.** The warehouse's five standing
  people don't move during either study run, so the person-distance branch
  of `risk_speed_governor` and the person-consequence weighting in the
  predictive costmap are exercised by proximity only, never by a closing
  person — a materially easier case than a person actually walking into the
  robot's path.

### Spatiotemporal risk maps (SRM) and MPPI in (x, y, t) (2026-09-10)

User B's prompt for this work was Thomas, Piat & Charpillet, "Learning
Spatiotemporal Occupancy Grid Maps for Efficient Decision-Making" (ICRA
2022, arXiv 2108.10585): each mover carries a **wide, graded risk comet**
ahead of it along its velocity (their eq. 3, `risk = max(0, 1 - d/d0)` off
the nearest occupied cell, `d0 = 2 m`, combined across occupancy levels with
a p=3 p-norm), and their modified TEB planner optimises a whole trajectory
in **(x, y, t)** against it, so "slow down and pass behind" is a solution
the planner can find at all. `panoptex_msgs/RiskStack` (21 layers, 0-6 s,
the CV rollout described above) already *is* a SOGM in shape, but each
layer was a narrow Gaussian (extent-capped ≈ 0.45 m) — a controller feels
nothing until it is almost inside the comet, which is why the
2026-09-08/09 avoidance runs above show the X3 stopping late or crawling
into the lane rather than visibly slowing early and passing behind.

**What landed**, on top of the existing `/risk_stack` pipeline (no changes
to STAGE 5 itself):

- **`risk_perception/risk_perception/srm.py`** — `stack_to_srm(stack,
  resolution, d0_m, levels=(0.2, 0.5, 0.8))`, pure numpy/scipy, eq. 3 in its
  `p → ∞` (max) form: for each layer and each level `L`, cells `≥ L` count
  as "occupied at level `L`"; `d_L` is the Euclidean distance transform to
  that set (metres); the level's term is `L · clip(1 - d_L/d0, 0, 1)`; the
  layer's SRM is the max of that term over all three levels. **Plus**: an
  occupied cell also reads its own raw occupancy value directly (not just
  the highest level it clears) — added after the WP-B bench found that,
  with only the level-max rule, a mobile robot's consequence-weighted core
  (0.75, between the 0.5 and 0.8 levels) read `0.5` at zero distance and
  could never trip the critic's collision threshold. `window_indices(...)`
  clips an axis-aligned window to the grid and reports the window's own
  `MapMetaData.origin`. 9 unit tests in `test/test_srm.py` (single-cell
  linear falloff, level-graded peak, empty/all-occupied layers, per-layer
  independence, window clipping including a centre entirely outside the
  grid, and a timing bound).
- **`predictive_risk_costmap_node.py`** computes the SRM from the raw stack
  every tick (after the STAGE 5 publish) over a `srm_window_m` (12.0 m)
  window centred on the robot, and publishes it three ways: a second
  `panoptex_msgs/RiskStack` on `srm_topic` (`/risk_stack_srm`, the window's
  own `info.origin`); `srm_now_topic` (`/risk_srm_now`,
  `nav_msgs/OccupancyGrid`, layer 0 only, for a plain RViz map display); and
  `srm_marker_topic` (`/risk_perception/srm_markers`, a `MarkerArray` over
  every layer whose cells clear `srm_marker_min` (0.3), coloured red
  ("now") fading to yellow across the horizon) — the "comet" figure the
  paper's own video shows. All gated by `publish_srm` (default true).
  `srm_d0_m` defaults to 1.5 m (the paper's 2 m would span half of the
  3.7 m aisle here); `srm_levels` defaults to `[0.2, 0.5, 0.8]`. Measured
  compute cost against a real sim run (`unit2_srm_yield_1`'s log, 21
  layers, a 121×121-cell window): **p50 ≈ 12.9 ms, p90 ≈ 14.5 ms** per
  tick — consistently ~12-13 ms across all six unit runs, comfortably under
  the `test_srm.py` 150 ms CI bound but *not* the "~9 ms" figure floated
  earlier in planning; use the measured number.
- **`panoptex_nav::PredictedRiskMppiCritic`** (`predicted_risk_mppi_critic.
  {hpp,cpp}`) now scores `/risk_stack_srm` instead of the raw `/risk_stack`:
  `cost = cost_weight · Σ_t time_discount^t · srm(x_t, y_t, t)^cost_power`,
  and a predicted collision (`srm ≥ collision_threshold`, outside
  `escape_radius_m` (0.3 m) of the rollout's own first pose) adds
  `collision_cost` (5000) once and stops scoring that rollout.
  `collision_threshold` **replaces** `lethal_threshold` — the old name is
  kept as a deprecated alias (a params file still setting only
  `lethal_threshold` keeps working, with a startup warning; one that sets
  both takes `collision_threshold`) — because on a smooth distance field
  "how close is this to an occupied core" is a geometric statement, not a
  class-confidence one. Shipped in `config/nav2_x3_panoptex_mppi.yaml`:
  `cost_weight: 15.0` (bench-chosen — see below; the C++ struct's compiled
  default is `30.0`), `cost_power: 2.0`, `time_discount: 0.97`,
  `collision_threshold: 0.6` (the compiled default is `0.90`; 0.6 is low
  enough to trip on a mobile robot's own consequence-weighted core once the
  "reads its own occupancy" fix above is in the stack it's scoring),
  `skip_first_s: 0.3`, `escape_radius_m: 0.3`. gtest coverage in
  `test/test_predicted_risk_mppi_critic.cpp` includes the **pass-behind**
  property: for a mover crossing the path at t≈2 s, "go now" at 0.26 m/s
  (drives into the core) costs **5135.6**, "wait 3 s then cross behind"
  costs **181.5**, and — the test's own logged surprise — "dash across
  ahead at 0.60 m/s" costs even less, **123.3**: at these speeds the robot
  cannot out-wait a comet 2·d0 = 3 m across within a 6 s horizon, so
  dashing keeps more true clearance than loitering just short of the
  crossing. All three are far below the ~5000 collision floor except the
  first.
- **Bench** (`results/avoidance/bench_srm.md`, domain 199, no Isaac,
  `tools/bench_mppi_arm.py` driving a real `controller_server`): crossing
  hazard min distance **1.66 m** with the SRM (`cost_weight 15`) vs.
  **0.49 m** with the old narrow field; head-on **1.58 m** vs. **0.45 m**.
  Both SRM manoeuvres cost about ⅓ more time-to-goal, inside the bench's
  2× budget. `cost_weight` 15/30/60 were within 5% of each other on every
  metric — the pick is not delicate once the quadratic term is large
  enough to out-pull `PathFollowCritic`. The bench's own caveat, now
  resolved: it flagged that `stack_to_srm` peaked at `0.8` (the highest
  level), not `1.0`, on a fully-occupied cell — below the shipped
  `collision_threshold`. The "occupied cell reads its own occupancy value"
  fix above (same day, after the bench) closes that gap for a real
  `/risk_stack_srm`; the bench numbers themselves used a renormalised
  field and are unaffected.

### Oracle perception mode (2026-09-10)

`x3_nav.launch.py` gained `perception:=panoptex|oracle` (default
`panoptex`, i.e. every study run to date). `perception:=oracle` replaces
the entire camera chain (overhead cams, RGB-D, GDINO/SAM2,
`object_tracker_node`) with **`gt_tracks`** (`panoptex_nav/gt_tracks_node.
py`, installed as the `gt_tracks` console script): it subscribes Isaac's
`/gt_tf` (`World → carterN`, ~60 Hz — `World` *is* the `map` frame, no
localization needed) and republishes ground-truth `vision_msgs/
Detection3DArray` on the same `/risk_perception/world_objects` topic the
real tracker uses, in the same packed `class_id` convention, so nothing
downstream (predictive costmap, critics, `mission_supervisor`) can tell
the difference. Per tracked robot (`robots` param, default `["carter1"]`):
position and velocity (EMA of a 0.5 s displacement-window finite
difference, `vel_window_sec`), `pmot = 1.0` at/above `moving_speed_mps`
(0.15 m/s) else `0.0`, label `"mobile robot"`, bbox `0.75 × 0.50` m,
position covariance `0.01`. This isolates prediction + planning from
detection entirely — the point being to validate the SRM/MPPI work above
on a clean signal before re-introducing the perception noise/dropouts
documented in the 2026-09-09 results.

**`gt_centre_offset_m` (default `-0.23`).** `/gt_tf`'s per-Carter frame is
`chassis_link`, confirmed against `warehouse/nav/carter_nav_params.yaml`'s
footprint (`[[0.14, 0.25], [0.14, -0.25], [-0.607, -0.25], [-0.607,
0.25]]`) to be the drive-axle origin, not the body centre — the body
centre sits `(0.14 + -0.607)/2 ≈ -0.23` m along the local +x axis, i.e.
~0.23 m *behind* the axle. The default shifts the reported point back by
that much along the heading (velocity direction while genuinely moving,
the `/gt_tf` quaternion's yaw otherwise). This resolves the "-0.17 m
along-track bias, suspected GT reference ≠ body centre" open item left in
`docs/probes_2026-09.md`'s Addendum 5 — the earlier suspicion had the
offset roughly right in magnitude but backwards in sign/interpretation.
Unit-tested in `test_gt_tracks.py` with no `rclpy` spin (yaw/velocity/EMA/
classify/offset/build-fields are all pure functions).

### Unit-crossing scenario (v1 → v2, why)

To validate the SRM/MPPI work in minutes rather than the ~45 min four-arm
study, a minimal scenario removes everything but one crossing: one Carter
shuttling a short lane, the X3 walking a two-waypoint back-and-forth
straddling it (`laps:=2` or more = 2× that many crossings), oracle
perception, no cameras.

- **v1** (`warehouse/nav/carter_shuttle.yaml`, `panoptex_nav/config/
  x3_unit_cross.yaml`): carter1 shuttles the real patrol lane, `x = 1.27`,
  between U-turns 3 m from the crossing at `y = 0.5`; the X3's west
  waypoint sits at `x = 0.65`. **Abandoned**: `x = 0.65` is only ~0.67 m
  from the lane centre — on the edge of any sensible risk field rather than
  a real pocket outside it, so the MPPI arm parked there and grazed the
  Carter anyway (`unit_srm_2`, min gap 0.40 m).
- **v2** (`warehouse/nav/carter_shuttle2.yaml`, `panoptex_nav/config/
  x3_unit_cross2.yaml`): carter2 shuttles its own west lane, `x = 2.2`,
  between `y = -1.0` and `y = 4.0`; carter1 is parked out of the way at
  `(1.27, -4.0)`. The X3's west waypoint moves out to `x = 1.0` — a real
  0.9 m body clearance from the moving Carter and 0.67 m from the shelf
  face — making the west side an actual pocket the planner can retreat
  into rather than a graze waiting to happen.

`results/avoidance/run_unit.sh <arm> <name> [sim_seconds=150]` runs either
scenario headlessly against Isaac Sim (domain 44): env `LAPS` (default 2),
`X3_WPS`/`CARTER_WPS` (waypoint file overrides — v2 needs both set),
`PERCEPTION` (default `oracle`), `YIELD` (`mission_supervisor`'s
`yield_enabled`, default `false`), `PANOPTEX_NO_CM` (`1` disables the
collision monitor — used for pure planner tests so it can't mask a
critic-side miss), `RUNNER_DELAY`. Ends on `/mission/state` reporting
"complete" or `sim_seconds` elapsing; wall time ≈ 5-10 min at RTF ≈ 0.42
(no GDINO/SAM2 in the oracle-perception loop). Analysis: `tools/
analyze_run.py --lane-x <x>` adds per-crossing labels (X3 lane-entry/exit
vs. the Carter's own along-lane position at that instant — `behind`/
`ahead`/`waited`/`contact`), min GT gap, and X3 approach speed;
`tools/unit_summary.py <run>...` collects each run's `analysis.md` into
`results/avoidance/UNIT_SUMMARY.md`.

**Known issue in `tools/unit_summary.py`'s "crossings" column.** Its regex
for counting per-carter crossing-label rows also matches that carter's row
in the *other*, single-row summary table above the crossing table in
`analysis.md` (both start `| carterN | <number> | <word> |`), adding one
spurious "crossing" with a numeral label to every run. It never lands on
`waited` or `contact` (those counts are correct), only on the total. Real
crossing counts, read directly from each run's own `analysis.md`: `unit_
srm_1b` 2 (not 3), `unit_srm_2` 4 (not 5), `unit2_srm_1b` 8 (not 9), `unit2_
srm_yield_1` 8 (not 9), `unit2_base_2` 6 (not 7). Not fixed here
(documentation-only pass) — flagging for whoever next touches that script.

### Results — unit-crossing runs (2026-09-10)

**Oracle perception, below.** Every run in the table below used
`perception:=oracle` (ground-truth Carter tracks, no cameras/GDINO/SAM2) —
it costs RTF ≈ 0.3 against oracle's 0.42 and, before the ghost-mover fix
just below, would have reintroduced the tracking/coverage errors documented
in the 2026-09-09 results above. Treat this table as "the planner and
prediction work correctly given a clean track," not "the full stack avoids
this Carter." **The `perception:=panoptex` variant on scenario v2 has since
been run once** (`unit2_pan_1`, first real-perception unit run) — see
**Real-perception ghost movers and map-filling comets** and **Real-perception
unit run (unit2_pan_1)** below for that result; it is kept out of this
oracle-only table because it is not a like-for-like comparison (fewer laps,
carter2 mostly stalled under load).

Six laps, domain 44, headless Isaac, collision monitor off
(`PANOPTEX_NO_CM=1`), same scenario per row unless noted (crossing counts
below are corrected per the known `unit_summary.py` issue just above):

| run | arm | scenario | RTF | crossings | waited | yields | contacts (<0.45 m) | min gap (m) | aborts |
|---|---|---|---|---|---|---|---|---|---|
| `unit2_base_2` | `baseline_mppi` | v2 | 0.4196 | 6 | 0 | 0 | 2 | 0.34 | 0 |
| `unit2_srm_1b` | `panoptex_mppi`, yield off | v2 | 0.4191 | 8 | 5 | 0 | 1 | 0.31 | 0 |
| `unit2_srm_yield_1` | `panoptex_mppi`, yield on | v2 | 0.4197 | 8 | 7 | 10 (5 refuge/24.0 s, 3 hold/5.2 s) | 0 | 0.81 | 0 |
| `unit_srm_1b` | `panoptex_mppi`, yield off | v1, 2 laps | 0.4189 | 2 | 1 | 0 | 0 | 2.02 | 0 |
| `unit_srm_2` | `panoptex_mppi`, yield off | v1 | 0.4196 | 4 | 1 | 0 | 2 | 0.40 | 0 |

`unit2_base_1` is invalid (carter2 never left its parked start — 0 real
passes) and kept only as `unit2_base_1_INVALID_carter_parked/`.

**Headline (scenario v2, identical 6-lap runs, oracle tracks): adding the
SRM critic alone cuts contacts 2→1 and raises min gap 0.34→0.31 m only
slightly, but adding the corridor-yield supervisor on top reaches 0
contacts and 0.81 m min gap — the first clean run of this study.** Reading
the three v2 rows as one progression: `baseline_mppi` never waits and hits
the Carter twice; the SRM critic alone (no yield behaviour) makes the X3
wait 5 of 8 crossings and drops to one contact, but still doesn't budget
enough clearance on its own; adding `mission_supervisor`'s corridor-yield
policy (10 yields — 5 refuges averaging 4.8 s, 3 holds averaging 1.7 s)
removes the remaining contact entirely.

**The residual failure class (both remaining contacts, `unit2_base_2` and
`unit2_srm_1b`): the same event.** Both are carter2 leaving its parked
south U-turn only ~1.5 m from the crossing — while parked, gt_tracks
reports no velocity comet at all (a stationary track's stack contribution
is a narrow blob, not the wide SRM field), so a mover that starts moving
close to the crossing is invisible as a hazard until it is already close.
This is exactly the gap the paper's *learned* SOGM would close — it learns
"objects tend to appear at doorways/junctions" as a prior over the map
itself, independent of any single track's current motion state. Our stack
has no such prior yet.

v1 (`unit_srm_2`, min gap 0.40 m) shows the pocket-placement failure mode
described above: with the X3's retreat point only ~0.67 m from the lane
centre, the MPPI arm treated "parked at the pocket" as itself inside the
risk field and grazed the Carter rather than truly stepping clear.

**Next steps**, in order of expected payoff: (1) run `perception:=
panoptex` once on scenario v2 to measure what real perception costs
against this oracle baseline; (2) move v2's south U-turn from `y = -1.0`
to `y = -3.0` (more clearance from the crossing, closing the
parked-Carter-restarts-close failure above without touching the critic);
(3) a learned "appearance" prior — extend `spatial_prior_node`'s existing
per-cell traffic statistics into a "movers restart here" density that the
SRM conversion (or the crossing policy directly) can add to a currently-
stationary track's otherwise-empty comet, closing the gap the paper's
learned SOGM was built for and this fixed-CV-rollout stack cannot express
on its own.

### Real-perception ghost movers and map-filling comets (fix, 2026-09-10)

Attempting the "run `perception:=panoptex` once on scenario v2" next step
listed above surfaced a new problem before the SRM/MPPI work could even be
judged with real cameras in the loop: the SRM filled roughly half the map
with comets, visible in RViz screenshots taken during the attempt. Three
independent root causes, all in `risk_perception`:

- **Unbounded rollout uncertainty.** `predictive_risk_costmap_node.py`'s
  `paint_track` grows each layer's positional variance as `P + t²·P_v +
  vel_inflation·t`; `object_tracker_node.py` seeds a freshly spawned
  track's velocity covariance `P_v` at `1.0 (m/s)²` (the Kalman filter's
  own prior, not a deliberate choice about rollout uncertainty), so by the
  6 s far end of the horizon that prior alone inflates sigma to roughly
  6 m — a track barely two updates old paints a comet almost as wide as
  the map, regardless of how implausible its estimated velocity is.
- **Ghost movers.** `lidar_cluster` tracks spawned off shelf edges (the
  scan clusterer's old `static_margin_m` 0.15 m with no persistence
  requirement let a single noisy scan spawn a track), and **every**
  category showed association-jump velocity spikes — an audit before the
  fix found 36 of 122 tracks moving faster than their own class's speed
  cap, including parked forklifts, tables, chairs, people and Carters.
  `stack_paint_unknown_moving` (on at the time) painted 15 of those as
  unknown-category movers.
- **Stationary objects drawn as moving.** The same stationary hypothesis
  was replicated per rollout layer, so a parked object with any nonzero
  `p_motion` estimate got drawn yellow (moving) at every layer instead of
  staying a static blob at layer 0 only.

**Fixes** (`config/risk_perception.yaml`, verified against the file):
`predictive_risk_costmap_node`'s `sigma_v_max_mps: 0.5` and
`max_sigma_m: 0.6` clamp the velocity-covariance prior before it can feed
the rollout's `t²` growth term, independently capping the resulting spatial
spread; `min_track_age_for_motion_sec: 1.0` and `min_hits_for_motion: 5`
keep a young track's hypothesis stationary-only; `max_speed_person_mps: 2.0`
/ `max_speed_robot_mps: 1.5` / `max_speed_wheeled_mps: 2.0` /
`max_speed_unknown_mps: 1.2` are per-class speed caps read by both the
predictive node and the tracker; `stack_paint_unknown_moving: false` (was
on) stops the stack painting any UNKNOWN-category track as moving at all,
gated by `unknown_moving_pmot_min: 0.8` / `unknown_moving_min_hits: 8` /
`unknown_moving_min_age_sec: 2.0` for the rare case it's re-enabled;
`srm_marker_delta: 0.05` restricts comet markers to layers `k` whose SRM
value exceeds layer 0's by at least that much, i.e. only where the track is
actually predicted to move into new territory. `object_tracker_node.py`
adds `max_speed_*_mps` (matching the predictive node's caps) plus
`max_speed_default_mps: 2.5` for any class not listed (furniture) and
`jump_reset_factor: 2.0`: a post-update speed between the cap and
`jump_reset_factor × cap` is clamped to the cap (direction kept, treated as
a genuine but noisy estimate); above `jump_reset_factor × cap` is treated
as an association jump (e.g. a lidar-cluster track re-associating with a
different shelf edge) and both velocity and `p_motion` are reset to zero,
logged at `WARN`. `class_id` now carries `|hits=N|age=S` in addition to the
existing fields, parsed by `risk_visualization.parse_class_id`.
`scan_cluster_detector_node.py` / `scan_clustering.py` raise
`static_margin_m` to `0.35` (must clear AMCL's own error budget, not just
be non-zero), add `cov_base_m2: 0.05`, and require `min_consecutive_scans:
3` within `persist_match_m: 0.3` of a cluster's previous position before it
counts as a track candidate (`scan_clustering.ClusterPersistence`).
`risk_perception`'s pytest suite: 252 passing (re-verified this pass with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`; two pre-existing,
unrelated `flake8`/`pep257` style-lint failures).

This fixed the *spatial* runaway (comets no longer fill the map). It did
not fix a second, independent problem in the same real-perception run —
**a parked object's own Kalman velocity reading as motion** — found the
same evening once `unit2_pan_1` (below) was analysed; see **Phantom
velocity on parked tracks (fix, 2026-09-10 late)** further down.

**Before/after** (`results/avoidance/unit/AUDIT_BEFORE_AFTER.md`,
`tools/track_audit.py`, scenario v2, 120 s sim, 1 lap, cameras + lidar):

| capture | tracks | ghost movers (speed > class cap) | unknown movers (pmot>0.5) | SRM area ≥0.45, layer 0 (p50) | SRM area ≥0.45, +6 s (p50) |
|---|---|---|---|---|---|
| before | 122 | 36 | 15 | 7.0 % | 13.6 % |
| after | 41 | 1 | 1 | 0.2 % | 0.4 % |

The one remaining "ghost" after the fix is a lidar cluster clamped *at* its
1.2 m/s class cap, not an unbounded spike; unknown-category tracks are no
longer painted as movers at all by default (`stack_paint_unknown_moving:
false`). Per-track tables: `results/avoidance/unit/audit_{before,after}/
track_audit.md`.

### Real-perception unit run (unit2_pan_1)

`unit2_pan_1` (2026-09-10 18:26, `results/avoidance/UNIT_SUMMARY.md`,
detail in `results/avoidance/unit/unit2_pan_1/`) is the **first
real-perception unit-crossing run** (`perception:=panoptex`, cameras +
lidar, `panoptex_mppi` + SRM critic + corridor-yield). RTF 0.27, 4 laps in
220 s (vs. the oracle table's 6), 11 yields, 0 aborts. The ghost-mover fix
held under a real run: SRM area ≥0.45 1.7 % / 1.6 % (layer 0 / last layer),
in the same range as the before/after table's "after" column, nowhere near
the pre-fix 7.0 % / 13.6 %.

**Result: 1 contact (5.8 s, min gap 0.23 m).** carter2 moved for only 11 %
of the run — its own nav2 aborted or failed to plan repeatedly under load —
so there was exactly one real encounter with a moving Carter, and it ended
in contact. Compare the oracle baseline on the same scenario,
`unit2_srm_yield_1`: 0 contacts, 0.81 m min gap.

**Perception quality**: carter2 was tracked within 1 m of ground truth only
56 % of the time (vs. carter1's 99.4 %), median offset 0.33 m (p90 0.86 m),
across 7 distinct track ids for the one Carter; most of the run's yields
were released by "track lost" after a 3.6 s timeout.

**Failure mechanism**: the ~0.3 m eastward track offset, combined with the
0.9 m-wide danger corridor `mission_supervisor` sweeps around a tracked
user, left no legal refuge cell on the east column at all — every cell
there needs 0.45 m clearance from the shelf beyond `x ≈ 3.6`, and none of
them had it once the corridor's own extra width was subtracted. `find_refuge`'s
least-bad fallback (see `panoptex_nav/README.md`'s **Mission supervisor**
section for the search logic) answered with a refuge on the **west** side
of the lane, and the X3 crossed in front of carter2 to reach it — the run's
one contact.

**Honest status.** The oracle result (`unit2_srm_yield_1`, 0 contacts)
still stands as a validation that the planner and prediction logic work
correctly given a clean track. Real perception now produces clean,
physically plausible comets — the ghost-mover/map-filling problem above is
fixed — but the closed loop is **not yet collision-free with cameras in
it**. Next: (i) the refuge search needs to tolerate track offset — widen
the search radius, relax clearance on the far side, or simply never choose
a refuge across the lane from the approaching user regardless of what the
side-preference ranking says; (ii) track continuity (7 ids for one Carter,
56 % coverage) needs work in the tracker's association/coverage, not just
the refuge logic; (iii) Carter-side nav2 needs to be more robust at low RTF
(it stalled for 89 % of this run), or the unit test should drive Carter
with a scripted mover instead of its own nav2 stack.

### Phantom velocity on parked tracks (fix, 2026-09-10 late)

**Symptom.** Even with the ghost-mover/map-filling fix above in place,
under `perception:=panoptex` the SRM "comet" around a **parked** object
still grew to a 10-13 m radius / 30-50 m² region — oracle mode never shows
this.

**Evidence** (bag `unit2_pan_1`, read offline): carter2's ground truth
(`/gt_tf`) was parked 90 % of the run, but its track #39 ("ground mobile
robot") reported speed p50 0.41 / p90 1.23 / max 2.00 m/s, `pmot > 0.5` in
44 % of samples, and some 20 s bins pinned at the 1.5 m/s class cap with
`pmot = 1.0` while GT = 0.00. Every other static object — tables, carts,
forklifts, the X3's own self-track — showed the same pattern. Published
`Pvx` ran 0.6-2.0 (m/s)², `Pxx` up to 8 m², and the tracker log had 7
"speed jump reset" lines with 10-25 m association displacements.

**Root cause.** The Kalman filter's velocity state is a finite difference
of jittery measurements: `process_noise 0.5` and a `P_v` seeded at
`1.0 (m/s)²`, against ~0.3 m inter-camera ground-plane offsets and a
persistent lidar-vs-camera offset, gave a steady-state `sigma_v ≈ 0.9 m/s`
on an object that never moved. `Track.update_beliefs`' motion test fired
on speed ≥ 0.3 m/s combined with only a 0.5-sigma Mahalanobis score —
jitter alone cleared it. `_gate_for` was uncapped, so a long-coasting
track could re-associate 10-25 m away. `paint_track_planner`
(`predictive_risk_costmap_node.py`) never received the earlier
`sigma_v_max_mps`/`max_sigma_m` clamps or the `moving_allowed` gate that
`paint_track` already had, so `/risk_costmap_planner` kept fanning out
raw `pmot`/`Pvx`. `"ground mobile robot"` was missing from
`risk_visualization.CLASS_BASE_RISK` — consequence 0.40 instead of the
oracle's `"mobile robot"` 0.75, a silent oracle/panoptex divergence.
`_srm_markers` drew comet layers `k ≥ 1` with only a 0.05 delta gate,
stacking 21 translucent layers into what looked like a solid region.

**Fix.** A new pure module, `src/risk_perception/risk_perception/
motion_evidence.py`, gives the tracker displacement-window motion evidence
— the same idea `panoptex_nav/gt_tracks_node.py` already uses for oracle
tracks — instead of trusting the Kalman filter's own instantaneous
velocity. `Track` keeps a deque of raw `(stamp, x, y, cov)` measurements
over `motion_window_s`; `displacement_evidence()` (default method
`half_median`) splits that window in half **by time**, takes the
coordinate-wise **median** position of each half, and compares the two —
a persistent inter-source offset that's roughly constant within each half
cancels out of the median-to-median difference instead of alternating
into inflated path length, which is what the original `endpoints` method
(newest-vs-oldest displacement gated by a straightness/path-length ratio,
kept selectable via `motion_disp_method: endpoints`) suffered from: with
two fused sources (a 10 Hz lidar-cluster stream and a slower camera
stream) at a persistent ~0.3 m relative offset, a genuine straight-line
mover read as near-zero straightness (reads as zigzag) and got rejected.
`p_motion` now comes from this displacement evidence
(`motion_require_displacement: true`), the Kalman velocity state is
damped toward zero below `p_motion < motion_vel_damp_below`, and the
published velocity is the displacement estimate itself
(`velocity_source: displacement`) rather than the (now-damped) Kalman
state. `class_id` gains a `|disp=<net_disp_m>` field alongside the
existing `|hits=|age=` fields. On the predictive side, a pure
`resolve_moving_allowed()` adds a `min_speed_for_motion_mps` floor
(matching oracle's own motion threshold), `paint_track_planner` gets the
same `moving_allowed`/`sigma_v_max_mps`/`max_sigma_m` clamps
`paint_track` already had, `risk_visualization.CATEGORY_BASE_RISK` adds a
label→category consequence fallback so `"ground mobile robot"` reads 0.75
like the oracle's `"mobile robot"`, and a pure `srm_marker_mask()` adds an
absolute `srm_marker_min` floor to layers `k ≥ 1` (not just the delta) plus
a `srm_marker_layer_stride` so only every 3rd predicted layer draws (7
layers instead of 21). RViz's reactive "Risk Costmap" display (`/risk_
costmap`, which double-paints every halo already scored via "SRM (now)")
is now off by default.

| param | file | old | new |
|---|---|---|---|
| `process_noise` | `object_tracker` | 0.5 | 0.05 |
| `velocity_var_init` | `object_tracker` | 1.0 (hardcoded) | 0.25 |
| `gate_speed_mps` | `object_tracker` | 1.0 | 0.6 |
| `gate_max_m` | `object_tracker` | uncapped | 2.5 |
| `motion_require_displacement` | `object_tracker` | n/a (new) | true |
| `motion_window_s` | `object_tracker` | n/a (new) | 2.0 |
| `motion_min_displacement_m` | `object_tracker` | n/a (new) | 0.3 |
| `motion_disp_k_sigma` | `object_tracker` | n/a (new) | 1.5 |
| `motion_min_samples` | `object_tracker` | n/a (new) | 2 |
| `motion_disp_method` | `object_tracker` | n/a (new) | half_median |
| `motion_hold_s` | `object_tracker` | n/a (new) | 3.0 |
| `motion_vel_damp_below` | `object_tracker` | n/a (new) | 0.5 |
| `velocity_source` | `object_tracker` | n/a (new, was always `kf`) | displacement |
| `min_speed_for_motion_mps` | `predictive_risk_costmap_node` | n/a (new) | 0.15 |
| `srm_marker_delta` | `predictive_risk_costmap_node` | 0.05 | 0.15 |
| `srm_marker_layer_stride` | `predictive_risk_costmap_node` | n/a (new, 1 = every layer) | 3 |

**Offline harness.** `tools/retrack_bag.py --bag <bag_dir> --mode {original,retrack,both} [--param k=v]* [--assert]`
replays a bag's `/risk_perception/detections_3d_map` through a headless
`ObjectTrackerNode` (`ROS_DOMAIN_ID=199`, never spun) plus the same
`paint_track`/`stack_to_srm` pipeline the live node uses, so a
tracker/predictive-param change can be judged against ground truth
(`/gt_tf`) in seconds, without Isaac. `--assert` exits 1 unless: phantom
rate < 5 %, missed-mover rate < 20 %, each tracked Carter > 80 % within
1 m with ≤ 3 ids, comet area p90 < 10 m², comet radius p90 < 4 m.

**Before/after** (`unit2_pan_1`'s bag, `tools/retrack_bag.py`):

| capture | phantom-mover rate | missed-mover rate | comet area p50/p90/max (m²) | carter continuity |
|---|---|---|---|---|
| `original` (before, as-recorded) | 25.4 % | 12.4 % | 2.9 / 9.7 / 29.7 (legacy delta-0.05 mask max 57.7) | carter2 57 % / 9 ids |
| `retrack` (after, new defaults) | 2.4 % (Carter-adjacent 1.3 %, far static 2.5 %) | 33.3 % | — / 8.6 / — | carter1 100 % / 6 ids; carter2 62 % / 5 ids |
| `audit_after` bag, `retrack` | 0.6 % | 19.3 % | — / 0.0 / — | — |
| `unit2_srm_yield_1` (oracle bag), `original` | 0.0 % | — | — | — |

Parked-object net displacement (`unit2_pan_1`, retrack) p90 0.33 m. The
`endpoints` method with only a straightness gate reached 4.9 % phantom at
best — still worse than `half_median`'s 2.4 %, for the two-source-zigzag
reason above. Phantom rate and comet area both now pass `--assert`;
missed-mover rate and carter2 continuity do **not** yet — that's the
known camera-coverage/continuity bottleneck documented in `unit2_pan_1`
above, not a threshold-tuning problem, and is tracked as a separate open
item. Tests: 337 pytest passed (`risk_perception` + `tools`), 1 skipped;
`flake8`/`pep257` style-lint tests fail as before (pre-existing repo-wide
debt, unrelated to this change).

**RESULT unit2_pan_2** (2026-09-10 22:03 EDT, confirmation sim run, same
recipe as `unit2_pan_1`: scenario v2, `panoptex_mppi`, `perception:=
panoptex`, yield on, 4 laps, collision monitor off; RTF 0.276; the
harness's first `carters_patrol` launch stalled in Configuring as usual,
retry activated in 11 s). **The comet fix held live.**
`tools/retrack_bag.py --mode original` on the new bag: phantom-mover rate
2.6 % (`unit2_pan_1`: 25.4 %), Carter-adjacent 1.4 %; comet area
p50/p90/max 0.2/7.4/20.1 m² (was 2.9/9.7/29.7), comet radius p50/p90/max
1.1/2.8/4.1 m (was 2.3/3.7/6.8), legacy delta-0.05 mask max 37.0 m² (was
57.7); 0 "speed jump reset" lines in the tracker log (was 7).
`track_audit`: 15 tracks over class cap (was 26), 2 unknown movers painted (was
6) — its "SRM ≥0.45 area" figure itself rose 1.7 % → 5.3 %, but only
because `"ground mobile robot"` now carries consequence 0.75 instead of
0.40 (this fix's own class fallback) and carter2 actually moved this run;
that metric isn't comparable across the two runs, the comet-mask numbers
above are.

Mission stats: 4 laps, cmd_vel activity 0.955, 16 yields, 4 holds (10 s),
12 refuges (76 s), 0 aborts, 0 planner failures, 0 MPPI fail lines.
Perception continuity improved but fragmented: carter2 tracked within 1 m
87 % (`analyze_run`) / 100 % (`retrack_bag`, wider matching) of the time
vs 56 % in `unit2_pan_1`, median offset 0.31 m — but split across 17-26
distinct ids (was 7); carter1 91 % / 8 ids. `pmot>0.5` while moving fell
to 66.6 % for carter2 (was 83.2 %) — the displacement test's delayed
onset.

**But contacts went up, not down: 7 contact episodes with carter2
(<0.45 m, 28.7 s total, min GT gap 0.179 m) vs 1 in `unit2_pan_1`.** The
comparison isn't apples-to-apples — carter2's own nav2 worked this run
(roughly 10 real passes through the crossing) instead of moving only 11 %
of `unit2_pan_1`, so exposure is far higher, not the fix regressing.
Per-contact forensics (`world_objects` vs `/gt_tf`, 10 s before each
contact) found a new, distinct failure class: in 5 of 7 contacts carter2
was tracked as **moving correctly** (`pmot` 1.0, speed ≈0.5 m/s,
displacement ≈0.6 m) but under a **furniture label** (`"table"` ids
102/208, `"chair"` id 28) — and furniture is excluded from the risk stack
(`stack_categories`), the speed governor (`_ACTIONABLE_CATEGORIES`), and
the corridor-yield users (`corridor.CORRIDOR_CATEGORIES`), so it produced
no comet and no yield. 1 contact was a stale-track hand-over (`"ground
mobile robot"` id 103 sat at `pmot=0` while detections moved to a new id
that only reached `pmot=1` 3 s before contact). The remaining contact
(t≈1282 s) had `"ground mobile robot"` id 278 tracked correctly —
`pmot=1.0`, correct velocity — for the whole approach: the known
refuge-geometry blocker from `unit2_pan_1` above, not a perception
failure.

**Both true at once, again**: the comet/phantom-velocity problem is fixed
and confirmed live, and the closed loop is still not collision-free with
cameras in the loop — the dominant new failure class is a mislabelled
mover (GroundingDINO calling a moving Carter a table or a chair). **The
fix has since landed (uncommitted):** `object_tracker_node.py`'s
`Track.update_promotion`/`published_label` promote a furniture/unknown
track with sustained displacement evidence to label `"moving object"`
(`LABEL_CATEGORIES` → `wheeled`, `CLASS_BASE_RISK` 0.65) once
`mover_promote_pmot_min: 0.8`, `mover_promote_min_s: 1.0`,
`mover_promote_min_disp_m: 0.5`, and `mover_promote_min_hits: 5` are all
met (`mover_promote_enabled: true`), demoting after `mover_demote_s: 60`
without renewed evidence; the raw label/category is kept for association,
only the *published* label changes, and `class_id` gains a
`|promoted=0/1` field.

`tools/retrack_bag.py` gained matching metrics — "actionable-mover rate"
(Carter-adjacent, GT-moving samples whose published category is
`robot`/`wheeled` and `pmot>0.5`) and a "mislabelled" share, with an
`--assert` target of >70 % actionable — and the fix is **offline-verified
but not yet run live**. `unit2_pan_2`'s bag, `original`→`retrack`:
actionable 41.1 % → 63.3 %, mislabelled 39.7 % → 11.0 %, phantom rate
2.6 % (unchanged), comet area p90 7.4 → 0.7 m², 21 promotions (raw labels
table/chair/ground/cable). `unit2_pan_1`: actionable 66.7 %, mislabelled
1.3 %, 6 promotions. `audit_after`: actionable 77.0 %, 2 promotions. The
>70 % actionable target is **not** met on the two larger bags — the
remaining gap tracks the missed-mover rate (33-35 %) almost exactly, i.e.
motion-onset latency plus id fragmentation (carter2 still 17-26 ids), a
continuity problem, not class exclusion. Tests: 344 passed, 1 skipped,
same 2 pre-existing style-only failures. Next: a live confirmation run
with this fix in place.

### Class-agnostic (motion-first) risk map and stable tracking (2026-09-11)

**Design decision.** User B asked whether the SRM/tracking should be
class-agnostic like Thomas, Piat & Charpillet (2021)'s dynamic/movable
framing — a mover is a mover, the noun never gates or weights risk — and
whether union-by-max composition was the right way to combine the four
priors (semantic, behavioural, relation, spatial). Verified against the
code: **union-by-max is correct everywhere** track blobs merge
(`_splat_into`, `risk_costmap_node`, `spatial_prior_node`), SRM levels
merge (`srm.py`), and both nav2 costmap layers (`risk_layer`, `lane_layer`)
merge — one flaw found and fixed: inside a single track,
`encounter_geometry.combine_severity` (`consequence * encounter_factor +
relation_bonus`) was never re-clipped, so a high-consequence track on a
close course could exceed 1.0 and reach the SRM core reading as "more than
fully occupied" — now `min(1.0, ...)`. CPA/TTC stays deliberately out of
the stack/SRM: the critic indexes SRM layers by rollout time, so TTC is
already implicit in the time-indexed lookup; CPA/TTC only feeds the
collapsed DWB grid and the governor's `closing` cap. `p_movable` is parsed
by the tracker but not consumed by the risk map (only the spatial prior
reads it) — unchanged by this round.

The class-agnostic change itself is **motion-first, not class-blind**:
every confirmed track paints the stack/planner/collapsed grids at one flat
consequence, `stack_consequence_agnostic` (0.75 — between the old
"wheeled" 0.65 and "person" 0.90), regardless of category.
`stack_categories` no longer gates entry. The semantic prior
(`CLASS_BASE_RISK`/`CATEGORY_BASE_RISK`) becomes an **optional modifier**,
off by default (`semantic_modifier_enabled: false`); when on, it can only
push the reading up — `max(agnostic, class_value)`, never a substitution,
never a veto. The governor acts on any mover
(`risk_speed_governor.act_on_any_mover`, default true): any track clearing
`moving_pmot_min` enters the `closing`/mover branch that used to require
category `robot`/`wheeled`; `static_slow` stays restricted to
robot/wheeled/promoted, since a static chair is Nav2's own obstacle
layer's job. Corridor users
(`corridor.select_corridor_users`/`corridor_users_any_mover`) are any
mover except `corridor.NON_CORRIDOR_CATEGORIES = ("person",)` — a person's
swept-rectangle prediction is fiction (people stop, turn on the spot,
step around things), handled by the governor's distance-only branch
instead, not a class veto on risk. All three ablation knobs
(`stack_motion_first`, `act_on_any_mover`, `corridor_users_any_mover`)
default to the new behaviour but can be flipped back to the pre-2026-09-11
category-gated behaviour for regression/comparison runs.

**Tracker (WP-B).** `object_tracker_node`'s `association_mode: agnostic`
(new default; `legacy` kept, byte-for-byte the old behaviour) replaces
exact/category-keyed association with a Mahalanobis cost on the
age-corrected predicted position (`d² ≤ assoc_chi2_gate` 9.21, the 2-DOF
99% chi-square gate) plus `assoc_size_weight` 2.0·|Δsize| and a *soft*
`assoc_label_penalty` 2.0 when categories mismatch and neither is
"unknown" — a phrase-churn mismatch still associates if the motion
evidence is strong enough, it just costs more, so Hungarian still prefers
a same-category match when one is available within gate. Category no
longer decides identity, only nudges cost. `Track.label_votes`
(score-weighted, 30 s half-life Counter) decides the *published* category
by majority vote instead of latest/most-confident-wins, so single-frame
GDINO phrase flicker doesn't relabel a track; `_revive` is label-blind
(Mahalanobis against `P + QΔt`, velocity-extrapolated). Confirmation is
N-of-M (`confirm_n` 3 of `confirm_m` 6 most recent association outcomes,
not "hits ≥ min_hits ever") so an intermittently-observed track can
confirm without consecutive good ticks; a miss is charged **at most once
per `confirm_miss_grace_s`** (1.5 s) of continuous silence via
`_charge_stale_misses` in `_tick`, not per incoming detection message —
see the regression below for why that distinction mattered. An
unconfirmed track with ≥ `graveyard_min_hits` (2) hits goes to the
graveyard instead of being dropped outright, revivable label-blind later.

| param | file | default |
|---|---|---|
| `stack_motion_first` | `predictive_risk_costmap_node` | true |
| `stack_consequence_agnostic` | `predictive_risk_costmap_node` | 0.75 |
| `semantic_modifier_enabled` | `predictive_risk_costmap_node` | false |
| `stack_moving_pmot_min` | `predictive_risk_costmap_node` | 0.5 |
| `max_speed_default_mps` | `predictive_risk_costmap_node` | 2.0 |
| `act_on_any_mover` | `risk_speed_governor` | true (1.0) |
| `corridor_users_any_mover` | `mission_supervisor` | true (1.0) |
| `association_mode` | `object_tracker` | agnostic |
| `assoc_chi2_gate` | `object_tracker` | 9.21 |
| `assoc_size_weight` | `object_tracker` | 2.0 |
| `assoc_label_penalty` | `object_tracker` | 2.0 |
| `label_vote_half_life_s` | `object_tracker` | 30.0 |
| `confirm_n` / `confirm_m` | `object_tracker` | 3 / 6 |
| `confirm_miss_grace_s` | `object_tracker` | 1.5 |
| `graveyard_min_hits` | `object_tracker` | 2 |
| `merge_chi2` | `object_tracker` | 1.0 |
| `activity_enabled` | `spatial_prior` | false (dropped from goals — see below) |

**WP-B regression and fix.** The first cut of the miss-charging logic
lived in `_detections_cb` and charged every unmatched pre-existing track a
miss on **every incoming message from any source** (3 cameras + 10 Hz
lidar share one callback) — a track seen by only one 0.5 Hz camera
collected ~5 misses per real hit and essentially never held enough recent
hits to confirm. That sank carter2 from 57 % tracked (legacy) to 21 %
(agnostic) on `unit2_pan_1`, not the confirm_n/confirm_m ratio itself.
Fixed by moving miss-charging out of `_detections_cb` into `_tick`,
gated by `confirm_miss_grace_s` and decoupled from message arrival rate
entirely (`_charge_stale_misses`) — one miss per grace interval of
genuine silence, never per message.

**Offline results** (`tools/retrack_bag.py --mode retrack`, tracked % /
distinct ids / id-switches per Carter):

| bag | carter1 | carter2 | carter2 (legacy) |
|---|---|---|---|
| `unit2_pan_1` | 97.8 % / 5 ids / 5 switches | 57.7 % / 4 ids / 7 switches | 62.4 % / 5 ids / 6 switches |
| `audit_after` | 100 % / 1 id / 0 switches | 99 % / 1 id / 0 switches | 100 % / 4 ids / 6 switches |
| `unit2_pan_2` | 94.9 % / 12 ids / 16 switches | 99.9 % / 12 ids / 27 switches | — |

Phantom / missed / actionable-mover rates: `unit2_pan_1` 3.9 % / 31.2 % /
68.8 %; `audit_after` 0.9 % / 16.9 % / 79.3 %; `unit2_pan_2` 3.9 % / 19.5 %
/ 77.4 % (pre-fix `unit2_pan_2` was 35.2 % phantom / 41.1 % actionable).
**Open gap:** the id-stability targets (ids ≤ 3, switches ≤ 3) are **not**
met on `unit2_pan_1`/`unit2_pan_2` — the same relaxation that rescues
sparse single-camera tracks (N-of-M, label-blind revive, a looser
`merge_chi2`) also lets marginal, barely-converged fragments confirm and
then split again instead of merging back. `audit_after` (shorter capture,
better coverage) clears the target cleanly, which is itself evidence the
gap tracks coverage/continuity, not the association math in general.

**Harness bug found and fixed.** `run_headless_tracker`'s `tracks_by_tick`
held **live** `Track` object references at each tick instead of a
point-in-time copy, so the harness's prediction stage — which walks
`tracks_by_tick` after the whole replay finishes — read every tick's
tracks as they stood at the **end of replay**, not as they stood at that
tick (comet-area sampling silently read 0.0 m² for everything). Fixed
with `TrackSnapshot.of(tr)`, an immutable per-tick copy of the fields the
prediction stage needs. After the fix, `audit_after`'s comet area p90 is
6.7 m² (previously read as 0.0 m², i.e. not measuring anything).

These are all **offline retrack numbers, not a sim result** — the
post-merge replay and a live confirmation run are still pending; see the
"RESULT" line below for the live run once it exists.

**Activity prior (5-min half-life channel): implemented, dropped from the
goals.** `spatial_prior_node` gained an `A` channel — same
deposit/decay/publish machinery as the existing 2 h lifelong `S` channel,
a separate, non-persisted, run-time memory that decays even while `S` is
frozen for a study — intended to seed the SRM's lowest level with recent
traffic at doorways/junctions (the paper's "latent risk" case a fixed-CV
rollout can't see). User B dropped it from this round's goals after
implementation; it stays in the tree **disabled by default**
(`activity_enabled: false`, `activity_weight`/`activity_weight_planner`/
`activity_weight_grid` all 0.0) so it cannot affect any result documented
here, and it's out of scope for verification. Document it as
available-but-off, not as part of this round's shipped behaviour.

**Merge: `origin/user-a/sandbox` → `user-b/sandbox` (local only, nothing
pushed).** 46 commits merged locally: `4612097` (pre-merge WIP snapshot:
SRM, motion-first consumers, agnostic tracker, activity prior off,
retrack harness) → `3b8ba96` (the merge commit) → `f383437` (evaluation
node docs) → `ac635e9` (harness speed-ups) → the newest commit, a merge of
the `wpb-confirm-fix` branch (tracker confirmation fix + harness
motion-first painting). User B tests (including the evaluation node above)
and pushes/merges back to `user-a/sandbox` himself; nothing here has been
pushed.

Taken from User A: class-aware NMS (`nms.py`), `collapse_compound_labels`
in the GDINO node, the research-CSV logging pipeline (`debug_log.py`,
`tools/prior_report.py`), the evaluation node/metrics/`evaluate_run.py`
(see the section right below), `nav2_sim.yaml` + the `_nav2()` ablation
include, `SPLAT_SIGMA_CUTOFF 4.5` (square-splat fix), the reactive costmap
`alpha 0.88` / `max_falloff_radius_m 1.50` fix, tracker `min_hits 2` /
`min_confidence 0.22` / `dynamic_half_life_s 6.0`, the rviz refresh, and
the `pallet`/`rolling chair` labels. Rejected: her category-keyed
association gate and best-score `label_score` adoption (superseded by the
agnostic Mahalanobis association + score-weighted label voting above),
`max_sigma_m 1.0` as a hard covariance cap, a hard `motion_gate: speed`,
`forget_half_life_s 300` applied to the **lifelong S channel** (this
would break `lane_layer`, the corridor lane band, and the frozen-run
contract — the working tree's answer to that idea is the separate,
non-persisted 5-min `A` channel above, not repurposing `S`), `door:
"uncertain"`, her single-word GDINO prompts (ours kept; flagged for User B
to compare), and `cpa_gain 0.5`.

**Fast-verification recipe.** The harness got faster in parallel:
`CARTER_FIRST_ATTEMPT_TIMEOUT` (60 s — the first `carters_patrol` launch
stalls in `Configuring` every time, the existing retry activates in
~11 s, this alone saves ~4 min/run over waiting out the old 300 s
timeout), a new short scenario `unit3`
(`warehouse/nav/carter_shuttle3.yaml`: carter2 shuttles a 3 m lane,
y = -1.0 … 2.0, a crossing every ~6 s instead of unit2's ~10 s; X3's
A (3.4, 0.5) ↔ C (1.0, 0.5) unchanged; `LAPS=2`, 150 s sim window), and a
watchdog (`WATCHDOG=1`: aborts with a reason file if carter2's GT hasn't
moved 0.5 m in 90 s, `object_tracker`/`predictive_risk_costmap` died,
`x3_nav.log` shows > 20 planner/controller failure lines, or `cmd_vel` is
silent 60 s after the runner delay — writes `ABORT_REASON.txt` and stops,
exit 2), plus `probe_monitor.py --progress-json/--check-progress` and
`results/avoidance/test_probe_monitor_watchdog.py`.

```bash
pytest -q src/risk_perception/test src/panoptex_nav/test tools
for b in unit2_pan_2 unit2_pan_1 audit_after; do
  python3 tools/retrack_bag.py --bag ~/panoptex_runs/runs/$b/bag --mode both --assert \
    --param association_mode=agnostic --param stack_motion_first=true; done

CARTER_WPS=$HOME/workspace/warehouse/nav/carter_shuttle3.yaml \
X3_WPS=$HOME/workspace/Panoptex/src/panoptex_nav/config/x3_unit_cross2.yaml \
LAPS=2 PERCEPTION=panoptex YIELD=true PANOPTEX_NO_CM=1 \
bash results/avoidance/run_unit.sh panoptex_mppi unit3_pan_1 150
```

**RESULT unit3_pan_1** (2026-09-11 14:31-14:42 EDT, 11 min wall incl.
bring-up; Carters active on the FIRST attempt this time, 16 s; RTF 0.279;
merged tree `30c2273` = pre-merge snapshot + user-a merge + tracker-fix
merge). **Mission:** only 1 of the 2 configured laps completed in the
150 s window (lap 123 s), cmd_vel activity 0.976, 2 yields, 1 hold (3.8 s),
1 refuge, 0 aborts, 0 planner/controller failures — carter2 completed 23
shuttle goals then stalled at its south waypoint for the last ~90 s
(`carter_loop` re-sending "heading to waypoint 1 (north)" every 50 ms;
Carter-side nav2 at low RTF, the known issue from earlier runs, not this
stack's). **Perception held live** (`tools/retrack_bag.py --mode
original`, vs `unit2_pan_2` in parens): phantom 2.9 % (2.6 %), missed-mover
26.6 % (35.2 %), actionable-mover 68.6 % (41.1 %), mislabelled 11.9 %
(39.7 %), comet area p50/p90/max 0.0/6.4/28.3 m² (0.2/7.4/20.1), comet
radius p90 3.0 m; carter1 100 % tracked / 1 id / 0 switches, carter2
96.7 % tracked / 9 ids / 10 switches, median offset 0.22 m (`analyze_run`'s
own matching: 81.5 %, 7 ids, `pmot > 0.5` while moving 75.5 %);
`track_audit`: 4 ghost movers (was 15), 0 unknown movers painted, 0
speed-jump resets. The class-agnostic + confirmation changes are confirmed
live; id stability is confirmed as the still-open gap (not an offline-only
artifact). **Avoidance:** 3 contact episodes (5.9 s total, min gap 0.30 m)
across 5 carter2 crossings — 101 s: carter2 parked at its north turnaround
(y≈2.25, only 1.75 m from the crossing) while the X3 waited nearly
stationary (approach speed 0.02 m/s); 154 s: carter2 moving north at
0.6 m/s, X3 crossing, gap 0.30 m; 174 s: carter2 parked at the south end,
X3 approaching, gap 0.41 m. The 3 m lane puts carter2's U-turn directly on
top of the refuge/hold cells, so `unit3` reads as a **perception**
regression check, not a fair avoidance test — recommendation: keep
`unit3` for the fast (~11 min) perception check, use the v2 lane
(y = -1 … 4) at `LAPS=2` / 200 s for avoidance comparisons. **Watchdog
caveat:** `WATCHDOG_MIN_WINDOW_S` auto-computes to `RUNNER_DELAY +
max(stall_s, silent_s)` = 180 s, which exceeded this run's 150 s window,
so no watchdog check ever ran this run; a fix (start the grace window from
the first `cmd_vel` activity instead of `RUNNER_DELAY`, stall window 60 s, plus a 15 s contact-push abort) landed right after the run,
not yet exercised live. Full analysis:
`results/avoidance/unit/unit3_pan_1/{analysis.md,track_audit.md,retrack_original.md}`.

### Evaluation node (from user-a/sandbox)

Merged 2026-09-11. `evaluation_node` is the **live** counterpart to
`tools/evaluate_run.py`: it collects raw samples during a trial (tf robot
pose, nearest-object distance, risk at the robot's position, detection /
costmap stamps) and, at trial end, calls the *same* tested batch functions in
`risk_perception/evaluation_metrics.py` that the bag-replay path calls — there
is no second "streaming" implementation of any metric.

**Enable it.** It starts with `evaluation_reference.launch.py`, whose
`enable_evaluation` arg defaults to `true`:

```bash
# terminal 1 -- the ablation under test (or panoptex_sim.launch.py enable_nav2:=true)
ros2 launch risk_perception risk_perception.launch.py <ablation flags>
# terminal 2 -- the fixed reference costmap + the live collector
ros2 launch risk_perception evaluation_reference.launch.py use_sim_time:=true
```

That launch also starts a **second** `predictive_risk_costmap_node`
(`predictive_risk_costmap_node_reference`) forced to full-system settings,
publishing `/risk_costmap_reference`. That grid is a scoring yardstick only
and is never wired into Nav2 — sampling exposure from whichever costmap
actually drove Nav2 would let a weaker arm self-report lower exposure just
because it computed less risk. `panoptex_sim.launch.py` refuses
`risk_topic:=/risk_costmap_reference` for the same reason.
`use_sim_time:=true` is **required** under a `/clock` source: the node mixes
`get_clock().now()` with message header stamps, so wall-vs-sim time corrupts
both latency and realtime-factor.

**Topics.** No custom messages — trial control is one `std_msgs/String` pair:

| topic | direction | payload |
|---|---|---|
| `/evaluation/trial_control` | subscribed | `start <label>` (clear buffers, begin collecting) / `end` (compute, publish, ready for the next `start`) |
| `/evaluation/trial_result` | published | the finished trial's `RunReport.summary()`, JSON |

Inputs it samples: `risk_topic` (`/risk_costmap_reference`),
`world_objects_topic`, `detections_topic` (`/risk_perception/detections_2d`,
for detection→costmap latency) and `replan_topic` (`/plan`, replans counted
off header stamps). All in `risk_perception.yaml`'s `evaluation_node:` block,
along with `sample_rate_hz` 10, `configured_period_s` 0.2 (keep == the
predictive node's `pred_dt`), `engagement_threshold_m` 2.0 and
`stop_speed_mps` 0.05. `tools/run_trials.py --condition-prefix <name>` drives
the pair and writes one JSON per leg under `--out-dir`.

**The CSVs are a separate thing.** `evaluation_node` writes no files; the
per-tick research CSVs come from the prior nodes themselves, switched on by
`prior_log_dir:=<dir>` (plus `tracker_log_unconfirmed:=true` to include tracks
below `min_hits`/`min_confidence`). Each node names its own file so successive
runs never clobber each other:

| file | one row per | what it answers |
|---|---|---|
| `object_tracker_<UTC>.csv` | track per tick | every prior's in/out: category, `p_movable_prior`, `motion_score` vs `motion_threshold`, `p_motion`, `p_movable`, `relconf_in`, `relation_bonus`, consequence, hits/misses/age, `confirmed`/`alive` |
| `predictive_costmap_<UTC>.csv` | track per tick | why Stage 2b did or didn't smear: `pmot`, raw `(vx, vy)`, consequence / CPA / relation multipliers, `w_stat` vs `w_move_step0`, `n_move_steps`, plus this tree's `moving_allowed` and `stack_consequence` |
| `spatial_prior_<UTC>.csv` | `kind=deposit`: track per tick; `kind=grid`: category per publish | the category scope and both Behavioral gates spelled out, deposit radius/`k`, and how much S/F has actually accumulated |

**Consuming a run directory.**

```bash
# live-trial JSON (one file per condition) -- diff across arms
python3 tools/run_trials.py trials.json --condition-prefix full_system --out-dir results

# same metrics, post-hoc from a bag; record at minimum
#   /tf /tf_static /odom /risk_perception/world_objects
#   /risk_perception/detections_2d /risk_costmap_reference /plan
python3 tools/evaluate_run.py ~/panoptex_runs/runs/<run>/bag \
    --label full_system --out results/full_system.json

# the per-tick priors, from whatever landed in prior_log_dir
python3 tools/prior_report.py --log-dir ~/.panoptex/logs
```

`evaluate_run.py` takes the bag directory positionally and emits
`RunReport.summary()` as a flat JSON dict — directly comparable, key for key,
with what `/evaluation/trial_result` publishes live, so a live trial and a
replay of its bag are the same numbers. `prior_report.py` takes the CSV
directory (`--log-dir`, or `--tracker`/`--spatial`/`--predictive` globs) and
prints percentile tables and per-label breakdowns per prior, writing
histograms and per-track timelines next to the CSVs (or `--out-dir`) when
matplotlib is available.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | conda env not active, **or** the package was built without it | `conda activate panoptex`; then check `head -1 install/risk_perception/lib/risk_perception/gdino_detector` — if it says `#!/usr/bin/python3`, colcon was run outside the env and stamped the wrong interpreter into every node. Rebuild with the env active (§4). Silent at build time, only shows up at launch |
| `Failed to load custom C++ ops. CPU mode Only!` | `_C` can't find `libc10.so` | `export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"` |
| `name '_C' is not defined` | GDINO extension not built | rebuild GDINO in the `panoptex` env with `CUDA_HOME` set (already exported globally), `--no-build-isolation` |
| `ModuleNotFoundError: risk_perception.<node>` | new file, not rebuilt | `colcon build` again (symlink gotcha, §4) |
| launch file "not found in share directory" | package not rebuilt after adding a file | `colcon build --packages-select risk_perception`, open a fresh terminal |
| `robot_localization not found` | ran the base/SLAM launch with the conda env active | plain `robot_localization` is a system-Python ROS package; conda's `PYTHONNOUSERSITE`/env isolation can shadow it. `robot_base.launch.py` on its own does not need the env. (`panoptex_pc.launch.py` does, and both `nav2_map_server` and `ekf_node` were verified to start fine with it active — if you hit this anyway, use §5.1's two-terminal split) |
| 2D Goal Pose does nothing | Nav2 not running | `ros2 topic info /goal_pose` — 0 subscribers means RViz published into the void. Relaunch with `enable_nav2:=true` |
| `global_cam_optical_frame` vanishes from RViz | <2 floor tags decoded that frame | expected with marginal tags; the calibrator now re-sends the last extrinsic at 10 Hz and warns after 5 s. Check coverage with `global_cam_tag_monitor` |
| tf `map → odom` frozen / stale | SLAM on sim time, or a zombie | `use_sim_time:=false`; `pkill` strays; `ros2 daemon stop/start` |
| `Cannot transform map → camera` | `map → odom` missing | SLAM not publishing — see above |
| `detections_3d` empty | nothing detected | lower thresholds; check `/detection_image`; confirm object matches prompt |
| RealSense `Frames didn't arrive` | USB-2 link | move to USB-3 port/cable — `lsusb -t` must show 5000M, not 480M |
| `Cannot transform ... -> map` in global_cam_projector/localizer | global_cam_calibrator hasn't solved an extrinsic yet | check `/global_cam/apriltag/detections` sees floor tags 1/2/3, and that §6's calibration has been run (`config/floor_tags.yaml` still has placeholder positions otherwise) |
| overhead detections land offset/rotated from the RGB-D camera's on the RViz map | `floor_tags.yaml` is in the tag-defined frame (from `global_cam_survey`), not SLAM's `map` | run `global_cam_map_align` (§6) — the survey alone never touches the `map` frame |

---

## 10. Roadmap

- **Nav2 integration** — done, two layers now: `nav2_risk_layer` was
  originally wired into `yahboomcar_nav`'s `global_costmap.plugins`
  (`dwa_nav_params.yaml`), listening on `/risk_costmap` (the reactive,
  non-predictive grid); §8's `panoptex_nav` package adds a second,
  time-aware path that does not go through any costmap at all —
  `PredictedRiskCritic` subscribes `/risk_stack` directly and scores each
  DWB candidate trajectory pose against the layer matching *that pose's own
  time offset*, and `risk_speed_governor` caps speed straight off
  `/risk_perception/world_objects`. Only the **global** costmap carries a
  risk layer (its `global_frame` is `map`, matching the risk grid); the
  local costmap's frame is `odom`, which would need its own transform if
  wired up too — the predictive/time-layered path sidesteps that limitation
  entirely for the controller, since it never goes through the local
  costmap in the first place.

  In the sim the same layer is reached by `panoptex_sim.launch.py
  enable_nav2:=true` against this repo's `config/nav2_sim.yaml` (same layer
  block, plus `amcl_sim.yaml`'s amcl tuning and sim time);
  `risk_layer_enabled:=` and `risk_topic:=` expose the baseline / reactive /
  predictive ablation arms as launch args, which is what
  `tools/run_trials.py` varies per condition.

  Verified against the sim params with a synthetic max-risk blob and no Isaac
  running (static `map→odom→base_footprint`, `use_sim_time:=false`): the whole
  Nav2 lifecycle reaches *Managed nodes are active* and `global_costmap` loads
  the layer in the order `static → obstacle → risk → inflation`. Measured on
  `/global_costmap/costmap`, all three arms against the same published grid:

  | arm | non-zero cells | risk bucket |
  |---|---|---|
  | reactive (`/risk_costmap`) | 47643 | **482 cells at 77** |
  | baseline (`risk_layer_enabled:=false`) | 47161 | none |
  | predictive (`/risk_costmap_predictive`) | 47161 | none (nothing publishing that topic in the test) |

  77 is Nav2's `Costmap2DPublisher` encoding of raw cost 200 — the layer's
  `max_cost` — so the 482-cell delta is exactly the layer's contribution and
  nothing else. **The layer only loads if this workspace's
  `install/setup.bash` is sourced in the launching terminal** (costmap plugins
  load in-process); `~/.bashrc` does it, `warehouse/env_sim.sh` does not add it
  on its own.
- **Dual-camera fusion** — done: `global_cam_*` nodes (bridge, calibrator,
  projector, localizer — see §2/§5 Terminal C) publish onto the same
  `map`-frame detection topic as the robot cam, after intrinsic
  (`camera_calibration`) + extrinsic (floor AprilTags, solvePnP)
  calibration. **The one-time calibration in §6 must be completed first** —
  the committed `config/floor_tags.yaml` is a placeholder.
- **Persistent world model** — done: `object_tracker` does data
  association, per-track Kalman, and per-class semantic decay; both cameras
  fuse through it via the shared `/risk_perception/detections_3d_map` topic.
  It now also publishes `vx`/`vy` in `class_id`, which is what makes the
  predictive stage anything other than inert.
- **Predictive risk (Stage 2/3/4)** — done:
  `predictive_risk_costmap_node` rolls a two-hypothesis future occupancy
  (stationary weighted `1 - p_motion`, constant-velocity weighted
  `p_motion`) over a horizon, time-discounts it by `gamma^step`, weights it
  by class consequence, and amplifies tracks on a collision course with the
  robot via CPA/TTC against `/odom` + tf. Publishes
  `/risk_costmap_predictive`, deliberately NOT `/risk_costmap`, so it runs
  beside the reactive node and the ablation is a matter of repointing
  `risk_layer.topic`.
- **Predictive risk into the controller (Stage 5 / WP3, §8)** — done, sim
  only: `predictive_risk_costmap_node` also publishes the ego-independent,
  time-layered `/risk_stack` (`panoptex_msgs/RiskStack`); `panoptex_nav`'s
  `PredictedRiskCritic` (DWB) and `risk_speed_governor` (`SpeedLimit`) are
  built, unit-tested, and wired end-to-end via `x3_nav.launch.py`'s two
  study arms against the Isaac Sim X3. Baseline-arm results are in; the
  Panoptex-arm run is still pending (see §8's results table).
- **Spatial persistence** — done: `spatial_prior_node` accumulates a
  map-aligned "things were observed moving here" grid across sessions
  (`~/.panoptex/spatial_prior.npz`) and publishes it as
  `/risk_perception/spatial_prior`; the predictive costmap blends it as a
  weak floor (`spatial_prior_weight`, default 0 = off). Learning-free
  counterpart to SOGM's learned latent risk. Note it is an observed-motion
  histogram with no visibility normalisation — unobserved cells and never-
  crossed cells both read 0.
- **Long-term re-identification** — not started. Tracks still die for good
  when they leave view, so "persistent" currently means persistent *places*
  and *within-session identity*, not appearance-based revival.
- **Robot localization via the overhead camera** — partially done.
  `global_cam_localizer` publishes `/global_cam/robot_pose` from tag 0 on
  the robot; compare it against the SLAM pose in RViz before going further.
  `config/ekf_global.yaml` + `robot_base.launch.py`'s `enable_ekf_global`
  arg (default **false**) sketch the fusion step (a second, map-frame EKF
  replacing slam_toolbox as `map → odom`'s owner) but this needs validation
  on real hardware before flipping on — only one node may publish
  `map → odom` at a time.

For the moving robot, switch `map_frame_projector`'s tf lookup from latest
(`Time()`) to the message stamp (`Time.from_msg(msg.header.stamp)`) so detections
aren't smeared by pose change during motion. (`global_cam_projector` and
`global_cam_localizer` already use `Time()` deliberately, since the overhead
camera itself is static — only the robot-mounted camera chain has this
issue.)
