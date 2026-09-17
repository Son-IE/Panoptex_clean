# Panoptex

Risk-aware navigation for a mobile robot in a warehouse-like environment.
The subject of this work is the **predictive risk field**: what a tracked
object carries with it, a set of contexts (called "priors" in this local code works) about whether it can move, whether
it is moving, what it is attached to, and what has moved through that place
before,  and how those priors are turned into risk projected forward in
time.

Everything downstream of that field is deliberately thin. The field is
handed to stock [Nav2](https://docs.nav2.org/) through its normal extension
points. No planner or controller was modified, and the navigation side is
not a contribution of this work.

This repository is the ROS 2 side of the system: five packages, no
simulation assets. It is built and evaluated against a companion Isaac Sim
warehouse scenario (a separate repository, referenced but not included
here).

## Architecture

```
          cameras (RGB-D + overhead) + lidar
                          │
                          ▼
              detection / segmentation
                          │
                          ▼
  ┌──────────────────────────────────────────────────────┐
  │ object_tracker_node  +  spatial_prior_node           │
  │ persistent tracks, each carrying FOUR PRIORS         │
  │                                                      │
  │   semantic      class-conditioned seed for           │
  │                 "could this ever move?"              │
  │   behavioral    p_movable (slow memory) and          │
  │                 p_motion (fast), learned from        │
  │                 observed motion, not from the label  │
  │   relation      a person standing *on* a machine     │
  │                 makes that machine a mover           │
  │   spatial-flow  place memory: where a class has      │
  │                 moved before (S) and how (F)         │
  └───────────────────────┬──────────────────────────────┘
                          │  per track: pose, velocity,
                          │  covariance, the four priors
                          ▼
  ┌──────────────────────────────────────────────────────┐
  │ predictive_risk_costmap_node                         │
  │   two future-occupancy hypotheses per track —        │
  │     stationary (1 - p_motion) vs. constant-velocity  │
  │     rollout (p_motion), covariance growing with t    │
  │   severity = consequence × encounter (CPA/TTC        │
  │     against the robot's own motion)                  │
  │   → a time-layered stack, one grid per horizon step  │
  │     (srm.py converts each layer to a risk falloff)   │
  └───────────────────────┬──────────────────────────────┘
                          │
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                          ▼
        stock Nav2, unmodified — the field enters as a costmap
        layer, a trajectory critic and a speed limit (appendix)
```

`risk_costmap_node` produces a second, *reactive (semantic-only)* per-frame field from the
same tracks, with no prediction in it. It exists as the within-system
comparison for the predictive field, not as a separate mechanism.

## Packages

| package | what it is |
|---|---|
| `risk_perception` | The perception/tracking/risk-field core: detectors, `object_tracker_node` (fuses detections into persistent, prior-carrying tracks), `risk_costmap_node` (reactive field), `predictive_risk_costmap_node` (predictive field + SRM), `spatial_prior_node` (learned place/motion memory), plus the evaluation and research-logging tooling. |
| `nav2_risk_layer` | A `nav2_costmap_2d::Layer` C++ plugin. Subscribes to an `OccupancyGrid` risk topic and folds it into Nav2's global costmap (`setCost(max(current, risk))`) so the planner deflects around risk instead of only around hard obstacles. |
| `panoptex_nav` | The Nav2-side consumers: `PredictedRiskCritic` (DWB) and `PredictedRiskMppiCritic` (MPPI), which score controller trajectory rollouts directly against the predictive field's time-layered SRM; `risk_speed_governor` (publishes `nav2_msgs/SpeedLimit` off live tracks); `mission_supervisor` (drives the waypoint tour for a study run); the four study-arm Nav2 parameter sets (baseline / panoptex, each with a DWB and an MPPI variant); and `gt_tracks_node`, an oracle-perception bypass for bench-testing the risk/planning stack without camera noise in the loop. |
| `panoptex_msgs` | One message definition, `RiskStack.msg` — the time-layered predictive risk field the controller-level critics consume. |
| `risk_map_publisher` | A small standalone tool: publishes a synthetic `OccupancyGrid` on the risk topic so `nav2_risk_layer` can be exercised and verified with no perception stack, no tracker, and no simulator running at all. |

## Setup

```bash
./setup.sh                 # creates/updates the `panoptex` conda env, installs
                            # GroundingDINO/SAM2 weights (auto-detects CUDA)
conda activate panoptex
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Build inside the activated conda env, every time.** `colcon` stamps each
installed Python node's shebang with whatever `python3` is on `PATH` at
build time. Build outside the env and the two model-loading nodes
(GroundingDINO/SAM2 detectors) silently get the system interpreter instead
of the conda one — no `torch`, and they crash on launch while every other
node in the same `ros2 launch` keeps running normally, so the log looks
healthy. Symptom: camera and TF work, but no detections, no risk, ever.

## Running

This repo assumes a running Isaac Sim warehouse scenario (separate
repository) publishing the robot's `/scan`/`/odom`/RGB-D topics and the
overhead camera feeds. Against that:

```bash
conda activate panoptex
source install/setup.bash
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true
```

Toggle the risk layer independently of everything else for a quick
baseline/risk-aware comparison on the same scene:

```bash
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true \
    risk_layer_enabled:=false                        # baseline: layer loaded, contributes nothing
ros2 launch risk_perception panoptex_sim.launch.py enable_nav2:=true \
    risk_topic:=/risk_costmap_predictive              # plan against the predictive field instead
```

To exercise `nav2_risk_layer` alone, with no perception stack at all:

```bash
ros2 run risk_map_publisher synthetic_risk_publisher
```

### The four study arms

`panoptex_nav/launch/x3_nav.launch.py` is the full study harness — one of
four Nav2 configurations against the same scenario:

| arm | costmap | controller |
|---|---|---|
| `baseline` | pure lidar (no risk layer) | DWB |
| `panoptex` | risk layer + lane layer | DWB + `PredictedRiskCritic` + speed governor |
| `baseline_mppi` | pure lidar | MPPI |
| `panoptex_mppi` | risk layer + lane layer | MPPI + `PredictedRiskMppiCritic` + speed governor |

```bash
ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex
```

The `panoptex` arms' Nav2-side pieces — the `RiskStack` wire format, the two
critics' parameters, the speed governor's gates — are documented in the
[appendix](#appendix-nav2-risk-consumer-interface).

`perception:=oracle` (instead of the default `perception:=panoptex`) skips
the camera/detector/tracker chain entirely and feeds the risk fields
straight from Isaac's ground truth via `gt_tracks_node` — useful for
bench-testing the prediction/planning/critic stack in isolation from
perception noise.

## Overhead camera calibration — one-time (real robot only)

This procedure is for the **real robot with a physical overhead camera**.
The Isaac Sim path in Running above exports the overhead camera's pose as
ground truth directly from the stage and does not need any of this.

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

# J3 -- RGB-D camera. NOT needed for calibration; start it later, once
# you're running the full stack normally.
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
`bench_global_cam.launch.py` is built on exactly this.

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
## Evaluation

```bash
# one repeat: drive a waypoint tour, score every leg against a fixed
# full-system reference field so every arm is judged on the same yardstick
ros2 launch risk_perception evaluation_reference.launch.py
python3 tools/run_trials.py trials_x3_tour.json --condition-prefix panoptex --out-dir results/panoptex

# N repeats, headless, one fresh Isaac session per repeat
./tools/run_ablation_repeats.sh panoptex 1 3

# combine repeats into stabilized per-arm numbers
python3 tools/aggregate_trials.py results/panoptex
```

`tools/scenario_publisher.py` and `tools/bench_mppi_arm.py` drive the risk
fields and the MPPI critic respectively from synthetic tracks — no camera,
no robot, no Isaac Sim — for validating the prediction/scoring math in
isolation.

`tools/prior_report.py`, `tools/prior_contribution_timeseries.py`, and
`tools/spatial_flow_heatmap.py` turn each prior node's own research-logging
CSVs (`prior_log_dir`, off by default) into readable reports and figures —
the way to inspect *why* a given track carries the risk it does.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | conda env not active, **or** the package was built without it | `conda activate panoptex`; then check `head -1 install/risk_perception/lib/risk_perception/gdino_detector` — if it says `#!/usr/bin/python3`, colcon was run outside the env and stamped the wrong interpreter into every node. Rebuild with the env active (see Setup above). Silent at build time, only shows up at launch |
| `Failed to load custom C++ ops. CPU mode Only!` | `_C` can't find `libc10.so` | `export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"` |
| `name '_C' is not defined` | GDINO extension not built | rebuild GDINO in the `panoptex` env with `CUDA_HOME` set (already exported globally), `--no-build-isolation` |
| `ModuleNotFoundError: risk_perception.<node>` | new file, not rebuilt | `colcon build` again (rebuild after adding a file -- colcon symlink-install gotcha) |
| launch file "not found in share directory" | package not rebuilt after adding a file | `colcon build --packages-select risk_perception`, open a fresh terminal |
| `robot_localization not found` | ran the base/SLAM launch with the conda env active | plain `robot_localization` is a system-Python ROS package; conda's `PYTHONNOUSERSITE`/env isolation can shadow it. `robot_base.launch.py` on its own does not need the env. — if you hit this anyway, run localization and the perception stack from separate terminals so only the perception one has the conda env active |
| 2D Goal Pose does nothing | Nav2 not running | `ros2 topic info /goal_pose` — 0 subscribers means RViz published into the void. Relaunch with `enable_nav2:=true` |
| `global_cam_optical_frame` vanishes from RViz | <2 floor tags decoded that frame | expected with marginal tags; the calibrator now re-sends the last extrinsic at 10 Hz and warns after 5 s. Check coverage with `global_cam_tag_monitor` |
| tf `map → odom` frozen / stale | SLAM on sim time, or a zombie | `use_sim_time:=false`; `pkill` strays; `ros2 daemon stop/start` |
| `Cannot transform map → camera` | `map → odom` missing | SLAM not publishing — see above |
| `detections_3d` empty | nothing detected | lower thresholds; check `/detection_image`; confirm object matches prompt |
| RealSense `Frames didn't arrive` | USB-2 link | move to USB-3 port/cable — `lsusb -t` must show 5000M, not 480M |
| `Cannot transform ... -> map` in global_cam_projector/localizer | global_cam_calibrator hasn't solved an extrinsic yet | check `/global_cam/apriltag/detections` sees floor tags 1/2/3, and that the overhead-camera calibration has been run (`config/floor_tags.yaml` still has placeholder positions otherwise) |
| overhead detections land offset/rotated from the RGB-D camera's on the RViz map | `floor_tags.yaml` is in the tag-defined frame (from `global_cam_survey`), not SLAM's `map` | run `global_cam_map_align` — the survey alone never touches the `map` frame |

## Appendix: Nav2 risk-consumer interface

The message and the two Nav2-side consumers that read the predictive field.
This is the *interface* between the risk stack above and Nav2 — it is
deliberately the thinnest part of the system.

**On prior work.** Two things in this appendix come from the spatiotemporal
occupancy grid map of Thomas, Piat & Charpillet, ["Learning Spatiotemporal
Occupancy Grid Maps for Efficient
Decision-Making"](https://arxiv.org/abs/2108.10585) (ICRA 2022): keeping
predicted risk *time-layered* — one grid per future time step — rather than
flattening it into a single static grid, and the conversion of each layer's
occupancy into a graded risk falling off with distance to occupied space
(their eq. 3, implemented in `srm.py`). Both sit at the very end of the
pipeline, on the controller-facing side.

What fills those layers does not. The four priors, the tracking they ride
on, and the two-hypothesis future occupancy they produce are developed in
`risk_perception` and have no counterpart in that paper — `srm.py` consumes
that output, it does not produce it. None of the paper's code, models or
parameters are used; everything here is written from scratch against the
published description.

### `panoptex_msgs/RiskStack`

```
std_msgs/Header header          # frame_id = the risk grid frame (map)
nav_msgs/MapMetaData info       # resolution / width / height / origin, all layers
float32 dt                      # seconds between consecutive layers
uint8 steps                     # number of layers (>= 1)
float32 horizon_start           # seconds from header.stamp to layer 0 (usually 0.0)
int8[] data                     # steps * info.height * info.width
```

Layer `k` covers the instant `header.stamp + horizon_start + k*dt`; layer 0
is "now". `data` is row-major per layer, the same convention as
`nav_msgs/OccupancyGrid.data`, so one layer slices straight into an
`OccupancyGrid` with no reshaping (0-100 risk, `-1` = unknown). Published
`RELIABLE + TRANSIENT_LOCAL, KeepLast(1)` — a subscriber with mismatched
QoS gets nothing at all, silently.

### `panoptex_nav::PredictedRiskCritic` (DWB) / `PredictedRiskMppiCritic` (MPPI)

For each candidate-trajectory pose at time offset `t`: transform into the
risk grid's frame, look up layer `k = clamp(round((t + stack_age -
horizon_start) / dt), 0, steps-1)`, read that cell's risk `r` (0-1). At or
above `lethal_threshold` the trajectory is rejected outright; otherwise the
pose contributes `time_discount^t * r^cost_power` to the trajectory's score.
**Fail-soft:** no stack yet, a stale one, or a failed TF lookup all just
score every trajectory `0.0` rather than blocking navigation.

| Parameter | Default | Meaning |
|---|---|---|
| `topic` | `/risk_stack` | The `RiskStack` topic |
| `cost_power` | `1.0` | Exponent on the 0-1 risk value |
| `time_discount` | `0.9` | Per-second discount on a pose at time `t` |
| `lethal_threshold` | `0.85` | Risk at/above which the trajectory is rejected outright |
| `stale_timeout_s` | `2.0` | Stack older than this -> critic goes inactive |
| `max_horizon_s` | `3.3` | Poses further out than this are ignored |
| `risk_frame` | `map` | Frame of the risk grid |
| `scale` | `1.0` | Nav2's own critic weight; needs tuning against the other critics |

### `risk_speed_governor`

Publishes `nav2_msgs/SpeedLimit` straight from tracked objects
(`/risk_perception/world_objects` — no costmap, no `RiskStack`). Where the
critic reshapes trajectory *scoring*, this puts a hard ceiling on *how fast
any trajectory may go*. `speed_limit == 0.0` always means "no limit," never
"stop" — a hard stop stays lidar/costmap territory.

| Category | Gate | Speed cap |
|---|---|---|
| person | within `person_crawl_radius_m` (1.0 m) | `person_crawl_pct` = 15% |
| person | within `person_slow_radius_m` (2.0 m) | `person_slow_pct` = 40% |
| any mover except person (`act_on_any_mover`, default on) | moving and closing (`t_cpa <= robot_ttc_s`, `d_cpa <= robot_cpa_m`) | `robot_closing_pct` = 60% |
| robot/wheeled | stationary, within `static_obstacle_slow_radius_m` | `static_slow_pct` = 60% |
| furniture/unknown | ignored — Nav2's own lidar costmap layers already cover static clutter | — |

The most-restrictive applicable cap wins, floored at `min_pct` (15%); a cap
releases back to "no limit" only after `release_hold_s` (0.5 s) with no
qualifying track, to stop a track flickering across a threshold from
chattering the speed limit every control cycle.
