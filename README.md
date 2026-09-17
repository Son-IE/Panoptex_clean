# Panoptex

Risk-aware navigation for a mobile robot in a warehouse-like environment. A
perception + tracking pipeline builds a persistent, semantically-labeled
world model; that world model is turned into risk fields two different ways
(a reactive per-frame field and a predictive, motion-hypothesis field); and
those risk fields are injected into [Nav2](https://docs.nav2.org/) at three
independent points — the global costmap, the local controller's trajectory
scoring, and a speed governor — so the robot both *plans around* risk and
*reacts to* it in real time.

This repository is the ROS 2 side of the system: five packages, no
simulation assets. It is built and evaluated against a companion Isaac Sim
warehouse scenario (a separate repository, referenced but not included
here).

## Architecture

```
   RGB-D / overhead cameras
            │
            ▼
   detection + tracking  (risk_perception: object_tracker_node)
            │
            │  persistent tracks: label, pose, velocity, covariance,
            │  four priors (semantic, behavioral, relation, spatial-flow)
            │
      ┌─────┴──────────────────────────────┐
      ▼                                     ▼
reactive risk field                 predictive risk field
(risk_costmap_node)                 (predictive_risk_costmap_node)
      │                                     │
      │                          ┌──────────┼──────────────┐
      │                          ▼          ▼              ▼
      │                  flattened R(c)  SRM (time-       world_objects
      │                  costmap grid    layered stack)   (raw tracks)
      ▼                          │          │              │
┌─────────────┐                  ▼          ▼              ▼
│  Nav2        │        nav2_risk_layer  panoptex_nav::   panoptex_nav::
│  global      │◄───────(costmap plugin) PredictedRisk*   risk_speed_governor
│  costmap     │                          Critic (DWB/MPPI)      │
└──────┬───────┘                          │                      │
       ▼                                  ▼                      ▼
  planner_server                   controller_server       controller_server
  (path bends around risk)         (trajectories scored     (SpeedLimit —
                                    against future risk)     robot slows near risk)
```

The reactive and predictive fields are genuinely independent — the
predictive field's flattened grid and its time-layered SRM are two
different outputs of the same node, consumed by two unrelated mechanisms
(a costmap layer vs. a controller-level critic). None of the three
injection points depend on either of the other two; each can be enabled or
disabled independently, which is what the study arms below actually vary.

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

`perception:=oracle` (instead of the default `perception:=panoptex`) skips
the camera/detector/tracker chain entirely and feeds the risk fields
straight from Isaac's ground truth via `gt_tracks_node` — useful for
bench-testing the prediction/planning/critic stack in isolation from
perception noise.

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

See `README_old.md` for the full development history this repository was
distilled from.
