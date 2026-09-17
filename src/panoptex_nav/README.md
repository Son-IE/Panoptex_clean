# panoptex_nav

Nav2 integration for the Panoptex predictive risk stack: takes the
time-layered risk field `risk_perception`'s `predictive_risk_costmap_node`
publishes and turns it into (a) trajectory scoring inside DWB **or MPPI** and
(b) a hard speed cap on top of the controller — see `../../README.md` §8 for how this fits
into the rest of the repo, the architecture diagram, and the two-arm sim
study's verified results.

Hybrid `ament_cmake` + `ament_cmake_python` package:

| Path | What it is |
|---|---|
| `include/panoptex_nav/predicted_risk_critic.hpp`, `src/predicted_risk_critic.cpp` | `panoptex_nav::PredictedRiskCritic` — the DWB (`dwb_core`) trajectory critic |
| `include/panoptex_nav/predicted_risk_mppi_critic.hpp`, `src/predicted_risk_mppi_critic.cpp` | `panoptex_nav::PredictedRiskMppiCritic` — the MPPI (`nav2_mppi_controller`) critic (WP2) |
| `include/panoptex_nav/risk_stack_lookup.hpp`, `src/risk_stack_lookup.cpp` | the RiskStack geometry / freshness / layer-index / batch-scoring rules both critics share |
| `panoptex_nav/risk_speed_governor.py`, `scripts/risk_speed_governor` | the speed-governor rclpy node (thin console-script launcher + the real module — see the launcher's own docstring for why this package, unlike an `ament_python` one, needs both) |
| `panoptex_nav/gt_tracks_node.py`, `scripts/gt_tracks` (WP-C, 2026-09-10) | oracle-perception node: ground-truth `Detection3DArray` tracks straight off `/gt_tf`, for `perception:=oracle` — see **Oracle perception (`gt_tracks`)...** below |
| `config/risk_speed_governor.yaml` | governor policy defaults |
| `config/nav2_x3_baseline.yaml`, `config/nav2_x3_panoptex.yaml` | the DWB study arms' full nav2 params for the sim X3 |
| `config/nav2_x3_baseline_mppi.yaml`, `config/nav2_x3_panoptex_mppi.yaml` | the same two arms on MPPI (WP2) |
| `tools/check_arm_params.py` | deep-diffs an arm pair (`--baseline`/`--arm`); wired into `colcon test` as `check_arm_params` + `check_arm_params_mppi` |
| `config/x3_sim_waypoints.yaml` | the waypoint loop `x3_nav.launch.py` drives (Scenario v3) |
| `config/x3_unit_cross.yaml`, `config/x3_unit_cross2.yaml` (2026-09-10) | the unit-crossing scenario's X3 waypoints, v1/v2 — see **Oracle perception...** below |
| `launch/x3_nav.launch.py` | the switch launch: `arm:=baseline\|panoptex\|baseline_mppi\|panoptex_mppi`, `perception:=panoptex\|oracle` |
| `plugins.xml` | pluginlib export for both critics (`dwb_core` and `nav2_mppi_controller` loaders) |
| `test/test_predicted_risk_critic.cpp`, `test/test_risk_stack_lookup.cpp`, `test/test_predicted_risk_mppi_critic.cpp`, `test/test_risk_speed_governor.py`, `test/test_gt_tracks.py` | gtest / pytest suites |

## `panoptex_nav::PredictedRiskCritic`

Subscribes `panoptex_msgs/RiskStack` — a **time-layered** predictive risk
field in the `map` frame, where layer `k` describes the instant
`header.stamp + horizon_start + k*dt` (see `panoptex_msgs/RiskStack.msg` for
the exact index/layout contract, and `risk_perception`'s
`predictive_risk_costmap_node.py` module docstring — STAGE 5 — for how the
stack is built and why it is ego-independent).

For every candidate trajectory DWB generates, each pose is scored against
the layer matching **that pose's own time offset**, not against a single
snapshot:

1. poses with `t > max_horizon_s` are ignored;
2. the pose is transformed from the local costmap global frame (`odom` on
   the X3) into `risk_frame` (`map`) using the TF cached once per control
   cycle;
3. the point is converted to a grid cell via `info.origin` (position **and**
   yaw) and `info.resolution`; poses outside the grid are ignored;
4. layer `k = clamp(lround((t + time_shift_s - horizon_start) / dt), 0, steps - 1)`;
5. `r = data[k*H*W + row*W + col] / 100` (`-1`/unknown counts as `0`);
6. if `r >= lethal_threshold` the trajectory is rejected with
   `dwb_core::IllegalTrajectoryException`;
7. otherwise `score += time_discount^t * r^cost_power`.

The consequence that matters: a trajectory may legally drive through a cell
that is lethal *now*, as long as it arrives after the hazard has moved on —
a static costmap critic cannot express this.

`time_shift_s` is the stack's own age (`now - stack.header.stamp`, floored
at 0), computed by `prepare()` every control cycle — **not** a settable
parameter. A pose's `t` is relative to the current control cycle, but the
`RiskStack` it's scored against may be up to `stale_timeout_s` seconds old;
adding that age before picking a layer keeps "the layer for `t` seconds from
now" meaning the same instant whether the stack just arrived or is a bit
stale.

### Fail-soft behaviour

`prepare()` **never returns false** and never blocks DWB. If no `RiskStack`
has arrived, the stack is stale, the costmap frame is unknown, or the TF
lookup fails, the critic goes inactive, logs a throttled warning
(`warn_period_s`) and scores every trajectory `0.0`.

Staleness is measured against `header.stamp` (so it behaves under
`use_sim_time`); if the publisher left the stamp at zero it falls back to
the receive time.

### Parameters

Declared under `<dwb_plugin_name>.<critic_name>.` — e.g.
`FollowPath.PredictedRisk.topic`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `topic` | string | `/risk_stack` | `panoptex_msgs/RiskStack` topic. Subscribed RELIABLE + TRANSIENT_LOCAL + KeepLast(1) to match the publisher (a mismatched durability means the latched message is never delivered). |
| `cost_power` | double | `1.0` | Exponent applied to the 0–1 risk value. `>1` makes the critic tolerate low risk and punish high risk harder. |
| `time_discount` | double | `0.9` | Per-second discount: a pose at time `t` is weighted `time_discount^t`. |
| `lethal_threshold` | double | `0.85` | Risk (0–1 scale) at or above which the trajectory is rejected outright. |
| `stale_timeout_s` | double | `2.0` | Older than this ⇒ inactive, score 0. |
| `max_horizon_s` | double | `3.3` | Poses with a larger time offset are ignored. |
| `risk_frame` | string | `map` | Frame of the risk grid. Identity fast-path when it equals the costmap global frame. |
| `warn_period_s` | double | `5.0` | Throttle period for the "no stack / stale / no TF" warnings. |
| `scale` | double | `1.0` | Declared by `dwb_core::TrajectoryCritic` itself; DWB multiplies the returned raw score by it. |
| `skip_first_s` | double | `0.5` | Poses earlier than this along a candidate are not scored at all. DWB's own trajectory pose 0 is `t=0`, the robot's own current cell — rejecting on it makes EVERY candidate illegal the instant the robot's own cell reads lethal, which froze the X3 and aborted `NavigateToPose` goals 263 times in `avoid_panoptex_1` (2026-09-09, `results/avoidance/`) before this existed. |
| `escape_radius_m` | double | `0.3` | Poses within this distance of the candidate's own first pose (the robot's current position) add graded cost but never throw `IllegalTrajectoryException` — leaving a cell the stack calls lethal must stay a legal option, or the robot can never escape one. 0 occurrences of "No valid trajectories" in the three runs after this landed. |
| *(`time_shift_s`)* | — | — | **Not a parameter** — computed internally each cycle from the stack's age (see above); listed here only so it isn't mistaken for a missing YAML key. |

### Listing it in the nav2 params

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

The critic short name (`PredictedRisk`) is arbitrary; the `.class` parameter is
what maps it onto the pluginlib class. `scale` needs tuning against the other
critics — the raw score is a discounted sum of 0–1 risk values over the
trajectory poses, so it is typically well under 1 for a mildly risky path.

The values above are `config/nav2_x3_panoptex.yaml`'s **current**
`scale`/`lethal_threshold` (40.0 / 0.45), not the critic's compiled-in
defaults (see the Parameters table below — the code still defaults to
`scale: 1.0`, `lethal_threshold: 0.85`). They were retuned after the
`panoptex_1` run because `risk_perception.risk_visualization.CLASS_BASE_RISK`
caps a mobile robot's consequence at `0.75 × confidence`, so a `RiskStack`
cell for a Carter can never reach the default `0.85` — `panoptex_1`'s stack
peaked at `0.37` on carter1's cell. `scale` was doubled alongside the lower
threshold so the sub-lethal scoring term (not just the
`IllegalTrajectoryException` path) has enough weight to actually move DWB's
trajectory choice.

## `panoptex_nav::PredictedRiskMppiCritic`

The MPPI sibling of the critic above (WP2), for the `baseline_mppi`/
`panoptex_mppi` arms. Same `RiskStack` *wire type* — rollout step `j` is
scored against the layer covering that step's own instant — but as of
**2026-09-10 it scores a different, derived stream**: `/risk_stack_srm`, a
**Spatiotemporal Risk Map** (Thomas, Piat & Charpillet 2021 eq. 3;
`risk_perception/risk_perception/srm.py`'s `stack_to_srm`, computed by
`predictive_risk_costmap_node` from the same STAGE 5 `/risk_stack` — see
`../../README.md` §8's **Spatiotemporal risk maps (SRM) and MPPI in (x, y,
t)** for the full derivation, the comet-visualisation topics, and the bench
that picked `cost_weight`), not the raw class/confidence `/risk_stack`
`PredictedRiskCritic` (DWB) still reads. Every SRM cell is a smooth
distance-to-occupied field in `[0, 1]` — high near an occupied core, 0 at
`d0` (1.5 m by default, set by the publisher) — rather than a class
probability, which is why `cost_power`/`collision_threshold` below mean
something different from their DWB namesakes. MPPI-shaped scoring:

- **additive, never an exception.** A predicted collision adds
  `collision_cost` (5000) instead of invalidating the sample; MPPI always has a
  distribution to update and `data.fail_flag` is never set by this critic.
- **the whole batch in one pass** over the raw xtensor buffers
  (`data.trajectories.x/y`), with the per-step layer index and time discount
  hoisted out of the batch loop. 1000 samples × 60 steps costs ≈ 2.5 ms
  (`test_predicted_risk_mppi_critic`'s `BatchScoringIsFastEnough`), i.e. a
  quarter of one 10 Hz control period.
- **why it exists at all.** DWB samples constant-velocity arcs over ~3 s, so
  its only answer to a predicted hazard is "go slower on the same path"
  (`avoid_panoptex_1`–`4`: the stack predicted carter1 correctly and DWB
  stopped *in* its lane). MPPI optimises a whole velocity sequence, so "brake
  now, pass behind the mover at t = 3 s" scores against the layer where the
  mover has already gone, and `motion_model: Omni` lets the mecanum base
  strafe out of the lane. Against a *narrow* risk field this bought little —
  the robot felt nothing until nearly inside the comet; the SRM's wide,
  graded field (peaking at `0.3 m` core + `d0`) is what gives the quadratic
  cost term several seconds of runway to actually bend the trajectory (bench:
  crossing min distance 0.49 m → 1.66 m, head-on 0.45 m → 1.58 m — see the
  README §8 section linked above for the full table).

Per rollout: `t = j * model_dt`; steps with `t < skip_first_s` are skipped;
`k = layerIndex(t, stack_age, horizon_start, dt, steps)` (clamped);
`s = srm_cell` (outside the grid ⇒ ignored, unknown ⇒ 0);
`s ≥ collision_threshold` **and** the pose is farther than `escape_radius_m`
from the rollout's own first pose ⇒ `costs[b] += collision_cost` and this
rollout stops being scored further; otherwise
`accum += time_discount^t · s^cost_power`, and finally
`costs[b] += cost_weight · accum`. `escape_radius_m` exists for the same
reason `skip_first_s` does on the DWB critic: a robot already standing in a
cell the SRM calls hot must keep a way out, or every rollout is charged the
same 5000 and the softmax goes flat.

Fail-soft exactly like the DWB critic: no stack, a stale stack, an unknown
costmap frame or a failed TF ⇒ inactive for that cycle, throttled warning,
`costs` untouched.

### Parameters

Flat dotted keys under `controller_server.FollowPath.<critic name>.`, read
through MPPI's shared `ParametersHandler` (so they are dynamically
reconfigurable).

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `true` | Declared by `mppi::critics::CriticFunction` itself. |
| `topic` | string | `/risk_stack_srm` | `panoptex_msgs/RiskStack` topic, subscribed RELIABLE + TRANSIENT_LOCAL + KeepLast(1) to match the publisher. Was `/risk_stack` before 2026-09-10. |
| `cost_weight` | double | `30.0` (C++ struct default; **`15.0` shipped** in `config/nav2_x3_panoptex_mppi.yaml`, bench-chosen — `results/avoidance/bench_srm.md`) | Multiplies the discounted risk sum. Compare with `PathAlignCritic` (14.0) and `PathFollowCritic` (5.0). Not comparable to the pre-SRM `5.0` — a squared SRM term already carries more weight per unit than a linear class-confidence one did. |
| `cost_power` | double | `2.0` | Exponent on the 0–1 SRM value. `2.0` (was `1.0` pre-SRM) — on a linear distance field, squaring is what turns "keep some clearance" into a gradient the softmax can actually follow. |
| `time_discount` | double | `0.97` | Per-second discount: step at time `t` weighs `time_discount^t`. (Was `0.95`.) |
| `collision_threshold` | double | `0.90` (C++ struct default; **`0.6` shipped**) | SRM value at/above which, outside `escape_radius_m`, `collision_cost` is charged. **Replaces `lethal_threshold`** — kept as a deprecated alias (still honoured if `collision_threshold` is absent, with a startup warning) since a distance-field threshold is a geometric statement ("within `~0.15 m` of an occupied core at `d0 = 1.5`"), not a class-confidence one. `0.6`, not the `0.90` default, because the shipped SRM's occupied-cell value is a mobile robot's consequence weight (`0.75`), not `1.0`. |
| `collision_cost` | double | `5000.0` | Added once per rollout on a predicted collision — deliberately half `ObstaclesCritic.collision_cost` (10000), i.e. a *predicted* collision is expensive but cheaper than a live one. |
| `escape_radius_m` | double | `0.30` | Poses within this distance of the rollout's own first pose (the robot's current position) never get charged `collision_cost`, however hot the SRM there — the MPPI analogue of the DWB critic's `escape_radius_m`. |
| `skip_first_s` | double | `0.3` | Steps earlier than this are not scored: standing in a cell the stack calls lethal must never make every rollout equally bad (the DWB equivalent ended 263 aborted goals on 2026-09-09). |
| `stale_timeout_s` | double | `2.0` | Older than this ⇒ inactive, nothing scored. |
| `risk_frame` | string | `map` | Frame of the risk grid; identity fast-path when it equals the costmap global frame, otherwise one cached TF lookup per `score()` call. |
| `warn_period_s` | double | `5.0` | Throttle period for the "no stack / stale / no TF" warnings. |

### Listing it in the nav2 params

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      critics: ["ConstraintCritic", "ObstaclesCritic", "GoalCritic", "GoalAngleCritic",
                "PathAlignCritic", "PathFollowCritic", "PathAngleCritic", "PredictedRiskCritic"]
      PredictedRiskCritic.class: "panoptex_nav::PredictedRiskMppiCritic"   # documentation only
      PredictedRiskCritic.enabled: true
      PredictedRiskCritic.topic: "/risk_stack_srm"
      PredictedRiskCritic.cost_weight: 15.0
      PredictedRiskCritic.cost_power: 2.0
      PredictedRiskCritic.time_discount: 0.97
      PredictedRiskCritic.collision_threshold: 0.6
      PredictedRiskCritic.collision_cost: 5000.0
      PredictedRiskCritic.skip_first_s: 0.3
      PredictedRiskCritic.escape_radius_m: 0.3
      PredictedRiskCritic.stale_timeout_s: 2.0
      PredictedRiskCritic.risk_frame: "map"
      PredictedRiskCritic.warn_period_s: 5.0
```

**The critic's short name is NOT arbitrary here.** Unlike DWB, MPPI has no
`.class` key: `nav2_mppi_controller`'s `CriticManager` resolves every entry of
`critics:` as `"mppi::critics::" + <name>`. So `plugins.xml` registers this
plugin under the *lookup name* `mppi::critics::PredictedRiskCritic` (with the
C++ *type* `panoptex_nav::PredictedRiskMppiCritic`, plus a same-type alias
under that fully qualified name for direct `pluginlib` loads), and the params
file must list it as exactly `PredictedRiskCritic`. The `.class` key above is
kept purely as documentation and is never read.

## `risk_speed_governor`

A plain rclpy node (no ROS diagnostics beyond its own logs) that turns
Panoptex's semantic world model into a single percentage speed cap, with
**zero Nav2 modification**: `nav2_controller`'s `controller_server` already
subscribes `nav2_msgs/SpeedLimit` on whatever topic `speed_limit_topic`
names and honours it via `setSpeedLimit()`. This node is the *only*
publisher on that topic (`speed_limit` by default — also nav2's own
built-in default, so pointing `controller_server.speed_limit_topic` at it
is a documented no-op key, kept explicit rather than relied on implicitly).
It is independent of, and complementary to, `PredictedRisk` above: the
critic reshapes trajectory **scoring**; this node puts a hard ceiling on
**how fast any trajectory is allowed to go**, straight off
`/risk_perception/world_objects` — no costmap, no `RiskStack`.

`nav2_msgs/SpeedLimit` semantics (read the `.msg` before touching this
node): `percentage: true` means `speed_limit` is 0–100, a percent of the
robot's configured max speed, and **`speed_limit == 0.0` always means "no
limit"** — it is never used to mean "stop". A hard stop is out of scope
here; that's costmap/lidar territory (`nav2_costmap_2d` obstacle layers,
`Oscillation`/`BaseObstacle` critics).

### Policy

All thresholds are ROS parameters — see `DEFAULT_PARAMS` in
`risk_speed_governor.py` and `config/risk_speed_governor.yaml`, kept as one
source of truth between the two.

| Category | Gate | Cap |
|---|---|---|
| `person` | distance-only — people change direction far faster than any CPA model can track, so this category never looks at velocity or CPA. `d <= person_crawl_radius_m` (1.0 m) | `person_crawl_pct` = 15% |
| `person` | `d <= person_slow_radius_m` (2.0 m) | `person_slow_pct` = 40% |
| `robot` / `wheeled` | CPA-gated: moving (`p_motion >= moving_pmot_min` = 0.5) **and** closing on the robot's own path (`0 < t_cpa <= robot_ttc_s` = 3.0 s **and** `d_cpa <= robot_cpa_m` = 1.0 m) | `robot_closing_pct` = **60%** |
| `robot` / `wheeled` | not moving (or not closing), `d <= static_obstacle_slow_radius_m` (1.0 m) | `static_slow_pct` = **60%** |
| `furniture` / `unknown` | ignored entirely — Nav2's own lidar-based costmap layers already handle static clutter; duplicating that here would just fight the local planner | — |

Both robot-category caps were raised from 30%/50% to 60% after
`panoptex_2` (`config/risk_speed_governor.yaml`'s own comment): a closing
AMR the Carters cannot see needs to be *evaded*, not just approached more
slowly, and a robot crawling at 30% cannot get out of the way in time —
`panoptex_2` capped at 30% from 2.5 m and still collided at 0.17 m. See
`../../README.md` §8 for what happened with 60% in place across
`panoptex_3`/`panoptex_4` — it did not, on its own, solve the problem.

A track is dropped before any policy check if its detection score is below
`min_track_score` (0.15) or its header stamp is older than
`max_track_age_sec` (3.0 s, measured against this node's own clock — behaves
under `use_sim_time`).

The final limit is the **minimum** percentage over every track that
currently applies (most restrictive wins), floored at `min_pct` (15%) so the
robot is never capped to a crawl slower than that. If no track applies, the
published value is `0.0` — "no limit", per the semantics above, not a
request to stop.

**Hysteresis**: a cap is applied the instant some track satisfies it (no
delay going more restrictive), but is only released — allowed to drop back
to "no limit" — after `release_hold_s` (0.5 s) with no applicable track at
all. Without this, a track flickering in and out of its threshold radius
(sensor noise, a person pausing exactly at 2.0 m) would chatter the robot's
speed limit every control cycle.

Robot pose comes from TF (`map_frame` → `base_frame`); robot body-twist
comes from `odom_topic` (`nav_msgs/Odometry`, in `base_frame`) and is
rotated into `map_frame` by the TF yaw before it is handed to
`cpa_geometry` — the same pattern
`risk_perception.predictive_risk_costmap_node.PredictiveRiskCostmapNode.
_update_robot_state` uses (see that method's docstring for why: CPA/TTC is a
relation between the robot's own motion and the object's, both must be
expressed in the same frame).

The policy core is the free function `compute_cap()` in
`risk_speed_governor.py` — no rclpy, no Node, pure data in / data out,
unit-tested directly in `test/test_risk_speed_governor.py` with no ROS
graph needed. The `RiskSpeedGovernor` class is a thin ROS shell around it:
gather tracks + robot state each tick, call it, apply hysteresis, publish.

## RGB-D depth + collision monitor (WP5)

Sensing additions to catch what the A1 lidar's single 0.11 m scan plane
misses -- low pallets and rack shields under that plane (the shelf-stall trap
in memory: the map's z-band and the lidar's scan plane are not the same
thing). Identical in **all four arms**: sensing is not a Panoptex-arm
difference, only what the risk layers/critic do with the world model is.

**Local costmap depth source.** `local_costmap.obstacle_layer` in every
`config/nav2_x3_*.yaml` gains a second `observation_sources` entry, `depth`
(`PointCloud2`, `/camera/camera/depth/points`, `min/max_obstacle_height`
`0.05`/`0.60` m, 3 m range) alongside `scan`. The sim publishes that topic
(`warehouse/x3_sim/build_x3_graphs.py`'s `/Graph/ROS2_RGBD`) in frame
`camera_depth_optical_frame`, which is in the TF tree under `base_footprint`
via `base_footprint -> base_link -> camera_link` (URDF fixed joints,
`robot_state_publisher`) `-> camera_depth_optical_frame` (static TF,
`yahboomcar_nav/launch/x3_sim_bringup_launch.py`'s
`tf_camera_depth_optical`) -- no TF was missing, nothing to add there.

**`nav2_collision_monitor`** (`config/collision_monitor_x3.yaml`) is the
last-resort reactive layer, spawned by `x3_nav.launch.py` (behind
`enable_collision_monitor:=true`, the default) in all four arms and in both
`learn_lanes` modes, with its own `lifecycle_manager_collision`
(`node_names: [collision_monitor]`) independent of nav2_bringup's internal
lifecycle manager. Command chain:

```
controller_server --cmd_vel_nav--> velocity_smoother --cmd_vel--> collision_monitor --cmd_vel_safe--> base
```

The sim base must be started with `X3_CMD_VEL_TOPIC=cmd_vel_safe` for the
monitor to actually gate anything -- otherwise it publishes `cmd_vel_safe`
into the void while the base keeps reading `cmd_vel` directly. The harness
scripts (`results/wiring_check/run_arm.sh`, `results/avoidance/run_avoid.sh`,
`tools/warmup_lanes.sh`) export this before starting Isaac; set
`PANOPTEX_NO_CM=1` to leave the base on plain `cmd_vel` and pass
`enable_collision_monitor:=false` to the launch (monitor fully out of the
loop, e.g. to isolate a supervisor-hold vs. monitor-stop question).

Three polygons, tuned to stay clear of `mission_supervisor`'s refuge geometry
(a refuge sits ~0.27 m off a shelf; the stop box must not self-trigger
there) while a genuine ~0.30 m-wide X3 body (`robot_radius` 0.15 m in the
costmaps) still gets a hard stop before contact:

| Polygon | Shape | Action | Notes |
|---|---|---|---|
| `PolygonStop` | x ±0.22 m, y ±0.20 m | `stop` | `max_points: 3` (see below) |
| `PolygonSlow` | x ±0.45 m, y ±0.40 m | `slowdown`, ratio 0.4 | `max_points: 3` |
| `FootprintApproach` | live footprint | `approach`, 1.0 s lookahead | `max_points: 5` |

Observation sources: `scan` (`/scan`) and `pointcloud` (`/camera/camera/depth/points`,
same 0.05–0.60 m height band as the costmap's `depth` source).

**Installed-version gotchas (1.1.20, `/opt/ros/humble`), checked against
`collision_monitor_params.yaml` and `nav2_collision_monitor`'s installed
headers/`.so` before writing the config:**

- The per-polygon point-count parameter is `max_points`, not `min_points`,
  and its sense is inverted from the intuitive reading:
  `polygon.hpp`'s own doc comment is "Maximum number of data readings within
  a zone to **not** trigger the action" -- the action fires once the count is
  *strictly greater* than `max_points`. So a "trigger at >= N points" design
  target becomes `max_points: N - 1` in this version's YAML (N=4 -> 3 for the
  two static zones, N=6 -> 5 for the footprint-approach zone).
- This version has **no `state_topic` parameter and does not publish**
  `nav2_msgs/CollisionMonitorState` -- confirmed by `strings` on
  `libcollision_monitor_core.so` (no `state_topic`/`publish_state` symbol)
  and by `collision_monitor_node.hpp` declaring no such publisher, even
  though the message type itself is installed (`nav2_msgs` ships it
  independent of which nav2 packages use it). `run_arm.sh`/`run_avoid.sh`
  bag-record a `/collision_monitor_state` topic that will simply never
  appear on this nav2 version -- expected, not a wiring bug.

## Build and test

```bash
cd ~/workspace/Panoptex
source /opt/ros/humble/setup.bash
conda activate panoptex                # required for every build in this repo
colcon build --symlink-install --packages-select panoptex_msgs risk_perception panoptex_nav
colcon test --packages-select panoptex_nav && colcon test-result --verbose
```

Five independent suites run under `colcon test`:

- **`test_predicted_risk_critic`** (gtest, `test/test_predicted_risk_critic.cpp`):
  builds a synthetic 10 m × 10 m / 0.1 m stack with a lethal blob crossing
  the origin at t ≈ 2 s and covers clear trajectory, timed collision,
  slow/legal trajectory, the "hazard has already left this cell" case,
  horizon truncation, the frame transform, out-of-grid poses, the
  discount/power parameters, staleness, the inactive fast path, the
  `time_shift_s` layer shift, and pluginlib loadability of
  `panoptex_nav::PredictedRiskCritic`.
- **`test_risk_speed_governor`** (pytest, `test/test_risk_speed_governor.py`):
  `compute_cap()` directly — person crawl/slow/uncapped, closing vs.
  diverging robots, static-obstacle capping, category filtering
  (furniture/unknown ignored, wheeled treated like robot), score/age
  gating, multi-track minimum-wins, the `min_pct` floor, and that CPA
  geometry is evaluated relative to the *moving* robot's own velocity.
- **`test_risk_stack_lookup`** (gtest, `test/test_risk_stack_lookup.cpp`):
  the rules both critics share — layer indexing (rounding, the stack-age time
  shift, clamping at both ends, degenerate stacks), cell lookup through a grid
  origin carrying a yaw, out-of-grid ⇒ `-1`, and the freshness/age rule.
- **`test_predicted_risk_mppi_critic`** (gtest,
  `test/test_predicted_risk_mppi_critic.cpp`): a lethal cell present only in
  layer 10 charges a rollout that is there at t = 1.0 s and not one that
  arrives at t = 3.0 s; the stack-age shift moves which layer a step reads;
  `skip_first_s` protects the rollout's own first steps; the batch pass equals
  a per-pose reference loop sample by sample (with a non-identity odom→map
  transform) and *adds* to pre-existing costs; out-of-grid costs nothing;
  1000 × 60 stays inside the control period; and both pluginlib lookup names
  load. `score()` itself is a thin shell around `scoreStackBatch()`, so the
  scoring math is tested there rather than standing up a `LifecycleNode` +
  `ParametersHandler` + configured `Costmap2DROS` rig.
- **`check_arm_params` / `check_arm_params_mppi`** (plain ctests wrapping
  `tools/check_arm_params.py --baseline … --arm …`): see below.

## The sim study arms (WP3.3 / WP3.4, extended by WP2)

There are four arms, in two pairs — `nav2_x3_{baseline,panoptex}.yaml` (DWB)
and `nav2_x3_{baseline,panoptex}_mppi.yaml` (MPPI, WP2). Each pair is guarded
independently by `tools/check_arm_params.py` (`check_arm_params` and
`check_arm_params_mppi` under `colcon test`), and the MPPI pair is derived from
the DWB pair so that everything outside `controller_server.FollowPath` — AMCL,
both costmaps (including the local `static_layer`), planner, behaviors,
velocity smoother, BT — stays byte-identical. The MPPI pair's controller block
is `nav2_mppi_controller::MPPIController`, `motion_model: Omni`, 60 × 0.1 s =
6.0 s horizon, `batch_size: 1000`, the same 0.26 m/s envelope as DWB, and the
stock critic set minus `PreferForwardCritic` (it penalises exactly the
sideways escape a holonomic base exists to make). The `panoptex_mppi` arm adds
`PredictedRiskCritic` and points the global costmap's `risk_layer` at the
swept planner grid `/risk_costmap_planner` (`max_cost: 140`) instead of the
collapsed `/risk_costmap_predictive` the DWB arm reads.

`config/nav2_x3_baseline.yaml` and `config/nav2_x3_panoptex.yaml` are the DWB
study arms' nav2 params for the sim X3 in the Isaac Sim warehouse. Both are
derived from `yahboomcar_ws`'s vendor DWB config
(`params/dwa_nav_params.yaml`, untouched) plus the tuned AMCL block
(`warehouse/x3_sim/tests/amcl_tuned_params.yaml`) and the mecanum
strafing/`velocity_smoother` additions from
`params/exploration_nav_params_sim.yaml`. They are meant to be **identical**
except for:

- `global_costmap.global_costmap.plugins` and its `risk_layer` entry
  (panoptex only — sub-lethal `max_cost: 120` predictive risk layer on
  `/risk_costmap_predictive`).
- `controller_server.FollowPath.critics` plus the `PredictedRisk.*` keys
  (panoptex only — see "Listing it in the nav2 params" above).
- `controller_server.speed_limit_topic` (panoptex only — feeds the
  `risk_speed_governor` node's `nav2_msgs/SpeedLimit` publications into
  DWB's speed scaling; this happens to equal nav2's own built-in default of
  `"speed_limit"`, so it's a no-op in practice, but it documents the
  dependency explicitly and is the sanctioned per-arm diff either way).

`tools/check_arm_params.py` deep-diffs the two files and fails (non-zero
exit, full diff printed) if anything other than the above has drifted apart.
It's wired into `colcon test` as a plain ctest (`check_arm_params`) — run it
directly with `python3 tools/check_arm_params.py`, or via
`colcon test --packages-select panoptex_nav`.

`config/x3_sim_waypoints.yaml` is Scenario v3, a 3-waypoint triangle (format:
yahboomcar_nav's `waypoints/default_waypoints.yaml`) starting at the X3's
spawn pose — see **Scenario v3** below for the current geometry and why it
replaced the earlier 4-point rectangle, and the comment block at the top of
that file for the full clearance/person-distance numbers.

`launch/x3_nav.launch.py` is the switch launch: `arm:=baseline`, `panoptex`,
`baseline_mppi` or `panoptex_mppi` selects `config/nav2_x3_<arm>.yaml` (or
override with `params_file:=`),
brings up `risk_perception`'s `panoptex_sim.launch.py` (overhead cams +
predictive risk costmap + spatial prior, identical compute load in both
arms, localization OFF so nav2_bringup owns `map`→`odom`), `nav2_bringup`'s
`bringup_launch.py`, `risk_speed_governor` (both panoptex arms), and —
starting `runner_delay` seconds after its own launch — `mission_supervisor`
(**all arms**, `yield_enabled` on the two `panoptex*` arms; see **Mission
supervisor** below) driving `x3_sim_waypoints.yaml` as a `NavigateToPose`
goal per waypoint, `laps` (0 = unbounded) stopping the mission after that
many laps. It refuses to run outside `ROS_DOMAIN_ID=44` (the risk-aware
study domain, `warehouse/env_study.sh`) unless `PANOPTEX_NAV_ANY_DOMAIN=1`
is set:

```bash
ros2 launch panoptex_nav x3_nav.launch.py arm:=baseline laps:=5
ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex laps:=5
ros2 launch panoptex_nav x3_nav.launch.py arm:=baseline_mppi laps:=5
ros2 launch panoptex_nav x3_nav.launch.py arm:=panoptex_mppi laps:=5
```

Run `tools/warmup_lanes.sh` once first if `lane_layer`/the static lane band
need to mean anything — see **Lane warm-up and the frozen prior** below.

Verified (ROS_DOMAIN_ID=199, `PANOPTEX_NAV_ANY_DOMAIN=1`, no Isaac/no
`/clock`) that both arms bring up `controller_server`, `planner_server`,
`bt_navigator`, `waypoint_follower`, `velocity_smoother`, `amcl` and
`map_server`; that the panoptex arm's `global_costmap` loads `risk_layer`
listening on `/risk_costmap_predictive`, its `controller_server` loads the
`PredictedRisk` critic and subscribes to `/risk_stack` (with
`predictive_risk_costmap_node` already publishing it), and `risk_speed_governor`
publishes `/speed_limit`. Without `/clock`, `controller_server` blocks
mid-activation waiting on its costmap/TF (a generic nav2 behaviour, not
specific to either arm) — expected, see the launch file's own docstring.

**Verified live against Isaac Sim, 2026-09-08** (`arm:=baseline`, headless,
`ROS_DOMAIN_ID=44`, 240 s sim window, RTF 0.296x): the X3 drove the full
waypoint loop and collided with carter1 (min ground-truth centre gap
0.342 m at sim t≈237 s); carter1 first became a lethal cell in the local
costmap at 2.9–3.5 m true range.

**`arm:=panoptex`, run `panoptex_1` (same day, RTF 0.300x): also
collided** (min gap 0.141 m) — but not because the critic/governor failed
to help; `object_tracker_node.py` never told them there was a mover.
carter1 was tracked accurately throughout (within 1 m of ground truth 58%
of the time, including during the approach) as 44 distinct, short-lived
track ids (median lifetime 12 s) — GroundingDINO's phrase churn on the same
object ("mobile robot" / "ground mobile robot" / "cart") combined with a
fixed 0.6 m association gate that a 0.6 m/s Carter outruns at ~1 Hz — and
every one of those 44 tracks had `p_motion = 0.00`, because the
Mahalanobis-only motion test never reaches `motion_threshold` (3.0 σ) for a
sparsely-updated mover whose velocity σ stays 0.5–0.9 m/s. Consequence:
`/risk_stack` only ever carried carter1's stationary hypothesis (peak 0.37
on its cell), and `risk_speed_governor` only ever fired `static_slow` —
`robot_closing` never had a chance to. Both tracker bugs were fixed
(`association_key: category`, `gate_speed_mps`, `motion_speed_mps` +
`motion_speed_score` — see `../../README.md` §8's **Perception
prerequisites** subsection and `test/test_object_tracker_motion.py` in
`risk_perception`), and `PredictedRisk.lethal_threshold`/`.scale` were
retuned (0.85→0.45, 20.0→40.0 — see above) since a mobile robot's
consequence cap made 0.85 unreachable regardless.

Three more runs followed as each fix exposed the next problem underneath
it: `panoptex_2` (tracker fix in place) still collided (0.173 m) because
the `RiskStack` magnitude at carter1's true cell stayed too low (0.14-0.17)
under `stack_consequence_mode: scaled`; `panoptex_3`
(`stack_consequence_mode: class` + bbox extent) avoided a collision by the
plan's <0.6 m criterion (0.709 m) but only because the X3 nearly froze —
25% of the stack's layer 0 read lethal from mislabeled static clutter;
`panoptex_4` (per-category `extent_cap_*` + `stack_categories: [person,
robot, wheeled]`) got the predictive layers actually tracking carter1's
future position correctly, and still collided (0.232 m) — the X3 stopped
correctly but *in carter1's lane*, which stopping alone cannot fix against
a Carter that can't see it. None of the five runs (including the
`arm:=baseline` one above) avoided a collision while the X3 kept moving
normally. Full per-run numbers, interpretation, and a "What to do next"
list are in `../../README.md` §8's results table; raw evidence
(`monitor.json`, `x3_nav_excerpt.log`, `nodes.txt`, `risk_stack_info.txt`
per run, plus a one-line `SUMMARY.md`) is under
`../../results/wiring_check/<run>/`. `../../docs/probes_2026-09.md` has the
supporting probes, including why carter1's own lidar never sees the X3 at
all (so all collision avoidance load in this scenario is on the X3's side)
and why `panoptex_1`'s self-track count is not usable evidence about the
sim self-exclusion radii.

## Mission supervisor

`mission_supervisor` (package `panoptex_nav`, executable
`mission_supervisor`; thin launcher + `panoptex_nav/mission_supervisor.py`,
the pure geometry in `panoptex_nav/corridor.py`) is the mission executor for
**both** study arms, spawned by `launch/x3_nav.launch.py` in place of the
old `yahboomcar_nav` waypoint-follower include. It owns the waypoint route
(`NavigateToPose` per waypoint via a raw `rclpy.action.ActionClient` — not
`FollowWaypoints`, which cannot be interrupted and resumed mid-list, and not
`nav2_simple_commander.BasicNavigator`, which spins internally and does not
compose with this node's own timers/subscriptions; both patterns, plus the
`_StopRunner` sentinel-exception unwind for fatal startup errors, come from
`yahboomcar_nav`'s `waypoint_runner.py`) and, when `yield_enabled:=true`
(panoptex arm only), a corridor-yielding layer on top that gets the robot
out of an approaching Carter's way *before* Nav2's reactive layers have to.
The baseline arm runs the identical code path with every yield branch
compiled out behind that one flag — not a second implementation.

### Corridor model (`corridor.py`)

Pure functions over numpy arrays and plain dicts — no rclpy, no Node, no
clock, no TF — unit-tested directly with no ROS graph (see **Bench
evidence** below). A *corridor* is the swept lane a moving `robot`/`wheeled`
track (`pmot ≥ pmot_min`, `|v| ≥ min_user_speed_mps`) is about to occupy: a
rectangle starting `back_margin_m` behind the track, running `|v| ×
t_corridor_sec` ahead of it along its heading, `half_width` either side.
People are excluded on purpose — they stop and turn on the spot, so a 10 s
swept rectangle is fiction; `risk_speed_governor`'s distance-only branch
already covers them. The heading is the track's raw velocity, snapped to
the learned Spatial-Flow heading (see **Lane layer** below) when that
cell's flow confidence exceeds `flow_min_conf` (0.3) and agrees with the
observed velocity within `flow_snap_deg` (30°) — raw velocity jitters
10-20° frame to frame, which swings the far end of a 10 s corridor by
metres, while the flow field is an EMA of what actually travels through
that cell over the whole run. Disagreement beyond `flow_snap_deg` means the
agent is genuinely leaving the learned lane, so snapping is refused rather
than forced.

Two different half-widths answer two different questions and are kept
deliberately separate:

- `corridor_half_width_m` (0.55 m = 0.25 m Carter + 0.15 m X3 + 0.15 m
  margin) for **gap acceptance** ("can I cross before it arrives") — a
  crossing is transient and `t_margin_sec` slack absorbs the error, so being
  wrong by 10 cm here costs a crossing the robot could have made, not a
  collision. (0.65 m was tried first and rejected: it put the scenario's
  west-strip leg permanently inside lane 1's corridor.)
- `danger_half_width_m` (0.90 m = 0.55 + 0.35 m of track-position error,
  measured 0.1-0.4 m off ground truth in `avoid_panoptex_2`) for
  **containment** ("am I in the way") — used for the refuge trigger, the
  static lane-band dilation, and the hold/refuge re-check. Being wrong here
  decides whether the robot stands still or steps aside, so it is padded
  for the tracker's own measured error instead of the nominal geometry.

**Gap acceptance** (`plan_corridor_crossing` + `decide_crossing`): intersect
the current `/plan` with each corridor; for the first crossing ahead,

```
t_clear = (d_entry + 2*half_width + clear_margin_m) / v_cross_mps
```

— time to drive the remaining `d_entry` to the corridor edge plus the full
lane width plus a margin, at the deliberately pessimistic `v_cross_mps`
(0.20 m/s: Nav2 decelerates into and accelerates out of a crossing, so this
has to be a lower bound on the robot's own speed). `tta` is the corridor
user's own distance to the crossing over its speed. **Hold** iff `0 < tta <
t_clear + t_margin_sec` (2.0 s); `tta <= 0` (user past) or infinite
(stationary user) is always "go".

**Refuge** (`find_refuge`): the robot is already inside a corridor whose
user is approaching within `t_yield_sec` (8.0 s) — there is no gap to
accept, the robot has to leave the lane. Picks the nearest free,
obstacle-clear cell (`refuge_radius_m` 2.5 m disc, `refuge_clearance_m`
0.45 m from anything occupied/unmapped) outside every corridor *and* the
static lane band, preferring the robot's own side (crossing the lane to
reach the far side is a worse encounter than the one being avoided) and
directions perpendicular to the lane. If the lane band swallows every legal
cell, the search returns the least-bad in-band candidate rather than `None`
(logged WARN, so a run's analysis can count how often "least bad" fired)
— `avoid_panoptex_1` chose four refuges inside the aisle before the lane
band existed and thrashed through six recomputes as the second lane's
Carter came back into view.

**Lane-line memory** (2026-09-09, `mppi_panoptex_2`). A corridor is the
*instantaneous* swept window: `back_margin_m` behind the track,
`speed * t_corridor_sec` (~6 m) ahead of it. That is the right object for
"will this user sweep the point I am standing on" and the wrong one for "is
this a place to stand" — carter1 patrols the line `x = 1.27` from `y = -4`
to `y = 7`, so a cell on that line but 7 m behind the Carter is outside
every corridor and looks like a fine refuge. It was taken 10 times out of
25 (one refuge 0.01 m off the centre line), the window came back over it,
and the refuge goal was cancelled and recomputed — nine cycles in one
contact episode. The learned lane band did not catch these either (the
prior read 2–15 there against `lane_band_min_value` 5.0), and the X3 ended
up spending 312 s of a 701 s run within 0.6 m of the lane line, where a
clean crossing takes ~7 s.

So the session remembers, per corridor user, the **line** it has actually
been observed driving (`corridor.LaneMemory` / `ObservedLane`): a segment
from the first sighting of a track to the latest, merged with any collinear
segment within `lane_merge_dist_m` so six churned track ids accumulate into
one lane rather than six. `lane_line_clearance()` measures against those
segments *extended* by `lane_extension_m`, because the patrol runs further
than the cameras watch it. It is used only for "where may I stand", never
for "may I cross": `find_refuge` rejects any cell inside
`refuge_lane_clearance_m` of a remembered line (a hard filter with no
least-bad fallback — hence `refuge_radius_max_m`, and hence a
`refuge stage` line in the log saying which disc answered),
`hold_in_place_is_unsafe()` counts standing on one as being in the way, and
`refuge_recompute_reason()` uses it as the only lane-geometry reason to
disturb a committed refuge. The lanes are published as a `LINE_LIST`
`MarkerArray` on `~/observed_lanes`, and `/mission/state` carries `lanes`
(how many are remembered), `lanes_effective` (how many of them actually
count) and `waypoint_in_lane`.

**Never cross a lane to reach a refuge (`mppi_panoptex_4`, 2026-09-09).**
The side preference above was only a *preference*, ranked below the
lane-line clearance, and the sim aisle made it lose. carter1 patrols
`x = 1.27`, a shelf runs at `x ≈ 0.30`, the strip between them is 0.97 m
wide — and `refuge_lane_clearance_m` was a flat 1.0 m, so **nothing on the
robot's own side could ever be legal**. The search answered with
`(2.73, 1.93)`, on the far side, with the Carter `tta = 6.5 s` out;
crossing 1.9 m of danger band at ≤ 0.26 m/s takes ≥ 7 s. The X3 was met in
the middle of the lane (`t_cpa 0.24 s`, `d_cpa 0.39 m`), released standing
*on* the centre line at `(1.27, 1.85)`, and pushed to `(1.31, 6.87)`. Runs
2–3 show the same mechanism. Three changes:

- **The far side is a hard rejection** (`corridor.lane_guards` /
  `guard_blocks`). For every corridor user, and for every remembered lane a
  user is currently driving, a candidate whose perpendicular offset has the
  opposite sign to the robot's is rejected unless
  `tta > crossing_time + refuge_cross_margin_s` (2.0 s), with
  `crossing_time = (|robot offset| + |candidate offset| + 2*half_width) /
  v_cross_mps` — the gap-acceptance test of the crossing rule, applied to
  the one decision that was still crossing a lane without making it. A
  remembered lane with nobody on it is not a guard: it keeps refuges *off*
  itself, but nothing on it can hit us, and walling off every learned line
  would strand the robot.
- **The clearance floor is derived, not fixed** (`refuge_lane_clearance()`):
  `max(refuge_lane_clearance_min_m, refuge_user_half_width_m +
  refuge_robot_radius_m + refuge_lane_margin_m)` = `max(0.60, 0.30 + 0.15 +
  0.30)` = **0.75 m**, so it cannot exceed the aisle it has to fit inside.
  `refuge_lane_clearance_m` > 0 still overrides it outright. The same
  derived number is used by `find_refuge`, `hold_in_place_is_unsafe()`, the
  waypoint gate and `refuge_recompute_reason()` — a floor the call sites
  disagreed about is a robot that steps off a lane and is told it is still
  on one.
- **Wall hug** (`corridor.find_wall_hug`, logged `WALL-HUG` at WARN). When
  neither disc has a candidate, take the free, obstacle-clear cell on *our*
  side that stands furthest off the traffic and hold there — below the
  clearance floor, possibly still inside the corridor, but never across it.
  Only if even that fails does the supervisor hold in place. Ranking
  changed with it: same side now outranks lane clearance, so half a metre
  of extra clearance can no longer buy a lane crossing.

**Lane hygiene (`mppi_panoptex_3`, 2026-09-09).** That memory fragmented
into **32** lanes over the hall's two Carter patrol lines. Each fragment is
extended 2 m past its own ends and each demands `refuge_lane_clearance_m` of
its own, so almost nothing inside the 2.5 m disc stayed legal and every
refuge was found at the 4.0 m stage — 3.5 m away, with a Carter closing.
Three changes: the merge test now takes the new segment's **midpoint**
against the existing line (a 2–4 m piece of a jittering track tilts about
its centre, swinging an endpoint half a metre while the midpoint barely
moves) with the angle widened to 25°, plus a new `lane_merge_gap_m` bound on
along-line separation, which used to be ignored entirely; every insert is
followed by a **transitive compaction pass** (`LaneMemory._compact`), so two
lanes a later observation has made collinear no longer stay apart forever;
and a lane only counts for the clearance rule once it passes
`lane_min_length_m` / `lane_min_points` (`corridor.lane_is_effective`), with
`lane_max` capping the memory at a warehouse-plausible 8.

**Cancel then send (`mppi_panoptex_3`).** nav2's `SimpleActionServer` runs
one goal at a time and can take a freshly accepted goal down together with
the goal it replaced. The run logged `refuge goal ended with status 6` seven
milliseconds after `state: navigating -> refuge`, right behind the
superseded waypoint goal's late result — and the supervisor's answer for a
refuge goal was "staying put until the corridor releases", i.e. standing in
the lane while carter1 pushed the X3 five metres north. Every goal change
now goes through `_transition_goal`: the new goal is queued, the active one
cancelled, and the queued goal sent when that goal reports a **terminal
result** (or by `_tick`'s `cancel_timeout_s` watchdog if no result ever
comes), so the server is idle when the replacement arrives.

Waiting for the CancelGoal *service* response is not enough, which is what
`mppi_panoptex_4` cost: that response only says nav2 **accepted** the
cancel, which its work loop then applies asynchronously. Refuge goal #25 was
sent from it and aborted 21 ms later by that same pending cancel
(`Aborting handle.` while halting the BT, status 6). The collateral-detection
re-send is kept as a second line of defence and covers hold and refuge goals
too — re-sent once, then holding in place with a WARN — and it no longer
requires one of *our* cancels to be unanswered at send time (#25's was
already answered; `_last_cancel_t` is the test now, so the baseline arm,
which never cancels, is still untouched).

**Refuge hysteresis.** A committed refuge is worth something in itself, so
a corridor window merely sliding over it is no longer a reason to cancel
the goal. Only a user whose closest approach to the committed point is
inside `refuge_recompute_tta_s` (6 s), or a lane-line clearance that has
dropped 0.2 m below `refuge_lane_clearance_m`, may recompute — and never
more than once per `refuge_recompute_min_s` (2 s).

**Waypoint gate.** Waypoint C of Scenario v3 is `(0.65, 0.5)`, 0.62 m off
carter1's patrol line, so *arriving* there is a lane crossing that the
plan-crossing test cannot see (the plan ends there rather than passing
through). Whenever the current waypoint's own lane-line clearance is below
`refuge_lane_clearance_m` — reported as `waypoint_in_lane` — it is only
approached while no tracked user will sweep it within `t_yield_sec`, and
the lane-line clause of `hold_in_place_is_unsafe()` keeps a yield from
parking on it.

**Escalation.** A hold whose stand-off point turns out to be behind the
robot (or within `hold_cancel_radius_m`) degenerates to "stand still" —
which is only a real yield if standing still is actually out of the way.
`hold_in_place_is_unsafe()` checks the robot's position against the danger
corridor and the static lane band before accepting that; in
`avoid_panoptex_2`, 8 of 21 yields logged "the stand-off is already behind
us" and one of those was the run's closest approach (0.15 m ground truth) —
the robot obediently parked 0.65 m off lane 1's centre while a Carter came
down it. When hold-in-place would be unsafe the supervisor takes a refuge
instead.

**Release** (`user_passed` / `loss_release_guard`): released once the point
yielded for is `half_width + release_margin_m` (0.3 m) *behind* the user
(not simply `along < 0` — the user's body still occupies the crossing the
instant its centre passes it), or the track is lost for `lost_timeout_sec`
(3.0 s). A lost track is **not** evidence the lane is clear on its own — one
Carter carried six different track ids across a single run —
`loss_release_guard` ignores identity and asks only whether any *currently*
tracked corridor user still sweeps the yielded-for point within
`t_yield_sec`; if one does, the yield re-binds to it (whatever its id) and
continues, and only a true all-clear held for `confirm_ticks` ticks releases
the mission. A loss-triggered release then needs only one confirming tick
for `post_loss_cooldown_sec` (2.0 s) before the next yield, rather than the
usual `confirm_ticks` — the robot is by definition next to a lane that was
busy a moment ago, and re-identification (not departure) is the likelier
read. Resuming also re-checks `refuge_candidate()` first and takes a refuge
instead if the robot is still inside an approaching user's corridor — never
re-enter a lane whose user is still coming.

**Hysteresis**: `confirm_ticks` (3, i.e. 0.6 s at the supervisor's 5 Hz)
consecutive agreeing evaluations before a yield starts; never released
before `min_hold_sec` (1.0 s) after committing.

### State machine and `/mission/state`

```
paused  --(start_delay, waypoint_pause)-->  navigating
navigating --(plan crosses a lane, gap too small)--> holding
navigating --(robot already IN a lane, user approaching)--> refuge
holding | refuge --(user passed, or track lost + all-clear)--> resuming
resuming --(interrupted waypoint's goal accepted)--> navigating
navigating --(last waypoint of the last lap succeeded)--> complete
```

Published `std_msgs/String` JSON on `state_topic` (`/mission/state`), on
every transition and at `publish_rate_hz` regardless (so a recorder never
has to infer state from gaps between transitions):

```json
{"t": 1234.5, "waypoint": 2, "name": "wp_003", "lap": 0,
 "state": "holding", "yield_count": 1,
 "yield": {"user": "7", "kind": "hold", "tta": 10.2, "t_clear": 13.0,
           "xy": [1.32, -3.0]},
 "goal_xy": [0.62, -3.0]}
```

`goal_xy` is the goal **currently** pursued (the hold/refuge point during a
yield, not the waypoint); `waypoint`/`name` always name the leg being
executed — what an interrupted goal resumes to.

### Params (`config/mission_supervisor.yaml`)

The corridor block below is also `corridor.CORRIDOR_DEFAULTS` — the module,
the node's `declare_parameter` calls, and the yaml comments are kept from
drifting apart by construction (one dict, iterated in three places).

| Group | Param | Default | Meaning |
|---|---|---|---|
| mission | `waypoints_file` / `loop` / `laps` / `start_index` | — / `true` / `0` / `0` | `laps=0` = unbounded while `loop:=true`; a positive count stops the mission after that many laps regardless of `loop`. |
| mission | `waypoint_pause_sec` | `0.2` | Pause between waypoints. |
| mission | `yield_enabled` | `false` | The baseline/panoptex A/B switch. |
| mission | `state_topic` | `/mission/state` | Where the JSON status above is published. |
| plumbing | `world_objects_topic` / `plan_topic` / `map_topic` | `/risk_perception/world_objects` / `plan` / `map` | Only subscribed when `yield_enabled:=true`. |
| plumbing | `flow_enabled` / `flow_topic` | `true` / `/risk_perception/spatial_flow/robot_group` | Learned-heading snapping input, and `corridor.lane_speed`'s approach-zone speed; degrades to raw-velocity headings if unavailable. The **merged group** channel (S = max over `robot` + `wheeled`, F from the winning member): the sim's Carters are learned as `wheeled` (GroundingDINO says "cart"/"forklift" as often as "mobile robot"), so the bare `robot` channel is empty. One array serves every corridor user whatever its own category. |
| plumbing | `headway_topic` | `/risk_perception/spatial_headway/robot_group` | Learned headway statistic for the blind-crossing rule, same geometry (`flow_resolution` / `flow_origin_*`) and necessarily the same merged group as `flow_topic`. |
| plumbing | `lane_band_enabled` / `lane_band_topic` | `true` / `/risk_perception/spatial_prior` | Static lane keep-out (see **Lane layer** below); falls back to live corridors only if not yet received. |
| plumbing | `map_free_max` | `25` | OccupancyGrid value at/below which a cell counts as free floor for the refuge search (`-1`/unknown is never free). |
| gating | `self_exclusion_radius_m` | `0.7` | Tracks this close to the robot ARE the robot — the cameras see the X3 as just another "mobile robot". |
| gating | `pmot_min` / `min_user_speed_mps` | `0.5` / `0.2` | Minimum motion confidence / speed for a track to own a corridor. |
| shape | `t_corridor_sec` / `back_margin_m` | `10.0` / `1.0` | Corridor length ahead of the user / margin behind it. |
| shape | `corridor_half_width_m` | `0.55` | Gap-acceptance width (see above). |
| shape | `danger_half_width_m` | `0.90` | Containment width (see above). |
| flow | `flow_min_conf` / `flow_snap_deg` | `0.3` / `30.0` | Snap-to-learned-heading gate. |
| crossing | `clear_margin_m` / `v_cross_mps` / `t_margin_sec` | `0.3` / `0.20` / `2.0` | Gap-acceptance formula. |
| crossing | `hold_back_m` / `hold_cancel_radius_m` | `0.7` / `0.3` | Stand-off distance / "close enough, just stop" radius. |
| refuge | `t_yield_sec` | `8.0` | TTA threshold that turns "already in a lane" into a refuge move. |
| refuge | `refuge_radius_m` / `refuge_clearance_m` | `2.5` / `0.45` | Search disc / minimum clearance from anything occupied or unmapped. |
| refuge | `refuge_radius_max_m` | `4.0` | Widened to ONCE when the narrow disc has no legal cell, before the wall hug and then holding in place. |
| refuge | `refuge_cross_margin_s` | `2.0` | NEVER CROSS A LANE TO REACH A REFUGE: a candidate on the far side of a lane line is rejected unless `tta > (\|robot offset\| + \|candidate offset\| + 2*danger_half_width_m) / v_cross_mps + this`. |
| lane memory | `lane_memory_s` | `600.0` | How long a remembered lane line survives with nothing driving down it (~ a whole run). |
| lane memory | `lane_merge_angle_deg` / `lane_merge_dist_m` / `lane_merge_gap_m` | `25.0` / `1.0` / `3.0` | When two observed segments are the same lane: UNDIRECTED angle (a patrol drives both ways), the new segment's MIDPOINT within the distance of the existing line, and at most this much along-line gap between their extents. |
| lane memory | `lane_min_length_m` / `lane_min_points` | `1.5` / `5.0` | Evidence a lane needs before it counts for the lane-clearance rule at all. Shorter lanes are remembered but do not vote. |
| lane memory | `lane_max` | `8.0` | Hard cap on remembered lanes; the shortest/oldest are dropped first. |
| lane memory | `lane_extension_m` | `2.0` | Each remembered segment is extended this far past both observed ends before clearance is measured. |
| lane memory | `refuge_lane_clearance_m` | `0.0` | Explicit OVERRIDE of that hard minimum; `0.0` (shipped) means derive it from the three terms below. |
| lane memory | `refuge_user_half_width_m` / `refuge_robot_radius_m` / `refuge_lane_margin_m` | `0.30` / `0.15` / `0.30` | The DERIVED hard minimum distance from every remembered lane line for a refuge, a hold-in-place and the waypoint gate: their sum, `0.75` m, floored at `refuge_lane_clearance_min_m`. |
| lane memory | `refuge_lane_clearance_min_m` | `0.60` | Floor under that derived value. |
| hysteresis | `refuge_recompute_tta_s` / `refuge_recompute_min_s` | `6.0` / `2.0` | A committed refuge is disturbed only by a sweep this imminent, and at most this often. |
| lane band | `lane_band_min_value` | `5.0` | Spatial-prior value at/above which a cell counts as "somebody's lane" (deliberately low). |
| release | `release_margin_m` / `lost_timeout_sec` | `0.3` / `3.0` | Release-test margin / track-loss timeout. |
| release | `confirm_ticks` / `min_hold_sec` | `3` / `1.0` | Hysteresis (see above). |
| release | `post_loss_cooldown_sec` | `2.0` | Fast-confirm window after a loss-release. |
| sequencing | `cancel_timeout_s` | `1.0` | How long a goal queued behind a cancel waits for the cancelled goal's terminal result before being sent anyway. |
| sequencing | `collateral_window_s` | `5.0` | A goal that died this soon after being sent, unrun and during one of our own cancels, is re-sent rather than counted as failed. |

### Bench evidence

`test/test_corridor.py`, 44 pytest cases on the pure `corridor.py` module
(wired into `colcon test` via `CMakeLists.txt`'s `ament_add_pytest_test`):
corridor geometry sign conventions, user selection dropping self/slow/
wrong-category tracks, flow snapping and its refusal to snap on
disagreement, plan-crossing index/entry-distance extraction, the
gap-acceptance decision at its boundary, a worked `t_clear` example, the
release margin, refuge search (own-side preference, everything blocked,
everything inside a corridor, the second-lane-band rejection, the
least-bad band fallback, line-clearance preference), hold-point walk-back
including past the lane band, all three `loss_release_guard` behaviours,
and containment differing between the two half-widths.
`test/test_observed_lanes.py` (28 cases) covers the lane memory itself, and
`test/test_refuge_side.py` (19 cases) the **never cross a lane to reach a
refuge** rule: the derived clearance floor and its override, the crossing
clock, guards that only exist while somebody is on the lane, the far-side
refuge refused at `tta = 6.5 s` and taken at 30 s, and the wall hug
(furthest off the traffic on our own side, still `refuge_clearance_m` off
the shelf, never across the line). Plus a scripted-track
bench on domain 199 (fake `/plan`/`/map` — `nav2_map_server` on
`warehouse_x3_nav.yaml` — and a scripted moving "mobile robot" track):
`mission_supervisor` published `/mission/state` transitions `navigating →
holding → navigating` as expected.

**Real-perception weakness observed (`unit2_pan_1`, 2026-09-10).** All the
bench coverage above exercises the refuge/guard geometry against clean or
scripted tracks. The first real-perception (`perception:=panoptex`)
unit-crossing run found a case it does not yet cover: carter2's track ran
~0.3 m east of ground truth (56 % of the run tracked within 1 m, 7 distinct
track ids for one Carter — see the top-level README's **Real-perception
unit run (unit2_pan_1)**). That offset, combined with the danger corridor's
own width, left no cell on the east column satisfying `refuge_clearance_m`
from the shelf at all; `find_refuge`'s least-bad path returned a refuge on
the **west** side of the lane, and the X3 crossed in front of carter2 to
reach it, producing the run's one contact (5.8 s, min gap 0.23 m). This is
a fact about observed behaviour, not a diagnosed code fix: the "never cross
a lane to reach a refuge" guard (above) is designed to hard-reject exactly
this kind of far-side candidate, and `test_refuge_side.py` covers it against
clean tracks, so whether the failure is the guard's side-classification
being thrown off by the track offset, the guard being inactive because the
user briefly read as "not on the lane," or something else has not yet been
isolated. Not fixed as of this pass.

## Lane layer

`config/nav2_x3_panoptex.yaml`'s global costmap adds a SECOND
`nav2_risk_layer::RiskLayer` instance, `lane_layer`, alongside the existing
`risk_layer` (panoptex arm only — `check_arm_params.py`'s allow-list covers
both). It reads `/risk_perception/spatial_prior` — `spatial_prior_node`'s
learned AMR-lane occupancy (S channel), not the instantaneous predictive
costmap — at `max_cost: 90` / `min_risk_value: 5`, deliberately sub-lethal
(under `risk_layer`'s 120, well under lethal 254) so NavFn's Dijkstra search
stops routing *along* a learned lane (parallel travel touches many cells at
that cost) while a perpendicular crossing of that same lane stays cheap (a
crossing touches one cell's width) — the same per-cell cost penalizes the
two cases very differently with no explicit direction term needed. The same
learned occupancy also builds `mission_supervisor`'s **static lane band**
(`corridor.make_lane_band`: cells ≥ `lane_band_min_value`, dilated by
`danger_half_width_m`) — the fix for refuges landing in a lane whose Carter
merely isn't currently in view (see **Mission supervisor** above). See
`config/x3_sim_waypoints.yaml`'s header for the two lane centres Scenario v3
is checked against (carter1 x=1.32, carter2 west leg x=2.15) and **Lane
warm-up** below for how S actually gets populated before either consumer
has anything to route/refuge around.

## Lane warm-up and the frozen prior

`x3_nav.launch.py learn_lanes:=true` starts ONLY `risk_perception`'s
`panoptex_sim.launch.py` (overhead cams, world model, predictive costmap,
`spatial_prior_node` — forced `enable_spatial_prior:=true` regardless of
that arg's own value) — no nav2_bringup, no `risk_speed_governor`, no
`mission_supervisor`. Intended to run against Carter patrol traffic with no
X3 nav stack competing for the domain, so `spatial_prior_node`'s S channel
accumulates a clean read of the two AMR lanes into `spatial_prior_path`
(default `~/.panoptex/spatial_prior_sim.npz` — never the lab's
`~/.panoptex/spatial_prior.npz`). `tools/warmup_lanes.sh [sim_seconds=600]
[spatial_prior_path]` drives one full warm-up session (Isaac headless →
`/clock` → X3 bringup → Carter patrol → this launch mode → wait
`sim_seconds` of **sim** time) and reports S-channel coverage along both
lanes from the saved `.npz` afterwards (max S within ±0.30 m of each lane's
line, every 0.5 m of `y`), reusing `tools/spatial_flow_heatmap.py`'s
npz-loading code where available. **Run it, never `source` it** — it sets
`-o pipefail` and traps `EXIT` to tear its own processes down, which hits
the calling shell instead if sourced; an earlier version's blanket `pkill -f
run_headless.py` in that trap killed a different, concurrently-running
session's Isaac, so the current version only reaps an Isaac it can prove
(via its own pidfile) that it started. Clean 600 s warm-up result
(`results/avoidance/spatial_prior_sim_warmup.npz`): carter1's lane at S ≥
0.05 for 11 of 23 sampled points along it (max 0.40), carter2's west leg
for 10 of 23 (max 0.27) — patchy, but sufficient for `lane_band_min_value`
to treat both as lanes end to end once dilated.

**The prior must be frozen during a study run.** `x3_nav.launch.py` forwards
`spatial_prior_learn_rate` / `spatial_prior_autosave_sec` to
`panoptex_sim.launch.py`'s `spatial_prior_node` overrides, pinned to `0.0` /
`1.0e9` whenever `learn_lanes:=false` (every study run — learning/autosave
`0.20`/`60.0` stay on only in `learn_lanes:=true` warm-up mode). In
`avoid_panoptex_2` the prior kept learning *during* the run: the X3 itself
is tracked as a "mobile robot" and deposited its own path into S alongside
genuine Carter traffic and static-clutter junk, saturating to 38,828 cells
≥ 0.05 (64% of the map) by the end, against ~1,500 after a clean warm-up —
and the run's own autosave then overwrote the warm-up `.npz` on disk. Always
point `spatial_prior_path` at the same file the warm-up wrote, and never run
a study with `learn_lanes:=true`.

## Scenario v3

`config/x3_sim_waypoints.yaml` (frame `map`) is a 3-waypoint triangle: spawn
(0.38, 0.07) → **A** (3.40, 2.50) → **B** (3.40, 0.50) → **C** (0.65, 0.50)
→ (loop back to A). `mission_supervisor` drives the first leg from spawn
onto A, as before. A→B runs the east column outside both Carter lanes;
**B→C and C→A each cross both lanes** (carter1 x=1.32, carter2's west leg
x=2.15) — exactly twice per lap. The crossing sits at y = 0.5, moved off the
v1/v2 scenarios' y = -3.0 after `avoid_panoptex_3`: at y = -3 the crossing
sat 1 m from carter1's U-turn point (`aisle_south`, y = -4.0), so the Carter
reversed direction right beside the X3, the constant-velocity corridor
model pointed the wrong way, and one contact there turned into the Carter
pushing the X3 roughly 8 m up the lane (see that file's own header comment,
written from the bag forensics, and `../../README.md` §8's **Results**
section). At y = 0.5 both Carters hold constant speed (carter1 turns at
y = -4/+7, carter2's west leg at y = -1) and overhead coverage is
65-78% — an improvement, though still short of full coverage (see
**Results** in the top-level README for why that gap still matters).

Earlier revisions, kept in the yaml's own header history: v2 (a 4-point
rectangle A→B→C→D, replacing the original loop that started directly at the
X3's spawn pose) and v2.1 (the west-strip leg C→D became touch-and-return,
because at x=0.60/0.65 it sat close enough to lane 1's edge that the
supervisor spent 118 s of one run in refuge on that leg alone). v3 was
re-checked against `warehouse/maps/warehouse_gt_x3.{yaml,pgm}` (≥0.30 m
obstacle clearance) and `warehouse/carters/tests/people_positions.json`
(≥1.5 m person distance) for every point and segment — v3 clears both with
≥0.35 m / ≥4.5 m margin; see the yaml header for the full numbers.

## Oracle perception (`gt_tracks`) and the unit-crossing scenario (2026-09-10)

`launch/x3_nav.launch.py`'s `perception:=panoptex|oracle` argument (default
`panoptex`) swaps the entire camera/detection chain for **`gt_tracks`**
(`panoptex_nav/gt_tracks_node.py`, executable `gt_tracks`): it reads
Isaac's `/gt_tf` (`World → carterN`) directly and republishes ground-truth
`vision_msgs/Detection3DArray` on the same `/risk_perception/world_objects`
topic and packed `class_id` convention the real `object_tracker_node`
uses, so every downstream consumer (predictive costmap, both critics,
`mission_supervisor`) runs unmodified against a clean signal. See
`../../README.md` §8's **Oracle perception mode** for the parameter table
(`robots`, `moving_speed_mps`, `gt_centre_offset_m` — the axle-vs-body-
centre correction, `-0.23` default) and **Spatiotemporal risk maps (SRM)
and MPPI in (x, y, t)** for why this scenario exists at all (validating the
SRM/MPPI-critic work above cheaply before re-adding perception noise).
Unit-tested in `test/test_gt_tracks.py`, no `rclpy` spin needed.

**The unit-crossing scenario** trims the four-arm study down to one Carter
crossing one X3 back-and-forth: `results/avoidance/run_unit.sh <arm>
<name> [sim_seconds=150]` (copy of `run_avoid.sh`'s Isaac bring-up /
activation-wait / cleanup-trap orchestration — read that script first).
Env overrides: `LAPS` (default `2`), `X3_WPS`/`CARTER_WPS` (waypoint file
paths — `config/x3_unit_cross.yaml` + `../../warehouse/nav/
carter_shuttle.yaml` for scenario v1, `config/x3_unit_cross2.yaml` +
`../../warehouse/nav/carter_shuttle2.yaml` for v2 — see the top-level
README's **Unit-crossing scenario (v1 → v2, why)** for why v2 replaced
v1), `PERCEPTION` (default `oracle`), `YIELD` (`mission_supervisor`'s
`yield_enabled`, default `false`), `PANOPTEX_NO_CM` (`1` disables
`nav2_collision_monitor` so a pure planner test can't be masked by it),
`RUNNER_DELAY`. **Must be executed, never sourced** (same reason as
`run_avoid.sh`/`warmup_lanes.sh` — see this repo's `.claude/CLAUDE.md`).
`tools/analyze_run.py --lane-x <x>` labels each crossing `behind`/`ahead`/
`waited`/`contact`; `tools/unit_summary.py <run>...` builds `../../
results/avoidance/UNIT_SUMMARY.md`.

**Harness changes (2026-09-10).** `run_unit.sh` now kills every
`/dev/shm/fastrtps_*` holder by PID before clearing stale segments (never a
`pkill -f` pattern — that also matches the calling automation shell's own
command line) and settles 20 s after `/clock` comes up before the Carter/X3
ROS graphs are trusted to be publishing. Separately, the first
`carters_patrol` launch after Isaac stalled in lifecycle `Configuring` in
3 of 3 attempts on 2026-09-10 (cause not isolated); the harness's existing
relaunch-once retry (`wait_carters_active`, kill by pidfile then relaunch)
activated both Carters cleanly each time it was needed.

## Results

`../../README.md` §8's **Results — avoidance runs** has the full
`avoid_panoptex_1`-`4` table, the critic-escape-rule fix, the push
artefact, the per-run contact-episode counts, the camera-coverage
diagnosis, and what the learned priors can and cannot do. Raw evidence
(`analysis.txt`, `x3_nav_excerpt.log`, `monitor.json` for runs 1-3) is under
`../../results/avoidance/<run>/`, one line per run in
`../../results/avoidance/SUMMARY.md`. `../../docs/probes_2026-09.md`'s
Addendum 4 has the supporting bag-forensics detail (lane geometry, the
coverage-along-lane table, the push artefact's numbers).

§8's **Results — unit-crossing runs (2026-09-10)** has the SRM/MPPI-critic
+ corridor-yield unit-scenario table (oracle perception): baseline 2
contacts, the SRM critic alone 1, SRM + corridor-yield 0 contacts / 0.81 m
min gap over 6 laps — plus the residual failure class (a Carter restarting
motion close to the crossing, no comet while parked) and next steps. §8's
**Real-perception ghost movers and map-filling comets** and **Real-perception
unit run (unit2_pan_1)**, right below that table, cover the same-day
ghost-mover fix and the first `perception:=panoptex` unit run (1 contact,
carter2 mostly stalled under load) — see also the refuge-side weakness note
above. Per-run detail: `../../results/avoidance/UNIT_SUMMARY.md` and
`../../results/avoidance/unit/<run>/analysis.md`; the SRM bench that chose
`cost_weight` is `../../results/avoidance/bench_srm.md`.
