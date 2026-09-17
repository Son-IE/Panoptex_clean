#!/usr/bin/env python3
"""
predictive_risk_costmap_node.py  --  STAGE 2 (prediction) + STAGE 3 (rasterization)
                                     + STAGE 4 (consequence / relative-motion)
                                     + STAGE 5 (time-layered risk stack)

Consumes /risk_perception/world_objects from the tracker and produces:
    /risk_costmap_predictive               nav_msgs/OccupancyGrid  (for Nav2)
    /risk_stack                            panoptex_msgs/RiskStack (for a
                                            trajectory-scoring critic, see
                                            STAGE 5 below)
    /risk_costmap_planner                  nav_msgs/OccupancyGrid  (for the
                                            route/tactical layer, see
                                            STAGE 6 below)
    /risk_perception/prediction_markers    MarkerArray             (to eyeball)

It publishes to /risk_costmap_predictive, NOT /risk_costmap, so it can run
side by side with the live risk_costmap_node without either one fighting for
nav2_risk_layer's input. To hand Nav2 the predictive field instead, point
`risk_layer.topic` in dwa_nav_params.yaml at /risk_costmap_predictive; do not
make both nodes publish the same topic.

STAGE 2 -- two-hypothesis future occupancy, per track:
    stationary  (weight 1 - p_motion):  Gaussian at current (x,y), cov = P_pos.
    moving      (weight p_motion):       constant-velocity roll-out from (x,y,vx,vy),
                                         cov grows each step (t^2 * P_vel + inflation).
  The mixture weight is p_motion, NOT p_movable -- a standing person (movable but
  not moving) gets a tight blob, not a forward smear.

STAGE 3 -- time-discounted predictive risk field:
    each future step's Gaussian is splatted into the grid, discounted by gamma^step
    (near future weighs more), max-combined across steps/hypotheses/tracks.
    The grid is STATELESS -- rebuilt from scratch each message. All temporal
    reasoning lives in the prediction, so there is no second decay here (TPSO is
    now a pure renderer).

STAGE 4 -- consequence and relative motion, two independent multipliers on
each track's weight before it is splatted:

  consequence  severity by class, from risk_visualization.risk_score_from_label
               (person 0.90, cart 0.65, monitor 0.20 ...), the same table the
               live risk_costmap_node keys off. Without it every track painted
               peak cost 100 and a chair read as lethal as a forklift.

  encounter    CPA/TTC amplification against the ROBOT's own motion. Risk is a
               relation, not a property: a person walking parallel to a parked
               robot is harmless, the same person walking into its path is not,
               and a *static* pillar becomes dangerous the moment the robot
               drives at it. With p_rel = p_obj - p_robot and
               v_rel = v_obj - v_robot,

                   t_cpa = -(p_rel . v_rel) / |v_rel|^2      (>0 <=> closing)
                   d_cpa = |p_rel + v_rel * t_cpa|
                   factor = 1 + gain * exp(-d_cpa/d0) * exp(-t_cpa/t0)

               Diverging tracks (t_cpa <= 0) and tracks with no relative motion
               get factor 1.0 exactly, so a stationary robot in an empty room
               behaves exactly as it did before this stage existed.

               The factor can push a weight above 1.0. That saturates at cost
               100 and *widens* the saturated core rather than making the peak
               taller -- intentional: a collision-course track should own a
               bigger keep-out region, not just a redder one.

  two_hypothesis_cpa (off by default, ablation knob): when False (the
  historical behaviour), the encounter `factor` above is computed ONCE from
  the track's raw (vx, vy) and reused for BOTH the stationary and moving
  hypotheses' severity -- the stationary blob ends up scored against a CPA
  it does not itself experience (a track believed parked still gets
  amplified by whatever the tracker's velocity estimate happens to read).
  When True, `factor` is computed TWICE, once per hypothesis, before
  severity is fused:

      factor_stat = cpa_geometry(x, y, 0,  0,  robot...)   # v_obj = 0
      factor_mov  = cpa_geometry(x, y, vx, vy, robot...)   # v_obj = (vx, vy)
      C_stat = combine_severity(consequence, factor_stat, relbonus)
      C_mov  = combine_severity(consequence, factor_mov,  relbonus)
      w_stat = (1 - p_motion) * C_stat
      w_move = p_motion * gamma**step * C_mov

  p_motion still decides how much mass falls on each hypothesis; the CPA
  factor now separately decides, per hypothesis, how dangerous that
  hypothesis's own geometry is. This matters most for an uncertain track
  (p_motion near 0.5): under the single-factor scheme one CPA reading
  amplifies both hypotheses together or neither, so a track that is
  probably moving away (factor_mov ~ 1.0) but might actually be parked
  directly in the robot's path (factor_stat > 1.0) has its real
  stationary-collision risk diluted by sharing the safe moving-hypothesis
  reading. See encounter_geometry.py and _encounter/_encounter_stationary.

  STAGE 5's stack and STAGE 6's planner grid are UNCHANGED by this flag --
  both already pin factor to 1.0 for every hypothesis, for the unrelated
  ego-independence reason documented below, so there is no second CPA to
  split there either way.

STAGE 5 -- time-layered risk stack, published on /risk_stack as
panoptex_msgs/RiskStack (see that .msg for the exact index/layout
contract) alongside the collapsed grid above:

  Layer 0 is "now"; layer k covers header.stamp + k * pred_dt. Per track,
  the stationary hypothesis is splatted into EVERY layer with weight
  (1 - pmot) * C_stack -- an object we believe is standing still looks the
  same whether the question is "now" or "N steps from now". The moving
  hypothesis's step-k rollout position is splatted into layer k ONLY, with
  weight pmot * C_stack, where C_stack = combine_severity(consequence, 1.0,
  relbonus) -- i.e. the SAME consequence/relation-bonus fusion as the
  collapsed grid's C, but with the CPA/TTC `factor` pinned to 1.0 for BOTH
  hypotheses and NO gamma^step discount.

  This is deliberate, not an oversight: the stack is EGO-INDEPENDENT (see
  panoptex_msgs/RiskStack.msg) so a controller critic that already knows
  its own candidate trajectory's speed can score each layer against the
  robot's OWN motion at that point in time, rather than inheriting a CPA
  factor computed against whatever the robot was doing at splat time. The
  gamma discount is also dropped for the same reason -- it exists so the
  collapsed grid's single max-combined cost prefers near-future risk over
  far-future risk, which is meaningless once time is its own axis; a
  critic scoring pose k at layer k should see that layer's true severity,
  undiscounted.

  unknown-but-moving (stack_paint_unknown_moving, OFF by default as of
  2026-09-10 -- see the WP-A "SRM fills half the map" finding below): a
  track whose category is NOT one of stack_categories is normally excluded
  from the stack entirely (stack_consequence forced to 0 -- see
  _predict_and_splat). The one exception, when the flag is explicitly
  turned on, is a track of category "unknown" (risk_visualization.
  label_category's fallback for anything not in LABEL_CATEGORIES --
  notably the lidar-cluster detector's unmatched clusters, which spawn
  category="unknown" tracks) that clears ALL of: pmot >=
  unknown_moving_pmot_min (0.8, was 0.5 -- a lidar-only track has no class
  to sanity-check its motion belief against, so this exception needs a
  much more confident "moving" read than a labeled track ever has to
  clear), hits >= unknown_moving_min_hits (8) and age >=
  unknown_moving_min_age_sec (2.0) (the same "don't trust a freshly
  spawned track's velocity" reasoning as min_hits_for_motion/
  min_track_age_for_motion_sec below, just a stricter bar for a track with
  no class to price by), and |v| <= max_speed_unknown_mps (the same
  per-category speed-plausibility cap _predict_and_splat applies to every
  track). A track failing any of those (including a genuinely STATIONARY
  one -- a static, unlabeled lidar cluster is the reactive obstacle
  layers' job, not the stack's) is excluded outright. One that clears
  every gate has no class to price by CLASS_BASE_RISK, so
  resolve_stack_consequence() substitutes a flat unknown_moving_risk (0.60
  default) for stack_consequence, which paint_track then still splits the
  usual way: the moving hypothesis carries it weighted by pmot, the
  stationary hypothesis by (1 - pmot) -- i.e. this changes WHAT magnitude
  gets used, not the mixture math itself. `hits`/`age` come from the same
  parsed class_id keys moving_allowed reads (see is_track_mature) --
  missing on both is read as "no evidence against the gate" per the same
  key, since an oracle/older-publisher unknown track (rare, but possible)
  must not silently and permanently fail this stricter gate just because
  its publisher predates the hits/age fields. See resolve_stack_consequence()
  and test_risk_stack.py's unknown-moving tests. Furniture (and every
  other category that isn't "unknown") stays excluded regardless of pmot.

WP-A -- CLASS-AGNOSTIC CONSEQUENCE (motion-first, 2026-09-11): the
unknown-but-moving exception above is a narrow patch for one category
("unknown", lidar-only tracks). `stack_motion_first` (on by default)
replaces it with the general answer Thomas, Piat & Charpillet (2021)'s own
SOGM framing gives: the stack does not grade risk by NOUN at all -- a
"table", a "forklift" and a "mobile robot" are all just "dynamic"/"movable"
occupancy, and the class label never vetoes or weights what a confirmed,
sufficiently-confident mover paints. `stack_categories` (the person/robot/
wheeled allow-list above) stops gating the stack, the collapsed grid
(/risk_costmap_predictive) AND the planner grid (/risk_costmap_planner)
alike -- every confirmed track paints all three, not just the ones whose
label happened to be in the list.

The flat consequence every track paints under motion-first is
`stack_consequence_agnostic` (0.75 default -- between the old "wheeled"
0.65 and "person" 0.90 class values: high enough that a parked mover is
still a real keep-out, not so high that a chair reads as lethal as a
forklift). The SEMANTIC prior -- CLASS_BASE_RISK, the same table
`risk_score_from_label` already keys off -- becomes an OPTIONAL modifier,
`semantic_modifier_enabled` (off by default): when on, the painted value is
`max(stack_consequence_agnostic, class_value)`, never a straight
substitution and never `min` -- the noun may only push the reading UP (a
genuine person still reads 0.90, not a diluted 0.75); it can never veto or
lower what motion evidence already established. This is the class-agnostic
/ class-aware split User B asked about: the *risk map* is class-agnostic by
default (a mover is a mover); the semantic prior stays available as a
separate, deliberately opt-in refinement, never load-bearing for whether a
track paints at all.

`stack_min_track_score` still applies under motion-first -- a track whose
detection confidence hasn't cleared that floor paints nothing (0.0),
agnostic value or not; motion-first widens WHICH categories can paint, it
does not relax the existing low-confidence veto. `stack_moving_pmot_min`
(0.8 default -- raised from 0.5, see its own declare_parameter comment: a
single frame of sensor misalignment could otherwise spike pmot enough to
trigger the moving hypothesis) is a separate, additional condition on the
MOVING hypothesis specifically (see `moving_allowed` in
`_predict_and_splat`): a track that is mature and plausible
(is_track_mature, speed_cap_for_category, min_speed_for_motion_mps -- see
resolve_moving_allowed above) but whose own motion belief currently reads
"probably stationary" (pmot below this floor) still gets the agnostic
consequence, just at its STATIONARY blob -- motion-first does not mean
"assume everything is moving," it means "don't let the class NOUN decide,"
and a track's own pmot is still the only evidence for which hypothesis it
paints. `stack_moving_pmov_min` (0.0 default, off) is the same gate applied
to `pmov` (p_movable) instead -- a slow, class-seeded companion to
`stack_moving_pmot_min` that is largely immune to single-frame noise; raise
it for a specific class that keeps false-triggering the moving hypothesis
without dulling every category's instant-motion sensitivity.
`speed_cap_for_category`'s
fallback for any category with no entry in the per-category cap table
(furniture, unknown, ...) is now `max_speed_default_mps` (2.0), not
`max_speed_unknown_mps` (1.2) -- the latter stays reserved for the
unknown-moving exception path's OWN speed check above, which is
deliberately stricter because that path (still reachable under
`stack_motion_first: false`, a regression/ablation knob) has no class at
all to sanity-check velocity against; a genuinely moving "table" should be
capped like any other wheeled-ish mover, not throttled to the
lidar-cluster-only bar.

See resolve_agnostic_consequence()/resolve_stack_consequence()'s
`motion_first`/`agnostic_value`/`semantic_modifier` kwargs, and
test_risk_stack.py's agnostic-consequence tests.

  semantic keep-out floor (min_extent_{person,robot,wheeled,forklift,
  other}_m): use_object_extent already widens a track's splat by its own
  bbox half-extent, capped per category (extent_cap_*_m) because the
  floor-projected bbox is unreliable for large/flat objects. The floor
  adds the opposite bound: a LOWER limit on that half-extent, the "class
  movability -> consequence severity" Semantic prior's context-aware
  keep-out radius (a person or a forklift needs a minimum berth even when
  its detected bbox collapses to a point). resolve_extent_half() combines
  the two as max(floor, min(cap, bbox_half)) -- the cap still bounds an
  oversized bbox, but if a category's floor exceeds its cap (the forklift
  floor 0.80 m is bigger than the wheeled cap 0.50 m by design -- a
  forklift's keep-out radius should not shrink just because a small bbox
  was measured), the floor wins outright, cap or no cap. select_min_extent()
  picks the floor: the "forklift" LABEL (not just its "wheeled" category)
  gets its own, bigger floor when the string "forklift" appears in the raw
  label; every other track floors by label_category(). Applies to every
  splat this node paints -- the stack AND both collapsed grids
  (/risk_costmap_predictive and /risk_costmap_planner) -- because Pxx/Pyy
  is computed once in _predict_and_splat and shared by all three callees.

  extrapolate_to_now (on by default): a track's message can be milliseconds
  to seconds old by the time this node ticks (tracker latency, network,
  publish_rate vs. the tracker's own rate). Before rolling out the moving
  hypothesis, its (x, y) start point is advanced by its own (vx, vy) times
  the track's age (now - that Detection3D's own header.stamp), clamped to
  [0, max_track_age_sec] so a genuinely stale or never-updated track (age
  0, or a message whose header was never stamped) doesn't extrapolate
  unboundedly. The STATIONARY hypothesis's position is never extrapolated
  -- something we believe hasn't moved hasn't moved just because our
  information about it is old. A track older than max_track_age_sec is
  still painted (clamped, not dropped); it just logs a throttled warning
  with the mean age, since that usually means the tracker or the
  transport between it and this node is falling behind.

STAGE 6 -- planner grid, published on planner_topic (/risk_costmap_planner
default) alongside the STAGE 3/4 collapsed grid, when publish_planner_grid
is true (on by default):

  A SECOND collapsed (single-snapshot, not time-layered) OccupancyGrid,
  same geometry as every other grid this node publishes, built by
  paint_track_planner() -- structurally the same two-hypothesis mixture as
  paint_track()'s grid half, but with three independent parameters so the
  route/tactical layer (global costmap risk_layer) can see farther and fade
  slower than either /risk_costmap_predictive (near-term, gamma-discounted,
  CPA-amplified) or /risk_stack (undiscounted but only 0-6 s):

    - planner_horizon_steps (20 default) instead of horizon_steps -- how
      many rollout steps the moving hypothesis sweeps, independent of the
      stack's own layer count.
    - planner_gamma (0.97 default, vs. the collapsed grid's gamma 0.9) --
      far-future barely fades, so a mover's whole swept lane stays visible
      to NavFn instead of thinning out a few steps in.
    - EGO-INDEPENDENT like the stack: the CPA/TTC `factor` is pinned to 1.0
      for both hypotheses (paint_track_planner has no `factor` argument at
      all -- there is nothing to pin wrong). The planner grid answers "where
      does this mover's lane go", not "is the robot currently closing on
      it" -- that framing is what /risk_costmap_predictive is for.
    - planner_flow_blend_weight (0.5 default) instead of flow_blend_weight
      -- REUSES sample_flow_grid()/blend_velocity() (the same functions
      paint_track()'s moving-hypothesis loop calls), just parameterised by
      a different weight, so the planner grid leans on the learned
      Spatial-Flow heading (paper eq. 16-17: v_eff_k = (1-beta_k)*v +
      beta_k*F_c[x_{k-1}], beta_k = min(1, weight * (k/N) * S_c[x_{k-1}]))
      for its far-horizon steps while /risk_stack and
      /risk_costmap_predictive keep flow_blend_weight at its 0.0 default
      (pure constant-velocity) -- a Carter approaching a U-turn gets swept
      along its learned lane on the grid NavFn plans against, without
      changing what the near-term stack or DWB/MPPI critic sees.

  Severity uses the same class-weighted, confidence-scaled `consequence`
  STAGE 4a computes for the collapsed grid (not stack_consequence -- no
  stack_categories filtering, no unknown-moving substitution: the planner
  grid paints every category the collapsed grid does), combined via
  combine_severity(consequence, 1.0, relbonus) -- factor pinned exactly
  like C_stack. Max-combined across detections, same as every other grid
  here. See paint_track_planner()'s own docstring and
  test_planner_grid.py.

WP-A -- Spatiotemporal Risk Map (SRM), published on srm_topic
(/risk_stack_srm default) when publish_srm is true (on by default):

  Thomas, Piat & Charpillet, "Learning Spatiotemporal Occupancy Grid Maps
  for Efficient Decision-Making" (2021), eq. 3: convert the STAGE 5 stack
  (already exactly their SOGM -- a time-layered occupancy field) into a
  risk field that falls off LINEARLY with distance to occupied space,
  risk(i) = max(0, 1 - d(i)/d0), instead of the Gaussian falloff the raw
  stack itself carries. See risk_perception/srm.py's stack_to_srm() for the
  exact per-level, max-combined formula and test_srm.py for its unit tests
  -- this node only windows the stack and calls that pure function.

  Computed over a srm_window_m x srm_window_m window (12.0 m default)
  centred on the robot's own map-frame position (self.robot_xy, the SAME
  tf lookup Stage 4b already resolves in _update_robot_state -- reused,
  not re-resolved, so this costs nothing extra; if it is unknown --
  enable_relative_motion is off, or the map->base_frame tf hasn't
  appeared yet -- the window centres on the grid's own centre instead,
  same "degrade gracefully" pattern the module docstring already
  documents for Stage 4). The window keeps the per-tick EDT cost bounded
  regardless of how large width_m/height_m grow, and matches Thomas et
  al.'s own local-window framing (their video shows a robot-centred
  "comet," not a whole-map field).

  Published as a SECOND panoptex_msgs/RiskStack (srm_topic) with
  info.origin set to the WINDOW's own origin (from
  risk_perception.srm.window_indices) and info.width/height the window's
  cell size -- NOT the full grid's -- while dt/steps/horizon_start and the
  header stamp are identical to the raw stack's, since the SRM is a
  per-layer transform of the same K layers, not a different horizon.  Also
  published: srm_now_topic (/risk_srm_now, nav_msgs/OccupancyGrid = SRM
  layer 0, "now," for RViz -- the raw stack has no single-grid analogue at
  all, so this is the SRM's only "drop straight into an existing display"
  form) and srm_marker_topic (/risk_perception/srm_markers,
  visualization_msgs/MarkerArray, one POINTS marker per layer k covering
  every cell >= srm_marker_min, coloured red (k=0, "now") fading to yellow
  (k=steps-1, ~6 s out) and alpha proportional to the cell's SRM value --
  the "comet" Thomas et al.'s own video shows). See _publish_srm() and
  _srm_layer_markers().

  bounded rollout / young-track / speed-plausibility fixes (2026-09-10,
  "SRM fills half the map with real perception"): three independent root
  causes, three independent fixes, all upstream of stack_to_srm() itself
  (which needed no changes -- it was always correctly rendering whatever
  the stack handed it):

  (1) paint_track() grows each rollout step's variance as
      var = P + t^2*P_v + vel_inflation*t, and object_tracker_node seeds a
      freshly spawned track's P_v at 1.0 (m/s)^2 -- the KALMAN FILTER'S
      OWN PRIOR, not evidence the object is actually moving that fast. At
      t=6s (the far end of the stack's horizon) that prior alone inflates
      sigma to ~6 m. Fixed by two independent clamps in paint_track(): (a)
      Pvx/Pvy are clamped to sigma_v_max_mps^2 (0.5 m/s default) BEFORE
      feeding the t^2 term, so the prior itself can't blow up the rollout;
      (b) var_x/var_y are clamped to max_sigma_m^2 (0.6 m default) in
      EVERY layer this function paints -- grid, stack stationary, and
      stack moving alike -- as a second, independent backstop (see (b)'s
      own effect on the stationary hypothesis too, since it shares Pxx/
      Pyy with the moving one). min_sigma_m (the pre-existing floor on
      Pxx/Pyy) is untouched; this only adds the matching ceiling.
  (2) A track with an implausible velocity (usually an association jump
      mid-track) got painted as a mover at that velocity regardless. Fixed
      by per-category speed caps (max_speed_{person,robot,wheeled,
      unknown}_mps) in _predict_and_splat/_resolve_moving_allowed: a track
      whose |v| exceeds its category's cap is painted stationary-only
      (moving_allowed=False), logged once per track id.
  (3) A freshly spawned track's pmot/velocity belief is itself unreliable
      before the tracker has accumulated enough hits/age to trust it (the
      same P_v=1.0 prior issue as (1), but for the MIXTURE WEIGHT rather
      than the rollout's spread). Fixed by is_track_mature() gating
      moving_allowed on min_track_age_for_motion_sec (1.0 s) and
      min_hits_for_motion (5): a track below either reads its pmot as 0
      for painting purposes, both grid and stack -- see moving_allowed on
      paint_track(). `hits`/`age` are read from the tracker's class_id
      string (object_tracker_node's WP-B addition); a track with NEITHER
      key (gt_tracks_node's oracle tracks, or any older publisher) is
      treated as mature unconditionally -- this gate must not silently
      break perception:=oracle runs, which is exactly the scenario this
      whole SRM/MPPI stack was validated against on 2026-09-10.

  Separately, the "comet" markers (_srm_markers) used to draw the
  STATIONARY hypothesis's contribution to every layer with that layer's
  own time-colour, so a parked object -- whose every SRM layer is
  identical to layer 0 by construction (see STAGE 5's stationary-hypothesis
  splat) -- showed a full yellow (k=steps-1) disc exactly like a genuine
  mover. Fixed by gating layers k>=1 on srm[k]-srm[0] >= srm_marker_delta
  (0.05 default): only a cell where THIS layer reads meaningfully above
  "now" gets a time-coloured point, so a parked object now shows only
  layer 0's red disc with no tail. Layer 0 itself is unaffected (still
  every cell >= srm_marker_min, coloured red).

Robot state comes from tf (map -> base_frame) for position/heading and /odom
for body-frame twist, which is then rotated into map. tf, not odom's own pose:
odom drifts, and the grid is in map. If either is missing the node degrades to
"robot stationary at the tf position" and, failing that, skips Stage 4
entirely -- prediction and consequence keep working.

Optionally blends in spatial_prior_node's persistent dynamics map as a weak
floor (`spatial_prior_weight`), so a doorway that is habitually busy carries
some risk even in a frame where nothing is detected. Off unless the topic is
actually publishing.

WP-C -- CLASS-AGNOSTIC RECENT-ACTIVITY FLOOR (2026-09-11): a second, faster
floor, from spatial_prior_node's `/risk_perception/activity_prior`
(`activity_topic`) -- an EMA of any mover, no category channels at all
(reuses spatial_prior_node's own `deposit_into_grid`/decay/publish
machinery; see that node's docstring), with a 5-minute half-life instead of
the multi-hour lifelong prior above. Two things distinguish it from
`spatial_prior_weight`'s floor:

  - it is CLASS-AGNOSTIC by construction, not a per-category channel merged
    down -- the same "a mover is a mover" framing as `stack_motion_first`
    above, just applied to a place-indexed memory instead of a single
    track's own consequence.
  - it keeps decaying even while the lifelong prior is FROZEN for a study
    run (see the repo's "the spatial prior must be frozen during a study
    run" convention) -- activity is run-time state, not a learned prior,
    and spatial_prior_node never saves it to disk. A run that must not
    LEARN a new lane can still legitimately want "something walked past
    this doorway 90 seconds ago" to matter for the next few minutes.

Resampled onto this node's own grid geometry by `_resample_occupancy_grid`
(the same nearest-cell, through-world-coordinates logic `_prior_cb` always
used, factored out so both callbacks share it -- see that method's
docstring) and cached UNWEIGHTED in `self.activity_baseline`, because --
unlike `spatial_prior_weight`, one weight baked in at ingestion -- three
INDEPENDENT weights apply it to three different consumers in `_tick`,
BEFORE any track is painted:

  - `activity_weight` seeds every layer of the STAGE 5 stack (bounded, by
    design, at <= `srm_levels[0]` + 0.05 -- an empty cell's activity floor
    should only ever reach about the SRM's lowest graded level, never
    read like a real detection).
  - `activity_weight_planner` seeds the STAGE 6 planner grid.
  - `activity_weight_grid` seeds the STAGE 3/4 collapsed grid
    (`/risk_costmap_predictive`) -- by MAX with the existing
    `spatial_prior_weight` floor `grid` already starts at (`grid =
    self.prior_baseline.copy()`, see `_tick`'s own top), not a second,
    additive floor stacked on top of it.

Every one of these three is a MAX-seed, never an addition: a track's own
painted core on a busy cell keeps its own (usually higher) value exactly as
before this feature existed; only a cell with NOTHING currently painted in
it is raised, to `weight * activity`. This composes for free with the rest
of the union-of-priors design the same way two tracks' splats already
compose with each other (`_splat_into`'s own `np.maximum`) and the way
nav2's own costmap layers (`risk_layer`, `lane_layer`) max-merge on top of
whatever this node publishes -- a recent-activity floor here is simply one
more thing being max-combined into the same field, never double-counted
against `lane_layer`'s own (much slower-moving) learned-lane floor.

Optionally ALSO reshapes STAGE 2b's rollout using spatial_prior_node's
per-category Spatial-Flow heading channel (`flow_blend_weight`, 0.0 = off
by default): for each track, its own category's (s, fx, fy) is sampled at
its current predicted cell every step, and its own (vx, vy) is blended
toward (fx, fy) -- growing with horizon step (near-term stays pure
constant-velocity; only far-term steps, where CV is least trustworthy
anyway, lean on the learned prior) and scaled by s (a cell with no deposit
history contributes nothing, regardless of flow_blend_weight -- see
blend_velocity). This is a DIFFERENT consumption of spatial_prior_node's
output than the floor above: the floor uses the aggregate S published on
spatial_prior_topic; the rollout blend uses the CATEGORY-SPECIFIC (s, fx,
fy) published per category on /risk_perception/spatial_flow/<category>,
because blending a track toward a place's average velocity is only
sensible against that track's own category's history.

`flow_category_map` (default ["robot:robot_group", "wheeled:robot_group"])
redirects which CHANNEL a category samples: a category that is a key in
the map samples spatial_prior_node's merged /risk_perception/spatial_flow/
<channel> topic (see that node's merge_group_flow) instead of its own
per-category one; a category absent from the map is unaffected. This
exists because object_tracker_node's `association_groups` fuses robot and
wheeled into one identity group (a mislabeled Carter flips label from
frame to frame), so either category's OWN per-category channel alone can
be missing the very evidence the other category deposited under a
different label that instant -- the merged channel already carries
whichever member has the stronger S at each cell. Applies to BOTH the
collapsed/stack blend (flow_blend_weight) and the planner grid's own
(planner_flow_blend_weight) -- see resolve_flow_channel/
parse_flow_category_map and test_predictive_costmap_flow.py.

Honest naming: this is a time-discounted PREDICTIVE RISK FIELD, not path-conditioned
collision probability (rasterizing into one 2D grid collapses the time axis).
CPA/TTC here only *shapes the costmap Nav2 plans against*; true reactive
avoidance (velocity obstacles / RVO / CBF) is a planner-level concern and is
not what this node does.

REQUIRES the tracker to embed velocity in class_id:
    "label|pmov=0.90|pmot=0.05|vx=0.12|vy=-0.03"
and position+velocity covariance in pose.covariance (cov[0],cov[7],cov[21],cov[28]).
Optionally also "hits=<int>" / "age=<sec>" (object_tracker_node's WP-B
addition, 2026-09-10) -- consumed by is_track_mature()'s young-track gate
(see the WP-A bug-fix section above); a class_id with NEITHER key (oracle
gt_tracks_node, or any older publisher) is treated as mature unconditionally,
never as young, so perception:=oracle keeps working unchanged.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException

from panoptex_msgs.msg import RiskStack

from risk_perception.risk_visualization import (
    risk_score_from_label, label_category, parse_class_id)
from risk_perception.encounter_geometry import cpa_geometry, combine_severity
from risk_perception.srm import stack_to_srm, window_indices
from risk_perception.debug_log import open_debug_csv, open_latency_csv, log_latency


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def sample_flow_grid(flow_array, wx: float, wy: float, resolution: float,
                     origin_x: float, origin_y: float, rows: int, cols: int
                     ) -> Tuple[float, float, float]:
    """Nearest-cell (s, fx, fy) from a (rows, cols, 3) per-category
    Spatial-Flow array; (0.0, 0.0, 0.0) outside bounds or when flow_array
    is None (category never subscribed, or flow_blend_weight is 0). s=0
    means "no evidence," which blend_velocity treats as zero confidence
    regardless of flow_blend_weight. Pure geometry -- no ROS, no Node --
    see test_predictive_costmap_flow.py."""
    if flow_array is None:
        return 0.0, 0.0, 0.0
    col = int((wx - origin_x) / resolution)
    row = int((wy - origin_y) / resolution)
    if 0 <= row < rows and 0 <= col < cols:
        s, fx, fy = flow_array[row, col]
        return float(s), float(fx), float(fy)
    return 0.0, 0.0, 0.0


def parse_flow_category_map(entries: List[str]) -> Dict[str, str]:
    """Parse `flow_category_map` entries ("category:channel") into a
    {category: channel} dict. Default
    ["robot:robot_group", "wheeled:robot_group"] routes both categories to
    spatial_prior_node's merged /risk_perception/spatial_flow/robot_group
    channel instead of their own per-category topics -- the tracker's
    `association_groups` fuses robot/wheeled into one identity group
    (a mislabeled Carter flips between the two label from frame to frame),
    so sampling either category's OWN topic alone would miss whichever
    label lost that instant; the merged channel already carries the better
    of the two per cell (see spatial_prior_node.merge_group_flow). A
    category with no entry here keeps its own channel, unchanged from
    before this feature. Malformed entries (no ':', or a blank half) are
    skipped. Pure function -- see test_predictive_costmap_flow.py."""
    out: Dict[str, str] = {}
    for entry in entries:
        if ":" not in str(entry):
            continue
        category, channel = str(entry).split(":", 1)
        category, channel = category.strip(), channel.strip()
        if not category or not channel:
            continue
        out[category] = channel
    return out


def resolve_flow_channel(category: str, flow_category_map: Dict[str, str]) -> str:
    """Which /risk_perception/spatial_flow/<channel> a track of `category`
    samples: the mapped channel if `category` is a key in
    `flow_category_map`, else `category` itself (its own per-category
    channel, unchanged from before this feature). Pure function -- see
    test_predictive_costmap_flow.py."""
    return flow_category_map.get(category, category)


def blend_velocity(vx: float, vy: float, flow_s: float, flow_fx: float,
                   flow_fy: float, step: int, horizon_steps: int,
                   flow_blend_weight: float) -> Tuple[float, float]:
    """Blend a track's own (vx, vy) toward a category-matched Spatial-Flow
    cell's (flow_fx, flow_fy) -- growing with horizon step (near-term stays
    pure constant-velocity; only far-term steps, where CV is least
    trustworthy anyway, lean on the learned prior) and scaled by flow_s (a
    cell with no deposit history contributes nothing, regardless of
    flow_blend_weight). Returns (vx, vy) UNCHANGED whenever
    flow_blend_weight <= 0 -- the off-by-default path this system ships
    with today. Pure arithmetic -- see test_predictive_costmap_flow.py."""
    if flow_blend_weight <= 0.0:
        return vx, vy
    step_frac = (step + 1) / horizon_steps
    blend = min(1.0, flow_blend_weight * step_frac * flow_s)
    return ((1.0 - blend) * vx + blend * flow_fx,
            (1.0 - blend) * vy + blend * flow_fy)


def resolve_agnostic_consequence(base_consequence: float, agnostic_value: float,
                                 semantic_modifier: bool) -> float:
    """WP-A class-agnostic consequence override (Thomas et al. 2021's
    "dynamic"/"movable" framing -- a mover is a mover; the class NOUN never
    vetoes or weights the risk it reads, only the semantic prior may nudge
    it up when explicitly turned on).

    `base_consequence` is whatever the class-keyed lookup would otherwise
    have produced (risk_score_from_label for the collapsed/planner grids'
    own `consequence`, _stack_consequence's class/scaled magnitude for the
    stack's `base_stack_consequence`) -- this is the ONE shared helper both
    call, so the SAME agnostic value paints /risk_costmap_predictive,
    /risk_costmap_planner AND /risk_stack alike for one track, not just
    whichever grid happened to be touched first (see
    PredictiveRiskCostmapNode._predict_and_splat and
    resolve_stack_consequence's motion_first branch below).

    semantic_modifier=False (semantic_modifier_enabled off, the default):
    `agnostic_value` outright -- the class value is discarded entirely, the
    noun neither lowers nor raises the reading.
    semantic_modifier=True: `max(agnostic_value, base_consequence)` -- the
    semantic prior may only push the value UP (a person's 0.90 class value
    beats the flat 0.75), never down; a modifier, never a veto.

    Pure arithmetic -- see test_risk_stack.py's agnostic-consequence tests."""
    if semantic_modifier:
        return max(agnostic_value, base_consequence)
    return agnostic_value


def resolve_stack_consequence(
    category: str, stack_categories, pmot: float, base_stack_consequence: float,
    stack_paint_unknown_moving: bool, unknown_moving_risk: float,
    unknown_moving_pmot_min: float,
    unknown_moving_min_hits: Optional[float] = None,
    unknown_moving_min_age_sec: Optional[float] = None,
    hits: Optional[float] = None,
    age: Optional[float] = None,
    speed: float = 0.0,
    max_speed_unknown_mps: float = 1.2,
    motion_first: bool = False,
    agnostic_value: float = 0.75,
    semantic_modifier: bool = False,
    score: Optional[float] = None,
    stack_min_track_score: Optional[float] = None,
) -> float:
    """STAGE 5 category gate for the time-layered stack (see module
    docstring's 'unknown-but-moving' section). `base_stack_consequence` is
    whatever _stack_consequence() already computed from the track's own
    label/score (class-level or confidence-scaled per stack_consequence_mode).

    - category in stack_categories -> pass it through unchanged (the
      existing person/robot/wheeled behaviour).
    - category == "unknown" (risk_visualization.label_category's fallback --
      notably a lidar-cluster detector's unmatched track) AND
      stack_paint_unknown_moving AND pmot >= unknown_moving_pmot_min AND
      (when given) hits >= unknown_moving_min_hits AND age >=
      unknown_moving_min_age_sec AND speed <= max_speed_unknown_mps -> the
      track reads as genuinely MOVING, mature enough to trust, and at a
      plausible speed, with no class to price it by, so substitute the
      flat unknown_moving_risk magnitude instead of whatever
      base_stack_consequence carried (that value came from a
      CLASS_BASE_RISK lookup keyed on a label like "lidar_cluster" that
      isn't in the table -- meaningless here). The last four checks are
      the 2026-09-10 "SRM fills half the map" fix's stricter bar for this
      exception specifically (unknown_moving_pmot_min raised 0.5 -> 0.8;
      hits/age/speed added) -- a lidar-only track has no class to
      sanity-check its motion belief against, so this path needs to be
      more confident than a labeled track ever has to be.
      unknown_moving_min_hits/unknown_moving_min_age_sec default to None
      (gate skipped) and hits/age default to None (checked only when both
      the threshold AND the value are given) purely so this function's
      pre-2026-09-10 callers/tests don't have to change; the node always
      passes all four.
    - everything else (a STATIC unknown, or any category that isn't
      "unknown" at all -- furniture included) -> 0.0, excluded from the
      stack exactly as before this feature existed.

    `motion_first`/`agnostic_value`/`semantic_modifier` (WP-A, 2026-09-11,
    default False/0.75/False so every pre-WP-A caller/test is unaffected):
    when motion_first is True this function's whole category-gate logic
    below is BYPASSED -- see the module docstring's "WP-A CLASS-AGNOSTIC
    CONSEQUENCE" section. `stack_categories`/`stack_paint_unknown_moving`
    stop mattering; every category (furniture, unknown, robot, wheeled,
    person alike) is priced by resolve_agnostic_consequence(
    base_stack_consequence, agnostic_value, semantic_modifier) instead,
    UNLESS `score`/`stack_min_track_score` are both given and score falls
    below that floor, in which case the track still paints 0.0 -- the
    existing low-confidence veto survives motion-first unchanged, only the
    category gate is removed. `score`/`stack_min_track_score` default to
    None (veto skipped) purely so this function's pre-2026-09-11 test
    calls, which never pass them, are unaffected; the node always passes
    both.

    Pure function -- no self, no rclpy -- see test_risk_stack.py's
    unknown-moving and agnostic-consequence tests."""
    if motion_first:
        if (score is not None and stack_min_track_score is not None
                and score < stack_min_track_score):
            return 0.0
        return resolve_agnostic_consequence(
            base_stack_consequence, agnostic_value, semantic_modifier)
    if category in stack_categories:
        return base_stack_consequence
    if not (stack_paint_unknown_moving and category == "unknown"
            and pmot >= unknown_moving_pmot_min):
        return 0.0
    if unknown_moving_min_hits is not None and hits is not None \
            and hits < unknown_moving_min_hits:
        return 0.0
    if unknown_moving_min_age_sec is not None and age is not None \
            and age < unknown_moving_min_age_sec:
        return 0.0
    if speed > max_speed_unknown_mps:
        return 0.0
    return unknown_moving_risk


def is_track_mature(kv: Dict[str, float], min_track_age_for_motion_sec: float,
                    min_hits_for_motion: float) -> bool:
    """WP-A young-track gate (2026-09-10 "SRM fills half the map" finding,
    root cause 3): object_tracker_node seeds a freshly spawned Kalman
    track's velocity covariance P_v at 1.0 (m/s)^2 -- the FILTER'S OWN
    PRIOR, not evidence the object is actually moving that fast -- and its
    pmot belief is similarly unreliable before enough measurement updates
    have accumulated. A track needs BOTH enough age AND enough hits before
    its motion belief is trusted for a moving-hypothesis splat; short of
    either, the caller should treat pmot as 0 for painting (see
    moving_allowed on paint_track).

    `kv` is parse_class_id()'s returned dict; `hits`/`age` are
    object_tracker_node's WP-B addition to the class_id string. A class_id
    with NEITHER key (gt_tracks_node's oracle tracks under
    perception:=oracle, or any publisher predating this feature) is
    treated as mature (True) -- this gate must never regress oracle mode,
    which has no notion of "hits" at all. A class_id with only ONE of the
    two keys checks only that one (the other reads as "no evidence against
    maturity" on its own dimension). Pure function -- see
    test_risk_stack.py."""
    age = kv.get("age")
    hits = kv.get("hits")
    if age is None and hits is None:
        return True
    if age is not None and age < min_track_age_for_motion_sec:
        return False
    if hits is not None and hits < min_hits_for_motion:
        return False
    return True


def speed_cap_for_category(category: str, speed_caps: Dict[str, float],
                           default_cap: float) -> float:
    """Per-category |v| plausibility cap (2026-09-10 finding, root cause
    2: an association jump mid-track hands it an implausible velocity that
    then gets painted as a mover at that speed). `speed_caps` is keyed by
    risk_visualization.label_category()'s categories (person/robot/
    wheeled); any other category (unknown, furniture, ...) falls back to
    `default_cap` (max_speed_unknown_mps). Pure function -- see
    test_risk_stack.py."""
    return speed_caps.get(category, default_cap)


def resolve_moving_allowed(
    kv: Dict[str, float], speed: float, category: str,
    min_track_age_for_motion_sec: float, min_hits_for_motion: float,
    speed_caps: Dict[str, float], default_cap: float,
    min_speed_mps: float,
) -> Tuple[bool, str]:
    """Pure WP2 extraction of the moving_allowed decision for paint_track/
    paint_track_planner (2026-09-10 "SRM fills half the map" finding, root
    causes 2+3, plus the 2026-09-10 evening min-speed guard below) -- no
    self, no rclpy, no logging (the node wrapper, _resolve_moving_allowed,
    owns the once-per-id log lines; this function only decides).

    Checked in order, first match wins:
    - not is_track_mature(kv, ...) -> (False, "immature"): a freshly
      spawned track's pmot/velocity belief is still the tracker's own
      seeded prior, not evidence (see is_track_mature's own docstring).
    - speed > speed_cap_for_category(category, ...) -> (False, "over_cap"):
      an association jump handed the track an implausible velocity for its
      class (see speed_cap_for_category's own docstring).
    - speed < min_speed_mps -> (False, "below_min_speed"): new 2026-09-10
      evening guard. A camera-jitter track can clear both checks above yet
      still read a few cm/s of "motion" that isn't real; oracle's own
      gt_tracks_node.classify_motion already only sets pmot=1 at
      speed >= 0.15 m/s (its motion_speed_mps), so gating panoptex's own
      painting at the SAME floor (min_speed_for_motion_mps, default 0.15)
      makes real and oracle perception agree on "how slow is stationary"
      without touching is_track_mature or the speed cap -- and does not
      regress oracle itself, since an oracle track never has pmot=1 below
      that speed in the first place.
    - otherwise -> (True, "ok").

    `kv` is parse_class_id()'s dict (drives is_track_mature); `speed_caps`/
    `default_cap` are the node's per-category cap table and its fallback
    (max_speed_unknown_mps), same arguments speed_cap_for_category takes.
    Pure function -- see test_risk_stack.py."""
    if not is_track_mature(kv, min_track_age_for_motion_sec, min_hits_for_motion):
        return False, "immature"
    cap = speed_cap_for_category(category, speed_caps, default_cap)
    if speed > cap:
        return False, "over_cap"
    if speed < min_speed_mps:
        return False, "below_min_speed"
    return True, "ok"


def select_min_extent(
    label: str, category: str, min_extent_person: float, min_extent_robot: float,
    min_extent_wheeled: float, min_extent_forklift: float, min_extent_other: float,
) -> float:
    """Pick the semantic keep-out floor (see module docstring) for one
    track. The "forklift" LABEL wins over its "wheeled" CATEGORY -- a
    forklift needs a bigger minimum berth than an arbitrary cart, and
    label_category() alone can't distinguish the two (both map to
    "wheeled"). Substring match on the raw label, case-insensitive, so
    "forklift", "toy forklift", etc. all qualify. Everything else floors
    by category; a category with no floor of its own (i.e. anything but
    person/robot/wheeled) uses min_extent_other. Pure function -- see
    test_risk_stack.py's extent-floor tests."""
    if "forklift" in label.lower():
        return min_extent_forklift
    if category == "person":
        return min_extent_person
    if category == "robot":
        return min_extent_robot
    if category == "wheeled":
        return min_extent_wheeled
    return min_extent_other


def resolve_extent_half(bbox_half: float, cap: float, floor: float) -> float:
    """Combine the existing extent_cap_* upper bound with the new semantic
    keep-out floor: max(floor, min(cap, bbox_half)). The cap still bounds
    an oversized/unreliable bbox (min(cap, bbox_half)); the floor then
    lower-bounds THAT -- so an undersized or point-like bbox still gets at
    least `floor`, and if a category's floor is itself bigger than its cap
    (the forklift floor 0.80 m vs. the wheeled cap 0.50 m, by design), the
    floor wins outright regardless of bbox size, since max(floor, x) can
    never fall below floor. Pure arithmetic -- see test_risk_stack.py."""
    return max(floor, min(cap, bbox_half))


# How many sigma out to actually rasterize each Gaussian (ported from
# user-a/sandbox 2026-09-11; was 3.0). 3.0 clipped the tail hard: for a
# saturated track (value >> 1, e.g. consequence*CPA*100 ~ 100-250) the
# Gaussian at 3 sigma is still value*exp(-4.5) ~ 1-3, which rounds to a
# painted cost, so the whole +/-3 sigma bounding box showed up as a
# hard-edged SQUARE of >=1 cost with a graded core inside. 4.5 sigma pushes
# the edge to value*exp(-10.1) ~ 4e-5*value, i.e. < 1 even for a fully
# saturated splat, so the boundary fades to 0 and reads as a round halo.
# ~2.25x the cells per splat -- still cheap at this grid size.
SPLAT_SIGMA_CUTOFF = 4.5


def _splat_into(grid: np.ndarray, mx: float, my: float, var_x: float, var_y: float,
                value: float, res: float, ox: float, oy: float,
                rows: int, cols: int) -> None:
    """Max-combine an axis-aligned Gaussian (peak=value) into `grid`, in
    place. Pure function -- no self, no rclpy -- shared by every caller
    that needs to paint one blob into one 2D array: the collapsed grid and
    each layer of the STAGE 5 stack (see paint_track), so the footprint/
    bounds math lives in exactly one place."""
    if value <= 0.001:
        return
    sx, sy = math.sqrt(var_x), math.sqrt(var_y)
    k = SPLAT_SIGMA_CUTOFF
    c0 = int((mx - k * sx - ox) / res)
    c1 = int((mx + k * sx - ox) / res)
    r0 = int((my - k * sy - oy) / res)
    r1 = int((my + k * sy - oy) / res)
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(cols - 1, c1), min(rows - 1, r1)
    if c1 < c0 or r1 < r0:
        return
    cs = np.arange(c0, c1 + 1)
    rs = np.arange(r0, r1 + 1)
    wx = ox + (cs + 0.5) * res
    wy = oy + (rs + 0.5) * res
    dx2 = ((wx - mx) ** 2) / var_x
    dy2 = ((wy - my) ** 2) / var_y
    g = value * np.exp(-0.5 * (dy2[:, None] + dx2[None, :]))
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    np.maximum(sub, g, out=sub)


def paint_track(
    stat_x: float, stat_y: float, move_x: float, move_y: float,
    vx: float, vy: float, pmot: float,
    Pxx: float, Pyy: float, Pvx: float, Pvy: float,
    consequence: float, factor: float, relbonus: float,
    res: float, ox: float, oy: float, rows: int, cols: int,
    N: int, pred_dt: float, gamma: float, vel_infl: float,
    grid: Optional[np.ndarray] = None, stack: Optional[np.ndarray] = None,
    want_stack: bool = True,
    flow_array: Optional[np.ndarray] = None, flow_blend_weight: float = 0.0,
    stack_consequence: Optional[float] = None,
    moving_allowed: bool = True,
    sigma_v_max_mps: float = 0.5,
    max_sigma_m: float = 0.6,
    factor_stat: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[int, float, float, float, float, float]]]:
    """Paint one track's two hypotheses into the collapsed, gamma-discounted,
    CPA-amplified `grid` (STAGE 3/4 -- what /risk_costmap_predictive
    publishes for Nav2) and, when `want_stack`, the undiscounted, CPA-free
    STAGE 5 time-layered `stack` -- a (N+1, rows, cols) array where layer 0
    is "now" and layer k is k*pred_dt seconds ahead (see module docstring
    and panoptex_msgs/RiskStack.msg).

    Pure numpy: no self, no rclpy. `grid`/`stack` may be passed in already
    painted by earlier tracks (this max-combines into them, in place, and
    returns the same objects) -- that is how PredictiveRiskCostmapNode
    accumulates multiple detections per tick; pass None to have this
    function allocate fresh zero arrays instead (what the STAGE 5 unit
    tests do, since the Node itself can't be constructed without
    rclpy.init -- it subscribes/publishes and starts a tf listener + timer
    in __init__). Pass want_stack=False to skip STAGE 5 entirely (returned
    stack is then None, whatever was passed in) -- for when publish_stack
    is off and the extra array isn't worth allocating.

    `stat_x/stat_y` is where the STATIONARY hypothesis is splatted -- the
    observed position, never extrapolated: an object believed stationary
    hasn't moved just because the message about it is old. `move_x/move_y`
    is where the MOVING hypothesis's Euler rollout STARTS -- the caller
    extrapolates this to "now" for a stale track (see extrapolate_to_now
    in the module docstring); (vx, vy) is still the track's raw velocity,
    unaffected by that extrapolation.

    `factor` is the STAGE 4b CPA/TTC multiplier against the track's raw
    (vx, vy) -- historically applied to BOTH hypotheses. `factor_stat`
    (two-hypothesis CPA, see module docstring's STAGE 4 section; keyword-
    only, default None) overrides it for the STATIONARY hypothesis alone,
    with `factor` then feeding only the moving one -- None (the default,
    every pre-existing caller) makes the stationary hypothesis reuse
    `factor` too, exactly the historical single-factor behaviour. Kept as
    an additive keyword rather than widening the required positional
    signature so every existing positional call (test_risk_stack.py,
    test_planner_grid.py) keeps working unmodified. The two hypotheses'
    severities (C_stat, C_mov below) are fused with their own factor
    BEFORE the (1 - pmot)/pmot split -- see combine_severity's docstring
    and encounter_geometry.py for why splitting late instead would force
    both hypotheses to share one encounter geometry.

    `moving_allowed` (2026-09-10 "SRM fills half the map" fix, default
    True so every pre-existing caller/test is unaffected): False makes
    this function treat pmot as 0 for EVERY weight below (both hypotheses,
    grid AND stack) regardless of what pmot itself says -- the entire
    mixture mass falls on the stationary blob. The caller decides False
    for a track that is too young (is_track_mature) or too fast for its
    category (speed_cap_for_category) to trust its motion belief; paint_track
    itself has no opinion on WHY, same division of labour as stat_x/move_x.

    `sigma_v_max_mps`/`max_sigma_m` (same fix, defaults 0.5 m/s and 0.6 m
    matching the node's own declared-parameter defaults): Pvx/Pvy is the
    Kalman filter's OWN PRIOR at track init (1.0 (m/s)^2), not evidence, so
    it is clamped to sigma_v_max_mps^2 before it can feed the rollout's
    t^2 growth term; var_x/var_y is additionally clamped to max_sigma_m^2
    in EVERY layer this function paints (both hypotheses, grid and stack
    alike -- the stationary hypothesis shares Pxx/Pyy with the moving
    one's t=0 base term, so it is capped too). min_sigma_m -- the
    pre-existing floor baked into Pxx/Pyy by the caller (_predict_and_
    splat's max(cov, min_var)) -- is untouched; this only adds the
    matching ceiling. Clamping variance changes a splat's SPREAD, never
    its PEAK height (a Gaussian's value at its own mean is `value`
    regardless of var_x/var_y -- see _splat_into), so this cannot change
    what an existing caller reads at a track's exact position, only how
    far the blob reaches beyond it.

    Returns (grid, stack, rollout): `rollout` is the moving-hypothesis
    trajectory as a list of (step, mx, my, var_x, var_y, w_move) for every
    step that was actually splatted into `grid` (i.e. STAGE 3/4's
    early-cutoff once gamma^step decays the weight below threshold), so a
    caller building markers doesn't have to recompute the Euler rollout.
    """
    if grid is None:
        grid = np.zeros((rows, cols), dtype=np.float32)
    if want_stack and stack is None:
        stack = np.zeros((N + 1, rows, cols), dtype=np.float32)
    if not want_stack:
        stack = None

    # moving_allowed=False -> the caller has decided this track's motion
    # belief isn't trustworthy yet (too young) or isn't plausible (too
    # fast for its category) -- zero pmot's effect on every weight below,
    # both hypotheses, grid and stack alike, rather than skip painting
    # outright: the (1 - eff_pmot) stationary term then carries the FULL
    # weight, so the track still gets painted, just stationary-only.
    eff_pmot = pmot if moving_allowed else 0.0

    # Clamp the velocity-covariance PRIOR before it can feed the t^2
    # rollout term (root cause 1 of the 2026-09-10 finding -- see this
    # function's own docstring above).
    pv_cap = sigma_v_max_mps * sigma_v_max_mps
    Pvx_c = min(Pvx, pv_cap)
    Pvy_c = min(Pvy, pv_cap)
    max_var = max_sigma_m * max_sigma_m
    # Matching ceiling on the BASE covariance too, so the stationary
    # hypothesis (which never enters the t^2 loop below) is bounded the
    # same way as every rolled-out step.
    Pxx_b = min(Pxx, max_var)
    Pyy_b = min(Pyy, max_var)

    # Two-hypothesis CPA: each hypothesis is fused with its OWN encounter
    # factor before the (1 - pmot)/pmot split below, not a single shared C
    # (see this function's docstring and module docstring STAGE 4).
    # factor_stat defaulting to `factor` (None -> reuse) is what makes
    # every pre-existing caller -- which only ever passed one `factor` --
    # byte-identical to before: C_stat == C_mov == the old single `C`.
    resolved_factor_stat = factor if factor_stat is None else factor_stat
    C_stat = combine_severity(consequence, resolved_factor_stat, relbonus)
    C_mov = combine_severity(consequence, factor, relbonus)
    # STAGE 5: same consequence + relation bonus, WITHOUT the CPA/TTC
    # multiplier (factor pinned to 1.0) -- the stack is ego-independent by
    # design. combine_severity(consequence, 1.0, relbonus) keeps the
    # "relation_bonus is additive, outside the multiplier" ordering
    # convention in the one place (encounter_geometry.py) instead of
    # re-deriving "consequence + relbonus" here.
    # stack_consequence lets the caller hand the stack a DIFFERENT severity
    # than the collapsed grid's (see _stack_consequence: class-level, not
    # confidence-scaled). None keeps the two identical.
    C_stack = combine_severity(
        consequence if stack_consequence is None else stack_consequence, 1.0, relbonus)

    # --- stationary hypothesis: one blob, into the grid AND every stack layer ---
    # The grid weight carries the CPA factor (via C); the stack weight uses
    # C_stack so the WHOLE stack, not just its moving hypothesis, is
    # ego-independent -- a parked cart must read the same in every layer no
    # matter how fast the robot is driving at it (the controller critic
    # supplies the ego motion itself).
    w_stat = (1.0 - eff_pmot) * C_stat
    _splat_into(grid, stat_x, stat_y, Pxx_b, Pyy_b, w_stat, res, ox, oy, rows, cols)
    if stack is not None:
        w_stat_stack = (1.0 - eff_pmot) * C_stack
        if w_stat_stack > 0.001:
            for k in range(N + 1):
                _splat_into(stack[k], stat_x, stat_y, Pxx_b, Pyy_b, w_stat_stack,
                            res, ox, oy, rows, cols)

    # --- moving hypothesis, rolled out over the horizon ---
    # w_move (grid) carries gamma^step discount and the CPA factor via C;
    # w_move_stack does neither, so unlike w_move it is the SAME every
    # step -- compute it once instead of inside the loop.
    w_move_stack = eff_pmot * C_stack
    paint_stack_moving = stack is not None and w_move_stack > 0.001

    # Layer 0 is "now": a mover occupies its CURRENT position at t=0 whatever
    # its p_motion, so the moving hypothesis contributes its step-0 position
    # there too (the stationary hypothesis already covers the 1-pmot share).
    # Without this a pmot=1 track left layer 0 empty (WP-A smoke test,
    # 2026-09-10), which is wrong for any consumer that looks at "now".
    if paint_stack_moving:
        _splat_into(stack[0], move_x, move_y, Pxx_b, Pyy_b, w_move_stack,
                    res, ox, oy, rows, cols)

    rollout: List[Tuple[int, float, float, float, float, float]] = []
    cur_x, cur_y = move_x, move_y
    for step in range(N):
        t = (step + 1) * pred_dt
        discount = gamma ** step
        w_move = eff_pmot * discount * C_mov

        flow_s, flow_fx, flow_fy = sample_flow_grid(
            flow_array, cur_x, cur_y, res, ox, oy, rows, cols)
        eff_vx, eff_vy = blend_velocity(
            vx, vy, flow_s, flow_fx, flow_fy, step, N, flow_blend_weight)
        cur_x += eff_vx * pred_dt
        cur_y += eff_vy * pred_dt
        mx, my = cur_x, cur_y

        var_x = min(Pxx_b + (t * t) * Pvx_c + vel_infl * t, max_var)
        var_y = min(Pyy_b + (t * t) * Pvy_c + vel_infl * t, max_var)

        if w_move > 0.001:
            _splat_into(grid, mx, my, var_x, var_y, w_move, res, ox, oy, rows, cols)
            rollout.append((step, mx, my, var_x, var_y, w_move))

        if paint_stack_moving:
            _splat_into(stack[step + 1], mx, my, var_x, var_y, w_move_stack,
                        res, ox, oy, rows, cols)

        if w_move <= 0.001 and not paint_stack_moving:
            # gamma^step only shrinks further from here -- once the grid's
            # weight has underflowed and the stack doesn't need this
            # track's moving hypothesis either, nothing later in the
            # horizon can matter.
            break

    return grid, stack, rollout


def paint_track_planner(
    stat_x: float, stat_y: float, move_x: float, move_y: float,
    vx: float, vy: float, pmot: float,
    Pxx: float, Pyy: float, Pvx: float, Pvy: float,
    consequence: float, relbonus: float,
    res: float, ox: float, oy: float, rows: int, cols: int,
    N: int, pred_dt: float, gamma: float, vel_infl: float,
    grid: Optional[np.ndarray] = None,
    flow_array: Optional[np.ndarray] = None, flow_blend_weight: float = 0.0,
    moving_allowed: bool = True,
    sigma_v_max_mps: float = 0.5,
    max_sigma_m: float = 0.6,
) -> np.ndarray:
    """STAGE 6 -- paint one track's two hypotheses into the PLANNER grid
    (see module docstring). Structurally the same mixture as paint_track()'s
    collapsed-grid half -- stationary hypothesis weight (1 - pmot) * C,
    moving hypothesis weighted pmot * gamma**step * C and rolled out with
    the SAME sample_flow_grid()/blend_velocity() helpers paint_track() uses
    -- but every one of N/gamma/flow_blend_weight is independently supplied
    by the caller (planner_horizon_steps/planner_gamma/
    planner_flow_blend_weight, not the stack/grid's own horizon_steps/
    gamma/flow_blend_weight), and there is no `factor` argument at all: C
    is combine_severity(consequence, 1.0, relbonus), the CPA/TTC multiplier
    pinned to 1.0 exactly like the stack's C_stack -- the planner grid is
    EGO-INDEPENDENT by the same reasoning (see RiskStack contract in the
    module docstring and README.md Sec.8).

    `moving_allowed`/`sigma_v_max_mps`/`max_sigma_m` (WP2, 2026-09-10
    evening) are applied EXACTLY as paint_track() applies them -- same
    eff_pmot substitution, same Pvx/Pvy clamp to sigma_v_max_mps^2 before
    it feeds the t^2 rollout term, same Pxx/Pyy base clamp and per-step
    var clamp to max_sigma_m^2 -- because this grid shares the SAME
    covariance-blowup failure mode paint_track's own docstring documents
    (a freshly seeded Kalman P_v is the filter's own prior, not evidence);
    the module docstring's claim that this function's N/gamma/flow are
    independent of paint_track's is unaffected by this -- only the shared
    clamps and the moving/stationary gate are common between the two, not
    the horizon/discount/flow shaping. Defaults preserve every pre-existing
    caller/test (moving_allowed=True is a no-op eff_pmot=pmot, and the
    clamp values match the node's own sigma_v_max_mps/max_sigma_m
    defaults).

    `grid` may already be painted by earlier tracks (max-combined into, in
    place, same convention as paint_track); pass None to allocate a fresh
    zero array. Pure numpy -- no self, no rclpy. See
    test_planner_grid.py."""
    if grid is None:
        grid = np.zeros((rows, cols), dtype=np.float32)

    # Same moving_allowed gate as paint_track() -- see that function's
    # docstring for the "too young"/"too fast"/"too slow" reasoning
    # (resolve_moving_allowed) this substitutes for.
    eff_pmot = pmot if moving_allowed else 0.0

    # Same two clamps as paint_track(): the velocity-covariance PRIOR
    # before it feeds the t^2 term, and a matching ceiling on the BASE
    # covariance so the stationary hypothesis is bounded the same way.
    pv_cap = sigma_v_max_mps * sigma_v_max_mps
    Pvx_c = min(Pvx, pv_cap)
    Pvy_c = min(Pvy, pv_cap)
    max_var = max_sigma_m * max_sigma_m
    Pxx_b = min(Pxx, max_var)
    Pyy_b = min(Pyy, max_var)

    C = combine_severity(consequence, 1.0, relbonus)

    w_stat = (1.0 - eff_pmot) * C
    _splat_into(grid, stat_x, stat_y, Pxx_b, Pyy_b, w_stat, res, ox, oy, rows, cols)

    cur_x, cur_y = move_x, move_y
    for step in range(N):
        t = (step + 1) * pred_dt
        discount = gamma ** step
        w_move = eff_pmot * discount * C

        flow_s, flow_fx, flow_fy = sample_flow_grid(
            flow_array, cur_x, cur_y, res, ox, oy, rows, cols)
        eff_vx, eff_vy = blend_velocity(
            vx, vy, flow_s, flow_fx, flow_fy, step, N, flow_blend_weight)
        cur_x += eff_vx * pred_dt
        cur_y += eff_vy * pred_dt

        var_x = min(Pxx_b + (t * t) * Pvx_c + vel_infl * t, max_var)
        var_y = min(Pyy_b + (t * t) * Pvy_c + vel_infl * t, max_var)

        if w_move <= 0.001:
            # gamma**step only shrinks further from here (planner_gamma is
            # normally close to 1.0, so this rarely trips before N runs out).
            break
        _splat_into(grid, cur_x, cur_y, var_x, var_y, w_move, res, ox, oy, rows, cols)

    return grid


def srm_marker_mask(srm: np.ndarray, k: int, srm_min: float, delta: float,
                    stride: int) -> np.ndarray:
    """WP3 pure extraction of one SRM comet layer's per-cell gate (see
    _srm_markers() and the module docstring's "Comet-vs-parked-disc fix"
    section) -- no self, no rclpy, no marker construction, just the
    boolean mask.

    - k == 0 ("now"): srm[0] >= srm_min, unconditionally -- layer 0 is
      never stride-gated (it's always drawn when publish_srm is on).
    - k >= 1: (srm[k] - srm[0] >= delta) & (srm[k] >= srm_min) -- a cell
      needs to clear the SAME absolute floor srm_min layer 0 uses AND read
      meaningfully above "now" (delta) before it counts as a "predicted
      motion" point at all; a parked object's srm[k] equals srm[0] in
      every layer (the stationary hypothesis is splatted into every stack
      layer, see STAGE 5), so delta alone would already exclude it, but
      the srm_min floor additionally keeps a near-zero srm[k] with a
      near-zero srm[0] (delta trivially satisfied by two tiny numbers)
      from drawing a point nobody would call "risk."
    - stride (srm_marker_layer_stride, default 3, 2026-09-10 evening): for
      k >= 1, only layers with k % stride == 0 are drawn at all -- returns
      an all-False mask (same shape as srm[0]) otherwise. 21 stacked
      translucent layers (the pre-stride behaviour) made even a
      barely-over-threshold cell's tail look solid where several
      low-alpha layers overlapped; skipping 2 of every 3 layers thins the
      stack to 7 without changing which cells qualify in the layers that
      DO draw. k == 0 is exempt from the stride (always evaluated) since
      it's the one layer every consumer treats as "now," never a tail.

    Pure function, no ROS -- see test_srm_markers.py / test_srm.py."""
    if k != 0 and (stride <= 0 or k % stride != 0):
        return np.zeros(srm[0].shape, dtype=bool)
    if k == 0:
        return srm[0] >= srm_min
    return (srm[k] - srm[0] >= delta) & (srm[k] >= srm_min)


class PredictiveRiskCostmapNode(Node):
    def __init__(self):
        super().__init__("predictive_risk_costmap_node")

        self.declare_parameter("input_topic", "/risk_perception/world_objects")
        self.declare_parameter("costmap_topic", "/risk_costmap_predictive")
        self.declare_parameter("marker_topic", "/risk_perception/prediction_markers")
        self.declare_parameter("map_frame", "map")

        # STAGE 5 -- time-layered risk stack (see module docstring)
        self.declare_parameter("stack_topic", "/risk_stack")
        self.declare_parameter("publish_stack", True)
        self.declare_parameter("extrapolate_to_now", True)
        # tracks older than this are still painted (clamped, not dropped);
        # exceeding it only trips the throttled staleness warning below.
        self.declare_parameter("max_track_age_sec", 3.0)
        # STAGE 5 magnitude: "class" paints CLASS_BASE_RISK[label] (0.75 for a
        # mobile robot) for any track above stack_min_track_score; "scaled"
        # multiplies by the track's decayed confidence like the collapsed grid.
        # With ~1 Hz sightings the confidence idles at 0.2-0.3, so "scaled"
        # painted a Carter at 0.14-0.17 -- below any usable lethal threshold
        # (2026-09-08 sim run). Where an object IS is the Kalman covariance's
        # job, not the detector score's.
        self.declare_parameter("stack_consequence_mode", "class")
        self.declare_parameter("stack_min_track_score", 0.1)
        # Widen every splat by the object's own half-extent (from the track's
        # bbox): a 0.75 m Nova Carter is not a point. Capped at 1.0 m.
        self.declare_parameter("use_object_extent", True)
        # Half-extent per category (m). The tracker's bbox size comes from
        # floor-projected masks and is unreliable for large/flat things, so
        # the category prior bounds it: extent = min(bbox/2, cap[category]).
        self.declare_parameter("extent_cap_person_m", 0.30)
        self.declare_parameter("extent_cap_robot_m", 0.45)
        self.declare_parameter("extent_cap_wheeled_m", 0.50)
        self.declare_parameter("extent_cap_other_m", 0.25)
        # Which categories the STAGE 5 stack paints at all. Static furniture
        # and unknown-label junk ("ground", "table" mislabels of floor
        # patches) are the lidar layers' business; painting them at class
        # magnitude with an extent made the whole aisle lethal and froze the
        # robot (2026-09-08 panoptex_3: 3 missed waypoints, no collision only
        # because it never moved).
        self.declare_parameter("stack_categories", ["person", "robot", "wheeled"])
        # Unknown-but-moving exception to the category gate above -- see
        # module docstring's 'unknown-but-moving' section and
        # resolve_stack_consequence(). A category=="unknown" track (e.g. a
        # lidar-cluster detector's unmatched detection) that is genuinely
        # MOVING (pmot >= unknown_moving_pmot_min) AND mature/plausible
        # enough (hits/age/speed gates below) still paints the stack, at a
        # flat unknown_moving_risk magnitude since it has no class to
        # price by CLASS_BASE_RISK; a STATIC unknown stays excluded (the
        # reactive obstacle layers' job). Default flipped True -> False and
        # unknown_moving_pmot_min raised 0.5 -> 0.8 (2026-09-10 "SRM fills
        # half the map with real perception" finding): a lidar-only track
        # has no class to sanity-check its motion belief against, so this
        # is now an opt-in exception with a much stricter bar, not an
        # always-on one.
        self.declare_parameter("stack_paint_unknown_moving", False)
        self.declare_parameter("unknown_moving_risk", 0.60)
        self.declare_parameter("unknown_moving_pmot_min", 0.8)
        # Same 2026-09-10 finding -- additional gates required (alongside
        # pmot) whenever stack_paint_unknown_moving is turned on: enough
        # measurement hits, enough age, and a plausible speed for an
        # UNKNOWN-category track (max_speed_unknown_mps below), mirroring
        # is_track_mature()'s general young-track gate but stricter, since
        # this path has no class at all to price consequence by.
        self.declare_parameter("unknown_moving_min_hits", 8)
        self.declare_parameter("unknown_moving_min_age_sec", 2.0)

        # WP-A -- class-agnostic (motion-first) consequence, 2026-09-11 --
        # see module docstring's "WP-A CLASS-AGNOSTIC CONSEQUENCE" section
        # and resolve_agnostic_consequence()/resolve_stack_consequence().
        # ON by default: replaces stack_categories/stack_paint_unknown_moving
        # above as the primary gate for the stack, the collapsed grid
        # (/risk_costmap_predictive) AND the planner grid
        # (/risk_costmap_planner) alike -- every confirmed, sufficiently
        # confident track paints all three regardless of its class label.
        # False restores the pre-2026-09-11 category-gated behaviour above
        # (a regression/ablation knob, not the shipped default).
        self.declare_parameter("stack_motion_first", True)
        # The flat consequence every track paints under stack_motion_first.
        # 0.75 sits between the old "wheeled" (0.65) and "person" (0.90)
        # class values: high enough that a parked mover is a real keep-out,
        # not so high a chair reads as lethal as a forklift.
        self.declare_parameter("stack_consequence_agnostic", 0.75)
        # OFF by default: the class-keyed semantic prior may only push the
        # agnostic value UP (max(agnostic, class_value)), never veto or
        # lower it -- see resolve_agnostic_consequence()'s docstring.
        self.declare_parameter("semantic_modifier_enabled", False)
        # Additional gate on the MOVING hypothesis specifically (grid,
        # stack AND planner grid alike -- see moving_allowed in
        # _predict_and_splat): a track that is mature/plausible
        # (is_track_mature, speed cap, min_speed_for_motion_mps) but whose
        # own pmot still reads "probably stationary" paints only its
        # stationary blob, at the agnostic consequence -- motion-first
        # prices EVERY category, it does not assume every track is moving.
        #
        # Raised 0.5 -> 0.8 (2026-09-12, real-camera sim run): object_
        # tracker_node's p_motion is a snap-then-decay signal (snaps to 1.0
        # on evidence of motion, decays at p_motion_decay=3.0/s otherwise,
        # half-life ~0.23s) -- a genuinely continuous mover keeps
        # re-triggering "moving" every tick and sits pinned near 1.0
        # regardless of this gate, but a STATIC object whose displacement
        # evidence fires ONE spurious tick from inter-camera/lidar
        # misalignment only clears a low gate for as long as that decay
        # curve stays above it: exp(-3*t) crosses 0.5 at t=0.23s but 0.8 at
        # only t=0.074s, roughly a 3x shorter window in which that single
        # phantom tick can paint a moving-hypothesis blob into
        # /risk_costmap_predictive, /risk_stack (and therefore
        # /risk_stack_srm), or /risk_costmap_planner. Matches the value
        # already chosen for the analogous unknown_moving_pmot_min gate
        # above. If real slow/intermittent movers start getting painted
        # stationary-only under this, retune motion_disp_k_sigma/
        # motion_min_displacement_m in object_tracker instead -- those
        # fix the phantom pmot at its SOURCE rather than just shortening
        # how long a downstream consumer trusts it.
        self.declare_parameter("stack_moving_pmot_min", 0.8)
        # A SLOW, class-seeded companion to stack_moving_pmot_min above:
        # pmot is instant evidence ("is it moving THIS frame") and can spike
        # from one frame of sensor misalignment, which is exactly why
        # stack_moving_pmot_min was raised 0.5 -> 0.8. pmov (p_movable) is
        # object_tracker's slow-changing, category-seeded belief in whether
        # the object could EVER move (prior_furniture 0.1, prior_person 0.9,
        # ...) -- a single noisy frame barely moves it. Default 0.0 = off
        # (pmov is clamped to [0, 1] by object_tracker, so `pmov >= 0.0` is
        # always true): purely additive, no behavior change until raised.
        # Raise this instead of stack_moving_pmot_min if a specific class
        # (e.g. parked shelving) keeps false-triggering the moving
        # hypothesis -- it targets the class-level prior rather than numbing
        # every category's instant-motion sensitivity at once.
        self.declare_parameter("stack_moving_pmov_min", 0.0)
        # Relation-prior ablation knob (2026-09-13). The geometric relation
        # evidence is produced UNCONDITIONALLY upstream -- rgbd_projector and
        # global_cam_projector both call mask_relation.tag_operator_relations
        # on every frame, and object_tracker's Track.update_relation runs on
        # every tick -- so before this parameter existed there was no way to
        # run a condition WITHOUT the Relation prior contributing to risk,
        # and the paper's "relation disabled" arm was not actually
        # reproducible. False zeroes `relbonus` HERE, at the single point
        # where the prior enters severity (combine_severity's additive term,
        # outside the encounter multiplier), which leaves the evidence
        # pipeline and the research CSV's relbonus column intact for
        # logging/validation while removing the prior's effect on every
        # published grid and on the stack. True (default) is the shipped
        # behaviour.
        self.declare_parameter("use_relation_bonus", True)
        # speed_cap_for_category's fallback for any category with no entry
        # in the per-category cap table below (furniture, unknown, ...),
        # now that motion-first paints every category. Deliberately NOT
        # max_speed_unknown_mps (1.2) -- that stays reserved for the
        # unknown-moving exception path's own, stricter speed check (no
        # class at all to sanity-check velocity against); a genuinely
        # moving "table" should be capped like any other wheeled-ish
        # mover.
        self.declare_parameter("max_speed_default_mps", 2.0)

        # Semantic keep-out floor -- a LOWER bound on the splat half-extent,
        # applied alongside the extent_cap_* upper bounds above (see
        # resolve_extent_half() / select_min_extent() and the module
        # docstring's 'semantic keep-out floor' section): a person or
        # forklift needs a minimum berth even when its detected bbox
        # collapses to a point. "forklift" is matched on the raw LABEL
        # (its category is "wheeled", same as any cart) so it gets its own,
        # bigger floor. Applies to the stack AND both collapsed grids
        # (/risk_costmap_predictive and /risk_costmap_planner) -- the
        # shared Pxx/Pyy covariance all three paint from.
        self.declare_parameter("min_extent_person_m", 0.50)
        self.declare_parameter("min_extent_robot_m", 0.45)
        self.declare_parameter("min_extent_wheeled_m", 0.45)
        self.declare_parameter("min_extent_forklift_m", 0.80)
        self.declare_parameter("min_extent_other_m", 0.25)

        # grid geometry (must cover / align with the known floor)
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width_m", 12.0)
        self.declare_parameter("height_m", 12.0)
        self.declare_parameter("origin_x", -6.0)
        self.declare_parameter("origin_y", -6.0)

        # STAGE 2 prediction
        self.declare_parameter("horizon_steps", 20)      # N future steps
        self.declare_parameter("pred_dt", 0.3)           # seconds per step
        self.declare_parameter("gamma", 0.9)             # time discount per step
        self.declare_parameter("vel_inflation", 0.05)    # cov growth per second (m^2)
        self.declare_parameter("min_sigma_m", 0.15)      # floor on spatial spread
        # Ceiling on spatial spread, and a separate ceiling on the velocity
        # covariance that feeds it (2026-09-10 "SRM fills half the map"
        # finding, root cause 1 -- see paint_track's own docstring):
        # object_tracker_node seeds a freshly spawned track's Pvx/Pvy at
        # 1.0 (m/s)^2, the KALMAN FILTER'S OWN PRIOR, not evidence -- at
        # t=6s (this stack's far horizon) that prior alone inflates sigma
        # to ~6 m. sigma_v_max_mps clamps the prior before it feeds the
        # rollout's t^2 term; max_sigma_m is a second, independent
        # backstop on the resulting var_x/var_y in every layer paint_track
        # paints (grid AND stack, both hypotheses). min_sigma_m above is
        # untouched -- this only adds the matching upper bound.
        self.declare_parameter("sigma_v_max_mps", 0.5)
        self.declare_parameter("max_sigma_m", 0.6)
        self.declare_parameter("publish_rate", 5.0)
        # Behavioral prior ablation switch -- object_tracker keeps learning
        # pmot regardless (this only gates its CONSUMPTION here), so toggling
        # this doesn't disturb the other priors' inputs. False collapses the
        # mixture to 100% stationary: w_stat = C, and the moving-hypothesis
        # loop's w_move is 0 on its first iteration, so it never splats.
        # Independent of enable_relative_motion below -- CPA/TTC reads vx/vy
        # directly, not pmot, so it still runs on the raw Kalman velocity.
        self.declare_parameter("use_motion_mixture", True)
        # Young-track gate (same 2026-09-10 finding, root cause 3 -- see
        # is_track_mature()'s own docstring): a track below EITHER
        # threshold has its pmot treated as 0 for painting (moving_allowed
        # =False on paint_track), both grid and stack -- its motion belief
        # (and, via Pvx/Pvy, its rollout spread) is the tracker's own
        # freshly-seeded prior, not yet evidence. Read from the tracker's
        # class_id "hits="/"age=" fields (object_tracker_node's WP-B
        # addition); a class_id with NEITHER key (gt_tracks_node's oracle
        # tracks, or any older publisher) reads as mature unconditionally.
        self.declare_parameter("min_track_age_for_motion_sec", 1.0)
        self.declare_parameter("min_hits_for_motion", 5)
        # Per-category |v| plausibility cap (same finding, root cause 2):
        # an association jump mid-track hands it an implausible velocity
        # that then gets painted as a mover at that speed. A track above
        # its category's cap is also painted stationary-only
        # (moving_allowed=False), logged once per track id -- see
        # _resolve_moving_allowed. Keyed by risk_visualization.
        # label_category(); max_speed_unknown_mps is both the fallback for
        # any category not listed here (furniture, ...) and the cap the
        # unknown-but-moving exception above uses.
        self.declare_parameter("max_speed_person_mps", 2.0)
        self.declare_parameter("max_speed_robot_mps", 1.5)
        self.declare_parameter("max_speed_wheeled_mps", 2.0)
        self.declare_parameter("max_speed_unknown_mps", 1.2)
        # 2026-09-10 evening: a track can pass is_track_mature and its
        # category's speed cap and still be reading pure camera jitter as
        # "motion" -- a few cm/s, nowhere near the cap. gt_tracks_node's
        # oracle classify_motion already only sets pmot=1 at
        # speed >= 0.15 m/s (its own motion_speed_mps), so gating painting
        # at the SAME floor here (resolve_moving_allowed's "below_min_speed"
        # reason) makes real and oracle perception agree on "how slow is
        # stationary" and does not regress oracle -- an oracle track never
        # carries pmot=1 below that speed in the first place. See
        # resolve_moving_allowed()/test_risk_stack.py.
        self.declare_parameter("min_speed_for_motion_mps", 0.15)

        # STAGE 4a -- consequence weighting by class
        self.declare_parameter("use_class_consequence", True)
        self.declare_parameter("consequence_default", 0.40)

        # STAGE 4b -- relative-motion (CPA/TTC) amplification
        self.declare_parameter("enable_relative_motion", True)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("cpa_gain", 1.5)          # max extra multiplier
        self.declare_parameter("cpa_scale_m", 0.6)       # d0: miss distance that still scares us
        self.declare_parameter("ttc_scale_s", 3.0)       # t0: how far ahead we care
        self.declare_parameter("min_rel_speed", 0.05)    # m/s below which geometry is meaningless
        self.declare_parameter("odom_timeout_sec", 1.0)  # stale odom -> assume robot stopped
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("publish_cpa_markers", True)
        # Two-hypothesis CPA (ablation knob, default False = historical
        # single-factor behaviour): see module docstring's STAGE 4 section
        # and paint_track()'s docstring for what this changes.
        self.declare_parameter("two_hypothesis_cpa", False)

        # optional persistent spatial prior (spatial_prior_node) -- floor use
        self.declare_parameter("spatial_prior_topic", "/risk_perception/spatial_prior")
        self.declare_parameter("spatial_prior_weight", 0.0)

        # WP-C -- class-agnostic recent-activity floor, 2026-09-11
        # (spatial_prior_node's /risk_perception/activity_prior: a 5-min
        # half-life EMA of ANY mover, no per-category channels -- see that
        # node's docstring). Unlike spatial_prior_weight above (the
        # multi-hour LEARNED lane prior, deliberately frozen during a study
        # run -- see the repo's "spatial prior must be frozen" convention),
        # this channel keeps decaying even while frozen: it is run-time
        # state, not a learned prior, and is never saved. Seeds every grid
        # this node publishes by MAX, never addition -- see _tick(): a
        # track's own painted core on a busy cell keeps its own (usually
        # higher) value, this only raises what an otherwise-EMPTY cell
        # reads, the paper's "latent risk at busy doorways" even where
        # nothing is currently detected. No double counting with
        # lane_layer -- nav2's global costmap already max-merges every
        # risk layer it carries, lane_layer (spatial_prior_weight's own
        # consumer) included, so a second max-seed here composes with it
        # the same way two tracks compose with each other, never adds on
        # top of it.
        self.declare_parameter("activity_topic", "/risk_perception/activity_prior")
        # The STACK's own weight. Bounded at <= srm_levels[0] + 0.05 by
        # design (0.25 default against srm_levels[0]'s own 0.2 default):
        # the activity floor should only ever push an empty cell up to
        # about the SRM's lowest graded level, never far into "this reads
        # like a real detection" territory.
        self.declare_parameter("activity_weight", 0.0)
        # The planner grid's own weight -- independent of activity_weight
        # above, same reasoning as planner_flow_blend_weight/
        # flow_blend_weight being independently tunable (see STAGE 6 in
        # the module docstring).
        self.declare_parameter("activity_weight_planner", 0.0)
        # Deliberately lower than the two above: /risk_costmap_predictive
        # already carries every track's own CPA-amplified peak AND (when
        # spatial_prior_weight is on) the learned lane floor; a smaller
        # grid weight avoids stacking a second, redundant floor on top of
        # a much slower-moving one for the ONE grid Nav2's DWB critic reads
        # directly.
        self.declare_parameter("activity_weight_grid", 0.0)

        # optional Spatial-Flow rollout blend -- a DIFFERENT consumption of
        # spatial_prior_node's output than the floor above; see module
        # docstring. Off by default: this changes the shape of the
        # predictive costmap Nav2 plans against, not just an additive
        # floor, so it should not turn on silently.
        self.declare_parameter(
            "flow_categories", ["person", "robot", "wheeled"])
        self.declare_parameter("flow_blend_weight", 0.0)

        # Research logging (ported from user-a/sandbox) -- one row per track
        # per tick showing why STAGE 2b did or did not produce a forward
        # smear: pmot, the raw (vx, vy), the consequence / CPA / relation
        # multipliers, and the resulting w_stat vs w_move plus how many
        # rollout steps actually splatted. Off unless set. Consumed by
        # tools/prior_report.py.
        self.declare_parameter("debug_log_path", "")
        self.declare_parameter("debug_log_dir", "")
        # Category -> channel override -- see parse_flow_category_map's
        # docstring. Default routes robot/wheeled onto spatial_prior_node's
        # merged robot_group channel; a category absent from this map keeps
        # sampling its own /risk_perception/spatial_flow/<category> topic.
        self.declare_parameter(
            "flow_category_map", ["robot:robot_group", "wheeled:robot_group"])

        # STAGE 6 -- planner grid (see module docstring). A second
        # collapsed OccupancyGrid, published alongside the STAGE 3/4 one,
        # for the route/tactical layer: farther-seeing (planner_horizon_
        # steps), slower-fading (planner_gamma close to 1.0), and flow-
        # blended (planner_flow_blend_weight) independently of the stack/
        # collapsed grid's own gamma/flow_blend_weight, which stay pure
        # constant-velocity. EGO-INDEPENDENT like the stack -- no CPA
        # factor parameter exists for it at all.
        self.declare_parameter("publish_planner_grid", True)
        self.declare_parameter("planner_topic", "/risk_costmap_planner")
        self.declare_parameter("planner_horizon_steps", 20)
        self.declare_parameter("planner_gamma", 0.97)
        self.declare_parameter("planner_flow_blend_weight", 0.5)

        # WP-A -- Spatiotemporal Risk Map (see module docstring). A
        # windowed, per-layer transform of the STAGE 5 stack via
        # risk_perception.srm.stack_to_srm (Thomas et al. 2021 eq. 3).
        self.declare_parameter("publish_srm", True)
        self.declare_parameter("srm_topic", "/risk_stack_srm")
        self.declare_parameter("srm_now_topic", "/risk_srm_now")
        self.declare_parameter(
            "srm_marker_topic", "/risk_perception/srm_markers")
        # d0 in risk(i) = max(0, 1 - d(i)/d0) -- Thomas et al.'s own value.
        self.declare_parameter("srm_d0_m", 1.5)
        # Occupancy probability thresholds combined by max() in
        # stack_to_srm -- see srm.py's docstring for why max() approximates
        # the paper's p=3 p-norm across occupied cells.
        self.declare_parameter("srm_levels", [0.2, 0.5, 0.8])
        # Side length of the robot-centred window the SRM is computed over
        # (see module docstring) -- bounds the per-tick EDT cost
        # independently of the full grid's width_m/height_m.
        self.declare_parameter("srm_window_m", 12.0)
        # Marker cutoff: only cells >= this SRM value get a "comet" point.
        self.declare_parameter("srm_marker_min", 0.3)
        # Comet-vs-parked-disc fix (2026-09-10 finding): the stationary
        # hypothesis is splatted into EVERY stack layer (see STAGE 5), so
        # a parked object's SRM layer k is identical to layer 0 -- without
        # this, _srm_markers drew that as a full time-coloured "moving"
        # point in every layer, same as a genuine mover. Layer 0 keeps the
        # srm_marker_min cutoff above (it IS "now"); layers k>=1 additionally
        # require srm[k] - srm[0] >= srm_marker_delta, so only a cell whose
        # PREDICTED motion pushes it meaningfully above "now" gets a
        # time-coloured point at all. Raised 0.05 -> 0.15 (2026-09-10
        # evening): 0.05 alone let two near-zero SRM values (a barely-warm
        # cell trailing a barely-warmer one) clear the gate -- 0.15 is a
        # real absolute floor on "how much above now," not just "not
        # exactly equal to now." See srm_marker_mask()/test_srm_markers.py.
        self.declare_parameter("srm_marker_delta", 0.15)
        # 2026-09-10 evening: only draw comet layers k>=1 every Nth step
        # (k % stride == 0; k==0 "now" is always drawn) -- 21 stacked
        # translucent layers (steps=21 at the default horizon/pred_dt) made
        # even a low-value tail look solid where several low-alpha layers
        # overlapped in RViz. stride=3 thins that to 7 drawn layers without
        # changing which cells qualify within a drawn layer. See
        # srm_marker_mask().
        self.declare_parameter("srm_marker_layer_stride", 3)

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.stack_topic = str(gp("stack_topic").value)
        self.publish_stack = bool(gp("publish_stack").value)
        self.extrapolate_to_now = bool(gp("extrapolate_to_now").value)
        self.max_track_age_sec = float(gp("max_track_age_sec").value)
        self.stack_consequence_mode = str(self.get_parameter("stack_consequence_mode").value).strip().lower()
        if self.stack_consequence_mode not in ("class", "scaled"):
            raise ValueError("stack_consequence_mode must be 'class' or 'scaled'")
        self.stack_min_track_score = float(self.get_parameter("stack_min_track_score").value)
        self.use_object_extent = bool(self.get_parameter("use_object_extent").value)
        self.extent_caps = {
            "person": float(self.get_parameter("extent_cap_person_m").value),
            "robot": float(self.get_parameter("extent_cap_robot_m").value),
            "wheeled": float(self.get_parameter("extent_cap_wheeled_m").value),
        }
        self.extent_cap_other = float(self.get_parameter("extent_cap_other_m").value)
        self.stack_categories = set(
            str(c).strip().lower() for c in self.get_parameter("stack_categories").value)
        self.stack_paint_unknown_moving = bool(gp("stack_paint_unknown_moving").value)
        self.unknown_moving_risk = float(gp("unknown_moving_risk").value)
        self.unknown_moving_pmot_min = float(gp("unknown_moving_pmot_min").value)
        self.unknown_moving_min_hits = float(gp("unknown_moving_min_hits").value)
        self.unknown_moving_min_age_sec = float(gp("unknown_moving_min_age_sec").value)
        # WP-A class-agnostic (motion-first) consequence -- see module
        # docstring and resolve_agnostic_consequence()/
        # resolve_stack_consequence().
        self.stack_motion_first = bool(gp("stack_motion_first").value)
        self.stack_consequence_agnostic = float(gp("stack_consequence_agnostic").value)
        self.semantic_modifier_enabled = bool(gp("semantic_modifier_enabled").value)
        self.stack_moving_pmot_min = float(gp("stack_moving_pmot_min").value)
        self.stack_moving_pmov_min = float(gp("stack_moving_pmov_min").value)
        self.use_relation_bonus = bool(gp("use_relation_bonus").value)
        self.min_extent_person = float(gp("min_extent_person_m").value)
        self.min_extent_robot = float(gp("min_extent_robot_m").value)
        self.min_extent_wheeled = float(gp("min_extent_wheeled_m").value)
        self.min_extent_forklift = float(gp("min_extent_forklift_m").value)
        self.min_extent_other = float(gp("min_extent_other_m").value)
        self.res = float(gp("resolution").value)
        self.ox = float(gp("origin_x").value)
        self.oy = float(gp("origin_y").value)
        self.cols = int(round(float(gp("width_m").value) / self.res))
        self.rows = int(round(float(gp("height_m").value) / self.res))
        self.N = int(gp("horizon_steps").value)
        self.pred_dt = float(gp("pred_dt").value)
        self.gamma = float(gp("gamma").value)
        self.vel_infl = float(gp("vel_inflation").value)
        self.min_var = float(gp("min_sigma_m").value) ** 2
        self.sigma_v_max_mps = float(gp("sigma_v_max_mps").value)
        self.max_sigma_m = float(gp("max_sigma_m").value)
        self.use_class_consequence = bool(gp("use_class_consequence").value)
        self.c_default = float(gp("consequence_default").value)
        self.use_motion_mixture = bool(gp("use_motion_mixture").value)
        self.min_track_age_for_motion_sec = float(gp("min_track_age_for_motion_sec").value)
        self.min_hits_for_motion = float(gp("min_hits_for_motion").value)
        self.min_speed_for_motion_mps = float(gp("min_speed_for_motion_mps").value)
        # Keyed by label_category(); max_speed_unknown_mps is the fallback
        # for any category not listed (see speed_cap_for_category).
        self.max_speed_unknown_mps = float(gp("max_speed_unknown_mps").value)
        # WP-A: speed_cap_for_category's fallback for THE ACTUAL moving-
        # allowed decision (_resolve_moving_allowed) is this, not
        # max_speed_unknown_mps -- see stack_motion_first's own declare_
        # parameter comment above for why the two are kept separate.
        self.max_speed_default_mps = float(gp("max_speed_default_mps").value)
        self.speed_caps = {
            "person": float(gp("max_speed_person_mps").value),
            "robot": float(gp("max_speed_robot_mps").value),
            "wheeled": float(gp("max_speed_wheeled_mps").value),
        }
        # Track ids already logged once for exceeding their category's
        # speed cap -- see _resolve_moving_allowed. Never evicted: a track
        # id is unique for the life of the tracker (object_tracker_node
        # never reuses one), and the set stays far smaller than a run's
        # object count.
        self._speed_cap_warned: set = set()
        # Same dedup pattern, for the below_min_speed DEBUG line -- see
        # _resolve_moving_allowed.
        self._min_speed_warned: set = set()

        self.enable_rel = bool(gp("enable_relative_motion").value)
        self.cpa_gain = float(gp("cpa_gain").value)
        self.cpa_scale = max(1e-3, float(gp("cpa_scale_m").value))
        self.ttc_scale = max(1e-3, float(gp("ttc_scale_s").value))
        self.min_rel_speed = float(gp("min_rel_speed").value)
        self.odom_timeout = float(gp("odom_timeout_sec").value)
        self.tf_timeout = float(gp("tf_timeout_sec").value)
        self.publish_cpa = bool(gp("publish_cpa_markers").value)
        self.two_hypothesis_cpa = bool(gp("two_hypothesis_cpa").value)

        self.prior_weight = float(gp("spatial_prior_weight").value)

        # WP-C -- class-agnostic recent-activity floor.
        self.activity_topic = str(gp("activity_topic").value)
        self.activity_weight = float(gp("activity_weight").value)
        self.activity_weight_planner = float(gp("activity_weight_planner").value)
        self.activity_weight_grid = float(gp("activity_weight_grid").value)

        self.flow_categories = [str(c) for c in gp("flow_categories").value]
        self.flow_blend_weight = float(gp("flow_blend_weight").value)
        self.flow_category_map = parse_flow_category_map(
            [str(e) for e in gp("flow_category_map").value])

        self.publish_planner_grid = bool(gp("publish_planner_grid").value)
        self.planner_topic = str(gp("planner_topic").value)
        self.planner_N = int(gp("planner_horizon_steps").value)
        self.planner_gamma = float(gp("planner_gamma").value)
        self.planner_flow_blend_weight = float(gp("planner_flow_blend_weight").value)

        self.publish_srm = bool(gp("publish_srm").value)
        self.srm_topic = str(gp("srm_topic").value)
        self.srm_now_topic = str(gp("srm_now_topic").value)
        self.srm_marker_topic = str(gp("srm_marker_topic").value)
        self.srm_d0_m = max(1e-3, float(gp("srm_d0_m").value))
        self.srm_levels = [float(v) for v in gp("srm_levels").value]
        self.srm_window_m = float(gp("srm_window_m").value)
        self.srm_marker_min = float(gp("srm_marker_min").value)
        self.srm_marker_delta = float(gp("srm_marker_delta").value)
        self.srm_marker_layer_stride = int(gp("srm_marker_layer_stride").value)
        # Rolling buffer of stack_to_srm() wall-clock durations (seconds),
        # for the p50/p90 log line every ~5 s -- see _tick(). 50 samples at
        # the 5 Hz default publish_rate is exactly the last 10 s, plenty
        # for a stable percentile without growing unbounded.
        self._srm_times: deque = deque(maxlen=50)

        self.bridge = CvBridge()
        # category -> (rows, cols, 3) array of (s, fx, fy), at THIS node's
        # own grid resolution -- see _flow_cb for why resampling isn't
        # supported here the way it is for the OccupancyGrid-based prior.
        self.flow_grids: Dict[str, np.ndarray] = {}

        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(gp("debug_log_dir").value))

        self.log_writer, self.log_file, _ = open_debug_csv(
            self, str(gp("debug_log_path").value), str(gp("debug_log_dir").value),
            "predictive_costmap",
            ["t", "track_id", "label", "category", "x", "y", "vx", "vy", "speed",
             "pmot", "pmov", "use_motion_mixture",
             "consequence", "cpa_factor", "t_cpa", "d_cpa", "relbonus", "C",
             "w_stat", "w_move_step0", "n_move_steps",
             "rollout_dx", "rollout_dy",
             "flow_blend_weight", "spatial_prior_weight",
             # WP-A columns, not in user-a/sandbox's header: the two gates
             # that decide whether the moving hypothesis paints at all, and
             # the (possibly agnostic) consequence the STACK used.
             "moving_allowed", "stack_consequence",
             # Two-hypothesis CPA columns (appended, not interleaved, so
             # tools/prior_report.py's header-keyed reads of the columns
             # above are unaffected). cpa_factor/t_cpa/d_cpa/C above are
             # always the MOVING hypothesis's; these three plus the flag
             # are the STATIONARY hypothesis's own reading -- equal to the
             # moving columns whenever two_hypothesis_cpa is 0.
             "two_hypothesis_cpa", "cpa_factor_stat", "t_cpa_stat",
             "d_cpa_stat", "C_stat"])

        self.latest: Optional[Detection3DArray] = None

        # robot state, refreshed per tick; None means "unknown, skip Stage 4b"
        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_v: Tuple[float, float] = (0.0, 0.0)
        self.last_odom: Optional[Odometry] = None
        self.last_odom_t: float = -1.0
        self._warned_tf = False

        # spatial prior resampled onto THIS grid, cached until a new one arrives
        self.prior_baseline: Optional[np.ndarray] = None
        # WP-C: recent-activity grid, resampled the same way -- see
        # _resample_occupancy_grid/_activity_cb. UNWEIGHTED (unlike
        # prior_baseline, which bakes spatial_prior_weight in at ingestion):
        # activity_weight/activity_weight_planner/activity_weight_grid each
        # apply their own weight to this SAME array in _tick.
        self.activity_baseline: Optional[np.ndarray] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Detection3DArray, str(gp("input_topic").value),
                                 self._objects_cb, 10)
        if self.enable_rel:
            self.create_subscription(Odometry, str(gp("odom_topic").value),
                                     self._odom_cb, 10)
        if self.prior_weight > 0.0:
            self.create_subscription(OccupancyGrid, str(gp("spatial_prior_topic").value),
                                     self._prior_cb, 1)
        # WP-C: subscribed if ANY of the three consumer weights is on --
        # each grid this node publishes is seeded independently (see
        # _tick), so no single weight being zero should suppress the
        # subscription the other two still need.
        if (self.activity_weight > 0.0 or self.activity_weight_planner > 0.0
                or self.activity_weight_grid > 0.0):
            self.create_subscription(OccupancyGrid, self.activity_topic,
                                     self._activity_cb, 1)
        # Subscribed if EITHER the stack/collapsed-grid rollout blend or the
        # planner grid's own (independent) blend weight is on -- the
        # planner grid defaults to planner_flow_blend_weight=0.5 while
        # flow_blend_weight defaults to 0.0, so this must not gate on
        # flow_blend_weight alone or the planner grid would never get flow
        # data with the shipped defaults.
        if self.flow_blend_weight > 0.0 or self.planner_flow_blend_weight > 0.0:
            # Subscribe by CHANNEL, not category -- flow_category_map can
            # route several categories (robot, wheeled) onto the same
            # merged channel (robot_group), and a dict comprehension over
            # flow_categories naturally de-duplicates that down to one
            # subscription per unique channel rather than two identical
            # ones. self.flow_grids is keyed by channel from here on; see
            # resolve_flow_channel for how a category looks its channel up
            # again at consumption time.
            channels = {resolve_flow_channel(category, self.flow_category_map)
                        for category in self.flow_categories}
            for channel in channels:
                self.create_subscription(
                    Image, f"/risk_perception/spatial_flow/{channel}",
                    lambda msg, c=channel: self._flow_cb(c, msg), 1)

        # TRANSIENT_LOCAL to match nav2_risk_layer's transient_local+reliable
        # subscription -- required the day risk_layer.topic is repointed here,
        # and it hands late-joining subscribers (RViz) the last grid for free.
        grid_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid, str(gp("costmap_topic").value), grid_qos)
        # same QoS as grid_pub -- RELIABLE + TRANSIENT_LOCAL so a
        # late-joining critic/RViz gets the last stack for free too.
        self.stack_pub = self.create_publisher(
            RiskStack, self.stack_topic, grid_qos) if self.publish_stack else None
        # same QoS again -- RELIABLE + TRANSIENT_LOCAL + KeepLast(1), see
        # module docstring STAGE 6.
        self.planner_pub = self.create_publisher(
            OccupancyGrid, self.planner_topic, grid_qos) if self.publish_planner_grid else None
        # WP-A -- same QoS again, see module docstring's SRM section.
        self.srm_pub = self.create_publisher(
            RiskStack, self.srm_topic, grid_qos) if self.publish_srm else None
        self.srm_now_pub = self.create_publisher(
            OccupancyGrid, self.srm_now_topic, grid_qos) if self.publish_srm else None
        self.srm_marker_pub = self.create_publisher(
            MarkerArray, self.srm_marker_topic, 10) if self.publish_srm else None
        self.marker_pub = self.create_publisher(
            MarkerArray, str(gp("marker_topic").value), 10)
        self.create_timer(1.0 / float(gp("publish_rate").value), self._tick)

        planner_status = (
            f"on ({self.planner_topic}, {self.planner_N}x{self.pred_dt}s, "
            f"gamma={self.planner_gamma})"
        ) if self.publish_planner_grid else "off"
        srm_status = (
            f"on ({self.srm_topic}, d0={self.srm_d0_m}m, "
            f"levels={self.srm_levels}, window={self.srm_window_m}m)"
        ) if self.publish_srm else "off"
        self.get_logger().info(
            f"predictive_risk_costmap STAGE2+3+4+5+6 + WP-A SRM: "
            f"{self.rows}x{self.cols} @ {self.res} m/cell, "
            f"horizon {self.N}x{self.pred_dt}s, "
            f"srm {srm_status}, "
            f"relative-motion {'on' if self.enable_rel else 'off'}, "
            f"spatial prior weight {self.prior_weight:.2f}, "
            f"stack {'on (' + self.stack_topic + ')' if self.publish_stack else 'off'}, "
            f"planner grid {planner_status}, "
            f"extrapolate-to-now {'on' if self.extrapolate_to_now else 'off'} "
            f"(max_track_age_sec={self.max_track_age_sec:.1f})")

    # ------------------------------------------------------------------ inputs

    def _objects_cb(self, msg):
        self.latest = msg

    def _odom_cb(self, msg):
        self.last_odom = msg
        self.last_odom_t = self._now()

    def _resample_occupancy_grid(self, msg: OccupancyGrid) -> np.ndarray:
        """WP-C, 2026-09-11: the resample half of _prior_cb, factored out
        so /risk_perception/activity_prior's subscriber (_activity_cb) can
        reuse it without also inheriting _prior_cb's spatial_prior_weight
        multiplication -- activity has THREE independent consumer weights
        (activity_weight/activity_weight_planner/activity_weight_grid,
        applied in _tick), not one shared weight baked in at ingestion.

        Returns `msg`'s data as a [0, 1] array resampled (nearest-cell,
        through world coordinates, so a message saved on a different grid
        geometry still lands in the right physical place) onto THIS node's
        own geometry -- same resample contract _prior_cb has always had,
        just without the caller-specific weight multiply."""
        src = np.array(msg.data, dtype=np.float32).reshape(
            msg.info.height, msg.info.width) / 100.0
        np.clip(src, 0.0, 1.0, out=src)

        same = (abs(msg.info.resolution - self.res) < 1e-9
                and msg.info.width == self.cols and msg.info.height == self.rows
                and abs(msg.info.origin.position.x - self.ox) < 1e-6
                and abs(msg.info.origin.position.y - self.oy) < 1e-6)
        if same:
            return src

        # nearest-cell resample through world coordinates, so a prior saved on a
        # different grid geometry still lands in the right physical place
        wx = self.ox + (np.arange(self.cols) + 0.5) * self.res
        wy = self.oy + (np.arange(self.rows) + 0.5) * self.res
        sc = np.floor((wx - msg.info.origin.position.x) / msg.info.resolution).astype(int)
        sr = np.floor((wy - msg.info.origin.position.y) / msg.info.resolution).astype(int)
        ok_c = (sc >= 0) & (sc < msg.info.width)
        ok_r = (sr >= 0) & (sr < msg.info.height)
        out = np.zeros((self.rows, self.cols), dtype=np.float32)
        if ok_c.any() and ok_r.any():
            out[np.ix_(ok_r, ok_c)] = src[np.ix_(sr[ok_r], sc[ok_c])]
        return out

    def _prior_cb(self, msg: OccupancyGrid):
        """Resample the prior onto this node's grid once, on arrival."""
        self.prior_baseline = self._resample_occupancy_grid(msg) * self.prior_weight

    def _activity_cb(self, msg: OccupancyGrid) -> None:
        """WP-C: resample spatial_prior_node's class-agnostic recent-
        activity grid onto this node's geometry, cached UNWEIGHTED (see
        _resample_occupancy_grid's docstring) -- _tick applies
        activity_weight/activity_weight_planner/activity_weight_grid
        independently to this same array for the stack/planner grid/
        collapsed grid respectively, by MAX, never addition."""
        self.activity_baseline = self._resample_occupancy_grid(msg)

    def _flow_cb(self, channel: str, msg: Image) -> None:
        """Store one channel's (s, fx, fy) -- `channel` is a per-category
        name (e.g. "person") or a merged group name (e.g. "robot_group"),
        whichever this subscription was created for; see
        resolve_flow_channel for how a category maps onto a channel at
        consumption time. Unlike _prior_cb's OccupancyGrid, sensor_msgs/
        Image carries no resolution/origin metadata to resample from --
        this topic REQUIRES identical grid geometry between
        spatial_prior_node and this node (already the documented
        convention for the other topic). A dimension mismatch is logged
        and the update is dropped rather than guessed at."""
        if msg.height != self.rows or msg.width != self.cols:
            self.get_logger().warning(
                f"spatial_flow/{channel}: {msg.width}x{msg.height} does not "
                f"match this node's {self.cols}x{self.rows} grid -- dropping "
                "this update. Image carries no resolution/origin to resample "
                "from; align both nodes' resolution/width_m/height_m/"
                "origin_x/origin_y instead.",
                throttle_duration_sec=10.0)
            return
        array = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC3")
        self.flow_grids[channel] = np.asarray(array, dtype=np.float32)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _update_robot_state(self):
        """Robot pose from tf (map->base), body twist from odom rotated into map."""
        self.robot_xy = None
        self.robot_v = (0.0, 0.0)
        if not self.enable_rel:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time(),
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException as exc:
            if not self._warned_tf:
                self.get_logger().warning(
                    f"no {self.map_frame} -> {self.base_frame} tf ({exc}); "
                    "relative-motion amplification is inactive until it appears")
                self._warned_tf = True
            return
        self._warned_tf = False

        self.robot_xy = (float(tf.transform.translation.x),
                         float(tf.transform.translation.y))

        if self.last_odom is None:
            return
        if self._now() - self.last_odom_t > self.odom_timeout:
            return  # stale odom: pose is still good, treat the robot as stopped
        yaw = yaw_from_quat(tf.transform.rotation)
        bx = float(self.last_odom.twist.twist.linear.x)
        by = float(self.last_odom.twist.twist.linear.y)
        self.robot_v = (bx * math.cos(yaw) - by * math.sin(yaw),
                        bx * math.sin(yaw) + by * math.cos(yaw))

    # ------------------------------------------------------------------ stage 4

    def _consequence(self, label, score):
        if not self.use_class_consequence:
            return 1.0
        return risk_score_from_label(label, max(0.0, min(1.0, score)))

    def _stack_consequence(self, label, score):
        """STAGE 5 severity: class-level unless stack_consequence_mode=scaled."""
        if self.stack_consequence_mode == "scaled":
            return self._consequence(label, score)
        if score < self.stack_min_track_score:
            return 0.0
        if not self.use_class_consequence:
            return 1.0
        return risk_score_from_label(label, 1.0)

    def _resolve_moving_allowed(self, kv, speed, category, label, id_seed) -> bool:
        """Node wrapper around the pure resolve_moving_allowed() (WP2
        extraction, 2026-09-10 evening): owns the once-per-id log lines,
        the pure function owns the decision. self._speed_cap_warned /
        self._min_speed_warned persist for the node's lifetime so a
        chronically-too-fast or chronically-too-slow track (a bad
        association or a permanently jittery detection, not a one-off)
        doesn't spam the log every tick. "immature"/"ok" reasons log
        nothing -- is_track_mature already has no logging of its own, and
        "ok" needs none."""
        allowed, reason = resolve_moving_allowed(
            kv, speed, category,
            self.min_track_age_for_motion_sec, self.min_hits_for_motion,
            self.speed_caps, self.max_speed_default_mps,
            self.min_speed_for_motion_mps)
        if reason == "over_cap" and id_seed not in self._speed_cap_warned:
            self._speed_cap_warned.add(id_seed)
            cap = speed_cap_for_category(category, self.speed_caps, self.max_speed_default_mps)
            self.get_logger().info(
                f'speed {speed:.1f} m/s > cap {cap:.1f} for "{label}" '
                f'(id {id_seed}): painting stationary only')
        elif reason == "below_min_speed" and id_seed not in self._min_speed_warned:
            self._min_speed_warned.add(id_seed)
            self.get_logger().debug(
                f'speed {speed:.2f} m/s < min_speed_for_motion_mps '
                f'{self.min_speed_for_motion_mps:.2f} for "{label}" '
                f'(id {id_seed}): painting stationary only')
        return allowed

    def _encounter(self, x, y, vx, vy):
        """(factor, t_cpa, d_cpa) for the MOVING hypothesis (v_obj = vx, vy).
        factor == 1.0 exactly when Stage 4b is inert.

        Pure geometry lives in encounter_geometry.cpa_geometry -- see
        test_encounter_geometry.py for the unit tests (no ROS graph needed).
        """
        if self.robot_xy is None:
            return 1.0, float("inf"), float("inf")
        return cpa_geometry(
            x, y, vx, vy,
            self.robot_xy[0], self.robot_xy[1], self.robot_v[0], self.robot_v[1],
            self.cpa_gain, self.cpa_scale, self.ttc_scale, self.min_rel_speed)

    def _encounter_stationary(self, x, y):
        """(factor, t_cpa, d_cpa) for the STATIONARY hypothesis (v_obj = 0,
        so v_rel = -v_robot -- only the robot's own motion can close the
        gap). Only called when two_hypothesis_cpa is on; see module
        docstring's STAGE 4 section. Same cpa_gain/cpa_scale/ttc_scale/
        min_rel_speed as _encounter -- the two hypotheses share every
        constant, they differ only in which velocity they hand cpa_geometry."""
        if self.robot_xy is None:
            return 1.0, float("inf"), float("inf")
        return cpa_geometry(
            x, y, 0.0, 0.0,
            self.robot_xy[0], self.robot_xy[1], self.robot_v[0], self.robot_v[1],
            self.cpa_gain, self.cpa_scale, self.ttc_scale, self.min_rel_speed)

    # ------------------------------------------------------------------ raster

    def _track_age_sec(self, det, now: float) -> float:
        """now - det.header.stamp, in seconds. A never-stamped header
        (sec=0, nanosec=0 -- object_tracker_node always stamps it, but a
        hand-built test/tool detection may not) is treated as age 0.0
        rather than as ~55 years old."""
        stamp = det.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if t <= 0.0:
            return 0.0
        return max(0.0, now - t)

    def _log_track_ages(self, detections) -> None:
        """Info-level mean/max track age every 5s regardless of staleness
        (a health signal), plus a separate throttled WARNING once any
        track exceeds max_track_age_sec -- it is still painted (clamped,
        not dropped), this only flags that the tracker/transport is
        falling behind."""
        if not detections:
            return
        now = self._now()
        ages = [self._track_age_sec(det, now) for det in detections]
        mean_age = sum(ages) / len(ages)
        max_age = max(ages)
        self.get_logger().info(
            f"track age: mean {mean_age:.2f}s, max {max_age:.2f}s "
            f"over {len(ages)} track(s)",
            throttle_duration_sec=5.0)
        if max_age > self.max_track_age_sec:
            stale = sum(1 for a in ages if a > self.max_track_age_sec)
            self.get_logger().warning(
                f"{stale}/{len(ages)} track(s) older than "
                f"max_track_age_sec={self.max_track_age_sec:.1f}s "
                f"(mean age {mean_age:.2f}s) -- still painted with a "
                "clamped extrapolation, but the tracker or the transport "
                "to this node may be falling behind",
                throttle_duration_sec=5.0)

    def _tick(self):
        _t0 = time.perf_counter()
        if self.prior_baseline is not None:
            grid = self.prior_baseline.copy()
        else:
            grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        stack = np.zeros((self.N + 1, self.rows, self.cols), dtype=np.float32) \
            if self.publish_stack else None
        planner_grid = np.zeros((self.rows, self.cols), dtype=np.float32) \
            if self.publish_planner_grid else None

        # WP-C -- class-agnostic recent-activity floor, seeded by MAX
        # BEFORE any track is painted: a track's own painted core on a
        # busy cell keeps its own (usually higher) value untouched by
        # this, and only an otherwise-EMPTY cell is raised -- the paper's
        # "latent risk at busy doorways" even where nothing is currently
        # detected. Never addition -- see module docstring's WP-C section
        # and the activity_weight* declare_parameter comments for why each
        # of the three grids gets its own weight. No-op (activity_baseline
        # still None) until the first /risk_perception/activity_prior
        # message arrives.
        if self.activity_baseline is not None:
            if self.activity_weight_grid > 0.0:
                grid_floor = self.activity_weight_grid * self.activity_baseline
                np.maximum(grid, grid_floor, out=grid)
            if stack is not None and self.activity_weight > 0.0:
                stack_floor = self.activity_weight * self.activity_baseline
                for k in range(self.N + 1):
                    np.maximum(stack[k], stack_floor, out=stack[k])
            if planner_grid is not None and self.activity_weight_planner > 0.0:
                planner_floor = self.activity_weight_planner * self.activity_baseline
                np.maximum(planner_grid, planner_floor, out=planner_grid)

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        self._update_robot_state()

        if self.latest is not None:
            self._log_track_ages(self.latest.detections)
            for det in self.latest.detections:
                grid, stack, planner_grid = self._predict_and_splat(
                    det, grid, markers, stack, planner_grid)

        # One shared stamp for every publisher this tick -- the SRM is a
        # per-layer transform of the SAME stack the raw /risk_stack gets,
        # so their headers must agree exactly, not just to within the
        # microseconds between two separate now() calls.
        stamp = self.get_clock().now().to_msg()

        self._publish_grid(grid, self.grid_pub, stamp=stamp)
        if stack is not None:
            self._publish_stack(stack, stamp=stamp)
            # WP-A: the SRM is a per-layer transform of THIS SAME stack, so
            # it requires publish_stack -- there is nothing to window/
            # transform when the raw stack was never built (publish_srm
            # with publish_stack:=false is a no-op, not an error).
            if self.publish_srm:
                self._publish_srm(stack, stamp=stamp)
        if planner_grid is not None:
            self._publish_grid(planner_grid, self.planner_pub, stamp=stamp)
        self.marker_pub.publish(markers)
        log_latency(self.lat_writer, self.lat_file, time.perf_counter() - _t0)

    def _predict_and_splat(self, det, grid, markers, stack=None, planner_grid=None):
        x = float(det.bbox.center.position.x)
        y = float(det.bbox.center.position.y)
        label, kv = ("object", {})
        pmot = pmov = vx = vy = relbonus = 0.0
        score = 1.0
        if det.results:
            label, kv = parse_class_id(det.results[0].hypothesis.class_id)
            score = float(det.results[0].hypothesis.score)
            pmot = kv.get("pmot", 0.0) if self.use_motion_mixture else 0.0
            pmov = kv.get("pmov", 0.3)
            vx = kv.get("vx", 0.0)
            vy = kv.get("vy", 0.0)
            # relation prior (proposed) -- see object_tracker_node.Track.update_relation
            # use_relation_bonus False is the Relation-prior ablation arm:
            # the tag is still parsed and still logged, it just does not
            # reach combine_severity. See that declare_parameter comment.
            relbonus = kv.get("relbonus", 0.0) if self.use_relation_bonus else 0.0
            cov = det.results[0].pose.covariance
            Pxx = max(float(cov[0]), self.min_var)
            Pyy = max(float(cov[7]), self.min_var)
            Pvx = float(cov[21])
            Pvy = float(cov[28])
        else:
            Pxx = Pyy = self.min_var
            Pvx = Pvy = 0.0

        try:
            id_seed = int(det.id) if det.id else 0
        except ValueError:
            id_seed = 0

        # Object extent: the track's bbox half-size, added as variance so a
        # 0.75 m AMR paints a band as wide as itself, not a point. The
        # extent_cap_* upper bound and the min_extent_* semantic keep-out
        # floor are combined by resolve_extent_half() -- see module
        # docstring's 'semantic keep-out floor' section; this shared
        # Pxx/Pyy feeds the collapsed grid, the stack, AND the planner grid.
        category = label_category(label)
        # Which /risk_perception/spatial_flow/<channel> this track's
        # category samples -- the mapped merged channel (flow_category_map)
        # if one exists for this category, else the category's own topic.
        # Computed once, used by both the collapsed/stack blend below and
        # the planner grid's own (independently-weighted) blend.
        flow_channel = resolve_flow_channel(category, self.flow_category_map)
        if self.use_object_extent:
            cap = self.extent_caps.get(category, self.extent_cap_other)
            bbox_half = 0.5 * max(float(det.bbox.size.x), float(det.bbox.size.y))
            floor = select_min_extent(
                label, category, self.min_extent_person, self.min_extent_robot,
                self.min_extent_wheeled, self.min_extent_forklift, self.min_extent_other)
            half = resolve_extent_half(bbox_half, cap, floor)
            Pxx += half * half
            Pyy += half * half

        # WP-A (2026-09-10 "SRM fills half the map" finding, root causes
        # 2+3): is this track mature enough (hits/age) AND fast enough to
        # be plausible for its category to trust its motion belief at all?
        # If not, paint_track treats pmot as 0 -- stationary-only -- for
        # BOTH hypotheses, grid and stack alike.
        speed = math.hypot(vx, vy)
        allowed = self._resolve_moving_allowed(kv, speed, category, label, id_seed)
        # WP-A: an additional, independent gate on the MOVING hypothesis --
        # even a mature, plausible-speed track only paints as a mover when
        # its OWN motion belief (pmot) is confident enough; below that it
        # still paints (at the agnostic consequence under motion-first),
        # just stationary-only. See stack_moving_pmot_min's declare_
        # parameter comment and the module docstring's WP-A section.
        #
        # stack_moving_pmov_min is the same idea applied to the SLOW,
        # class-seeded belief (p_movable) instead of the instant one
        # (p_motion) -- see that param's own declare_parameter comment.
        # Default 0.0 makes this a no-op (pmov is always >= 0.0).
        moving_allowed = (
            allowed
            and pmot >= self.stack_moving_pmot_min
            and pmov >= self.stack_moving_pmov_min
        )

        # STAGE 4a: how bad is a collision with THIS class...
        consequence = self._consequence(label, score)
        if self.stack_motion_first:
            # WP-A: the SAME agnostic value paints the collapsed grid AND
            # the planner grid (both consume this `consequence` variable
            # below) -- see resolve_agnostic_consequence()'s docstring for
            # why this is a shared helper rather than two independent
            # overrides.
            consequence = resolve_agnostic_consequence(
                consequence, self.stack_consequence_agnostic,
                self.semantic_modifier_enabled)
        stack_consequence = resolve_stack_consequence(
            category, self.stack_categories, pmot, self._stack_consequence(label, score),
            self.stack_paint_unknown_moving, self.unknown_moving_risk,
            self.unknown_moving_pmot_min,
            unknown_moving_min_hits=self.unknown_moving_min_hits,
            unknown_moving_min_age_sec=self.unknown_moving_min_age_sec,
            hits=kv.get("hits"), age=kv.get("age"),
            speed=speed, max_speed_unknown_mps=self.max_speed_unknown_mps,
            motion_first=self.stack_motion_first,
            agnostic_value=self.stack_consequence_agnostic,
            semantic_modifier=self.semantic_modifier_enabled,
            score=score, stack_min_track_score=self.stack_min_track_score)
        # STAGE 4b: ...and how much is the current encounter geometry closing.
        # This is the collapsed grid's C only -- paint_track derives the
        # STAGE 5 stack's own C_stack (factor pinned to 1.0) internally.
        #
        # Two-hypothesis CPA (two_hypothesis_cpa, see module docstring):
        # factor_mov is against the track's own believed velocity, exactly
        # as before. When the flag is off, factor_stat is just an alias for
        # factor_mov -- a parked hypothesis scored by the SAME CPA reading
        # as the moving one, the historical (arguably conflated) behaviour.
        # When on, factor_stat is computed separately with v_obj = 0, so
        # the robot driving at a track it believes may be parked reads its
        # own closing geometry rather than borrowing the moving hypothesis's.
        factor_mov, t_cpa_mov, d_cpa_mov = self._encounter(x, y, vx, vy)
        if self.two_hypothesis_cpa:
            factor_stat, t_cpa_stat, d_cpa_stat = self._encounter_stationary(x, y)
        else:
            factor_stat, t_cpa_stat, d_cpa_stat = factor_mov, t_cpa_mov, d_cpa_mov

        if self.publish_cpa and max(factor_stat, factor_mov) > 1.05:
            self._add_cpa_marker(markers, id_seed, x, y, t_cpa_mov, d_cpa_mov,
                                 factor_mov, factor_stat if self.two_hypothesis_cpa else None)

        # STAGE 5 extrapolation: advance the MOVING hypothesis's rollout
        # start to "now" by this track's own age. The stationary hypothesis
        # keeps the observed (x, y) untouched -- see module docstring.
        move_x, move_y = x, y
        if self.extrapolate_to_now:
            age = min(self._track_age_sec(det, self._now()), self.max_track_age_sec)
            move_x = x + vx * age
            move_y = y + vy * age

        # Iterative (Euler) rollout rather than closed-form x + vx*t: the
        # Spatial-Flow blend can make the effective velocity vary step to
        # step, so position has to accumulate. At flow_blend_weight <= 0
        # (the default) blend_velocity returns (vx, vy) unchanged every
        # step, which reduces this to the same constant-velocity rollout as
        # before, to within floating-point rounding.
        flow_array = self.flow_grids.get(flow_channel) \
            if self.flow_blend_weight > 0.0 else None

        grid, stack, rollout = paint_track(
            x, y, move_x, move_y, vx, vy, pmot,
            Pxx, Pyy, Pvx, Pvy, consequence, factor_mov, relbonus,
            self.res, self.ox, self.oy, self.rows, self.cols,
            self.N, self.pred_dt, self.gamma, self.vel_infl,
            grid=grid, stack=stack, want_stack=self.publish_stack,
            flow_array=flow_array, flow_blend_weight=self.flow_blend_weight,
            stack_consequence=stack_consequence,
            moving_allowed=moving_allowed,
            sigma_v_max_mps=self.sigma_v_max_mps, max_sigma_m=self.max_sigma_m,
            factor_stat=factor_stat if self.two_hypothesis_cpa else None,
        )
        for step, mx, my, var_x, var_y, w_move in rollout:
            self._add_pred_marker(markers, id_seed, step, mx, my,
                                  math.sqrt(var_x), w_move)

        self._log_track(det, label, category, x, y, vx, vy, pmot, pmov,
                        consequence, factor_mov, t_cpa_mov, d_cpa_mov, relbonus,
                        rollout, moving_allowed, stack_consequence,
                        factor_stat, t_cpa_stat, d_cpa_stat)

        # STAGE 6: planner grid -- own flow_array lookup because
        # planner_flow_blend_weight is independent of (and non-zero by
        # default, unlike) flow_blend_weight above.
        if planner_grid is not None:
            planner_flow_array = self.flow_grids.get(flow_channel) \
                if self.planner_flow_blend_weight > 0.0 else None
            planner_grid = paint_track_planner(
                x, y, move_x, move_y, vx, vy, pmot,
                Pxx, Pyy, Pvx, Pvy, consequence, relbonus,
                self.res, self.ox, self.oy, self.rows, self.cols,
                self.planner_N, self.pred_dt, self.planner_gamma, self.vel_infl,
                grid=planner_grid,
                flow_array=planner_flow_array,
                flow_blend_weight=self.planner_flow_blend_weight,
                moving_allowed=moving_allowed,
                sigma_v_max_mps=self.sigma_v_max_mps, max_sigma_m=self.max_sigma_m,
            )

        return grid, stack, planner_grid

    def _log_track(self, det, label, category, x, y, vx, vy, pmot, pmov,
                   consequence, factor, t_cpa, d_cpa, relbonus,
                   rollout, moving_allowed, stack_consequence,
                   factor_stat=None, t_cpa_stat=None, d_cpa_stat=None) -> None:
        """One research-CSV row per track per tick (ported from
        user-a/sandbox; consumed by tools/prior_report.py). Her version read
        w_stat/w_move/n_move_steps out of an inline rollout loop; ours gets
        w_move_step0 from paint_track's returned `rollout` list instead. Off
        unless debug_log_dir/debug_log_path is set.

        `factor`/`t_cpa`/`d_cpa` here are always the MOVING hypothesis's
        (factor_mov from _predict_and_splat) -- the pre-existing "cpa_factor"/
        "t_cpa"/"d_cpa"/"C" columns keep their historical meaning whether or
        not two_hypothesis_cpa is on. `factor_stat`/`t_cpa_stat`/`d_cpa_stat`
        (two_hypothesis_cpa only; None otherwise, logged as the moving
        values so the row is still fully populated) feed the three NEW
        trailing columns, appended rather than interleaved so an existing
        header-keyed reader (tools/prior_report.py) is unaffected."""
        if not self.log_writer:
            return
        if factor_stat is None:
            factor_stat, t_cpa_stat, d_cpa_stat = factor, t_cpa, d_cpa
        C_mov = combine_severity(consequence, factor, relbonus)
        C_stat = combine_severity(consequence, factor_stat, relbonus)
        w_stat = (1.0 - pmot) * C_stat
        n_move_steps = len(rollout)
        w_move_step0 = rollout[0][5] if rollout else 0.0
        rollout_dx = rollout[-1][1] - x if rollout else 0.0
        rollout_dy = rollout[-1][2] - y if rollout else 0.0
        self.log_writer.writerow([
            f"{self._now():.3f}", det.id, label, category,
            f"{x:.3f}", f"{y:.3f}", f"{vx:.4f}", f"{vy:.4f}",
            f"{math.hypot(vx, vy):.4f}",
            f"{pmot:.3f}", f"{pmov:.3f}", int(self.use_motion_mixture),
            f"{consequence:.3f}", f"{factor:.3f}",
            "inf" if math.isinf(t_cpa) else f"{t_cpa:.3f}",
            "inf" if math.isinf(d_cpa) else f"{d_cpa:.3f}",
            f"{relbonus:.3f}", f"{C_mov:.3f}",
            f"{w_stat:.4f}", f"{w_move_step0:.4f}", n_move_steps,
            f"{rollout_dx:.3f}", f"{rollout_dy:.3f}",
            f"{self.flow_blend_weight:.2f}", f"{self.prior_weight:.2f}",
            int(bool(moving_allowed)), f"{stack_consequence:.3f}",
            int(self.two_hypothesis_cpa), f"{factor_stat:.3f}",
            "inf" if math.isinf(t_cpa_stat) else f"{t_cpa_stat:.3f}",
            "inf" if math.isinf(d_cpa_stat) else f"{d_cpa_stat:.3f}",
            f"{C_stat:.3f}"])
        self.log_file.flush()

    def _add_pred_marker(self, markers, id_seed, step, mx, my, sigma, weight):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns = "prediction"
        m.id = id_seed * 100 + step
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position = Point(x=mx, y=my, z=0.05)
        m.pose.orientation.w = 1.0
        d = max(0.1, 2.0 * sigma)          # ~2 sigma footprint
        m.scale.x = m.scale.y = d
        m.scale.z = 0.02
        # red comet-tail, fading with weight (near-future brightest)
        m.color = ColorRGBA(r=0.95, g=0.2, b=0.2, a=float(max(0.05, min(0.8, weight))))
        m.lifetime = DurationMsg(sec=0, nanosec=300_000_000)
        markers.markers.append(m)

    def _add_cpa_marker(self, markers, id_seed, x, y, t_cpa, d_cpa, factor,
                        factor_stat=None):
        """Readout of WHY a track got amplified -- the Stage 4b test
        instrument. `factor_stat` (two_hypothesis_cpa only) appends the
        stationary hypothesis's own factor so an ablation run can eyeball
        both readings, not just the moving one; None (single-CPA mode, or
        when the two happen to coincide) leaves the text exactly as before."""
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns = "encounter"
        m.id = id_seed
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = Point(x=x, y=y, z=0.85)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.20
        m.color = ColorRGBA(r=1.0, g=0.85, b=0.2, a=1.0)
        m.lifetime = DurationMsg(sec=0, nanosec=300_000_000)
        m.text = f"TTC {t_cpa:.1f}s  CPA {d_cpa:.2f}m  x{factor:.2f}"
        if factor_stat is not None:
            m.text += f"  (stat x{factor_stat:.2f})"
        markers.markers.append(m)

    def _fill_map_info(self, info) -> None:
        """Same resolution/width/height/origin on the OccupancyGrid's
        MapMetaData and the RiskStack's -- a layer sliced out of the
        stack's `data` drops straight into an OccupancyGrid with no
        coordinate conversion."""
        info.resolution = self.res
        info.width = self.cols
        info.height = self.rows
        info.origin.position.x = self.ox
        info.origin.position.y = self.oy
        info.origin.orientation.w = 1.0

    def _publish_grid(self, grid, publisher=None, stamp=None):
        """Publish one collapsed OccupancyGrid on `publisher` (defaults to
        self.grid_pub, the /risk_costmap_predictive publisher, so existing
        callers are unaffected) -- shared by both /risk_costmap_predictive
        and STAGE 6's /risk_costmap_planner, which differ only in which
        grid array and which publisher they use, not in how a grid is
        serialized. `stamp` defaults to "now" when the caller doesn't hand
        one in, but _tick() passes the SAME stamp to every publisher each
        tick (see _publish_srm) so the raw stack and its SRM never drift
        apart by the few microseconds between two separate now() calls."""
        if publisher is None:
            publisher = self.grid_pub
        msg = OccupancyGrid()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        self._fill_map_info(msg.info)
        # round, not truncate -- risk_costmap_node rounds, and the two grids
        # are meant to be diffed cell-for-cell in the ablation
        cost = np.clip(np.round(grid * 100.0), 0, 100).astype(np.int8)
        msg.data = cost.flatten(order="C").tolist()
        publisher.publish(msg)

    def _publish_stack(self, stack, stamp=None):
        """STAGE 5's time-layered field. `data` is stack.ravel(order="C")
        -- since stack is (steps, rows, cols), that is index
        k*rows*cols + r*cols + c, exactly the layout panoptex_msgs/
        RiskStack.msg documents. `stamp` -- see _publish_grid."""
        msg = RiskStack()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        self._fill_map_info(msg.info)
        msg.dt = self.pred_dt
        msg.steps = self.N + 1
        msg.horizon_start = 0.0
        cost = np.clip(np.round(stack * 100.0), 0, 100).astype(np.int8)
        msg.data = cost.ravel(order="C").tolist()
        self.stack_pub.publish(msg)

    def _srm_window_centre(self) -> Tuple[float, float]:
        """Where the SRM window is centred: the robot's own map-frame
        position if Stage 4b's tf lookup has resolved it this tick
        (self.robot_xy, set in _update_robot_state -- reused, not
        re-resolved), else this grid's own centre (see module docstring's
        WP-A section for why "degrade to the grid centre" is the right
        fallback here, matching Stage 4's own degrade pattern)."""
        if self.robot_xy is not None:
            return self.robot_xy
        return (self.ox + 0.5 * self.cols * self.res,
                self.oy + 0.5 * self.rows * self.res)

    def _publish_srm(self, stack, stamp=None) -> None:
        """WP-A: window the STAGE 5 `stack` around the robot, run it
        through risk_perception.srm.stack_to_srm (Thomas et al. 2021 eq.
        3), and publish the result on srm_topic/srm_now_topic/
        srm_marker_topic. See module docstring's WP-A section."""
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        r0, r1, c0, c1, win_origin = window_indices(
            (self.ox, self.oy), self.res, self.rows, self.cols,
            self._srm_window_centre(), self.srm_window_m)
        window_stack = stack[:, r0:r1, c0:c1]
        win_rows, win_cols = window_stack.shape[1], window_stack.shape[2]

        t0 = time.perf_counter()
        srm = stack_to_srm(window_stack, self.res, self.srm_d0_m, self.srm_levels)
        elapsed_s = time.perf_counter() - t0
        self._srm_times.append(elapsed_s)
        if self._srm_times:
            times_ms = np.asarray(self._srm_times) * 1000.0
            self.get_logger().info(
                f"SRM compute: p50={np.percentile(times_ms, 50):.2f} ms, "
                f"p90={np.percentile(times_ms, 90):.2f} ms "
                f"over last {len(self._srm_times)} tick(s) "
                f"({win_rows}x{win_cols} window)",
                throttle_duration_sec=5.0)

        if self.srm_pub is not None:
            msg = RiskStack()
            msg.header.stamp = stamp
            msg.header.frame_id = self.map_frame
            msg.info.resolution = self.res
            msg.info.width = win_cols
            msg.info.height = win_rows
            msg.info.origin.position.x = win_origin[0]
            msg.info.origin.position.y = win_origin[1]
            msg.info.origin.orientation.w = 1.0
            msg.dt = self.pred_dt
            msg.steps = self.N + 1
            msg.horizon_start = 0.0
            cost = np.clip(np.round(srm * 100.0), 0, 100).astype(np.int8)
            msg.data = cost.ravel(order="C").tolist()
            self.srm_pub.publish(msg)

        if self.srm_now_pub is not None:
            now_msg = OccupancyGrid()
            now_msg.header.stamp = stamp
            now_msg.header.frame_id = self.map_frame
            now_msg.info.resolution = self.res
            now_msg.info.width = win_cols
            now_msg.info.height = win_rows
            now_msg.info.origin.position.x = win_origin[0]
            now_msg.info.origin.position.y = win_origin[1]
            now_msg.info.origin.orientation.w = 1.0
            now_cost = np.clip(np.round(srm[0] * 100.0), 0, 100).astype(np.int8)
            now_msg.data = now_cost.flatten(order="C").tolist()
            self.srm_now_pub.publish(now_msg)

        if self.srm_marker_pub is not None:
            self.srm_marker_pub.publish(
                self._srm_markers(srm, win_origin, stamp))

    def _srm_markers(self, srm, win_origin, stamp) -> MarkerArray:
        """One POINTS marker per layer k, one point per cell that passes
        that layer's gate -- the "comet" Thomas et al.'s own video shows
        (see module docstring). Colour interpolates red (k=0, "now") to
        yellow (k=steps-1, ~6 s out): r=1.0 fixed, g grows 0->1 with k
        (red+green=yellow), b=0.0. Alpha is proportional to the cell's own
        SRM value, so a barely-over-threshold cell fades rather than
        popping fully opaque. z = 0.02 + 0.01*k stacks later layers
        slightly higher so overlapping comet segments don't z-fight in
        RViz.

        Per-layer gate is srm_marker_mask() (WP3 extraction, 2026-09-10
        evening) -- layer 0 is the plain srm_marker_min threshold
        (unchanged -- it IS "now"); layers k>=1 additionally require
        srm[k] - srm[0] >= srm_marker_delta (2026-09-10 "parked object
        reads as a mover" fix, delta raised 0.05 -> 0.15 the same evening)
        AND are only drawn every srm_marker_layer_stride-th layer (new
        2026-09-10 evening param, default 3 -- 21 stacked translucent
        layers made a low-value tail look solid where several low-alpha
        layers overlapped; see srm_marker_mask()'s own docstring). Alpha is
        no longer a flat clip(v, 0.1, 1.0): it now maps srm_marker_min
        itself to the alpha floor (0.05) and 1.0 to fully opaque, alpha =
        0.05 + 0.95*(v - srm_marker_min) / (1 - srm_marker_min), clipped
        to [0.05, 1] -- a cell just past the absolute floor barely shows,
        and only a cell close to fully occupied reads fully opaque,
        instead of every qualifying cell already starting at 0.1."""
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        K = srm.shape[0]
        ox, oy = win_origin
        alpha_denom = max(1e-6, 1.0 - self.srm_marker_min)
        for k in range(K):
            mask = srm_marker_mask(
                srm, k, self.srm_marker_min, self.srm_marker_delta,
                self.srm_marker_layer_stride)
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                continue
            frac = k / max(1, K - 1)
            xs = ox + (cols.astype(np.float32) + 0.5) * self.res
            ys = oy + (rows.astype(np.float32) + 0.5) * self.res
            values = srm[k][rows, cols]
            alphas = np.clip(
                0.05 + 0.95 * (values - self.srm_marker_min) / alpha_denom,
                0.05, 1.0)

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.map_frame
            m.ns = "srm"
            m.id = k
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = self.res
            m.points = [Point(x=float(x), y=float(y), z=0.02 + 0.01 * k)
                        for x, y in zip(xs, ys)]
            m.colors = [ColorRGBA(r=1.0, g=float(frac), b=0.0, a=float(a))
                        for a in alphas]
            m.lifetime = DurationMsg(sec=0, nanosec=400_000_000)
            markers.markers.append(m)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = PredictiveRiskCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
