#!/usr/bin/env bash
# run_ablation_repeats.sh -- headless, unattended repeats of the x3 waypoint
# tour (x3_waypoints_to_trials.py's trials_x3_tour.json) for ONE Nav2
# ablation arm. One repeat = one fresh headless Isaac session + one full
# 6-waypoint tour via run_trials.py, scored against evaluation_reference.
#
# Isaac is killed and relaunched headless between repeats so the Nova
# Carters reset to their authored patrol start (the alternative -- leaving
# Isaac running across repeats -- means each repeat meets the Carters at a
# random point in their loop, i.e. same route/different timing, not the
# same scenario; see the conversation this was built from).
# evaluation_reference.launch.py is (re)started fresh EVERY repeat, right
# alongside that repeat's own fresh Isaac session -- not once per arm
# invocation. It carries a live TF2 buffer, and reusing one across an Isaac
# restart poisons it (the new repeat's low /clock timestamps look like they
# go backwards relative to the previous repeat's high-water mark, so tf2
# silently drops everything -- "TF_OLD_DATA ignoring data from the past" --
# which starves lookup_transform(map, base_footprint) for the whole repeat:
# path_length_m/velocity_smoothness/stopped_time_s all silently read as
# 0/fully-stopped despite every Nav2 leg genuinely SUCCEEDED). Do NOT start
# it manually in a separate terminal or expect it to survive across
# invocations either: a leftover copy races the next repeat's own copy on
# /evaluation/trial_control and /evaluation/trial_result and silently
# corrupts recorded trial labels. Both failure modes root-caused
# 2026-09-14; see the pre-flight kill and the per-repeat launch below.
#
# Usage:
#   ./tools/run_ablation_repeats.sh baseline 1 3
#                                   ^arm     ^start rep ^end rep (inclusive)
#
# Arms (see the `case` block below for exactly what each one flips):
#   system:        baseline | reactive | predictive
#   leave-one-out: no_behavioral | no_semantic | no_relation | no_flow
#                  no_encounter
#
# Run once per arm, e.g.:
#   for arm in baseline reactive predictive \
#              no_behavioral no_semantic no_relation no_flow no_encounter; do
#       ./tools/run_ablation_repeats.sh "$arm" 1 3 || break
#   done
#
# 8 arms x 3 reps x ~10 min/rep is roughly 4 h unattended. The `predictive`
# arm IS the full system -- every no_* arm is that arm minus one mechanism,
# so `predictive` is the comparison row for all of them and must be rerun
# whenever the shipped defaults change.
#
# PRECONDITION for the two Spatial-Flow-sensitive arms (predictive, no_flow):
# the persisted prior must already be warmed and must stay frozen. Every
# trial here inherits panoptex_sim.launch.py's spatial_prior_learn_rate
# default -- pass spatial_prior_learn_rate:=0.0 (see EXTRA_FLAGS below) or
# the runs both read AND rewrite the prior, and no two arms see the same one.
#
# Refuses to overwrite an existing results/<arm>_rep<N> output dir or bag --
# exactly the mistake that cost a redo last time (two repeats both wrote to
# results/baseline_rep1). Delete it yourself first if you really want to
# redo one specific rep.
#
# Run the FIRST rep of the FIRST arm attended (watch the terminal, keep an
# eye on GPU/Isaac) before trusting the rest to run unsupervised -- headless
# Isaac startup timing hasn't been measured on this machine yet, and the
# fixed sleeps below are best-effort, not verified thresholds.

set -uo pipefail

WAREHOUSE="$HOME/Digital-Twin-Project/warehouse"
PANOPTEX="$HOME/workspaces/Panoptex"
# x3_sim/isaac.sh defaults PYTHON_SH to a hardcoded, machine-specific Isaac install
# path when ISAACSIM_PYTHON isn't set -- env_sim.sh only
# makes the ROS side of this PC-agnostic, not the Isaac Sim install path
# itself. Without this, isaac.sh silently fails 6/6 retries with "No such
# file or directory" (exit 127) while this script sails on past its /clock
# wait because a stale, already-broken Isaac session from earlier in the
# day was still publishing /clock with no working /scan behind it.
export ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-$HOME/isaacsim/python.sh}"
# Which USD stage to run headless -- override to point at a different scene.
# Default is the verified stage used throughout this repo (see the main
# README): static warehouse, no NPC/agent extensions to worry about.
#
#   SCENE_USD=$HOME/Digital-Twin-Project/warehouse/some_other_scene.usd \
#   SCENE_TAG=other_scene ./tools/run_ablation_repeats.sh baseline 1 3
#
# CAUTION if you point this at a scene with animated NPCs (e.g. one using
# Isaac's Replicator Agent extension): run_headless.py has no NPC-specific
# logic -- it just opens whatever --scene it's given and presses Play --
# so whether such NPCs actually animate headless (vs sitting static)
# depends on Isaac's own animation extensions (omni.anim.people /
# omni.anim.behavior.tree) being enabled at Play time. VERIFY this on the
# first supervised rep before trusting results from that scene; a study run
# where the NPCs silently sat static is not the study you think it is.
SCENE_USD="${SCENE_USD:-$HOME/Digital-Twin-Project/warehouse/Baseline_scenario_metric.usd}"
SCENE_TAG="${SCENE_TAG:-}"
# X3's map_yaml (panoptex_sim.launch.py) and the Carters' map (carters_patrol
# .launch.py) -- override either independently if you point SCENE_USD at a
# different stage with its own exported map pair:
#   X3_MAP_YAML=$HOME/Digital-Twin-Project/warehouse/maps/some_other_x3_nav.yaml \
#   CARTER_MAP_YAML=$HOME/Digital-Twin-Project/warehouse/maps/some_other_carter.yaml \
#   ./tools/run_ablation_repeats.sh ...
X3_MAP_YAML="${X3_MAP_YAML:-$WAREHOUSE/maps/warehouse_x3_nav.yaml}"
CARTER_MAP_YAML="${CARTER_MAP_YAML:-$WAREHOUSE/maps/warehouse_gt_carter.yaml}"
# X3's spawn pose in the stage/map frame -- MUST match the actual prim
# transform in SCENE_USD (warehouse/x3_sim/build_x3_graphs.py's X3_POSITION
# for a rebuilt-graphs stage, or wherever the prim was placed by hand for a
# pre-baked one) AND risk_perception/config/amcl_sim.yaml's initial_pose --
# all three are independent copies of the same number with no shared source
# of truth. A stale spawn here doesn't fail loudly: AMCL just localizes the
# robot onto whatever's at the OLD coordinates, /scan and the local costmap
# read as if still spawned there, and every subsequent Nav2 goal fails to
# plan. Hit this once already switching scenes -- if you point SCENE_USD at
# a different stage, update X3_SPAWN_X/Y AND amcl_sim.yaml's initial_pose
# together, not just one of them.
X3_SPAWN_X="${X3_SPAWN_X:-0.38}"
X3_SPAWN_Y="${X3_SPAWN_Y:-0.07}"
# Extra launch args appended to EVERY arm, whitespace-separated. Use this to
# pin the study-run contract rather than editing the launch defaults:
#
#   EXTRA_FLAGS="enable_spatial_prior:=true spatial_prior_learn_rate:=0.0 \
#                spatial_prior_autosave_sec:=1e9" ./tools/run_ablation_repeats.sh ...
#
# spatial_prior_learn_rate:=0.0 is what freezes the lane prior: it stops
# deposits, stops decay (decay_factor returns 1.0 when frozen) and stops the
# autosave from overwriting the warm-up file. Without it a study run learns
# from its own tracks and every later arm reads a different prior than the
# first one did.
read -r -a EXTRA_FLAGS <<< "${EXTRA_FLAGS:-}"
DOMAIN=62
ISAAC_READY_TIMEOUT=90
NAV2_SETTLE_S=5
BAG_TOPICS=(/tf /tf_static /odom /risk_perception/world_objects
            /risk_perception/detections_2d /risk_costmap_reference /plan
            /cmd_vel /local_costmap/costmap /scan)
# /scan added 2026-09-13: Complex_Scene_ForkliftNavMesh.usd's first warm-up
# repeat had a completely empty local_costmap (0% lethal) for the entire
# ~12 min run despite 1.4 m measured wall clearance at spawn -- amcl_pose
# updates 55 s apart, controller_server rejecting every DWB trajectory,
# planner producing plans the local costmap never contained a single point
# of. Root cause not yet confirmed because /scan wasn't in the bag to check
# post-hoc; strong suspicion is the scene's new forklift/worker NPC prims
# (added via Replicator Agent) breaking the lidar raycast somehow. Check
# live next run: `ros2 topic echo /scan --field ranges` -- all-inf/all-max
# confirms the sensor itself, a normal-looking spread points elsewhere.

ARM=${1:?"usage: $0 <baseline|reactive|predictive> <start_rep> <end_rep>"}
START_REP=${2:?"usage: $0 <baseline|reactive|predictive> <start_rep> <end_rep>"}
END_REP=${3:?"usage: $0 <baseline|reactive|predictive> <start_rep> <end_rep>"}

# --- Arms --------------------------------------------------------------
# Three SYSTEM arms (which costmap, if any, Nav2 plans against) and a set of
# LEAVE-ONE-OUT arms, each of which is the full predictive system with
# exactly ONE mechanism of Sec. IV switched off via panoptex_sim.launch.py's
# ablation args (see _ABLATION_ARGS there).
#
# Why leave-one-out and not cumulative-add: with cumulative rows you cannot
# tell whether a mechanism matters GIVEN the others, only whether it matters
# in the order you happened to add it. Each arm below answers "what does the
# full system lose without this one thing", which is the question "four
# priors, not one" actually needs answered.
#
# Why these are NOT crossed with baseline/reactive: risk_costmap_node (the
# reactive arm) consumes no rollout, no encounter term, no relation bonus and
# no flow, and the baseline arm consumes no risk layer at all -- crossing the
# knobs with either produces bit-identical runs under different labels. The
# ablation is predictive-only, by construction.
case "$ARM" in
    # --- system arms ---
    baseline)      NAV2_FLAGS=(risk_layer_enabled:=false) ;;
    reactive)      NAV2_FLAGS=(risk_topic:=/risk_costmap) ;;
    predictive)    NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive) ;;

    # --- leave-one-out arms (all predictive) ---
    # Behavioral prior: pmot forced to 0, so nothing paints a moving
    # hypothesis and the two-hypothesis mixture collapses to stationary.
    no_behavioral) NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               use_motion_mixture:=false) ;;
    # Semantic prior magnitude: the class consequence table stops
    # contributing and every track paints the class-agnostic value.
    no_semantic)   NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               semantic_modifier_enabled:=false) ;;
    # Relation prior: relbonus no longer reaches severity (still logged).
    no_relation)   NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               use_relation_bonus:=false) ;;
    # Spatial-Flow prior: BOTH consumptions off -- the additive floor and
    # the rollout blend. Off together because either alone leaves the prior
    # half-connected, which is an arm nobody asked about.
    no_flow)       NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               spatial_prior_weight:=0.0
                               flow_blend_weight:=0.0) ;;
    # Encounter geometry: enc pinned to 1.0, so risk stops depending on the
    # robot's own course at all.
    no_encounter)  NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               enable_relative_motion:=false) ;;
    # Behavioral + Encounter, COMBINED (2026-09-14, User A's call): these two
    # are not cleanly separable on their own -- cpa_geometry() is evaluated
    # twice per track (factor_stat with v_obj=0, factor_mov with the tracked
    # velocity) and blended by the SAME pmot the Behavioral two-hypothesis
    # mixture produces, so use_motion_mixture:=false alone already collapses
    # eff_pmot to 0 and kills the only branch that uses factor_mov -- i.e.
    # plain no_behavioral is secretly (Behavioral + part of Encounter)
    # already. Rather than report a misleadingly "pure" -Behavioral row next
    # to a -Encounter row that doesn't reciprocally touch it, ablate both
    # together as one row and report the combined effect honestly instead of
    # pretending they decompose.
    no_behavioral_encounter) NAV2_FLAGS=(risk_topic:=/risk_costmap_predictive
                               use_motion_mixture:=false
                               enable_relative_motion:=false) ;;

    *) echo "unknown arm: $ARM"
       echo "  system:        baseline | reactive | predictive"
       echo "  leave-one-out: no_behavioral_encounter | no_semantic | no_relation | no_flow | no_encounter"
       echo "                 (no_behavioral/no_encounter individually still work, just not"
       echo "                 recommended together -- see no_behavioral_encounter's comment)"
       exit 1 ;;
esac

# NOT arms, but still reachable as launch args (panoptex_sim.launch.py's
# _ABLATION_ARGS) for a one-off probe without adding a row to the table:
#   two_hypothesis_cpa:=false    one shared CPA reading for both hypotheses
#   stack_moving_pmov_min:=0.0   pmov gate off, pmot alone guards the mover
#   use_class_consequence:=false every class weighs 1.0
# e.g.  EXTRA_FLAGS="two_hypothesis_cpa:=false" ./tools/run_ablation_repeats.sh predictive 9 9

export ROS_DOMAIN_ID=$DOMAIN
mkdir -p "$PANOPTEX/results"

echo "=== precondition check: required packages on this shell's path ==="
for pkg in risk_perception yahboomcar_nav carter_navigation; do
    if ! ros2 pkg prefix "$pkg" > /dev/null 2>&1; then
        echo "FATAL: package '$pkg' not found on AMENT_PREFIX_PATH."
        echo "  carter_navigation missing usually means env_sim.sh was not"
        echo "  sourced in THIS shell before running this script -- run:"
        echo "  source $WAREHOUSE/env_sim.sh"
        echo "  then re-run this script from the same shell."
        exit 1
    fi
done
echo "  ok: risk_perception, yahboomcar_nav, carter_navigation all resolve"

if [[ ! -x "$ISAACSIM_PYTHON" ]]; then
    echo "FATAL: ISAACSIM_PYTHON=$ISAACSIM_PYTHON does not exist or isn't executable."
    echo "  Set it explicitly if Isaac Sim lives somewhere other than \$HOME/isaacsim:"
    echo "  ISAACSIM_PYTHON=/path/to/python.sh $0 $*"
    exit 1
fi
if [[ ! -f "$SCENE_USD" ]]; then
    echo "FATAL: SCENE_USD=$SCENE_USD does not exist."
    exit 1
fi
echo "  ok: scene = $SCENE_USD"
if [[ ! -f "$X3_MAP_YAML" ]]; then
    echo "FATAL: X3_MAP_YAML=$X3_MAP_YAML does not exist."
    exit 1
fi
if [[ ! -f "$CARTER_MAP_YAML" ]]; then
    echo "FATAL: CARTER_MAP_YAML=$CARTER_MAP_YAML does not exist."
    exit 1
fi
echo "  ok: x3 map = $X3_MAP_YAML"
echo "  ok: carter map = $CARTER_MAP_YAML"
echo "  ok: ISAACSIM_PYTHON=$ISAACSIM_PYTHON exists"

PIDS_TO_KILL=()

    # `setsid ros2 launch ... &` makes the launch process its own session/
    # group leader, but `ros2 launch` itself puts EACH node it spawns into
    # ITS OWN process group (so it can control signal propagation itself) --
    # verified empirically: `kill -TERM -$pid` was only killing the launch
    # orchestrator, leaving every actual node process (risk_costmap_node,
    # spatial_prior_node, gdino/sam2, amcl, ...) as an orphan that survived
    # into the next repeat. 11 risk_costmap_node + 10 spatial_prior_node
    # zombies accumulated this way across repeated attempts, exhausting DDS
    # shared-memory ports (RTPS_TRANSPORT_SHM Error) and leaving /amcl
    # "active" but never actually publishing /amcl_pose. Fix: walk the real
    # PPID tree (unaffected by process-group tricks) and kill every
    # descendant directly, not the process group.
collect_descendants() {
    local pid=$1
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        collect_descendants "$child"
        echo "$child"
    done
}

kill_tree() {
    local root=$1 sig=$2
    local pid
    for pid in $(collect_descendants "$root"); do
        kill "-$sig" "$pid" 2>/dev/null || true
    done
    kill "-$sig" "$root" 2>/dev/null || true
}

# Deny-list, not allow-list: an enumerated list of node executable names
# kept growing every time a repeat left behind something new (carter_loop,
# global_cam_projector_*, static_tf_global_cam_*, scan_cluster_detector,
# coverage_mask_node, base_link_to_laser, camera_link_to_camera_*,
# run_trials.py's own basic_navigator/run_trials_listener,
# rosbag2_recorder -- all missed by earlier versions of this function and
# left running, one of them stuck forever waiting on /amcl_pose from AMCL
# this same sweep had already killed). Scan every process this user owns
# instead, kill anything whose command line matches this stack's tools,
# and protect ONLY the two nodes that must survive every repeat plus this
# script itself.
sweep_repeat_processes() {
    local pid cmdline
    for pid in $(ps -u "$USER" -o pid= 2>/dev/null); do
        cmdline=$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" || true)
        [[ -z "$cmdline" ]] && continue
        # tf2_ros/robot_state_publisher/static_transform_publisher: raw
        # executables x3_sim_bringup_launch.py spawns whose own argv carries
        # no "yahboomcar"/"ros2"-recognizable string at all (only their own
        # tf2_ros/robot_state_publisher binary name + frame-id args) --
        # these survived every earlier version of this sweep.
        [[ "$cmdline" =~ (ros2|isaac|risk_perception|yahboomcar|carter|amcl|nav2|gdino|sam2|global_cam|run_trials|rosbag2|tf2_ros|robot_state_publisher|static_transform_publisher) ]] || continue
        [[ "$cmdline" == *reference* ]] && continue
        [[ "$cmdline" == *evaluation_node* ]] && continue
        [[ "$cmdline" == *run_ablation_repeats.sh* ]] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
}

cleanup_repeat() {
    for pid in "${PIDS_TO_KILL[@]}"; do kill_tree "$pid" TERM; done
    sleep 3
    for pid in "${PIDS_TO_KILL[@]}"; do kill_tree "$pid" KILL; done
    PIDS_TO_KILL=()
    sweep_repeat_processes
    sleep 2
}

cleanup_all() {
    # evaluation_reference.launch.py is launched per-repeat now (its PID is
    # in PIDS_TO_KILL like everything else) -- no separate REF_PID to track.
    cleanup_repeat
}
trap cleanup_all EXIT

wait_for_topic() {
    local topic=$1 timeout=$2 waited=0
    until ros2 topic list 2>/dev/null | grep -qx "$topic"; do
        sleep 2; waited=$((waited + 2))
        if (( waited >= timeout )); then
            echo "  TIMEOUT waiting for $topic after ${timeout}s"
            return 1
        fi
    done
}

# /scan EXISTING (wait_for_topic) is not the same as /scan being USABLE.
# 2026-09-13, deep into the first full overnight 8-arm run: 3 of the first
# 12 reps (reactive_rep2, predictive_rep3, no_behavioral_rep3 -- different
# arms, different rep positions, same SCENE_USD that worked fine in every
# OTHER rep) came back with local_costmap 100% empty (0% lethal) for their
# ENTIRE ~10+ min duration -- the exact signature the NavMeshVolume
# Purpose=guide fix was supposed to have killed for good. Since the SAME
# saved .usd worked in the other reps, this isn't the file regressing --
# it's Isaac's physics/collision registration occasionally not finishing
# before the lidar starts raycasting on a fresh headless launch (more
# likely under the load this scene carries deep into a multi-hour run).
# Sample one /scan message and reject it if too many ranges are the RTX
# lidar's -1.0 invalid sentinel -- same check done by hand that found the
# original NavMeshVolume bug.
check_scan_valid() {
    local tmp
    tmp=$(mktemp)
    timeout 10 ros2 topic echo /scan --field ranges --once > "$tmp" 2>/dev/null
    python3 -c "
import re, sys
text = open('$tmp').read()
vals = [float(v) for v in re.findall(r'-?\d+\.\d+', text)]
if len(vals) < 10:
    sys.exit(1)
invalid = sum(1 for v in vals if v < 0)
sys.exit(0 if (invalid / len(vals)) < 0.5 else 1)
"
    local rc=$?
    rm -f "$tmp"
    return $rc
}

echo "=== pre-flight: killing any leftover evaluation_reference/evaluation_node ==="
echo "  (sweep_repeat_processes protects evaluation_node so it survives BETWEEN"
echo "  reps of THIS invocation -- but at this point, before THIS invocation has"
echo "  launched its own copy below, any evaluation_node found here can only be"
echo "  a leftover from an earlier invocation or a manual 'ros2 launch' that never"
echo "  got cleaned up. It shares this invocation's /evaluation/trial_control and"
echo "  /evaluation/trial_result topics and races with the fresh one launched"
echo "  below, silently corrupting run_trials.py's recorded trial labels --"
echo "  root-caused 2026-09-14 after hours of results with doubled/missing"
echo "  waypoint labels despite every on-disk trials_x3_tour.json snapshot"
echo "  checking out correct. Kill unconditionally here; nothing legitimate"
echo "  should be running yet.)"
for _pid in $(ps -u "$USER" -o pid= 2>/dev/null); do
    _cmdline=$(tr '\0' ' ' 2>/dev/null < "/proc/$_pid/cmdline" || true)
    [[ "$_cmdline" == *evaluation_reference* || "$_cmdline" == *evaluation_node* \
       || "$_cmdline" == *predictive_risk_costmap_node_reference* ]] || continue
    echo "  killing leftover pid $_pid: $_cmdline"
    kill -9 "$_pid" 2>/dev/null || true
done
sleep 2

echo "=== clean-slate check: killing any pre-existing per-repeat processes ==="
echo "  (a leftover from an earlier manual run or an interrupted script run"
echo "  collides with the first repeat's fresh nodes -- same node name,"
echo "  same symptom as the orphaned-process trap in the launch layer)"
sweep_repeat_processes
sleep 2
DUPES=$(ros2 node list 2>/dev/null | sort | uniq -d || true)
if [[ -n "$DUPES" ]]; then
    echo "  FATAL: still duplicate node names after the sweep:"
    echo "$DUPES"
    echo "  something outside this script's reach is holding them up"
    echo "  (check other terminals/sessions). Resolve manually, then re-run."
    exit 1
fi
echo "  clean"

for REP in $(seq "$START_REP" "$END_REP"); do
    LABEL="${SCENE_TAG:+${SCENE_TAG}_}${ARM}_rep${REP}"
    OUTDIR="$PANOPTEX/results/$LABEL"
    BAGDIR="$PANOPTEX/results/${LABEL}_bag"

    if [[ -e "$OUTDIR" || -e "$BAGDIR" ]]; then
        echo ""
        echo "SKIPPING $LABEL -- $OUTDIR or $BAGDIR already exists (delete it first to redo this rep)"
        continue
    fi

    # --- generic stuck-rep watchdog (2026-09-14) ---------------------------
    # Point-fixing each hang as it's discovered (run_trials.py's own AMCL
    # wait, the /initialpose topic pub) is whack-a-mole -- baseline_rep6
    # alone hit TWO different unbounded waits in one session (18 min on
    # amcl/get_state, then 27 min on `ros2 topic pub -1` after that was
    # fixed). This is the blanket net for whatever hangs next, known or
    # not: if every log file this rep writes to goes quiet for
    # STUCK_LIMIT_S straight, kill everything for this rep and let the
    # outer loop move on to the next repeat -- a genuinely working rep
    # (7-8 min typical, longer with Isaac retries) keeps writing SOMETHING
    # well inside that window; only a truly stuck process goes silent this
    # long. Cheap insurance: a real 5-minute-long silent stall is never a
    # rep worth waiting out.
    STUCK_LIMIT_S=300
    (
        while true; do
            sleep 15
            newest=0
            for f in "$PANOPTEX/results/${LABEL}"*.log; do
                [[ -e "$f" ]] || continue
                mt=$(stat -c %Y "$f" 2>/dev/null || echo 0)
                (( mt > newest )) && newest=$mt
            done
            (( newest == 0 )) && continue   # no log written yet, still booting
            now=$(date +%s)
            if (( now - newest > STUCK_LIMIT_S )); then
                echo "  WATCHDOG: $LABEL has been silent ${STUCK_LIMIT_S}s+ -- forcing this rep to abandon"
                pkill -9 -f "run_trials.py .*--condition-prefix $LABEL " 2>/dev/null || true
                pkill -9 -f "ros2 topic pub -1 /initialpose" 2>/dev/null || true
                break
            fi
        done
    ) &
    REP_WATCHDOG_PID=$!

    echo ""
    echo "=================================================================="
    echo "=== $LABEL : fresh headless Isaac ==="
    echo "=================================================================="

    ISAAC_OK=false
    for ISAAC_ATTEMPT in 1 2 3; do
        echo "  --- Isaac attempt $ISAAC_ATTEMPT/3 ---"
        setsid "$WAREHOUSE/x3_sim/isaac.sh" "$WAREHOUSE/x3_sim/run_headless.py" --scene "$SCENE_USD" \
            > "$PANOPTEX/results/${LABEL}_isaac.log" 2>&1 &
        PIDS_TO_KILL+=($!)

        echo "  waiting up to ${ISAAC_READY_TIMEOUT}s for /clock ..."
        if ! wait_for_topic /clock "$ISAAC_READY_TIMEOUT"; then
            echo "  Isaac never came up, see ${LABEL}_isaac.log"
            cleanup_repeat
            continue
        fi
        # /clock alone isn't proof THIS repeat's Isaac is actually simulating
        # -- a stale/broken Isaac session from earlier can keep ticking
        # /clock with a dead sensor pipeline behind it (exactly what stalled
        # AMCL forever: isaac.sh was failing 6/6 with "No such file or
        # directory" every repeat, yet /clock kept passing because of an old
        # leftover session). /scan only exists once Isaac's lidar bridge is
        # actually alive.
        echo "  waiting up to ${ISAAC_READY_TIMEOUT}s for /scan (proves Isaac is actually simulating, not just ticking a stale /clock) ..."
        if ! wait_for_topic /scan "$ISAAC_READY_TIMEOUT"; then
            echo "  /scan never appeared, see ${LABEL}_isaac.log"
            cleanup_repeat
            continue
        fi
        sleep 5   # let physics/OmniGraphs settle past the first few ticks

        echo "  checking /scan content (not just existence) ..."
        if ! check_scan_valid; then
            echo "  /scan mostly invalid (-1.0 sentinel) -- physics/collider registration"
            echo "  likely didn't finish before the lidar started; retrying with a fresh Isaac"
            cleanup_repeat
            continue
        fi
        echo "  /scan looks real -- proceeding"
        ISAAC_OK=true
        break
    done
    if ! $ISAAC_OK; then
        echo "  Isaac failed all 3 attempts for $LABEL -- skipping this rep"
        kill "$REP_WATCHDOG_PID" 2>/dev/null || true
        wait "$REP_WATCHDOG_PID" 2>/dev/null || true
        continue
    fi

    echo "=== evaluation_reference.launch.py (fresh per repeat) ==="
    # MUST be launched fresh per repeat, not once per arm invocation: it
    # holds a live TF2 buffer, and Isaac Sim's /clock resets to ~0 on every
    # repeat's fresh headless relaunch above. A buffer that survived a prior
    # repeat has high-water timestamps from that repeat's later sim-time --
    # the new repeat's low timestamps then look like they go backwards, and
    # tf2 silently drops them ("TF_OLD_DATA ignoring data from the past"),
    # which starves evaluation_node's lookup_transform(map, base_footprint)
    # for the entire repeat: path_length_m, velocity_smoothness and
    # stopped_time_s all silently read as 0/full-duration-stopped despite
    # every Nav2 leg genuinely SUCCEEDED. Root-caused 2026-09-14 after
    # rep2 of both baseline and predictive read path_length_m=0.0 on
    # all 4 legs while rep1 (first repeat after a fresh evaluation_node)
    # was clean. Reusing one evaluation_node across an arm's repeats was
    # meant to give risk_exposure() a common "always full-system" yardstick
    # (see the launch file's docstring) -- restarting it per repeat keeps
    # that (same code, same forced-on config, independent of this arm's
    # ablation flags) without carrying a stale TF2 buffer across a clock
    # reset.
    setsid ros2 launch risk_perception evaluation_reference.launch.py \
        > "$PANOPTEX/results/${LABEL}_eval_reference.log" 2>&1 &
    PIDS_TO_KILL+=($!)
    sleep 3

    echo "=== x3 ROS bringup ==="
    setsid ros2 launch yahboomcar_nav x3_sim_bringup_launch.py \
        > "$PANOPTEX/results/${LABEL}_x3bringup.log" 2>&1 &
    PIDS_TO_KILL+=($!)
    sleep 3

    echo "=== carters patrol ==="
    setsid ros2 launch "$WAREHOUSE/launch/carters_patrol.launch.py" \
        map:="$CARTER_MAP_YAML" \
        > "$PANOPTEX/results/${LABEL}_carters.log" 2>&1 &
    PIDS_TO_KILL+=($!)
    sleep 5

    echo "=== panoptex_sim.launch.py ($ARM: ${NAV2_FLAGS[*]} ${EXTRA_FLAGS[*]}) ==="
    setsid ros2 launch risk_perception panoptex_sim.launch.py \
        enable_nav2:=true enable_rviz:=false "${NAV2_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
        map_yaml:="$X3_MAP_YAML" \
        prior_log_dir:="$HOME/.panoptex/logs/$LABEL" \
        > "$PANOPTEX/results/${LABEL}_panoptex.log" 2>&1 &
    PIDS_TO_KILL+=($!)
    sleep "$NAV2_SETTLE_S"

    echo "=== waiting for /amcl node before publishing initial pose ==="
    AMCL_WAIT=0
    until timeout 10 ros2 node list 2>/dev/null | grep -qx /amcl; do
        sleep 2; AMCL_WAIT=$((AMCL_WAIT + 2))
        if (( AMCL_WAIT >= 60 )); then
            echo "  /amcl never appeared after 60s -- publishing initial pose anyway, likely to be lost"
            break
        fi
    done
    sleep 2   # node up != subscriber callback group spun up yet -- small extra margin

    echo "=== resetting AMCL initial pose to X3 spawn ($X3_SPAWN_X, $X3_SPAWN_Y) -- publishing 5x, 1s apart ==="
    # A single -1 publish races /initialpose's subscriber callback coming up
    # (no transient-local QoS on that topic) and can be silently lost --
    # this is what stalled BasicNavigator's own _waitForInitialPose() retry
    # loop in the run this script was built to fix. Repeating costs nothing:
    # AMCL treats every /initialpose message as a full reset regardless of
    # how many arrive.
    #
    # `timeout 5` wraps EACH publish (2026-09-14): plain `ros2 topic pub -1`
    # blocks until it sees a matching subscriber before it sends anything
    # and exits -- if that match never happens (subscriber QoS mismatch, a
    # slow-to-spin-up AMCL, whatever) it hangs FOREVER with no deadline at
    # all. This is what stalled baseline_rep6 for 27+ minutes doing
    # nothing: the whole rep, including run_trials.py's own bounded-wait
    # fix, never even got a chance to run because THIS loop never returned.
    # A skipped publish here is cheap (4 more follow in the same loop, and
    # AMCL treats every one as a full reset) -- silently moving on beats
    # hanging the entire rep.
    for _ in 1 2 3 4 5; do
        if ! timeout 5 ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
            "{header: {frame_id: \"map\"}, pose: {pose: {position: {x: $X3_SPAWN_X, y: $X3_SPAWN_Y, z: 0.0}, orientation: {w: 1.0}}}}" \
            > /dev/null 2>&1; then
            echo "  WARN: one /initialpose publish timed out after 5s -- continuing (4 more attempts follow)"
        fi
        sleep 1
    done

    echo "=== bag recording -> $BAGDIR ==="
    setsid ros2 bag record -o "$BAGDIR" "${BAG_TOPICS[@]}" \
        > "$PANOPTEX/results/${LABEL}_bag.log" 2>&1 &
    BAG_PID=$!
    PIDS_TO_KILL+=("$BAG_PID")
    sleep 3

    # Audit trail for the 2026-09-13 corruption (see run_trials.py's own
    # duplicate-label check): snapshot exactly what's on disk RIGHT NOW,
    # immediately before run_trials.py reads it, so a future mismatch is
    # provable after the fact instead of only inferred from label patterns
    # in _all_trials.json days later.
    cp "$PANOPTEX/trials_x3_tour.json" "$PANOPTEX/results/${LABEL}_trials_used.json"

    echo "=== run_trials.py : $LABEL ==="
    # Teed to its own log (not just this script's combined stdout, whose
    # path this script has no handle on) so the watchdog above can see
    # this phase's activity too -- pipefail (set at the top of this file)
    # keeps `if` checking python's own exit code, not tee's.
    if python3 "$PANOPTEX/tools/run_trials.py" "$PANOPTEX/trials_x3_tour.json" \
        --condition-prefix "$LABEL" --out-dir "$OUTDIR" --ros-domain-id "$DOMAIN" \
        2>&1 | tee "$PANOPTEX/results/${LABEL}_run_trials.log"; then
        echo "  $LABEL tour finished"
    else
        echo "  $LABEL tour FAILED -- check ${LABEL}_panoptex.log and $OUTDIR/_all_trials.json"
    fi

    kill "$REP_WATCHDOG_PID" 2>/dev/null || true
    wait "$REP_WATCHDOG_PID" 2>/dev/null || true

    echo "=== stopping bag + this repeat's processes ==="
    kill_tree "$BAG_PID" TERM
    sleep 3
    cleanup_repeat

    echo "=== $LABEL done, settling before next repeat ==="
    sleep 5
done

echo ""
echo "all requested repeats of $ARM done."
echo "evaluation_reference.launch.py will be stopped automatically on exit."
echo "aggregate with:"
echo "  python3 tools/aggregate_trials.py \\"
for REP in $(seq "$START_REP" "$END_REP"); do
    REP_LABEL="${SCENE_TAG:+${SCENE_TAG}_}${ARM}_rep${REP}"
    echo "    --all-trials-json results/${REP_LABEL}/_all_trials.json \\"
done
for REP in $(seq "$START_REP" "$END_REP"); do
    REP_LABEL="${SCENE_TAG:+${SCENE_TAG}_}${ARM}_rep${REP}"
    echo "    --isaac-log results/${REP_LABEL}_isaac.log \\"
done
echo "    --label ${SCENE_TAG:+${SCENE_TAG}_}$ARM --out results/${SCENE_TAG:+${SCENE_TAG}_}${ARM}_aggregate.json"
echo "node compute-time breakdown with:"
echo "  python3 tools/node_latency_report.py \\"
for REP in $(seq "$START_REP" "$END_REP"); do
    REP_LABEL="${SCENE_TAG:+${SCENE_TAG}_}${ARM}_rep${REP}"
    echo "    --log-dir \$HOME/.panoptex/logs/${REP_LABEL} \\"
done
echo "    --label ${SCENE_TAG:+${SCENE_TAG}_}$ARM --out results/${SCENE_TAG:+${SCENE_TAG}_}${ARM}_latency.json"
