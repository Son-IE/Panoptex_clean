#!/usr/bin/env bash
# warmup_lanes.sh -- WP-C lane warm-up: run x3_nav.launch.py learn_lanes:=true
# against Isaac Sim + Carter patrol traffic for a while so
# spatial_prior_node's S channel actually learns the two AMR lanes before
# any study run relies on lane_layer, then report S-channel coverage along
# both lanes from the saved npz.
#
# Modelled on results/wiring_check/run_arm.sh (start_bg/cleanup/wait_for_topic
# helpers, same bring-up order: Isaac headless -> wait for /clock -> Carter
# patrol -> X3 bringup -> the panoptex_nav launch under test).
#
# Usage:
#   tools/warmup_lanes.sh [sim_seconds=600] [spatial_prior_path]
#
# spatial_prior_path defaults to ~/.panoptex/spatial_prior_sim.npz (the same
# default x3_nav.launch.py's own spatial_prior_path arg uses) -- NEVER the
# lab's ~/.panoptex/spatial_prior.npz.
#
# Does NOT run Isaac itself when sourced/executed by an agent that doesn't
# own the GPU -- see the orchestrator note below. This script is meant to be
# run BY the orchestrator, which reserves Isaac Sim; WP-C's own job was to
# get this script bash -n clean and its npz-reporting logic dry-run
# verified against a synthetic .npz, not to actually execute a sim session.
set -o pipefail

SIM_SECONDS="${1:-600}"
SPATIAL_PRIOR_PATH="${2:-$HOME/.panoptex/spatial_prior_sim.npz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"           # .../Panoptex
WH="$HOME/workspace/warehouse"
HERE="${PANOPTEX_RUNS_DIR:-$HOME/panoptex_runs}/warmup"
LOGDIR="$HERE/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
PIDS_FILE="$LOGDIR/pids.txt"; : > "$PIDS_FILE"

log() { echo "[warmup_lanes $(date +%H:%M:%S)] $*"; }

start_bg() {
    # Same pidfile-tracking pattern as run_arm.sh's start_bg: run detached
    # under setsid so `cleanup` below can signal the whole process group,
    # not just the immediate child (ros2 launch spawns a tree).
    local name="$1"; shift
    local pidfile="$LOGDIR/$name.pid" logfile="$LOGDIR/$name.log"
    rm -f "$pidfile"
    setsid bash -c "echo \$\$ > '$pidfile'; exec \"\$@\"" -- "$@" > "$logfile" 2>&1 &
    disown
    local t0=$(date +%s)
    while [ ! -s "$pidfile" ]; do sleep 0.2; [ $(( $(date +%s) - t0 )) -ge 10 ] && break; done
    [ -s "$pidfile" ] && { echo "$(cat "$pidfile")" >> "$PIDS_FILE"; log "$name pid=$(cat "$pidfile")"; } || log "WARNING: $name no pidfile"
}

cleanup() {
    log "cleanup"
    while read -r pid; do [ -n "$pid" ] && kill -TERM "-$pid" 2>/dev/null; done < "$PIDS_FILE"
    sleep 8
    while read -r pid; do [ -n "$pid" ] && kill -KILL "-$pid" 2>/dev/null; done < "$PIDS_FILE"
    # Only reap an Isaac THIS script started (its pidfile exists); a blanket
    # pkill here killed another run's Isaac on 2026-09-09 when this script
    # aborted at the "Isaac already running" guard.
    if [ -s "$LOGDIR/isaac.pid" ] || [ -s "$LOGDIR/isaac_retry.pid" ]; then pkill -f run_headless.py 2>/dev/null; sleep 2; fi
    log "cleanup done; remaining: $(pgrep -af 'run_headless|ros2 launch|component_container|probe_monitor' | grep -v pgrep | wc -l)"
}
trap cleanup EXIT

wait_for_topic() {
    local t0=$(date +%s)
    while ! timeout 10 ros2 topic list 2>/dev/null | grep -qx "$1"; do
        sleep 2
        [ $(( $(date +%s) - t0 )) -ge "$2" ] && return 1
    done
    return 0
}

# Kill just one start_bg-tracked process group (same TERM-then-KILL sequence
# cleanup() above uses), without tearing down the whole run -- used by
# wait_carters_active's relaunch-once retry.
kill_pidfile() {
    local pidfile="$1"
    [ -s "$pidfile" ] || return 0
    local pid; pid=$(cat "$pidfile")
    kill -TERM "-$pid" 2>/dev/null
    sleep 8
    kill -KILL "-$pid" 2>/dev/null
}

# 2026-09-09: carter1's lifecycle_manager_navigation was observed hung
# forever at "Waiting for service controller_server/get_state..." under
# Isaac + GDINO/SAM2 startup load, so /carter1/follow_waypoints never
# appeared and carter1 sat parked while carter2 patrolled normally -- a
# transient bring-up race, not a config bug. Poll until both Carters' nav2
# stacks are actually active before trusting carters_patrol is up (matters
# here too: a parked carter1 means the traffic being learned is half
# missing, silently degrading the warm-up).
CARTER_ACTIVATE_TIMEOUT="${CARTER_ACTIVATE_TIMEOUT:-180}"
wait_carters_active() {
    local t0=$(date +%s) elapsed al
    while :; do
        al="$(timeout 10 ros2 action list 2>/dev/null)"
        if echo "$al" | grep -qx "/carter1/follow_waypoints" \
            && echo "$al" | grep -qx "/carter2/follow_waypoints" \
            && grep -q "carter1.*lifecycle_manager_navigation.*Managed nodes are active" "$LOGDIR/carters_patrol.log" 2>/dev/null \
            && grep -q "carter2.*lifecycle_manager_navigation.*Managed nodes are active" "$LOGDIR/carters_patrol.log" 2>/dev/null; then
            elapsed=$(( $(date +%s) - t0 ))
            log "carters active after ${elapsed}s"
            return 0
        fi
        elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -ge "$CARTER_ACTIVATE_TIMEOUT" ]; then
            log "carters not active after ${elapsed}s (timeout ${CARTER_ACTIVATE_TIMEOUT}s)"
            return 1
        fi
        sleep 5
    done
}

# Measures elapsed SIM time (not wall time) off /clock, so SIM_SECONDS means
# what it says under Isaac's own real-time factor rather than however long
# wall-clock happens to take. Reads one /clock message, in nanoseconds, via
# `ros2 topic echo --once` -- no extra python dependency needed for this
# one-shot poll.
clock_sim_seconds() {
    # First `sec:` line of one /clock message, integer seconds only. The
    # `--once` echo also prints a trailing "---" separator, so grab just the
    # digits (2026-09-09: "4\n---" leaked into $(( )) and aborted the run).
    timeout 5 ros2 topic echo --once /clock 2>/dev/null \
        | grep -m1 -E '^\s*sec:' | grep -o -E '[0-9]+' | head -n1
}

wait_sim_seconds() {
    local target="$1"
    local t0_sim
    t0_sim="$(clock_sim_seconds)"
    if [ -z "$t0_sim" ]; then
        log "WARNING: could not read /clock; falling back to wall-clock sleep for ${target}s"
        sleep "$target"
        return
    fi
    log "sim t0=${t0_sim}s, waiting for +${target}s sim time"
    while :; do
        sleep 5
        local now_sim
        now_sim="$(clock_sim_seconds)"
        [ -z "$now_sim" ] && continue
        local elapsed=$(( now_sim - t0_sim ))
        log "sim elapsed=${elapsed}s / ${target}s"
        [ "$elapsed" -ge "$target" ] && break
    done
}

report_lane_coverage() {
    # Sample the SAVED npz's aggregate S (element-wise max across
    # categories -- same combine rule as spatial_prior_node._publish(), so
    # this matches what lane_layer actually sees on
    # /risk_perception/spatial_prior) along both lanes, y=-4..7 every 0.5 m.
    # Prefers tools/spatial_flow_heatmap.py's load() (it already knows the
    # npz schema -- resolution/origin_x/origin_y/width/height/categories/
    # s__<cat>/fx__<cat>/fy__<cat>, see that file and spatial_prior_node.py's
    # module docstring); falls back to reading the npz directly if that
    # import fails for any reason (e.g. run outside the conda env).
    local npz_path="$1"
    python3 - "$npz_path" "$REPO_ROOT" << 'PYEOF'
import sys
import numpy as np

npz_path, repo_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root + "/tools")

try:
    from spatial_flow_heatmap import load
    geom, grids = load(npz_path)
except SystemExit as exc:
    print(f"spatial_flow_heatmap.load: {exc}")
    sys.exit(1)
except Exception as exc:      # noqa: BLE001 -- fall back below
    print(f"spatial_flow_heatmap.load unavailable ({exc}); reading npz directly")
    data = np.load(npz_path)
    categories = [str(c) for c in data["categories"]]
    geom = dict(resolution=float(data["resolution"]), origin_x=float(data["origin_x"]),
                origin_y=float(data["origin_y"]), width=int(data["width"]),
                height=int(data["height"]))
    grids = {c: dict(s=np.asarray(data[f"s__{c}"])) for c in categories}

res, ox, oy = geom["resolution"], geom["origin_x"], geom["origin_y"]
W, H = geom["width"], geom["height"]

# Aggregate S: element-wise max across every category's grid, identical to
# spatial_prior_node._publish()'s combine rule for /risk_perception/spatial_prior
# (the topic lane_layer actually reads).
S = np.zeros((H, W), dtype=np.float32)
for g in grids.values():
    np.maximum(S, g["s"], out=S)


def s_band(x, y, half_band_m=0.30):
    """Max S within +/- half_band_m of the nominal lane line at this y --
    the Carters wander +/-0.17 m about the line and the tracker adds its own
    offset, so a single-cell probe on the line reads ~0 even when the lane
    IS learned (2026-09-09: reported 0.00 with 330 cells > 0.05 in the band).
    Same row/col convention as deposit_into_grid (row grows with y)."""
    row = int(round((y - oy) / res))
    c0 = int(round((x - half_band_m - ox) / res))
    c1 = int(round((x + half_band_m - ox) / res))
    if not (0 <= row < H):
        return float("nan")
    c0, c1 = max(0, c0), min(W - 1, c1)
    return float(S[row, c0:c1 + 1].max()) if c1 >= c0 else float("nan")


for label, lane_x in (("carter1 lane (x=1.32)", 1.32), ("carter2 west leg (x=2.15)", 2.15)):
    print(f"\n{label}: max S within +/-0.30 m of the line")
    vals = []
    y = -4.0
    while y <= 7.0 + 1e-9:
        v = s_band(lane_x, y)
        vals.append(v)
        print(f"  y={y:5.1f}  S={v:.3f}")
        y += 0.5
    good = [v for v in vals if v == v]
    print(f"  coverage: {sum(1 for v in good if v >= 0.05)}/{len(good)} samples >= 0.05, "
          f"median {sorted(good)[len(good)//2]:.2f}, max {max(good):.2f}")
PYEOF
}

main() {
    log "SIM_SECONDS=$SIM_SECONDS SPATIAL_PRIOR_PATH=$SPATIAL_PRIOR_PATH"

    # ORCHESTRATOR NOTE (see module header): Isaac Sim is a single shared
    # GPU resource reserved for the orchestrator across this whole study --
    # this script must not be run by two agents/terminals at once, same
    # rule as run_arm.sh.
    source "$WH/env_study.sh"
    eval "$(conda shell.bash hook)"; conda activate panoptex
    [ "$ROS_DOMAIN_ID" = "44" ] || { log "FATAL domain $ROS_DOMAIN_ID (expected 44 -- source env_study.sh)"; exit 1; }
    pgrep -f run_headless.py >/dev/null && { log "FATAL: an Isaac is already running"; exit 1; }

    # WP5: point the sim base's cmd_vel subscription at nav2_collision_monitor's
    # gated output (X3_CMD_VEL_TOPIC, see warehouse/x3_sim/build_x3_graphs.py
    # and panoptex_nav/README.md's "RGB-D depth + collision monitor") so the
    # warm-up drive exercises the same safety chain a study run does.
    # PANOPTEX_NO_CM=1 leaves the base on plain cmd_vel AND disables the
    # monitor node itself (enable_collision_monitor:=false below).
    if [ "${PANOPTEX_NO_CM:-0}" = "1" ]; then
        log "PANOPTEX_NO_CM=1: collision monitor disabled, base on plain cmd_vel"
        CM_LAUNCH_ARG="enable_collision_monitor:=false"
    else
        export X3_CMD_VEL_TOPIC=cmd_vel_safe
        CM_LAUNCH_ARG=""
    fi

    log "1. Isaac headless"
    start_bg isaac "${ISAACSIM_PYTHON:-$HOME/isaacsim/python.sh}" "$WH/x3_sim/run_headless.py" \
        --scene "$WH/Baseline_scenario_metric.usd" --rebuild-graphs
    wait_for_topic "/clock" 200 || { log "FATAL isaac not up in 200s"; tail -n 40 "$LOGDIR/isaac.log"; exit 1; }
    log "/clock up"

    log "2. X3 bringup"
    start_bg x3_bringup ros2 launch yahboomcar_nav x3_sim_bringup_launch.py
    sleep 5

    log "3. Carter patrol (the traffic being learned)"
    start_bg carters_patrol ros2 launch "$WH/launch/carters_patrol.launch.py" \
        map:="$WH/maps/warehouse_gt_carter.yaml"
    if ! wait_carters_active; then
        log "carters not active on first attempt; killing and relaunching carters_patrol once"
        kill_pidfile "$LOGDIR/carters_patrol.pid"
        log "3b. Carter patrol retry"
        start_bg carters_patrol ros2 launch "$WH/launch/carters_patrol.launch.py" \
            map:="$WH/maps/warehouse_gt_carter.yaml"
        if ! wait_carters_active; then
            log "FATAL: carterN nav2 never activated"
            exit 1
        fi
    fi

    log "4. x3_nav.launch.py learn_lanes:=true -- perception + spatial_prior only"
    start_bg x3_nav ros2 launch panoptex_nav x3_nav.launch.py \
        learn_lanes:=true spatial_prior_path:="$SPATIAL_PRIOR_PATH" $CM_LAUNCH_ARG
    wait_for_topic "/risk_perception/spatial_prior" 90 || log "WARNING /risk_perception/spatial_prior not seen"

    log "5. warming up for ${SIM_SECONDS}s sim time"
    wait_sim_seconds "$SIM_SECONDS"

    log "6. done warming; cleanup will save+exit spatial_prior_node (SIGINT under ros2 launch autosaves too, see that node's docstring)"
    cleanup
    trap - EXIT      # cleanup already ran; don't run it again on normal exit

    log "7. lane coverage from $SPATIAL_PRIOR_PATH"
    report_lane_coverage "$(python3 -c "import os,sys; print(os.path.expanduser(sys.argv[1]))" "$SPATIAL_PRIOR_PATH")"
}

main "$@"
