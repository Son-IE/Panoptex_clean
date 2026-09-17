#!/usr/bin/env python3
"""WP6: one reusable post-run analysis tool, replacing the ad-hoc copies
`results/wiring_check/analyze_bag.py` and `results/avoidance/analyze_bag.py`
(both kept verbatim for history; this supersedes them).

Scores a study run (rosbag2 bag + the harness's text/JSON logs) against the
metrics used throughout `results/wiring_check/SUMMARY.md` and
`results/avoidance/SUMMARY.md`: per-Carter contact/collision episodes and
minimum gap, mission-supervisor lap/yield/hold/refuge bookkeeping, cmd_vel
activity, tracker accuracy against ground truth, RiskStack lethal-area
sanity, MPPI/DWB controller health from the nav log, collision-monitor
stops (when that topic exists), measured real-time factor, and (WP-C) a
"crossings" section: every X3 lane-entry/exit episode against each
Carter's patrol lane (`--lane-x`/`--lane-half-width-m`, default 1.32 m /
0.55 m), each labelled "contact" (min GT gap < `--contact-thresh-m` during
the episode), "waited" (the X3 was slow and near the lane for
`--wait-min-s` at some point before entry), "behind" (the Carter had
already passed the crossing point, in its own direction of travel, by
more than `--crossing-behind-m`), or "ahead" (the fallback -- the Carter
had not yet reached it) -- see `lane_crossings()`'s own docstring.

Usage:
    python3 tools/analyze_run.py <run_dir> [--bag PATH] \\
        [--gt-carter-topics /gt_tf] [--json out.json] [--md out.md]

`<run_dir>` is one `results/<family>/<run_name>/` directory (e.g.
`results/avoidance/avoid_panoptex_3`). By default the tool looks for a bag
at `<run_dir>/bag`; pass `--bag` to point at a bag kept elsewhere (the run
harness currently keeps multi-GB bags in the session scratchpad only, see
`results/avoidance/SUMMARY.md`). The nav log defaults to
`<run_dir>/x3_nav_excerpt.log` (falls back to `<run_dir>/x3_nav.log`).

--- Ground-truth sources (checked against the bag topic lists the existing
scripts read and `warehouse/carters/tests/patrol_monitor.py`) ---
Isaac's own `/odom` is ground truth for the X3 (perfect simulated odometry,
offset by the fixed spawn pose -- see `--spawn`). The Carters are NOT
ground-truthed via a per-robot `/carterN/odom` topic (that is each Carter's
own noisy nav2 localization estimate, used for ITS OWN navigation, not
truth). The actual ground truth is the single `/gt_tf` topic (a
`tf2_msgs/TFMessage`, `World -> carter1` / `World -> carter2`, published by
`warehouse/carters/build_carter_graphs.py`'s `PubGT` OmniGraph node) --
confirmed by grepping `results/*/analyze_bag.py` (both read `/gt_tf`) and
`warehouse/carters/tests/patrol_monitor.py`'s own docstring. `--gt-carter-
topics` defaults to `/gt_tf` accordingly but stays overridable (each entry
is either a bare TFMessage topic, matched against `--carters` by
`child_frame_id` suffix, or `TOPIC=carterName` for a `nav_msgs/Odometry`
ground-truth source per carter) in case a future run captures ground truth
differently.

--- Timestamp handling ---
`/odom`, `/gt_tf` and `/risk_perception/world_objects` all carry a stamped
header from nodes that run under `use_sim_time:=true` (or, for `/gt_tf`,
directly off Isaac's own sim clock) -- their `header.stamp` IS sim time, so
gap/episode timing needs no wall<->sim conversion at all. `/mission/state`
carries its own `"t"` field, which is `mission_supervisor`'s sim clock
(confirmed against `mission_supervisor.py`'s docstring: "`t` is this node's
clock (sim time under use_sim_time:=true...)"). Only the raw
`geometry_msgs/Twist` topics (`/cmd_vel`, `/cmd_vel_nav`, `/cmd_vel_safe`)
carry no header at all; for those this tool converts consecutive bag-
receive-time deltas to sim seconds via the measured real-time factor (see
`--rtf-source` below), which only needs to be locally accurate (a ratio
between consecutive messages), not globally anchored.

RTF itself is measured from `/clock` if it was recorded in the bag
(`sim_span / wall_span` between the first and last `/clock` sample); if
`/clock` was not recorded (true of every run so far -- see
`results/*/run_arm.sh`), it falls back to `<run_dir>/monitor.json`'s
`real_time_factor` field (written by `probe_monitor.py`); if neither is
available, cmd_vel timing degrades to raw wall seconds and the tool says so
in the output instead of silently mislabeling units.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message as _rclpy_deserialize

_BAD_RECORDS = 0


def deserialize_message(data, msg_type):
    """deserialize_message that skips corrupt records (a bag killed mid-write
    leaves a truncated last message; Fast CDR raises RMWError on it)."""
    global _BAD_RECORDS
    try:
        return _rclpy_deserialize(data, msg_type)
    except Exception:  # noqa: BLE001 - any deserialization failure
        _BAD_RECORDS += 1
        return None


def _safe_deserialize():
    return _BAD_RECORDS

from rosidl_runtime_py.utilities import get_message

# Canonical class_id parser + label->category map, straight from the source
# that writes the packed string (object_tracker_node._publish_objects):
#     f"{tr.label}|pmov={tr.p_movable:.2f}|pmot={tr.p_motion:.2f}"
#     f"|vx={vx:.3f}|vy={vy:.3f}|relbonus={tr.relation_bonus:.3f}"
# Reusing risk_visualization's parse_class_id/label_category instead of
# re-deriving the format here keeps this tool in sync automatically if the
# packed string or the label->category table ever changes.
from risk_perception.risk_visualization import label_category, parse_class_id

DEFAULT_SPAWN = (0.70, 0.05)  # X3 spawn pose in map, yaw 0 (WP0, 2026-09-09 spawn move)
DEFAULT_CARTERS = ("carter1", "carter2")
DEFAULT_CMD_VEL_CANDIDATES = ("/cmd_vel_safe", "/cmd_vel", "/cmd_vel_nav")  # what the base actually receives first
# carter1's patrol lane centre / half-width, as used throughout
# config/x3_sim_waypoints.yaml's clearance comments ("lane1_dx", lane
# centre x = 1.32) -- see the "crossings" section below.
DEFAULT_LANE_X = 1.32
DEFAULT_LANE_HALF_WIDTH_M = 0.55


# --------------------------------------------------------------------- misc

def percentile(values, p):
    """Linear-interpolation percentile, p in [0, 1]. None on empty input."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def nearest_by_time(series, t, tol):
    """series: sorted list of (t, ...) tuples. Returns the entry nearest to
    t if within tol seconds, else None. O(log n)."""
    if not series:
        return None
    ts = [s[0] for s in series]
    i = bisect.bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(series) and abs(series[j][0] - t) <= tol:
            if best is None or abs(series[j][0] - t) < abs(best[0] - t):
                best = series[j]
    return best


# --------------------------------------------------------------- bag reader

def discover_bag(run_dir: Path, bag_arg: str | None) -> Path | None:
    if bag_arg:
        return Path(bag_arg)
    candidate = run_dir / "bag"
    if candidate.exists():
        return candidate
    # search for any metadata.yaml (a rosbag2 dir) or loose .db3/.mcap under run_dir
    for meta in run_dir.rglob("metadata.yaml"):
        return meta.parent
    for ext in ("*.db3", "*.mcap"):
        hits = list(run_dir.rglob(ext))
        if hits:
            return hits[0].parent
    return None


def resolve_storage_id(bag_path: Path) -> str:
    if any(bag_path.glob("*.mcap")):
        return "mcap"
    return "sqlite3"


def open_reader(bag_path: Path):
    storage_id = resolve_storage_id(bag_path)
    rd = SequentialReader()
    rd.open(StorageOptions(uri=str(bag_path), storage_id=storage_id),
            ConverterOptions("", ""))
    return rd


# ------------------------------------------------------------- gt-topic spec

def parse_gt_topic_spec(items, carters):
    """Each item is 'TOPIC' (TFMessage, matched by child_frame_id suffix
    against `carters`) or 'TOPIC=carterName' (nav_msgs/Odometry, ground
    truth for exactly that carter)."""
    specs = []
    for it in items:
        if "=" in it:
            topic, name = it.split("=", 1)
            specs.append((topic, name))
        else:
            specs.append((it, None))
    return specs


# ------------------------------------------------------------------- reading

class RunData:
    def __init__(self, carters):
        self.carters = list(carters)
        self.x3_series = []                       # (t, x, y) sim seconds, map frame
        self.carter_series = {c: [] for c in carters}  # (t, x, y)
        self.mission_states = []                   # parsed JSON dicts, in bag order
        self.risk_stack_lethal_frac = []            # one fraction per /risk_stack msg
        # SRM area-fraction metrics (WP-C): one fraction per message on
        # whichever topic actually supplied them (/risk_stack_srm, falling
        # back to /risk_stack if the SRM topic isn't in the bag -- see
        # read_bag's srm_actual_topic resolution).
        self.srm_topic_used = None
        self.srm_layer0_frac = []                   # area% >= threshold, layer 0
        self.srm_layerlast_frac = []                 # area% >= threshold, layer K-1
        self.log_lines_source = None
        self.clock_samples = []                    # (wall_ns, sim_sec) from /clock
        self.cmd_vel_topic_used = None
        self.cmd_vel_samples = []                   # (wall_ns, speed_mps)
        self.collision_monitor_events = []          # (t_wall_ns, action_type, polygon_name)
        self.collision_monitor_action_names = {}
        # world_objects on-the-fly accumulators, per carter:
        self.wo_total = {c: 0 for c in carters}
        self.wo_tracked_any = {c: 0 for c in carters}      # nearest robot/wheeled <= tracked_max_m
        self.wo_offsets = {c: [] for c in carters}          # dist when <= offset_max_m
        self.wo_moving_samples = {c: 0 for c in carters}    # gt speed > moving_speed
        self.wo_moving_pmot_hi = {c: 0 for c in carters}    # ... and tracked with pmot > 0.5
        self.wo_track_ids = {c: set() for c in carters}     # distinct ids while "tracked"
        # Inclusive companion (lidar-fusion probe follow-up, 2026-09-09):
        # same as wo_tracked_any/wo_track_ids above but ALSO counting the
        # nearest track when it is an "unknown"-category lidar_cluster
        # track (object_tracker_node's label-agnostic association pass, see
        # its module docstring) -- a Carter tracked as an unknown lidar
        # blob still means perception isn't blind to it, which the strict
        # robot/wheeled-only tracked_pct alone hides.
        self.wo_tracked_any_incl = {c: 0 for c in carters}
        self.wo_track_ids_incl = {c: set() for c in carters}


def read_bag(bag_path: Path, gt_specs, carters, odom_topic, mission_topic,
             world_objects_topic, risk_stack_topic, clock_topic,
             cmd_vel_candidates, collision_monitor_topic,
             tracked_max_m, offset_max_m, moving_speed_mps,
             match_tol_s, lethal_threshold, spawn, risk_stack_srm_topic=None):
    rd = open_reader(bag_path)
    all_types = {t.name: t.type for t in rd.get_all_topics_and_types()}
    msg_classes = {}

    def msg_class(topic):
        if topic not in msg_classes:
            msg_classes[topic] = get_message(all_types[topic])
        return msg_classes[topic]

    tf_topics = {t for t, name in gt_specs if name is None and t in all_types
                 and all_types[t] == "tf2_msgs/msg/TFMessage"}
    odom_gt_topics = {t: name for t, name in gt_specs if name is not None}

    cmd_vel_topic = None
    for cand in cmd_vel_candidates:
        if cand in all_types:
            cmd_vel_topic = cand
            break

    topics_of_interest = set()
    topics_of_interest |= tf_topics
    topics_of_interest |= set(odom_gt_topics)
    if odom_topic in all_types:
        topics_of_interest.add(odom_topic)
    if mission_topic in all_types:
        topics_of_interest.add(mission_topic)
    if world_objects_topic in all_types:
        topics_of_interest.add(world_objects_topic)
    if risk_stack_topic in all_types:
        topics_of_interest.add(risk_stack_topic)
    # SRM area% metric (WP-C): read /risk_stack_srm if the bag has it,
    # else fall back to risk_stack_topic itself (that message is then
    # handled once below and feeds both risk_stack_lethal_frac and the
    # srm_* accumulators).
    srm_actual_topic = None
    if risk_stack_srm_topic is not None:
        if risk_stack_srm_topic in all_types:
            srm_actual_topic = risk_stack_srm_topic
        elif risk_stack_topic in all_types:
            srm_actual_topic = risk_stack_topic
        if srm_actual_topic is not None:
            topics_of_interest.add(srm_actual_topic)
    if clock_topic in all_types:
        topics_of_interest.add(clock_topic)
    if cmd_vel_topic:
        topics_of_interest.add(cmd_vel_topic)
    if collision_monitor_topic in all_types:
        topics_of_interest.add(collision_monitor_topic)

    data = RunData(carters)
    data.cmd_vel_topic_used = cmd_vel_topic
    data.srm_topic_used = srm_actual_topic

    carter_latest = {c: None for c in carters}   # (t, x, y)
    carter_prev = {c: None for c in carters}      # previous distinct sample, for gt speed

    collision_action_cls = None
    if collision_monitor_topic in all_types:
        try:
            collision_action_cls = msg_class(collision_monitor_topic)
        except Exception:
            collision_action_cls = None

    while rd.has_next():
        topic, raw, tns = rd.read_next()
        if topic not in topics_of_interest:
            continue
        m = deserialize_message(raw, msg_class(topic))
        if m is None:
            continue

        if topic in tf_topics:
            for tr in m.transforms:
                for c in carters:
                    if tr.child_frame_id.endswith(c):
                        t = stamp_to_sec(tr.header.stamp)
                        x = tr.transform.translation.x
                        y = tr.transform.translation.y
                        data.carter_series[c].append((t, x, y))
                        if carter_latest[c] is not None:
                            carter_prev[c] = carter_latest[c]
                        carter_latest[c] = (t, x, y)
            continue

        if topic in odom_gt_topics:
            c = odom_gt_topics[topic]
            t = stamp_to_sec(m.header.stamp)
            p = m.pose.pose.position
            data.carter_series[c].append((t, p.x, p.y))
            if carter_latest[c] is not None:
                carter_prev[c] = carter_latest[c]
            carter_latest[c] = (t, p.x, p.y)
            continue

        if topic == odom_topic:
            t = stamp_to_sec(m.header.stamp)
            p = m.pose.pose.position
            data.x3_series.append((t, spawn[0] + p.x, spawn[1] + p.y))
            continue

        if topic == mission_topic:
            try:
                data.mission_states.append(json.loads(m.data))
            except Exception:
                pass
            continue

        if topic == risk_stack_topic or topic == srm_actual_topic:
            info = m.info
            W, H, steps = info.width, info.height, m.steps
            if W and H and steps:
                arr = np.asarray(m.data, dtype=np.int8).reshape(steps, H, W)
                if topic == risk_stack_topic:
                    frac = float(np.mean(arr[0] >= lethal_threshold))
                    data.risk_stack_lethal_frac.append(frac)
                if topic == srm_actual_topic:
                    data.srm_layer0_frac.append(float(np.mean(arr[0] >= lethal_threshold)))
                    data.srm_layerlast_frac.append(float(np.mean(arr[-1] >= lethal_threshold)))
            continue

        if topic == clock_topic:
            sim = m.clock.sec + m.clock.nanosec * 1e-9
            data.clock_samples.append((tns, sim))
            continue

        if cmd_vel_topic and topic == cmd_vel_topic:
            speed = math.hypot(m.linear.x, m.linear.y)
            data.cmd_vel_samples.append((tns, speed))
            continue

        if topic == collision_monitor_topic:
            action = getattr(m, "action_type", None)
            poly = getattr(m, "polygon_name", "")
            data.collision_monitor_events.append((tns, action, poly))
            continue

        if topic == world_objects_topic:
            t = stamp_to_sec(m.header.stamp)
            for c in carters:
                latest = carter_latest[c]
                if latest is None or (t - latest[0]) > match_tol_s:
                    continue
                gx, gy = latest[1], latest[2]
                data.wo_total[c] += 1
                best_dist = None
                best_track = None
                # Inclusive nearest: same as best_dist/best_track (strict
                # robot/wheeled) but also candidates a nearest "unknown"
                # lidar_cluster track -- see wo_tracked_any_incl's comment
                # above. Tracked separately (not just a looser filter on
                # best_dist) because the NEAREST strict track and the
                # NEAREST inclusive track can be two different detections
                # (e.g. a closer lidar_cluster blob with a farther labeled
                # track also in frame).
                best_dist_incl = None
                best_track_incl = None
                for d in m.detections:
                    if not d.results:
                        continue
                    class_id = d.results[0].hypothesis.class_id
                    label, kv = parse_class_id(class_id)
                    is_strict = label_category(label) in ("robot", "wheeled")
                    is_incl = is_strict or label.strip().lower() == "lidar_cluster"
                    if not is_incl:
                        continue
                    cx = d.bbox.center.position.x
                    cy = d.bbox.center.position.y
                    dist = math.hypot(cx - gx, cy - gy)
                    if is_strict and (best_dist is None or dist < best_dist):
                        best_dist = dist
                        best_track = (d.id, kv.get("pmot", 0.0))
                    if best_dist_incl is None or dist < best_dist_incl:
                        best_dist_incl = dist
                        best_track_incl = (d.id, kv.get("pmot", 0.0))
                gt_speed = None
                prev = carter_prev[c]
                if prev is not None and latest[0] > prev[0]:
                    dt = latest[0] - prev[0]
                    if dt > 1e-3:
                        gt_speed = math.hypot(gx - prev[1], gy - prev[2]) / dt
                if best_dist is not None and best_dist <= offset_max_m:
                    data.wo_offsets[c].append(best_dist)
                if best_dist is not None and best_dist <= tracked_max_m:
                    data.wo_tracked_any[c] += 1
                    data.wo_track_ids[c].add(best_track[0])
                    if gt_speed is not None and gt_speed > moving_speed_mps:
                        data.wo_moving_samples[c] += 1
                        if best_track[1] > 0.5:
                            data.wo_moving_pmot_hi[c] += 1
                if best_dist_incl is not None and best_dist_incl <= tracked_max_m:
                    data.wo_tracked_any_incl[c] += 1
                    data.wo_track_ids_incl[c].add(best_track_incl[0])
            continue

    for c in carters:
        data.carter_series[c].sort(key=lambda r: r[0])
    data.x3_series.sort(key=lambda r: r[0])
    if collision_action_cls is not None:
        for name in ("DO_NOTHING", "STOP", "SLOWDOWN", "APPROACH", "LIMIT"):
            v = getattr(collision_action_cls, name, None)
            if v is not None:
                data.collision_monitor_action_names[v] = name
    return data


# ------------------------------------------------------------------ metrics

def compute_rtf(data: RunData, monitor_json_path: Path):
    if len(data.clock_samples) >= 2:
        (w0, s0), (w1, s1) = data.clock_samples[0], data.clock_samples[-1]
        wall_span = (w1 - w0) * 1e-9
        if wall_span > 0:
            return (s1 - s0) / wall_span, "bag /clock"
    if monitor_json_path.exists():
        try:
            mj = json.loads(monitor_json_path.read_text())
            rtf = mj.get("real_time_factor")
            if rtf:
                return float(rtf), "monitor.json"
        except Exception:
            pass
    return None, "none (cmd_vel timing falls back to raw wall seconds)"


def gap_episodes(x3_series, carter_series, threshold, merge_gap_s, match_tol_s):
    """Contiguous stretches where the GT gap < threshold, adjacent stretches
    within merge_gap_s of each other merged into one episode (mirrors
    results/avoidance/analyze_bag.py's contact_episodes, generalised to any
    threshold and driven off sim-second header stamps instead of a wall-time
    + fudge-factor RTF)."""
    episodes = []
    cur = None
    for t, x, y in x3_series:
        near = nearest_by_time(carter_series, t, match_tol_s)
        if near is None:
            if cur is not None and t - cur[1] > merge_gap_s:
                episodes.append(cur)
                cur = None
            continue
        g = math.hypot(x - near[1], y - near[2])
        if g < threshold:
            if cur is None:
                cur = [t, t, g, (x, y)]
            else:
                cur[1] = t
                cur[2] = min(cur[2], g)
        elif cur is not None and t - cur[1] > merge_gap_s:
            episodes.append(cur)
            cur = None
    if cur is not None:
        episodes.append(cur)
    return [{"t_start_s": e[0], "t_end_s": e[1], "dur_s": round(e[1] - e[0], 2),
             "min_gap_m": round(e[2], 3), "x3_xy": [round(e[3][0], 2), round(e[3][1], 2)]}
            for e in episodes]


def min_gap(x3_series, carter_series, match_tol_s):
    best = None
    for t, x, y in x3_series:
        near = nearest_by_time(carter_series, t, match_tol_s)
        if near is None:
            continue
        g = math.hypot(x - near[1], y - near[2])
        if best is None or g < best:
            best = g
    return best


def _series_velocity(series, t, tol):
    """Finite-difference (vx, vy) at time `t` from a sorted (t, x, y)
    series -- the two samples straddling `t` (nearest pair if `t` sits at
    or past an edge). None if the series has fewer than 2 samples, the
    bracketing pair's dt is degenerate, or `t` is farther than `tol` from
    BOTH samples in the pair (i.e. there is no data anywhere near `t`)."""
    if len(series) < 2:
        return None
    ts = [s[0] for s in series]
    i = bisect.bisect_left(ts, t)
    i = min(max(i, 1), len(series) - 1)
    a, b = series[i - 1], series[i]
    dt = b[0] - a[0]
    if dt <= 1e-6:
        return None
    if abs(t - a[0]) > tol and abs(t - b[0]) > tol:
        return None
    return (b[1] - a[1]) / dt, (b[2] - a[2]) / dt


def lane_in_band(x, lane_x, half_width):
    return abs(x - lane_x) <= half_width


def lane_crossing_episodes(x3_series, lane_x, half_width):
    """Contiguous stretches of x3_series where the X3 is inside the lane
    band [lane_x - half_width, lane_x + half_width]. Returns a list of the
    episode's own (t, x, y) sample sublists (unlike gap_episodes, which
    only keeps start/end/min -- the full sublist is needed below both to
    interpolate the lane-CENTRE crossing time and to bound the min-gap
    check to exactly this episode)."""
    episodes = []
    cur = []
    for s in x3_series:
        if lane_in_band(s[1], lane_x, half_width):
            cur.append(s)
        elif cur:
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


def _interp_lane_crossing(samples, lane_x):
    """samples: one episode's (t, x, y) sublist (>=1 entries, all already
    inside the lane band -- see lane_crossing_episodes). Returns (t, y) at
    the point where x linearly crosses lane_x (the lane CENTRE, not just
    the band edge) between two consecutive samples. Falls back to the
    sample nearest lane_x if the episode never actually reaches the centre
    (e.g. it only grazes the band edge and turns back)."""
    for i in range(len(samples) - 1):
        t0, x0, y0 = samples[i]
        t1, x1, y1 = samples[i + 1]
        d0, d1 = x0 - lane_x, x1 - lane_x
        if d0 == 0.0:
            return t0, y0
        if (d0 < 0.0) != (d1 < 0.0):
            f = d0 / (d0 - d1) if (d0 - d1) != 0.0 else 0.0
            return t0 + f * (t1 - t0), y0 + f * (y1 - y0)
    best = min(samples, key=lambda s: abs(s[1] - lane_x))
    return best[0], best[2]


def _waited_before_entry(x3_series, entry_t, lane_x, half_width,
                          wait_speed_mps, wait_min_s, wait_radius_m):
    """True if any CONTIGUOUS stretch of x3_series strictly before
    entry_t has the X3 slower than wait_speed_mps (consecutive-sample
    speed) AND within wait_radius_m of the lane band, for a total
    duration >= wait_min_s. Not required to touch entry_t itself -- a
    robot that idled well short of the crossing and then crept the rest
    of the way in still counts as having waited."""
    max_run = 0.0
    run = 0.0
    prev = None
    for t, x, y in x3_series:
        if t >= entry_t:
            break
        if prev is not None:
            dt = t - prev[0]
            if dt > 1e-6:
                speed = math.hypot(x - prev[1], y - prev[2]) / dt
                near_band = (abs(x - lane_x) - half_width) <= wait_radius_m
                if speed < wait_speed_mps and near_band:
                    run += dt
                    max_run = max(max_run, run)
                else:
                    run = 0.0
        prev = (t, x, y)
    return max_run >= wait_min_s


def _approach_mean_speed(x3_series, entry_t, approach_dist_m):
    """Mean speed over the last `approach_dist_m` of path length (walking
    backward through x3_series) strictly before entry_t. None if fewer
    than 2 samples precede entry_t."""
    pre = [s for s in x3_series if s[0] < entry_t]
    if len(pre) < 2:
        return None
    path = 0.0
    idx = len(pre) - 1
    while idx > 0 and path < approach_dist_m:
        t1, x1, y1 = pre[idx]
        t0, x0, y0 = pre[idx - 1]
        path += math.hypot(x1 - x0, y1 - y0)
        idx -= 1
    dt = pre[-1][0] - pre[idx][0]
    return path / dt if dt > 1e-6 else None


def lane_crossings(
    x3_series, carter_series, lane_x=DEFAULT_LANE_X,
    half_width=DEFAULT_LANE_HALF_WIDTH_M, behind_m=0.6,
    wait_speed_mps=0.05, wait_min_s=2.0, wait_radius_m=1.5,
    approach_dist_m=3.0, contact_thresh_m=0.45, match_tol_s=0.5,
):
    """For each contiguous stretch of x3_series inside the Carter lane
    band [lane_x - half_width, lane_x + half_width] ("lane-entry/exit
    episode"), find the crossing point -- the interpolated time/position
    where the X3's path crosses the lane CENTRE (lane_x) -- the Carter's
    position/velocity there, and label the episode:

      contact  the min GT gap between the X3 and this Carter during the
               episode (entry to exit) is < contact_thresh_m. Checked
               FIRST -- a hit here always wins over the timing labels
               below, since actual contact is the thing that matters most.
      waited   not a contact, and the X3's speed was < wait_speed_mps for
               a contiguous >= wait_min_s stretch, within wait_radius_m of
               the lane band, at some point before entry. Checked second.
      behind   neither of the above: the Carter had already travelled
               > behind_m PAST the crossing point, in ITS OWN direction of
               travel, by the time the X3 reached the lane centre -- i.e.
               the X3 crossed safely behind (after) the Carter.
      ahead    the fallback label: the Carter had not yet reached (or was
               within behind_m of) the crossing point in its own
               direction of travel -- i.e. still oncoming or just arriving
               when the X3 crossed.

    "past the crossing point in its own direction of travel" is computed
    as (carter_y - crossing_y) * sign(carter_vy): positive means the
    Carter's own forward motion has already carried it beyond crossing_y,
    regardless of whether the Carter happens to be northbound or
    southbound at the time.

    Returns a list of dicts, one per episode, oldest first, with keys:
    t_mid_s, entry_t_s, exit_t_s, crossing_xy, carter_xy,
    carter_y_offset_m, carter_vy_mps, label, min_gt_gap_m,
    x3_mean_speed_approach_mps. carter_xy/carter_y_offset_m/carter_vy_mps
    are None if no Carter sample exists within match_tol_s of t_mid (in
    which case the label falls back to "ahead" unless contact/waited
    already applied, since there is no Carter-position evidence to
    justify "behind").

    Pure function -- (t, x, y) series in, no ROS/bag I/O -- unit-tested
    directly against a synthetic timeline in test_analyze_run.py.
    """
    results = []
    for ep in lane_crossing_episodes(x3_series, lane_x, half_width):
        entry_t, exit_t = ep[0][0], ep[-1][0]
        t_mid, crossing_y = _interp_lane_crossing(ep, lane_x)

        carter_now = nearest_by_time(carter_series, t_mid, match_tol_s)
        carter_v = _series_velocity(carter_series, t_mid, match_tol_s)

        mg = min_gap(ep, carter_series, match_tol_s)
        is_contact = mg is not None and mg < contact_thresh_m
        waited = (not is_contact) and _waited_before_entry(
            x3_series, entry_t, lane_x, half_width,
            wait_speed_mps, wait_min_s, wait_radius_m)
        mean_speed = _approach_mean_speed(x3_series, entry_t, approach_dist_m)

        if is_contact:
            label = "contact"
        elif waited:
            label = "waited"
        else:
            label = "ahead"
            if carter_now is not None and carter_v is not None:
                vy = carter_v[1]
                sign = 1.0 if vy >= 0.0 else -1.0
                signed_offset = (carter_now[2] - crossing_y) * sign
                if signed_offset > behind_m:
                    label = "behind"

        results.append({
            "t_mid_s": round(t_mid, 3),
            "entry_t_s": round(entry_t, 3),
            "exit_t_s": round(exit_t, 3),
            "crossing_xy": [round(lane_x, 3), round(crossing_y, 3)],
            "carter_xy": (
                [round(carter_now[1], 3), round(carter_now[2], 3)]
                if carter_now is not None else None),
            "carter_y_offset_m": (
                round(carter_now[2] - crossing_y, 3) if carter_now is not None else None),
            "carter_vy_mps": round(carter_v[1], 3) if carter_v is not None else None,
            "label": label,
            "min_gt_gap_m": round(mg, 3) if mg is not None else None,
            "x3_mean_speed_approach_mps": (
                round(mean_speed, 3) if mean_speed is not None else None),
        })
    return results


def cmd_vel_metrics(samples, rtf, stop_thresh, stop_min_s):
    total = len(samples)
    if total == 0:
        return {"n_msgs": 0, "activity_ratio": None, "mean_speed_moving_mps": None,
                "stop_time_s": None, "stop_episodes": None}
    rtf_eff = rtf if rtf else 1.0
    moving = 0
    speed_sum = 0.0
    stop_time = 0.0
    stop_episodes = 0
    run_start = None
    run_last = None
    for i, (tns, speed) in enumerate(samples):
        if speed > stop_thresh:
            moving += 1
            speed_sum += speed
            if run_start is not None:
                dur = (run_last - run_start) * 1e-9 * rtf_eff
                if dur >= stop_min_s:
                    stop_time += dur
                    stop_episodes += 1
                run_start = None
        else:
            if run_start is None:
                run_start = tns
            run_last = tns
    if run_start is not None:
        dur = (run_last - run_start) * 1e-9 * rtf_eff
        if dur >= stop_min_s:
            stop_time += dur
            stop_episodes += 1
    return {
        "n_msgs": total,
        "activity_ratio": round(moving / total, 3),
        "mean_speed_moving_mps": round(speed_sum / moving, 4) if moving else None,
        "stop_time_s": round(stop_time, 2),
        "stop_episodes": stop_episodes,
    }


def mission_metrics(states):
    if not states:
        return None
    laps_by_index = []
    last_lap = states[0].get("lap")
    lap_start_t = states[0].get("t")
    for s in states:
        lap = s.get("lap")
        if lap != last_lap:
            laps_by_index.append({"lap": last_lap, "dur_s": round(s.get("t", 0) - lap_start_t, 2)})
            lap_start_t = s.get("t")
            last_lap = lap
    final = states[-1]
    # hold / refuge episodes: contiguous stretches of state in {"holding","refuge"}
    hold_eps, refuge_eps = [], []
    cur_kind, cur_start = None, None
    for i, s in enumerate(states):
        st = s.get("state")
        kind = st if st in ("holding", "refuge") else None
        if kind != cur_kind:
            if cur_kind is not None:
                dur = round(states[i - 1].get("t", 0) - cur_start, 2)
                (hold_eps if cur_kind == "holding" else refuge_eps).append(dur)
            cur_kind, cur_start = kind, s.get("t")
    if cur_kind is not None:
        dur = round(states[-1].get("t", 0) - cur_start, 2)
        (hold_eps if cur_kind == "holding" else refuge_eps).append(dur)
    return {
        "source": "/mission/state",
        "final_state": final.get("state"),
        "laps_completed": final.get("lap"),
        "per_lap_time_s": laps_by_index,
        "yield_count_final": final.get("yield_count"),
        "hold_episodes": len(hold_eps),
        "hold_time_s": round(sum(hold_eps), 2),
        "refuge_episodes": len(refuge_eps),
        "refuge_time_s": round(sum(refuge_eps), 2),
    }


def mission_metrics_from_log(log_text):
    if log_text is None:
        return None
    m = re.search(r"mission complete: (\d+) lap", log_text)
    laps = int(m.group(1)) if m else None
    return {
        "source": "log fallback (/mission/state not in bag) -- per-lap timing and "
                  "hold/refuge episode durations are NOT recoverable from the log alone",
        "final_state": None,
        "laps_completed": laps,
        "per_lap_time_s": "n/a (log fallback)",
        "yield_count_final": None,
        "hold_episodes": None, "hold_time_s": None,
        "refuge_episodes": None, "refuge_time_s": None,
    }


def nav_log_health(log_text):
    if log_text is None:
        return {"source": "n/a (no log file found)"}
    no_valid_traj = len(re.findall(r"No valid trajectories", log_text))
    failed_plan = len(re.findall(r"failed to create plan", log_text, re.IGNORECASE))
    mppi_fail = len([ln for ln in log_text.splitlines()
                      if "mppi" in ln.lower() and "fail" in ln.lower()])
    mission_aborts = len(re.findall(r"aborted \(status", log_text))
    controller_errors = len([ln for ln in log_text.splitlines()
                              if "controller_server" in ln
                              and re.search(r"fail|abort|error", ln, re.IGNORECASE)])
    return {
        "source": "nav log text search",
        "dwb_no_valid_trajectories": no_valid_traj,
        "failed_to_create_plan": failed_plan,
        "mppi_fail_lines": mppi_fail,
        "mission_supervisor_waypoint_aborts": mission_aborts,
        "controller_server_error_lines": controller_errors,
    }


def collision_monitor_metrics(data: RunData, do_nothing_default=0):
    if not data.collision_monitor_events:
        return {"source": "n/a (/collision_monitor_state not in bag)"}
    names = data.collision_monitor_action_names
    do_nothing = None
    for v, n in names.items():
        if n == "DO_NOTHING":
            do_nothing = v
    if do_nothing is None:
        do_nothing = do_nothing_default
    stop_events = [(t, a, p) for t, a, p in data.collision_monitor_events if a != do_nothing]
    by_polygon = {}
    for _, a, p in stop_events:
        by_polygon[p or "?"] = by_polygon.get(p or "?", 0) + 1
    return {
        "source": "/collision_monitor_state",
        "n_msgs": len(data.collision_monitor_events),
        "n_stop_or_limit_events": len(stop_events),
        "by_polygon": by_polygon,
        "action_names": names,
    }


def track_accuracy(data: RunData, carters):
    out = {}
    for c in carters:
        total = data.wo_total[c]
        tracked = data.wo_tracked_any[c]
        tracked_incl = data.wo_tracked_any_incl[c]
        offsets = data.wo_offsets[c]
        moving = data.wo_moving_samples[c]
        moving_hi = data.wo_moving_pmot_hi[c]
        out[c] = {
            "n_gt_samples": total,
            "tracked_pct": round(100 * tracked / total, 1) if total else None,
            # Same "nearest track within tracked_max_m" test as tracked_pct,
            # but also counting an "unknown"-category lidar_cluster track --
            # see wo_tracked_any_incl's comment in RunData.__init__.
            "tracked_incl_lidar_pct": (
                round(100 * tracked_incl / total, 1) if total else None),
            "offset_p50_m": round(percentile(offsets, 0.5), 3) if offsets else None,
            "offset_p90_m": round(percentile(offsets, 0.9), 3) if offsets else None,
            "n_offset_samples": len(offsets),
            "pmot_gt0.5_while_moving_pct": (
                round(100 * moving_hi / moving, 1) if moving else None),
            "n_moving_samples": moving,
            "distinct_track_ids": len(data.wo_track_ids[c]),
            "distinct_track_ids_incl_lidar": len(data.wo_track_ids_incl[c]),
        }
    return out


# --------------------------------------------------------------------- main

def build_metrics(args, run_dir: Path):
    bag_path = discover_bag(run_dir, args.bag)
    if bag_path is None:
        raise SystemExit(
            f"no bag found under {run_dir} (checked {run_dir}/bag, metadata.yaml, "
            "*.db3/*.mcap). Pass --bag explicitly, or see results/*/SUMMARY.md -- "
            "bags may only exist in the session scratchpad.")

    gt_specs = parse_gt_topic_spec(args.gt_carter_topics, args.carters)
    log_path = None
    for cand in (args.log, run_dir / "x3_nav_excerpt.log", run_dir / "x3_nav.log"):
        if cand and Path(cand).exists():
            log_path = Path(cand)
            break
    log_text = log_path.read_text(errors="replace") if log_path else None

    data = read_bag(
        bag_path, gt_specs, args.carters, args.odom_topic, args.mission_topic,
        args.world_objects_topic, args.risk_stack_topic, args.clock_topic,
        DEFAULT_CMD_VEL_CANDIDATES if args.cmd_vel_topic is None else (args.cmd_vel_topic,),
        args.collision_monitor_topic, args.tracked_max_m, args.offset_max_m,
        args.moving_speed_mps, args.match_tol_s, args.lethal_threshold, args.spawn,
        risk_stack_srm_topic=args.risk_stack_srm_topic)

    rtf, rtf_source = compute_rtf(data, run_dir / "monitor.json")

    per_carter = {}
    for c in args.carters:
        cs = data.carter_series[c]
        mg = min_gap(data.x3_series, cs, args.match_tol_s)
        contact = gap_episodes(data.x3_series, cs, args.contact_thresh_m,
                                args.episode_merge_s, args.match_tol_s)
        collision = gap_episodes(data.x3_series, cs, args.collision_thresh_m,
                                  args.episode_merge_s, args.match_tol_s)
        per_carter[c] = {
            "n_gt_samples": len(cs),
            "min_gt_gap_m": round(mg, 3) if mg is not None else None,
            "contact_episodes": contact,
            "n_contact_episodes": len(contact),
            "contact_time_s": round(sum(e["dur_s"] for e in contact), 2),
            "collision_episodes": collision,
            "n_collision_episodes": len(collision),
        }
    track_acc = track_accuracy(data, args.carters)
    for c in args.carters:
        per_carter[c]["track_accuracy"] = track_acc[c]

    crossings = {
        c: lane_crossings(
            data.x3_series, data.carter_series[c], lane_x=args.lane_x,
            half_width=args.lane_half_width_m, behind_m=args.crossing_behind_m,
            wait_speed_mps=args.wait_speed_mps, wait_min_s=args.wait_min_s,
            wait_radius_m=args.wait_radius_m, approach_dist_m=args.approach_dist_m,
            contact_thresh_m=args.contact_thresh_m, match_tol_s=args.match_tol_s)
        for c in args.carters
    }

    if data.mission_states:
        mission = mission_metrics(data.mission_states)
    else:
        mission = mission_metrics_from_log(log_text)

    cmdvel = cmd_vel_metrics(data.cmd_vel_samples, rtf, args.stop_speed_mps,
                              args.stop_min_s)

    metrics = {
        "run_dir": str(run_dir),
        "bag": str(bag_path),
        "log": str(log_path) if log_path else None,
        "carters": list(args.carters),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "rtf_source": rtf_source,
        "n_x3_odom_samples": len(data.x3_series),
        "per_carter": per_carter,
        "crossings": {
            "lane_x": args.lane_x,
            "lane_half_width_m": args.lane_half_width_m,
            "behind_m": args.crossing_behind_m,
            "by_carter": crossings,
        },
        "mission": mission,
        "cmd_vel": {**cmdvel, "topic_used": data.cmd_vel_topic_used or "n/a"},
        "risk_stack": {
            "n_msgs": len(data.risk_stack_lethal_frac),
            "lethal_ge": args.lethal_threshold,
            "lethal_area_pct_p50": (
                round(100 * percentile(data.risk_stack_lethal_frac, 0.5), 2)
                if data.risk_stack_lethal_frac else None),
            "lethal_area_pct_p90": (
                round(100 * percentile(data.risk_stack_lethal_frac, 0.9), 2)
                if data.risk_stack_lethal_frac else None),
        },
        # SRM area% metrics (WP-C): from /risk_stack_srm, falling back to
        # /risk_stack if the bag has no SRM topic -- see read_bag's
        # srm_actual_topic resolution. "layer0" mirrors the existing
        # RiskStack lethal-area% row above; "last-layer" is the same
        # computation on the far end of the horizon (index -1), where an
        # SRM rollout's uncertainty has had the most time to spread.
        "srm": {
            "topic_used": data.srm_topic_used or "n/a",
            "n_msgs": len(data.srm_layer0_frac),
            "area_ge": args.lethal_threshold,
            "layer0_area_pct_p50": (
                round(100 * percentile(data.srm_layer0_frac, 0.5), 2)
                if data.srm_layer0_frac else None),
            "layer0_area_pct_p90": (
                round(100 * percentile(data.srm_layer0_frac, 0.9), 2)
                if data.srm_layer0_frac else None),
            "last_layer_area_pct_p50": (
                round(100 * percentile(data.srm_layerlast_frac, 0.5), 2)
                if data.srm_layerlast_frac else None),
            "last_layer_area_pct_p90": (
                round(100 * percentile(data.srm_layerlast_frac, 0.9), 2)
                if data.srm_layerlast_frac else None),
        },
        "nav_log_health": nav_log_health(log_text),
        "collision_monitor": collision_monitor_metrics(data),
    }
    return metrics


def na(v):
    return "n/a" if v is None else v


def render_markdown(m: dict) -> str:
    lines = []
    run = Path(m["run_dir"]).name
    lines.append(f"### {run}")
    lines.append("")
    lines.append(f"bag: `{m['bag']}` &nbsp; log: `{m['log']}` &nbsp; "
                 f"RTF: {m['rtf'] if m['rtf'] is not None else 'n/a'} "
                 f"({m['rtf_source']})")
    lines.append("")
    mission = m["mission"] or {}
    cv = m["cmd_vel"]
    rs = m["risk_stack"] or {}
    srm = m["srm"] or {}
    nav = m["nav_log_health"]
    cmon = m["collision_monitor"]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| laps completed | {na(mission.get('laps_completed'))} |")
    lines.append(f"| per-lap time (s) | {na(mission.get('per_lap_time_s'))} |")
    lines.append(f"| yield count (final) | {na(mission.get('yield_count_final'))} |")
    lines.append(f"| hold episodes / total s | "
                 f"{na(mission.get('hold_episodes'))} / {na(mission.get('hold_time_s'))} |")
    lines.append(f"| refuge episodes / total s | "
                 f"{na(mission.get('refuge_episodes'))} / {na(mission.get('refuge_time_s'))} |")
    lines.append(f"| cmd_vel topic | {na(cv.get('topic_used'))} |")
    lines.append(f"| cmd_vel activity ratio | {na(cv.get('activity_ratio'))} |")
    lines.append(f"| mean speed while moving (m/s) | {na(cv.get('mean_speed_moving_mps'))} |")
    lines.append(f"| total stop time (s) / episodes | "
                 f"{na(cv.get('stop_time_s'))} / {na(cv.get('stop_episodes'))} |")
    lines.append(f"| RiskStack lethal-area% (layer0 >= {na(rs.get('lethal_ge'))}) p50/p90 | "
                 f"{na(rs.get('lethal_area_pct_p50'))} / {na(rs.get('lethal_area_pct_p90'))} |")
    lines.append(f"| SRM area% >={na(srm.get('area_ge'))} layer0 p50/p90 ({na(srm.get('topic_used'))}) | "
                 f"{na(srm.get('layer0_area_pct_p50'))} / {na(srm.get('layer0_area_pct_p90'))} |")
    lines.append(f"| SRM area% >={na(srm.get('area_ge'))} last-layer p50/p90 ({na(srm.get('topic_used'))}) | "
                 f"{na(srm.get('last_layer_area_pct_p50'))} / {na(srm.get('last_layer_area_pct_p90'))} |")
    lines.append(f"| DWB \"No valid trajectories\" count | {na(nav.get('dwb_no_valid_trajectories'))} |")
    lines.append(f"| \"failed to create plan\" count | {na(nav.get('failed_to_create_plan'))} |")
    lines.append(f"| MPPI fail-line count | {na(nav.get('mppi_fail_lines'))} |")
    lines.append(f"| mission_supervisor waypoint aborts | {na(nav.get('mission_supervisor_waypoint_aborts'))} |")
    lines.append(f"| controller_server error lines | {na(nav.get('controller_server_error_lines'))} |")
    if cmon.get("source", "").startswith("n/a"):
        lines.append("| collision-monitor stop/limit events | n/a |")
    else:
        lines.append(f"| collision-monitor stop/limit events | {na(cmon.get('n_stop_or_limit_events'))} "
                     f"({cmon.get('by_polygon', {})}) |")
    lines.append("")
    lines.append("| carter | min GT gap (m) | contact eps (<{:.2f}m) | contact time (s) | "
                 "collision eps (<{:.2f}m) | tracked % | offset p50/p90 (m) | "
                 "pmot>0.5 while moving % | distinct track ids |"
                 .format(0.45, 0.6))
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c, d in m["per_carter"].items():
        ta = d["track_accuracy"]
        lines.append(
            f"| {c} | {na(d['min_gt_gap_m'])} | {na(d['n_contact_episodes'])} | {na(d['contact_time_s'])} | "
            f"{na(d['n_collision_episodes'])} | {na(ta['tracked_pct'])} | "
            f"{na(ta['offset_p50_m'])}/{na(ta['offset_p90_m'])} | "
            f"{na(ta['pmot_gt0.5_while_moving_pct'])} | {na(ta['distinct_track_ids'])} |")
    lines.append("")

    crossings = m.get("crossings") or {}
    by_carter = crossings.get("by_carter") or {}
    lines.append(
        f"#### Lane crossings (lane x={na(crossings.get('lane_x'))}, "
        f"half-width={na(crossings.get('lane_half_width_m'))} m)")
    lines.append("")
    lines.append("| carter | t_mid (s) | label | carter y-offset (m) | carter vy (m/s) | "
                 "min GT gap (m) | X3 approach speed (m/s) |")
    lines.append("|---|---|---|---|---|---|---|")
    any_rows = False
    for c, episodes in by_carter.items():
        for e in episodes:
            any_rows = True
            lines.append(
                f"| {c} | {na(e['t_mid_s'])} | {e['label']} | {na(e['carter_y_offset_m'])} | "
                f"{na(e['carter_vy_mps'])} | {na(e['min_gt_gap_m'])} | "
                f"{na(e['x3_mean_speed_approach_mps'])} |")
    if not any_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--bag", default=None, help="bag path/dir override "
                    "(default: <run_dir>/bag, else search for metadata.yaml/*.db3/*.mcap)")
    ap.add_argument("--log", default=None, help="nav log override "
                    "(default: <run_dir>/x3_nav_excerpt.log, else <run_dir>/x3_nav.log)")
    ap.add_argument("--json", default=None, help="write full metrics JSON here")
    ap.add_argument("--md", default=None, help="write a Markdown table block here")
    ap.add_argument("--carters", nargs="+", default=list(DEFAULT_CARTERS))
    ap.add_argument("--gt-carter-topics", nargs="+", default=["/gt_tf"],
                     help="TOPIC (TFMessage, matched by child_frame_id suffix against "
                          "--carters) or TOPIC=carterName (nav_msgs/Odometry ground truth "
                          "for that one carter). Default /gt_tf matches the sim harness.")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--mission-topic", default="/mission/state")
    ap.add_argument("--world-objects-topic", default="/risk_perception/world_objects")
    ap.add_argument("--risk-stack-topic", default="/risk_stack")
    ap.add_argument("--risk-stack-srm-topic", default="/risk_stack_srm",
                     help="SRM area%% metric source; falls back to --risk-stack-topic "
                          "if this topic isn't in the bag")
    ap.add_argument("--clock-topic", default="/clock")
    ap.add_argument("--collision-monitor-topic", default="/collision_monitor_state")
    ap.add_argument("--cmd-vel-topic", default=None,
                     help="default: first of /cmd_vel, /cmd_vel_nav, /cmd_vel_safe present in the bag")
    ap.add_argument("--spawn", nargs=2, type=float, default=list(DEFAULT_SPAWN),
                     metavar=("X", "Y"))
    ap.add_argument("--contact-thresh-m", type=float, default=0.45)
    ap.add_argument("--collision-thresh-m", type=float, default=0.6)
    ap.add_argument("--episode-merge-s", type=float, default=3.0,
                     help="episodes within this many seconds of each other are merged")
    ap.add_argument("--match-tol-s", type=float, default=0.5,
                     help="max age of a GT/track sample used to match against another topic's timestamp")
    ap.add_argument("--tracked-max-m", type=float, default=1.0)
    ap.add_argument("--offset-max-m", type=float, default=2.0)
    ap.add_argument("--moving-speed-mps", type=float, default=0.2)
    ap.add_argument("--stop-speed-mps", type=float, default=0.02)
    ap.add_argument("--stop-min-s", type=float, default=1.0)
    ap.add_argument("--lethal-threshold", type=int, default=45)
    ap.add_argument("--lane-x", type=float, default=DEFAULT_LANE_X,
                     help="carter1 lane centre x (map frame) for the crossings section")
    ap.add_argument("--lane-half-width-m", type=float, default=DEFAULT_LANE_HALF_WIDTH_M)
    ap.add_argument("--crossing-behind-m", type=float, default=0.6,
                     help="Carter offset past the crossing point (along its own direction "
                          "of travel) beyond which a crossing episode is labelled 'behind'")
    ap.add_argument("--wait-speed-mps", type=float, default=0.05,
                     help="X3 speed below which it counts as 'waiting' for the crossings 'waited' label")
    ap.add_argument("--wait-min-s", type=float, default=2.0,
                     help="minimum contiguous slow duration for the crossings 'waited' label")
    ap.add_argument("--wait-radius-m", type=float, default=1.5,
                     help="max distance from the lane band the X3 must be within while "
                          "'waiting' (crossings 'waited' label)")
    ap.add_argument("--approach-dist-m", type=float, default=3.0,
                     help="path length before lane entry over which the crossings section "
                          "reports the X3's mean approach speed")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metrics = build_metrics(args, args.run_dir)
    md = render_markdown(metrics)
    print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2, default=str))
        print(f"wrote {args.json}", file=sys.stderr)
    if args.md:
        Path(args.md).write_text(md)
        print(f"wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
