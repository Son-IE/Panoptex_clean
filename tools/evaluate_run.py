#!/usr/bin/env python3
"""
evaluate_run.py -- post-hoc evaluation of one recorded run against the
Efficiency / Safety / Latency / Clearance metrics in
risk_perception.evaluation_metrics.

Pure bag-reading/ROS-message plumbing lives HERE; every actual number is
computed by evaluation_metrics.py's pure functions, which are unit-tested
(test_evaluation_metrics.py). This script is deliberately thin and is NOT
unit-tested the way that module is -- it needs a real bag to exercise
meaningfully -- so keep new computation OUT of this file and IN that
module wherever possible.

WHY A SEPARATE REFERENCE COSTMAP TOPIC: see evaluation_reference.launch.py
and evaluation_metrics.py's module docstring. Risk exposure must be scored
against a risk field that is IDENTICAL across every condition in an
ablation study -- sampling it from whichever costmap actually drove Nav2
that run makes an ablated ("weaker") condition self-report lower exposure
merely because it computed less risk, not because the robot was actually
any safer.

Record, at minimum:
  ros2 bag record /tf /tf_static /odom /risk_perception/world_objects \\
      /risk_perception/detections_2d /risk_costmap_reference /plan
while BOTH risk_perception.launch.py (whatever ablation is under test) AND
evaluation_reference.launch.py (the fixed full-system scorer) are running.
/odom is only used to recompute risk_exposure_raw_* (CPA/TTC needs robot
velocity, not just position) -- everything else still works without it,
that field just comes back 0.0.

Usage:
  python3 tools/evaluate_run.py path/to/bag --label full_system \\
      --out results/full_system.json

Run once per condition (baseline, each ablation arm, full system), then
diff the resulting JSON files -- each is RunReport.summary()'s flat dict,
directly comparable across runs.

NOT covered here (needs your Nav2 config to say where): "time to goal" is
approximated as the bag's total duration on --risk-topic, and the
"replan" topic defaults to /plan -- both are reasonable defaults, not
guaranteed correct for every Nav2 setup. Override --replan-topic, or pass
--goal-time-s explicitly, if your setup differs.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer

from risk_perception import evaluation_metrics as em

# Nav2 convention: nav_msgs/Odometry's twist is body-frame (base_footprint),
# so it needs the same tf-derived yaw rotation into map frame that
# predictive_risk_costmap_node._update_robot_state applies live -- see
# robot_velocity_series() below.


def open_reader(bag_path: str) -> Tuple[rosbag2_py.SequentialReader, Dict[str, str]]:
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr")
    for storage_id in ("sqlite3", "mcap"):
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
        reader = rosbag2_py.SequentialReader()
        try:
            reader.open(storage_options, converter_options)
        except Exception:
            continue
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        return reader, topic_types
    raise RuntimeError(
        f"could not open {bag_path} as sqlite3 or mcap -- "
        "check the path and that it is a rosbag2 directory")


def _header_stamp_ns(msg) -> Optional[int]:
    """Nanoseconds from msg.header.stamp, or None for headerless messages
    (e.g. tf2_msgs/TFMessage -- its per-transform headers are handled
    separately by build_tf_buffer)."""
    header = getattr(msg, "header", None)
    if header is None:
        return None
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def read_all(bag_path: str, wanted_topics: List[str]) -> Dict[str, list]:
    """One sequential pass over the bag; {topic: [(stamp_ns, msg), ...]}
    for every requested topic actually present (missing ones warn and
    come back as an empty list -- callers must handle that gracefully,
    not every metric needs every topic).

    stamp_ns is the message's own header.stamp (same time base
    build_tf_buffer keys the tf2 Buffer with) wherever the message HAS a
    header, falling back to rosbag2's own record-time stamp otherwise.
    rosbag2 always timestamps by wall-clock arrival time regardless of
    use_sim_time -- with the sim clock driving this whole pipeline, that
    wall-clock stamp can be off from every message's own sim-time
    header.stamp by many orders of magnitude, which made every downstream
    lookup_transform() call against the tf buffer fail outright."""
    reader, topic_types = open_reader(bag_path)
    msg_classes = {}
    for topic in wanted_topics:
        if topic in topic_types:
            msg_classes[topic] = get_message(topic_types[topic])
        else:
            print(f"warning: topic {topic} not found in bag, skipping", file=sys.stderr)

    out: Dict[str, list] = {t: [] for t in wanted_topics}
    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        if topic not in msg_classes:
            continue
        msg = deserialize_message(data, msg_classes[topic])
        header_ns = _header_stamp_ns(msg)
        out[topic].append((header_ns if header_ns is not None else bag_stamp_ns, msg))
    return out


def build_tf_buffer(tf_msgs: list, tf_static_msgs: list) -> Buffer:
    """Replay /tf and /tf_static into a tf2_ros.Buffer so robot pose can be
    queried at arbitrary historical timestamps, the same
    lookup_transform() call the live nodes use -- not a separate pose
    approximation."""
    buffer = Buffer()
    for _, msg in tf_static_msgs:
        for transform in msg.transforms:
            buffer.set_transform_static(transform, "bag")
    for _, msg in tf_msgs:
        for transform in msg.transforms:
            buffer.set_transform(transform, "bag")
    return buffer


def robot_xy_series(buffer: Buffer, sample_stamps_ns: List[int],
                    map_frame: str, base_frame: str
                    ) -> List[Optional[Tuple[float, float]]]:
    out: List[Optional[Tuple[float, float]]] = []
    for stamp_ns in sample_stamps_ns:
        try:
            tf = buffer.lookup_transform(map_frame, base_frame, Time(nanoseconds=stamp_ns))
            out.append((tf.transform.translation.x, tf.transform.translation.y))
        except Exception:
            out.append(None)
    return out


def robot_velocity_series(buffer: Buffer, odom_msgs: list, sample_stamps_ns: List[int],
                          map_frame: str, base_frame: str
                          ) -> List[Tuple[float, float]]:
    """Body-frame /odom twist, nearest-message per sample, rotated into map
    frame by the TF yaw at that same instant -- same construction as
    predictive_risk_costmap_node._update_robot_state (tf pose + odom twist
    rotated by yaw), needed here so reference_severity_at_point's CPA/TTC
    term sees the same robot velocity the live node would have. (0.0, 0.0)
    for any sample with no odom yet or no resolvable tf."""
    odom_stamps = np.array([s for s, _ in odom_msgs], dtype=np.int64)
    out: List[Tuple[float, float]] = []
    for stamp_ns in sample_stamps_ns:
        if len(odom_stamps) == 0:
            out.append((0.0, 0.0))
            continue
        idx = min(int(np.searchsorted(odom_stamps, stamp_ns)), len(odom_stamps) - 1)
        _, odom = odom_msgs[idx]
        try:
            tf = buffer.lookup_transform(map_frame, base_frame, Time(nanoseconds=stamp_ns))
        except Exception:
            out.append((0.0, 0.0))
            continue
        q = tf.transform.rotation
        yaw = em.yaw_from_quat(q.x, q.y, q.z, q.w)
        bx = float(odom.twist.twist.linear.x)
        by = float(odom.twist.twist.linear.y)
        out.append((bx * np.cos(yaw) - by * np.sin(yaw),
                   bx * np.sin(yaw) + by * np.cos(yaw)))
    return out


def tracks_at_samples(world_objects_msgs: list, sample_stamps_ns: List[int]
                      ) -> List[List[Dict[str, float]]]:
    """Nearest world_objects message per sample, unpacked into the plain
    dicts reference_severity_at_point() expects -- label/score/x/y from the
    Detection3D fields directly, pmot/vx/vy/relbonus parsed out of class_id
    (em.parse_class_id, same contract as every node's own parser), Pxx/Pyy/
    Pvx/Pvy from pose.covariance[0, 7, 21, 28] (object_tracker_node's own
    packing -- see that node's _publish_objects)."""
    obj_stamps = np.array([s for s, _ in world_objects_msgs], dtype=np.int64)
    out: List[List[Dict[str, float]]] = []
    for stamp_ns in sample_stamps_ns:
        if len(obj_stamps) == 0:
            out.append([])
            continue
        idx = min(int(np.searchsorted(obj_stamps, stamp_ns)), len(obj_stamps) - 1)
        _, msg = world_objects_msgs[idx]
        tracks = []
        for det in msg.detections:
            if not det.results:
                continue
            h = det.results[0].hypothesis
            label, kv = em.parse_class_id(h.class_id)
            cov = det.results[0].pose.covariance
            tracks.append({
                "label": label,
                "score": float(h.score),
                "x": float(det.bbox.center.position.x),
                "y": float(det.bbox.center.position.y),
                "pmot": kv.get("pmot", 0.0),
                "vx": kv.get("vx", 0.0),
                "vy": kv.get("vy", 0.0),
                "relbonus": kv.get("relbonus", 0.0),
                "Pxx": float(cov[0]),
                "Pyy": float(cov[7]),
                "Pvx": float(cov[21]),
                "Pvy": float(cov[28]),
            })
        out.append(tracks)
    return out


def nearest_object_distance(world_objects_msgs: list, sample_stamps_ns: List[int],
                            robot_xy: List[Optional[Tuple[float, float]]]) -> List[float]:
    """Distance from the robot to the nearest track in the temporally
    closest world_objects message, per sample. No movability filter here
    -- clearance to ANY tracked object is the point; filter upstream by
    category if a "dynamic objects only" variant is wanted."""
    obj_stamps = np.array([s for s, _ in world_objects_msgs], dtype=np.int64)
    distances = []
    for stamp_ns, xy in zip(sample_stamps_ns, robot_xy):
        if xy is None or len(obj_stamps) == 0:
            distances.append(float("inf"))
            continue
        idx = min(int(np.searchsorted(obj_stamps, stamp_ns)), len(obj_stamps) - 1)
        _, msg = world_objects_msgs[idx]
        best = float("inf")
        for det in msg.detections:
            dx = det.bbox.center.position.x - xy[0]
            dy = det.bbox.center.position.y - xy[1]
            best = min(best, float(np.hypot(dx, dy)))
        distances.append(best)
    return distances


def risk_at_xy(costmap_msgs: list, sample_stamps_ns: List[int],
              robot_xy: List[Optional[Tuple[float, float]]]) -> List[float]:
    """Nearest-message, nearest-cell risk value (int8 0-100 -> float 0-1)
    at the robot's position, per sample."""
    cm_stamps = np.array([s for s, _ in costmap_msgs], dtype=np.int64)
    out = []
    for stamp_ns, xy in zip(sample_stamps_ns, robot_xy):
        if xy is None or len(cm_stamps) == 0:
            out.append(0.0)
            continue
        idx = min(int(np.searchsorted(cm_stamps, stamp_ns)), len(cm_stamps) - 1)
        _, grid = costmap_msgs[idx]
        col = int((xy[0] - grid.info.origin.position.x) / grid.info.resolution)
        row = int((xy[1] - grid.info.origin.position.y) / grid.info.resolution)
        if 0 <= row < grid.info.height and 0 <= col < grid.info.width:
            value = grid.data[row * grid.info.width + col]
            out.append(max(0, value) / 100.0)
        else:
            out.append(0.0)
    return out


def speed_series(xy: List[Tuple[float, float]], times_s: List[float]) -> List[float]:
    speed = [0.0]
    for i in range(1, len(xy)):
        dt = max(1e-6, times_s[i] - times_s[i - 1])
        dx = xy[i][0] - xy[i - 1][0]
        dy = xy[i][1] - xy[i - 1][1]
        speed.append(float(np.hypot(dx, dy)) / dt)
    return speed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_path")
    ap.add_argument("--label", default="run")
    ap.add_argument("--risk-topic", default="/risk_costmap_reference",
                    help="the FIXED reference costmap topic -- see module docstring")
    ap.add_argument("--world-objects-topic", default="/risk_perception/world_objects")
    ap.add_argument("--odom-topic", default="/odom",
                    help="for the raw-severity metric's CPA/TTC term -- see "
                         "risk_exposure_raw_* in the summary output")
    ap.add_argument("--detections-topic", default="/risk_perception/detections_2d",
                    help="for detection-to-costmap latency")
    ap.add_argument("--replan-topic", default="/plan",
                    help="Nav2 global plan republish topic -- verify against your "
                         "Nav2 config, adjust if different")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_footprint")
    ap.add_argument("--sample-rate-hz", type=float, default=10.0,
                    help="rate to resample robot pose/risk/clearance at")
    ap.add_argument("--configured-period-s", type=float, default=0.2,
                    help="1 / publish_rate of the reference costmap node, for realtime_factor")
    ap.add_argument("--engagement-threshold-m", type=float, default=2.0)
    ap.add_argument("--stop-speed-mps", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="write RunReport summary JSON here")
    args = ap.parse_args()

    topics = [args.risk_topic, args.world_objects_topic, args.detections_topic,
              args.replan_topic, args.odom_topic, "/tf", "/tf_static"]
    data = read_all(args.bag_path, topics)

    tf_buffer = build_tf_buffer(data.get("/tf", []), data.get("/tf_static", []))

    costmap_msgs = data.get(args.risk_topic, [])
    all_stamps = sorted(s for s, _ in costmap_msgs)
    if not all_stamps:
        print(f"no messages on {args.risk_topic} -- nothing to evaluate "
              "(did evaluation_reference.launch.py run during this recording?)",
              file=sys.stderr)
        sys.exit(1)

    t0, t1 = all_stamps[0], all_stamps[-1]
    period_ns = max(1, int(1e9 / args.sample_rate_hz))
    sample_stamps = list(range(t0, t1, period_ns)) or [t0]
    times_s = [(s - t0) / 1e9 for s in sample_stamps]

    robot_xy = robot_xy_series(tf_buffer, sample_stamps, args.map_frame, args.base_frame)
    if not any(xy is not None for xy in robot_xy):
        print("no valid map->base_frame tf found for any sample -- check "
              "--map-frame/--base-frame and that /tf, /tf_static were recorded",
              file=sys.stderr)
        sys.exit(1)
    xy_filled = [(0.0, 0.0) if xy is None else xy for xy in robot_xy]

    # --- safety: risk exposure (grid-sampled, clipped 0-100 -- see
    # risk_exposure_raw_* below for the unclipped counterpart) ---
    risk_samples = risk_at_xy(costmap_msgs, sample_stamps, robot_xy)
    dt = [times_s[0]] + list(np.diff(times_s))
    exposure_total, exposure_rate = em.risk_exposure(risk_samples, dt)
    length = em.path_length([xy for xy in robot_xy if xy is not None])
    exposure_per_m = em.risk_exposure_per_distance(exposure_total, length)

    # --- safety: RAW (unclipped) risk exposure -- recomputed directly from
    # world_objects/odom/tf instead of sampled from the published grid, so
    # a near-miss and a near-collision no longer both saturate to the same
    # value. See evaluation_metrics.py's "Raw (unclipped) severity" section.
    robot_v = robot_velocity_series(
        tf_buffer, data.get(args.odom_topic, []), sample_stamps,
        args.map_frame, args.base_frame)
    tracks = tracks_at_samples(data.get(args.world_objects_topic, []), sample_stamps)
    raw_samples = [
        em.reference_severity_at_point(trk, xy[0], xy[1], v[0], v[1])
        if xy is not None else 0.0
        for trk, xy, v in zip(tracks, robot_xy, robot_v)
    ]
    exposure_raw_total, exposure_raw_rate = em.risk_exposure(raw_samples, dt)
    exposure_raw_per_m = em.risk_exposure_per_distance(exposure_raw_total, length)

    # --- clearance ---
    distances = nearest_object_distance(
        data.get(args.world_objects_topic, []), sample_stamps, robot_xy)
    encounters = em.clearance_encounters(distances, times_s, args.engagement_threshold_m)
    clearance_stats = em.clearance_distribution_stats(encounters)

    # --- efficiency ---
    speed = speed_series(xy_filled, times_s)
    stop_count, stopped_time = em.stop_events(speed, times_s, args.stop_speed_mps)
    smoothness = em.velocity_smoothness(speed, times_s)
    replan_stamps_s = [(s - t0) / 1e9 for s, _ in data.get(args.replan_topic, [])]
    replans = em.replan_count(replan_stamps_s)

    # --- latency / compute ---
    det_stamps_s = [(s - t0) / 1e9 for s, _ in data.get(args.detections_topic, [])]
    costmap_stamps_s = [(s - t0) / 1e9 for s, _ in costmap_msgs]
    latencies = em.detection_to_costmap_latency(det_stamps_s, costmap_stamps_s)
    rtf = em.realtime_factor(costmap_stamps_s, args.configured_period_s)

    report = em.RunReport(
        label=args.label,
        time_to_goal_s=em.time_to_goal(0.0, times_s[-1]),
        path_length_m=length,
        stop_count=stop_count,
        stopped_time_s=stopped_time,
        velocity_smoothness=smoothness,
        replan_count=replans,
        risk_exposure_total=exposure_total,
        risk_exposure_rate=exposure_rate,
        risk_exposure_per_m=exposure_per_m,
        risk_exposure_raw_total=exposure_raw_total,
        risk_exposure_raw_rate=exposure_raw_rate,
        risk_exposure_raw_per_m=exposure_raw_per_m,
        clearance=clearance_stats,
        detection_to_costmap_latency_s=latencies,
        realtime_factor=rtf,
    )

    summary = report.summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
