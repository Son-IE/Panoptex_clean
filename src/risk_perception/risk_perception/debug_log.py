#!/usr/bin/env python3
"""
debug_log.py  --  one helper for the research CSVs the prior nodes emit.

object_tracker_node, spatial_prior_node and predictive_risk_costmap_node each
take a `debug_log_dir` parameter (and object_tracker also keeps its older
`motion_log_path`). When set, they dump one row per track per tick covering
every prior's inputs and outputs -- semantic / behavioral / spatial-flow /
relation -- for offline analysis with tools/prior_report.py.

No ROS import here: `node` is only used for get_logger(), duck-typed, so this
stays unit-testable and importable from anywhere.
"""

import csv
import os
from datetime import datetime, timezone


def resolve_log_path(explicit_path: str, log_dir: str, stem: str) -> str:
    """A directory wins over an explicit path and auto-names
    <dir>/<stem>_<UTC timestamp>.csv so repeated runs never clobber. Returns
    "" when neither is set (logging off)."""
    if log_dir:
        d = os.path.expanduser(log_dir)
        os.makedirs(d, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return os.path.join(d, f"{stem}_{ts}.csv")
    if explicit_path:
        return os.path.expanduser(explicit_path)
    return ""


def open_latency_csv(node, log_dir: str):
    """(writer, file_handle); (None, None) when log_dir is unset.

    One row per timed call -- stamp_utc, wall_ms -- for a per-node compute-
    time breakdown (how long THIS node's own hot path takes, independent of
    evaluation_metrics.detection_to_costmap_latency's message-timestamp
    proxy, which only sees when a costmap moved, not what any one node
    spent computing). Stem is the node's own ROS name (self.get_name()), so
    e.g. gdino_detector_global_cam_mid and gdino_detector_global_cam_exit
    -- same executable, different camera -- land in separate files instead
    of one writer racing across two node instances. See
    tools/node_latency_report.py for the reader side."""
    writer, fh, _ = open_debug_csv(
        node, "", log_dir, f"{node.get_name()}_latency",
        ["stamp_utc", "wall_ms"])
    return writer, fh


def log_latency(writer, fh, wall_s: float) -> None:
    """Write one (stamp_utc, wall_ms) row and flush. No-op if writer is
    None (logging off) -- callers can call this unconditionally."""
    if writer is None:
        return
    writer.writerow([datetime.now(timezone.utc).isoformat(), f"{wall_s * 1000.0:.3f}"])
    fh.flush()


def open_debug_csv(node, explicit_path: str, log_dir: str, stem: str, header):
    """(writer, file_handle, resolved_path); (None, None, "") when off."""
    path = resolve_log_path(explicit_path, log_dir, stem)
    if not path:
        return None, None, ""
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(list(header))
    fh.flush()
    try:
        node.get_logger().info(f"{stem}: research CSV -> {path}")
    except Exception:                                    # noqa: BLE001
        print(f"{stem}: research CSV -> {path}", flush=True)
    return writer, fh, path
