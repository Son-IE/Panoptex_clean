#!/usr/bin/env python3
"""
node_latency_report.py -- per-NODE compute-time breakdown from the
<node_name>_latency_<UTC timestamp>.csv files debug_log.open_latency_csv
writes when a node's debug_log_dir param is set (gdino_detector,
sam2_segmenter, risk_costmap, predictive_risk_costmap, spatial_prior,
object_tracker -- see each node's own comment on debug_log_dir).

This measures a DIFFERENT thing than evaluation_metrics.py's
detection_to_costmap_latency (a message-timestamp proxy for "how stale is
the costmap relative to the last detection", no node instrumentation): each
row here is a real time.perf_counter() wall-clock measurement of one node's
own hot-path call (GDINO/SAM2 inference, one _tick/_update/_publish_grid),
timed from inside that node.

Layout: one prior_log_dir per repeat (run_ablation_repeats.sh's
$HOME/.panoptex/logs/$LABEL), containing one <node_name>_latency_*.csv per
node instance that ran with logging on (per-camera for gdino/sam2, so
gdino_detector_global_cam_mid and gdino_detector_global_cam_exit -- same
executable, different camera -- are separate files/rows here).

Usage (one repeat):
  python3 tools/node_latency_report.py --log-dir ~/.panoptex/logs/forklift_baseline_rep1

Usage (pool several repeats of the same arm, mirrors aggregate_trials.py):
  python3 tools/node_latency_report.py \\
      --log-dir ~/.panoptex/logs/forklift_baseline_rep1 \\
      --log-dir ~/.panoptex/logs/forklift_baseline_rep2 \\
      --log-dir ~/.panoptex/logs/forklift_baseline_rep3 \\
      --label forklift_baseline --out results/forklift_baseline_latency.json
"""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

# <node_name>_latency_<YYYYmmdd>_<HHMMSS>.csv -- see
# debug_log.resolve_log_path's own naming (<dir>/<stem>_<UTC timestamp>.csv,
# stem = f"{node.get_name()}_latency" from open_latency_csv).
_FILENAME_RE = re.compile(r"^(?P<node>.+)_latency_\d{8}_\d{6}\.csv$")


def collect_node_samples(log_dir: str) -> Dict[str, List[float]]:
    """{node_name: [wall_ms, ...]} for every *_latency_*.csv found directly
    under log_dir. A node that never produced a message in this repeat
    (e.g. sam2 with zero detections) simply has no file -- not an error."""
    out: Dict[str, List[float]] = {}
    pattern = os.path.join(os.path.expanduser(log_dir), "*_latency_*.csv")
    for path in glob.glob(pattern):
        m = _FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        node = m.group("node")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            vals = [float(row["wall_ms"]) for row in reader if row.get("wall_ms")]
        out.setdefault(node, []).extend(vals)
    return out


def summarize(samples: List[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else float("nan"),
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "max_ms": float(arr.max()) if arr.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", action="append", required=True,
                    help="one repeat's prior_log_dir -- pass once per repeat to pool")
    ap.add_argument("--label", default="condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # node -> list of (rep_index, [wall_ms,...])
    per_rep: List[Dict[str, List[float]]] = [collect_node_samples(d) for d in args.log_dir]
    all_nodes = sorted({node for rep in per_rep for node in rep})
    if not all_nodes:
        raise SystemExit(
            f"no *_latency_*.csv found under any of {args.log_dir} -- "
            "was debug_log_dir/prior_log_dir actually set for this run?")

    result: Dict[str, dict] = {}
    print(f"=== {args.label} -- node compute-time breakdown, {len(args.log_dir)} repeat(s) ===\n")
    for node in all_nodes:
        pooled: List[float] = []
        reps_present = 0
        for rep in per_rep:
            if node in rep:
                pooled.extend(rep[node])
                reps_present += 1
        stats = summarize(pooled)
        stats["reps_present"] = reps_present
        result[node] = stats
        print(f"  {node}:")
        print(f"    mean {stats['mean_ms']:.2f} ms | p50 {stats['p50_ms']:.2f} ms | "
              f"p95 {stats['p95_ms']:.2f} ms | max {stats['max_ms']:.2f} ms | "
              f"n={stats['n']} (over {reps_present}/{len(args.log_dir)} repeat(s))")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "nodes": result}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
