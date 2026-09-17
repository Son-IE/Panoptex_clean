#!/usr/bin/env python3
"""
x3_waypoints_to_trials.py -- convert a yahboomcar_nav waypoints YAML
(schema: frame_id + waypoints: [{name, x, y, yaw}, ...], e.g.
warehouse/nav/x3_waypoints.yaml) into tools/run_trials.py's trial config
JSON, ONE WAYPOINT PER TRIAL.

Why one waypoint per trial rather than one trial for the whole tour:
evaluation_node resets its metric buffers on every "start <label>" and
finalizes + publishes on "end" (see evaluation_node.py's trial_control
protocol). Splitting the tour this way gets a PER-WAYPOINT
Efficiency/Safety/Latency/Clearance breakdown for free -- run_trials.py's
own outer loop already runs these labeled trials back-to-back as one
continuous tour (no Isaac/robot reset between them, see its module
docstring), so only the evaluation bookkeeping is segmented, not the
robot's actual path. No change needed to run_trials.py or
evaluation_node.py.

Usage:
  python3 tools/x3_waypoints_to_trials.py \\
      $HOME/Digital-Twin-Project/warehouse/nav/x3_waypoints.yaml \\
      --out trials_x3_tour.json
"""

import argparse
import json

import yaml


def convert(waypoints_yaml_path: str) -> dict:
    with open(waypoints_yaml_path) as f:
        data = yaml.safe_load(f)

    waypoints = data.get("waypoints") or []
    if not waypoints:
        raise SystemExit(
            f"no 'waypoints' list found in {waypoints_yaml_path} -- wrong "
            "file, or the ACTIVE TOUR block was commented out")

    trials = []
    for wp in waypoints:
        if "name" not in wp or "x" not in wp or "y" not in wp:
            raise SystemExit(f"waypoint entry missing name/x/y: {wp}")
        trials.append({
            "label": wp["name"],
            "waypoints": [[float(wp["x"]), float(wp["y"]), float(wp.get("yaw", 0.0))]],
        })
    return {"trials": trials}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("waypoints_yaml", help="yahboomcar_nav-schema waypoints YAML")
    ap.add_argument("--out", default="trials_x3_tour.json")
    args = ap.parse_args()

    config = convert(args.waypoints_yaml)
    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)
    names = ", ".join(t["label"] for t in config["trials"])
    print(f"wrote {len(config['trials'])} single-waypoint trials -> {args.out}")
    print(f"order: {names}")


if __name__ == "__main__":
    main()
