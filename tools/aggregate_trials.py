#!/usr/bin/env python3
"""
aggregate_trials.py -- turn N repeats of a waypoint-per-trial tour (see
x3_waypoints_to_trials.py) into stabilized, comparable numbers:

  1. PER-WAYPOINT: mean +/- std of each metric across repeats, for every
     waypoint label -- "how does wp3_west_mid behave, averaged over 3 runs".
  2. TOUR: each repeat's 6 waypoint legs rolled up into one whole-tour
     number, then mean +/- std of THAT across repeats -- "how does the
     whole tour behave, averaged over 3 runs".

Input: one --all-trials-json per repeat, each run_trials.py's own
"<out-dir>/_all_trials.json" output (a list of evaluation_node
RunReport.summary() dicts, one per waypoint-trial, plus "leg_outcomes").

Rolling up one repeat's per-waypoint legs into one tour number is NOT a
uniform "sum everything" -- RunReport.summary() already collapsed some
fields from raw samples into a mean/percentile (see evaluation_metrics.py's
RunReport.summary()), and summing a percentile across legs is meaningless.
Three categories, by field:

  SUM      -- true integrals/counts over disjoint time windows, so summing
              across legs IS the tour total: time_to_goal_s, path_length_m,
              stop_count, stopped_time_s, replan_count,
              risk_exposure_total, risk_exposure_raw_total,
              clearance_n_encounters.
  RATE     -- per-leg rate/per-m fields are ALSO not summable, but they ARE
              exactly reconstructible from the SUM fields above (rate =
              total/time, per_m = total/distance), so recompute rather than
              average the per-leg ratios: risk_exposure_rate,
              risk_exposure_per_m, risk_exposure_raw_rate,
              risk_exposure_raw_per_m.
  APPROX   -- fields already collapsed from raw samples inside ONE leg
              (std-of-jerk, latency percentile, clearance percentile,
              realtime-factor extrema) cannot be losslessly re-pooled from
              only the per-leg summary numbers -- evaluate_run.py/
              evaluation_node.py would need to expose raw per-sample lists
              for that, which they deliberately don't (see RunReport's own
              docstring: "lists collapsed to mean + p95"). Reported instead
              as a leg-duration-weighted mean across legs, clearly labeled
              approximate: velocity_smoothness, latency_mean_s,
              latency_p95_s, realtime_factor_mean, realtime_factor_max.
              clearance_min_m is the one exception that IS exact (the min of
              per-leg mins is the true tour min); clearance_p5_m/median_m
              are approximated the same weighted-mean way as the rest.

Usage:
  python3 tools/aggregate_trials.py \\
      --all-trials-json results/baseline_rep1/_all_trials.json \\
      --all-trials-json results/baseline_rep2/_all_trials.json \\
      --all-trials-json results/baseline_rep3/_all_trials.json \\
      --label baseline --out results/baseline_aggregate.json

Optionally add one --isaac-log per repeat (same order as --all-trials-json)
to also report Isaac's sim/wall-clock realtime ratio -- parsed straight from
run_headless.py's own "ratio X.XXx" print lines in run_ablation_repeats.sh's
<label>_isaac.log, not otherwise captured anywhere:
      --isaac-log results/baseline_rep1_isaac.log \\
      --isaac-log results/baseline_rep2_isaac.log \\
      --isaac-log results/baseline_rep3_isaac.log \\
"""

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# run_headless.py's periodic ("sim ... wall ... ratio X.XXx ... FPS") and
# final ("sim ..., wall ..., ratio X.XXXx, ... FPS") print lines both match
# this -- see run_headless.py's own f-strings. `ratio` there is CUMULATIVE
# sim_time/wall_time since this repeat's Isaac process started (not a
# per-interval rate), so the LAST match in the log is that repeat's overall
# average, and the MIN match is the worst window observed (a stall/overload
# drags the cumulative average down for the rest of the run, so min tends to
# sit late in the log too, but is reported separately since it is not
# guaranteed to be the last line specifically).
_ISAAC_RATIO_RE = re.compile(r"ratio\s+([0-9.]+)x")

SUM_FIELDS = [
    "time_to_goal_s", "path_length_m", "stop_count", "stopped_time_s",
    "replan_count", "risk_exposure_total", "risk_exposure_raw_total",
    "clearance_n_encounters",
]
APPROX_FIELDS = [
    "velocity_smoothness", "latency_mean_s", "latency_p95_s",
    "realtime_factor_mean", "realtime_factor_max",
    "clearance_p5_m", "clearance_median_m",
]
# label -> (numerator field, denominator field)
RATE_FIELDS = {
    "risk_exposure_rate": ("risk_exposure_total", "time_to_goal_s"),
    "risk_exposure_per_m": ("risk_exposure_total", "path_length_m"),
    "risk_exposure_raw_rate": ("risk_exposure_raw_total", "time_to_goal_s"),
    "risk_exposure_raw_per_m": ("risk_exposure_raw_total", "path_length_m"),
}


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def per_waypoint_stats(repeats: List[List[dict]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """repeats: one list of per-leg summary dicts per repeat (run_trials.py's
    _all_trials.json content). Groups by label (waypoint name), independent
    of repeat, then mean/std per field."""
    by_label: Dict[str, Dict[str, List[float]]] = {}
    for legs in repeats:
        for leg in legs:
            label = leg.get("label", "")
            # run_trials.py prefixes the condition onto the label
            # ("<condition_prefix>__<trial_label>") -- strip it back off so
            # the same waypoint groups across repeats with different
            # condition-prefixes (rep1/rep2/rep3).
            wp_name = label.split("__", 1)[-1]
            bucket = by_label.setdefault(wp_name, {})
            for key, value in leg.items():
                if isinstance(value, (int, float)):
                    bucket.setdefault(key, []).append(float(value))
    return {wp: {field: _mean_std(vals) for field, vals in fields.items()}
            for wp, fields in by_label.items()}


def tour_rollup(legs: List[dict]) -> Dict[str, float]:
    """One repeat's per-leg summaries -> one whole-tour number per field."""
    out: Dict[str, float] = {}
    for field in SUM_FIELDS:
        out[field] = float(sum(leg.get(field, 0.0) or 0.0 for leg in legs))
    for field, (num, den) in RATE_FIELDS.items():
        denom = out.get(den, 0.0)
        out[field] = out.get(num, 0.0) / denom if denom > 0 else 0.0
    weights = [max(leg.get("time_to_goal_s", 0.0) or 0.0, 1e-6) for leg in legs]
    total_w = sum(weights)
    for field in APPROX_FIELDS:
        vals = [leg.get(field) for leg in legs]
        pairs = [(w, v) for w, v in zip(weights, vals)
                if v is not None and not (isinstance(v, float) and np.isnan(v))]
        out[field] = (sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)
                     if pairs else float("nan"))
    leg_mins = [leg.get("clearance_min_m") for leg in legs
               if leg.get("clearance_min_m") is not None
               and not (isinstance(leg.get("clearance_min_m"), float)
                        and np.isnan(leg["clearance_min_m"]))]
    out["clearance_min_m"] = min(leg_mins) if leg_mins else float("nan")
    return out


def parse_isaac_realtime_ratio(isaac_log_path: str) -> Tuple[float, float]:
    """(last_ratio, min_ratio) parsed from one repeat's run_headless.py
    stdout log (run_ablation_repeats.sh's <label>_isaac.log). (nan, nan) if
    the file is missing or has no ratio lines at all (e.g. the repeat died
    before Isaac ever printed one) -- distinguishable from a real 0.0x by
    being NaN, same convention as every other "not computed" field here."""
    try:
        with open(isaac_log_path) as f:
            text = f.read()
    except OSError:
        return float("nan"), float("nan")
    ratios = [float(m) for m in _ISAAC_RATIO_RE.findall(text)]
    if not ratios:
        return float("nan"), float("nan")
    return ratios[-1], min(ratios)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-trials-json", action="append", required=True,
                    help="one repeat's run_trials.py _all_trials.json -- pass once per repeat")
    ap.add_argument("--isaac-log", action="append", default=None,
                    help="one repeat's run_ablation_repeats.sh <label>_isaac.log, same "
                         "order as --all-trials-json, to report Isaac's sim/wall-clock "
                         "realtime ratio alongside the nav metrics. Optional; omit entirely "
                         "to skip (older invocations keep working unchanged).")
    ap.add_argument("--label", default="condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.isaac_log and len(args.isaac_log) != len(args.all_trials_json):
        raise SystemExit(
            f"--isaac-log given {len(args.isaac_log)} time(s) but --all-trials-json "
            f"{len(args.all_trials_json)} time(s) -- pass one isaac.log per repeat, "
            "same order, or omit --isaac-log entirely")

    repeats = []
    for path in args.all_trials_json:
        with open(path) as f:
            repeats.append(json.load(f))

    per_wp = per_waypoint_stats(repeats)
    tours = [tour_rollup(legs) for legs in repeats]
    tour_fields = SUM_FIELDS + list(RATE_FIELDS) + APPROX_FIELDS + ["clearance_min_m"]
    tour_stats = {field: _mean_std([t[field] for t in tours]) for field in tour_fields}

    isaac_fields: List[str] = []
    if args.isaac_log:
        last_ratios, min_ratios = [], []
        for path in args.isaac_log:
            last_r, min_r = parse_isaac_realtime_ratio(path)
            last_ratios.append(last_r)
            min_ratios.append(min_r)
        tour_stats["isaac_realtime_ratio_last"] = _mean_std(last_ratios)
        tour_stats["isaac_realtime_ratio_min"] = _mean_std(min_ratios)
        isaac_fields = ["isaac_realtime_ratio_last", "isaac_realtime_ratio_min"]

    result = {
        "condition": args.label,
        "n_repeats": len(repeats),
        "per_waypoint": per_wp,
        "tour_per_repeat": tours,
        "tour": tour_stats,
    }

    print(f"=== {args.label} -- {len(repeats)} repeat(s) ===\n")
    print("-- per-waypoint (mean +/- std across repeats) --")
    for wp, fields in per_wp.items():
        print(f"  {wp}:")
        for key in ("time_to_goal_s", "path_length_m", "risk_exposure_raw_total",
                   "clearance_n_encounters", "clearance_min_m", "replan_count"):
            if key in fields:
                s = fields[key]
                print(f"    {key}: {s['mean']:.4g} +/- {s['std']:.4g} (n={s['n']})")
    print("\n-- whole tour (mean +/- std across repeats) --")
    for field in tour_fields:
        s = tour_stats[field]
        print(f"  {field}: {s['mean']:.4g} +/- {s['std']:.4g} (n={s['n']})")
    if isaac_fields:
        print("\n-- Isaac sim/wall-clock realtime ratio (from *_isaac.log) --")
        for field in isaac_fields:
            s = tour_stats[field]
            print(f"  {field}: {s['mean']:.4g} +/- {s['std']:.4g} (n={s['n']})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
