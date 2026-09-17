#!/usr/bin/env python3
"""
prior_contribution_timeseries.py -- per-prior risk contribution over time for
one recorded run, built entirely from predictive_risk_costmap_node's own
research CSV (prior_log_dir's predictive_costmap_<ts>.csv) plus
spatial_prior_node's grid CSV -- no bag replay, no recomputation through
evaluation_metrics' pure functions. Every value plotted is something the live
node already logged, read back rather than re-derived, so this is a faithful
read of what actually ran, not a second implementation that can drift out of
sync with it.

At each predictive_costmap tick, the DOMINANT track (max C, the same
`peak = max(peak, ...)` this node itself takes across tracks) stands in for
"risk near the robot": whichever object is currently the most severe threat
is the one setting the grid's peak wherever its splat lands nearest the
robot's path.

Four components, from combine_severity's own formula (encounter_geometry.py):
    C = min(1, consequence * cpa_factor + relbonus)

  Semantic   = (consequence - 1) * cpa_factor   -- marginal loss going to the
               class-agnostic baseline (use_class_consequence:=false, which
               sets consequence=1); negative for a low-consequence class.
  Encounter  = consequence * (cpa_factor - 1)   -- marginal loss disabling
               CPA/TTC (enable_relative_motion:=false, cpa_factor=1).
  Relation   = relbonus                          -- exact: purely additive.
  Behavioral = pmot * C                          -- NOT a marginal loss like
               the other three (disabling it doesn't change C, it changes
               WHERE the severity gets painted -- a stationary blob only,
               vs. + a moving rollout ahead of the track; see
               predictive_risk_costmap_node.py's own docstring). Read instead
               as "how much of this track's severity budget the moving
               hypothesis currently owns."

Spatial-Flow deliberately excluded from this plot: it's a GRID floor/blend,
not a per-track quantity, so it isn't track-centric like the four lines
above (it can't be read off the same row -- spatial_prior_<ts>.csv's own
"grid" rows only log whole-grid summary stats, no per-cell time series) --
mixing it in here would silently change what the line means partway through
the legend. Use spatial_flow_heatmap.py for that one, spatially, instead.

Sign is meaningful and intentionally kept (not plotted as magnitude):
positive = this mechanism is currently ADDING to risk versus the baseline
it's measured against; negative = it's DISCOUNTING risk below that baseline
(e.g. Semantic goes negative whenever the dominant track's class rates below
the class-agnostic 1.0 -- the prior is doing its job of not overstating a
low-consequence object's risk, not an error).

Usage:
  python3 tools/prior_contribution_timeseries.py \\
      --log-dir ~/.panoptex/logs/complex_forklift_predictive_rep1 \\
      --out out/prior_contributions_predictive_rep1.png
"""

import argparse
import csv
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "total": "#0b0b0b",
    "Semantic": "#1c3f6e",     # navy
    "Encounter": "#dd9a3a",    # amber
    "Behavioral": "#149a9c",   # teal (swapped from Relation)
    "Relation": "#9a85d6",     # violet, lightened (swapped from Behavioral)
}


def _read_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_dominant_track_per_tick(log_dir: str):
    matches = sorted(glob.glob(os.path.join(log_dir, "predictive_costmap_*.csv")))
    if not matches:
        raise SystemExit(f"no predictive_costmap_*.csv found under {log_dir}")
    rows = _read_csv(matches[0])
    if not rows:
        raise SystemExit(f"{matches[0]} has no rows")

    best_by_t: dict = {}
    for row in rows:
        t = float(row["t"])
        c = float(row["C"])
        if t not in best_by_t or c > float(best_by_t[t]["C"]):
            best_by_t[t] = row

    ticks = sorted(best_by_t)
    dom = [best_by_t[t] for t in ticks]
    return (
        np.array(ticks, dtype=np.float64),
        np.array([float(r["consequence"]) for r in dom]),
        np.array([float(r["cpa_factor"]) for r in dom]),
        np.array([float(r["relbonus"]) for r in dom]),
        np.array([float(r["pmot"]) for r in dom]),
        np.array([float(r["C"]) for r in dom]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True, help="prior_log_dir for one rep")
    ap.add_argument("--out", default="out/prior_contributions.png")
    args = ap.parse_args()

    t, consequence, cpa_factor, relbonus, pmot, C = \
        load_dominant_track_per_tick(args.log_dir)

    semantic = (consequence - 1.0) * cpa_factor
    encounter = consequence * (cpa_factor - 1.0)
    relation = relbonus
    behavioral = pmot * C

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(t, C, color=COLORS["total"], linewidth=2.2,
            label="Total severity", zorder=5)
    ax.plot(t, semantic, color=COLORS["Semantic"], linewidth=2,
            label="Semantic")
    ax.plot(t, encounter, color=COLORS["Encounter"], linewidth=2,
            label="Encounter (CPA/TTC)")
    ax.plot(t, relation, color=COLORS["Relation"], linewidth=2,
            label="Relation")
    ax.plot(t, behavioral, color=COLORS["Behavioral"], linewidth=2,
            label="Behavioral")

    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_xlabel("Time [s]", color="#52514e", fontsize=15)
    ax.set_ylabel("Risk contribution", color="#52514e", fontsize=15)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5,
              frameon=False, fontsize=13)

    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"n_ticks={len(t)}  t_range=({t.min():.1f}, {t.max():.1f})  "
          f"mean C={C.mean():.3f}  max C={C.max():.3f}")


if __name__ == "__main__":
    main()
