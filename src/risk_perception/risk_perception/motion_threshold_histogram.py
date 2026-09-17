#!/usr/bin/env python3
"""
motion_threshold_histogram.py  --  turn the tracker's CSV into the threshold plot.

Run the STATIC bag (nothing moves) through the tracker with motion_log_path set,
then:
    python motion_threshold_histogram.py motion_static.csv

Prints the percentiles of the static-object motion_score (your null distribution)
and suggests a threshold at the 99th percentile. That single number + this plot
is what justifies `motion_threshold` in the paper -- it's set to the noise floor,
not by hand.
"""
import sys
import csv

import numpy as np


def main(path):
    scores = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                scores.append(float(row["motion_score"]))
            except (KeyError, ValueError):
                pass
    if not scores:
        raise SystemExit("no motion_score values found")

    s = np.asarray(scores)
    print(f"static motion_score over {len(s)} samples:")
    for p in (50, 90, 95, 99, 99.9):
        print(f"  p{p:>4}: {np.percentile(s, p):.3f}")
    p99 = np.percentile(s, 99)
    print(f"\nSuggested motion_threshold = {p99 * 1.1:.2f}  "
          f"(99th pct x1.1 margin)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(s, bins=60, color="#378ADD", edgecolor="white")
        plt.axvline(p99, color="#E24B4A", linestyle="--",
                    label=f"99th pct = {p99:.2f}")
        plt.xlabel("motion_score  (Mahalanobis velocity)")
        plt.ylabel("count")
        plt.title("Static-object motion score (null distribution)")
        plt.legend()
        plt.tight_layout()
        out = path.rsplit(".", 1)[0] + "_hist.png"
        plt.savefig(out, dpi=130)
        print(f"saved plot -> {out}")
    except ImportError:
        print("(matplotlib not installed -- percentiles only)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python motion_threshold_histogram.py <csv>")
    main(sys.argv[1])