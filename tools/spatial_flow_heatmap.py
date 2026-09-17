#!/usr/bin/env python3
"""
spatial_flow_heatmap.py -- render spatial_prior_node's persisted state as a
figure: S (color, "where") and F (arrows, "which way"), one panel per
category.

No ROS needed. spatial_prior_node._save()/_load() write plain numpy
(resolution, origin_x, origin_y, width, height, categories, and per
category s__<cat>/fx__<cat>/fy__<cat>) to persist_path -- see that node's
module docstring -- so this reads the .npz directly and never has to touch
rclpy, cv_bridge, or a live topic. Safe to run against a file that a live
spatial_prior_node currently has open: _save() writes to a .tmp path and
os.replace()s it in, so the reader only ever sees a complete file.

Run any time after spatial_prior_node has autosaved (autosave_period_sec
in risk_perception.yaml, default 60 s) or shut down cleanly (Ctrl-C in the
launch terminal -- NOT `kill` under `ros2 run`, which skips the save; see
that node's docstring) to see the current learned state without stopping
the live system.

Usage:
    python3 tools/spatial_flow_heatmap.py
    python3 tools/spatial_flow_heatmap.py --npz ~/.panoptex/spatial_prior.npz \\
        --out out/spatial_flow --min-s 0.05 --stride 3

Color is a single-hue sequential ramp, light -> dark clay, matching
docs/figures/panoptex_style.py's "clay = contribution of this work" --
Spatial-Flow is one of this paper's contributions, and a monochrome ramp is
also the colorblind-safe choice for a continuous magnitude regardless (the
dataviz method: "sequential = one hue, light->dark, never a rainbow" --
a single-hue ramp needs no CVD pair check since there is no second hue to
confuse it with). Arrows carry a dark halo so they stay legible at both
ends of the ramp, and are length-scaled by sqrt(s) so a cell with only
marginal evidence (s just above --min-s) visibly reads as less certain
than a saturated one -- the same "s=0 contributes nothing" principle
predictive_risk_costmap_node.blend_velocity uses for the rollout blend.
"""

import argparse
import os
import sys

import numpy as np


def load(npz_path):
    if not os.path.exists(npz_path):
        sys.exit(
            f"no such file: {npz_path}\n"
            "spatial_prior_node hasn't saved anything yet -- it writes on "
            "clean shutdown and every autosave_period_sec (default 60 s). "
            "Let it run a bit longer, or check persist_path in "
            "risk_perception.yaml if you customised it."
        )
    data = np.load(npz_path)
    categories = [str(c) for c in data["categories"]]
    if not categories:
        sys.exit(
            f"{npz_path} exists but has no categories yet -- nothing has "
            "cleared both Behavioral-prior gates (p_movable >= "
            "p_movable_min AND p_motion >= p_motion_min, sustained) long "
            "enough to deposit. Walk a person through the camera's view "
            "for a few seconds and try again."
        )
    geom = dict(resolution=float(data["resolution"]),
                origin_x=float(data["origin_x"]), origin_y=float(data["origin_y"]),
                width=int(data["width"]), height=int(data["height"]))
    grids = {c: dict(s=data[f"s__{c}"], fx=data[f"fx__{c}"], fy=data[f"fy__{c}"])
             for c in categories}
    return geom, grids


def clay_cmap():
    """Single-hue sequential ramp, light -> dark clay. Anchor colors match
    panoptex_style.py's B_FILL2/B_LINE so this sits visually with the rest
    of the paper's figures rather than introducing a new palette."""
    from matplotlib.colors import LinearSegmentedColormap
    stops = ["#FBFAF8", "#F4EBE2", "#E7D5C4", "#C79A6E", "#8C5F3F", "#4A2F1E"]
    return LinearSegmentedColormap.from_list("clay_sequential", stops, N=256)


def smooth_for_display(S, sigma):
    """Blur a COPY of S for the imshow rendering only -- never touches the
    grid the console diagnostics (max S, cells>min_s) are computed from, and
    never touches the arrow mask/direction below, which still reads the raw
    grid. This is purely cosmetic: it softens the 0.1 m cell edges into
    something that reads as a continuous field, at the cost of literal
    per-cell accuracy -- fine for a quick look, not for reading exact S off
    a specific cell. sigma is in GRID CELLS, not metres."""
    if sigma <= 0.0:
        return S
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(S, sigma=sigma, mode="constant", cval=0.0)
    except ImportError:
        # numpy-only separable Gaussian, for an env without scipy. Kernel
        # radius 3*sigma is the same cutoff scipy uses by default.
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        out = np.apply_along_axis(
            lambda m: np.convolve(m, k, mode="same"), axis=1, arr=S)
        out = np.apply_along_axis(
            lambda m: np.convolve(m, k, mode="same"), axis=0, arr=out)
        return out


def resolve_cmap(name):
    """'clay' (default) is the paper-consistent single-hue ramp from
    clay_cmap(); anything else is passed straight to matplotlib, so
    --cmap inferno / magma / turbo / hot all work for a punchier, more
    eye-catching read while iterating -- those are still single-path
    sequential ramps (no hue-cycling), just with far more dynamic range
    than a muted print palette needs."""
    import matplotlib.pyplot as plt
    if name == "clay":
        return clay_cmap()
    return plt.get_cmap(name)


def plot(geom, grids, out_prefix, min_s, stride, dpi, quiver_scale, cmap_name,
         smooth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    res, ox, oy = geom["resolution"], geom["origin_x"], geom["origin_y"]
    w, h = geom["width"], geom["height"]
    extent = [ox, ox + w * res, oy, oy + h * res]
    cats = sorted(grids)
    cmap = resolve_cmap(cmap_name)
    # white fill + a near-black stroke reads against either end of any
    # sequential ramp, light-to-dark or dark-to-light -- see resolve_cmap.
    arrow_halo = "#2A1B10" if cmap_name == "clay" else "#111111"

    fig, axes = plt.subplots(1, len(cats), figsize=(5.4 * len(cats), 5.2),
                              squeeze=False)
    axes = axes[0]

    rows, cols = np.indices((h, w))
    wx_full = ox + (cols + 0.5) * res
    wy_full = oy + (rows + 0.5) * res

    for ax, cat in zip(axes, cats):
        S, FX, FY = grids[cat]["s"], grids[cat]["fx"], grids[cat]["fy"]
        # smoothing is cosmetic and display-only -- title, arrow mask and
        # arrow direction below all still read the raw S/FX/FY.
        S_disp = smooth_for_display(S, smooth)
        im = ax.imshow(S_disp, origin="lower", extent=extent, cmap=cmap,
                        vmin=0.0, vmax=1.0,
                        interpolation="nearest" if smooth <= 0.0 else "bilinear")
        ax.set_title(f"{cat}   (max S = {S.max():.2f})", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("map x (m)")
        ax.set_ylabel("map y (m)")
        ax.set_aspect("equal")

        Ss = S[::stride, ::stride]
        mask = Ss >= min_s
        if mask.any():
            conf = np.sqrt(np.clip(Ss[mask], 0.0, 1.0))
            qk = dict(color="white", width=0.006, headwidth=4, headlength=5,
                      path_effects=[pe.withStroke(linewidth=1.6,
                                                   foreground=arrow_halo)])
            if quiver_scale is not None:
                qk.update(scale_units="xy", angles="xy", scale=quiver_scale)
            ax.quiver(
                wx_full[::stride, ::stride][mask], wy_full[::stride, ::stride][mask],
                FX[::stride, ::stride][mask] * conf, FY[::stride, ::stride][mask] * conf,
                **qk)
        else:
            ax.text(0.5, 0.5,
                    f"no cell has S >= {min_s} yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#5C6875",
                    style="italic")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("S  —  P(category moves here)", fontsize=9)

    has_sub = smooth > 0.0
    fig.suptitle("Spatial-Flow prior  —  S (color)  +  F (arrows)",
                  fontsize=13, fontweight="bold", y=0.98)
    if has_sub:
        fig.text(0.5, 0.905,
                 f"display-smoothed, σ={smooth:g} cells — not the raw grid",
                 ha="center", fontsize=8.5, style="italic", color="#5C6875")
    fig.tight_layout(rect=[0, 0, 1, 0.86 if has_sub else 0.94])

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_prefix + ".png", dpi=dpi)
    fig.savefig(out_prefix + ".pdf")
    return out_prefix + ".png", out_prefix + ".pdf"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--npz", default=os.path.expanduser("~/.panoptex/spatial_prior.npz"),
        help="path to spatial_prior_node's persisted state "
             "(default: risk_perception.yaml's persist_path)")
    ap.add_argument("--out", default="out/spatial_flow_heatmap",
                     help="output file prefix; .png and .pdf both written")
    ap.add_argument("--min-s", type=float, default=0.05,
                     help="cells with S below this draw no arrow (default 0.05)")
    ap.add_argument("--stride", type=int, default=6,
                     help="sample every Nth cell for arrows, so the field "
                          "doesn't turn into a solid block of ink (default "
                          "6 -- tuned against a ~1 m/s walking-pace F on "
                          "spatial_prior_node's default 0.1 m grid; drop it "
                          "for a coarser grid or slower category)")
    ap.add_argument("--quiver-scale", type=float, default=None,
                     help="fixed data-units-per-velocity-unit scale (see "
                          "matplotlib quiver 'scale', scale_units='xy'); "
                          "default lets matplotlib auto-scale per plot")
    ap.add_argument("--cmap", default="turbo",
                     help="'clay' (default) matches this paper's other "
                          "figures; any matplotlib colormap name also works "
                          "(e.g. inferno, magma, turbo, hot) for a punchier, "
                          "higher-contrast read while iterating")
    ap.add_argument("--smooth", type=float, default=0.0,
                     help="Gaussian-blur S for DISPLAY ONLY, sigma in grid "
                          "cells (default 0.0 = off, exact per-cell values). "
                          "Cosmetic -- softens the 0.1 m cell edges into a "
                          "continuous-looking field; console diagnostics and "
                          "the arrows still read the raw grid. Try 1.0-1.5. "
                          "Uses scipy if available, else a numpy fallback.")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    geom, grids = load(args.npz)
    for cat, g in grids.items():
        nz = int((g["s"] > args.min_s).sum())
        print(f"  {cat:12s} max S={g['s'].max():.3f}  cells>{args.min_s}={nz}")
    png, pdf = plot(geom, grids, args.out, args.min_s, args.stride, args.dpi,
                     args.quiver_scale, args.cmap, args.smooth)
    print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
