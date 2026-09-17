#!/usr/bin/env python3
"""
srm.py -- WP-A: Spatiotemporal Risk Map (SRM).

Thomas, Piat & Charpillet, "Learning Spatiotemporal Occupancy Grid Maps for
Efficient Decision-Making" (2021), eq. 3: an occupancy layer is converted to
a risk map that falls off LINEARLY with distance to occupied space,

    risk(i) = max(0, 1 - d(i) / d0)

where d(i) is the Euclidean distance from cell i to the nearest occupied
cell and d0 is the distance at which risk reaches zero. The paper thresholds
occupancy at several probability LEVELS and combines them with a p-norm
(p=3) across occupied cells, which for well-separated occupied regions is
well approximated by taking the max over levels of the level-weighted
falloff -- that is what `stack_to_srm` below does: for each candidate level
L, cells with probability >= L are "occupied at level L", the falloff is
computed against THAT occupied set, scaled by L (so a barely-probable
occupancy contributes a shallower risk than a near-certain one), and the
final value is the max over levels. A cell whose stack probability equals
exactly one of `levels` therefore peaks at that level's value -- the "level-
graded" behaviour test_srm.py exercises.

This module is pure numpy/scipy: no ROS, no rclpy, so it can be imported and
unit-tested (test_srm.py) without a ROS graph, and reused as-is from
predictive_risk_costmap_node.py's STAGE 5 stack -> SRM conversion.
"""

import math
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def stack_to_srm(
    stack: np.ndarray, resolution: float, d0_m: float,
    levels: Sequence[float] = (0.2, 0.5, 0.8),
) -> np.ndarray:
    """Convert a (K, H, W) STAGE 5 risk stack (float, values in [0, 1]) into
    a (K, H, W) Spatiotemporal Risk Map (SRM), per-layer, per Thomas et al.
    2021 eq. 3.

    For each layer k and each level L in `levels`:
      - occupied = stack[k] >= L
      - if nothing is occupied at this level, this level's term is 0
        everywhere for this layer (skipped outright -- computing an EDT
        against an all-free mask is meaningless: scipy.ndimage.
        distance_transform_edt has no "background" pixel to measure from
        and returns garbage large distances rather than raising, so this
        case MUST be guarded rather than trusted to fall out of the
        formula naturally).
      - otherwise d_L = distance_transform_edt(~occupied) * resolution
        (metres from each cell to the nearest occupied-at-level-L cell;
        exactly 0 AT an occupied cell, by construction of
        distance_transform_edt against a boolean "is this cell background"
        mask -- the all-occupied case therefore already yields d_L == 0
        everywhere for free, no separate guard needed there).
      - term = L * clip(1 - d_L / d0_m, 0, 1)
    srm[k] = elementwise max of `term` over all levels.

    Levels are combined with max, not summed -- an approximation of the
    paper's p=3 p-norm across occupied cells that is exact when a level's
    occupied set is a subset of a coarser level's dilation (true for the
    default (0.2, 0.5, 0.8) thresholds against one Gaussian-splatted mover)
    and otherwise still a reasonable, cheap stand-in: the point of the
    p-norm is "dominated by the worst nearby occupancy," which max already
    captures.

    Vectorised over the (H, W) plane inside each level (numpy comparisons,
    one EDT call); the K layers and the levels within a layer are looped in
    plain Python because scipy's EDT is fundamentally a per-plane operation
    (passing a 3D array through it would let distance leak between layers,
    treating "close in time" as "close in space") -- see this file's
    top-of-module docstring and test_srm.py's timing assertion for why this
    is still fast enough (K=21, 120x120 well under 100 ms measured on this
    machine).

    Returns a fresh (K, H, W) float32 array; never mutates `stack`.
    """
    stack = np.asarray(stack, dtype=np.float32)
    K, H, W = stack.shape
    out = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        layer = stack[k]
        best = np.zeros((H, W), dtype=np.float32)
        for level in levels:
            occupied = layer >= level
            if not occupied.any():
                # No cell reaches this level in this layer -- this level's
                # term is 0 everywhere; skip the EDT call entirely (see
                # docstring: an EDT with no background pixel is garbage,
                # not an error).
                continue
            d = distance_transform_edt(~occupied).astype(np.float32) * resolution
            term = float(level) * np.clip(1.0 - d / d0_m, 0.0, 1.0)
            np.maximum(best, term, out=best)
        # At an occupied cell the risk is the occupancy value itself, not the
        # highest level it happens to clear: our stack values are consequence-
        # weighted (mobile robot 0.75, person 0.90), so with levels (0.2,0.5,0.8)
        # a Carter's core would read 0.5 and never trip the critic's collision
        # threshold (2026-09-10 WP-B finding). Falloff between cells still comes
        # from the level terms.
        np.maximum(best, layer.astype(np.float32, copy=False), out=best)
        out[k] = best
    return out


def window_indices(
    info_origin_xy: Tuple[float, float], resolution: float, H: int, W: int,
    centre_xy: Tuple[float, float], window_m: float,
) -> Tuple[int, int, int, int, Tuple[float, float]]:
    """Row/col slice bounds of an axis-aligned `window_m` x `window_m`
    window centred on `centre_xy` (world coords), clipped to the (H, W)
    grid whose origin (world coords of cell (0, 0)'s lower corner) is
    `info_origin_xy`.

    Returns (r0, r1, c0, c1, window_origin_xy) such that
    `grid[r0:r1, c0:c1]` IS the window (python half-open slice convention
    -- r1/c1 are exclusive) and `window_origin_xy` is the world (x, y) of
    that sub-grid's own (0, 0) cell origin, i.e. what a caller should put
    in a new MapMetaData.origin for the windowed grid.

    Clipping: a window that would extend past the grid edge (robot near a
    wall, or `centre_xy` outside the grid entirely) is truncated to
    [0, H) / [0, W); the returned slice is therefore always non-empty as
    long as the grid itself is non-empty (H > 0 and W > 0), even when
    `centre_xy` lies fully outside the grid -- it collapses to the nearest
    edge cell rather than an empty slice, since an empty SRM window would
    have no origin to report and no downstream consumer expects one.
    """
    ox, oy = info_origin_xy
    cx, cy = centre_xy
    half_cells = (window_m / 2.0) / resolution

    ccol = (cx - ox) / resolution
    crow = (cy - oy) / resolution

    c0 = int(math.floor(ccol - half_cells))
    c1 = int(math.ceil(ccol + half_cells))
    r0 = int(math.floor(crow - half_cells))
    r1 = int(math.ceil(crow + half_cells))

    c0 = min(max(c0, 0), max(W - 1, 0))
    r0 = min(max(r0, 0), max(H - 1, 0))
    c1 = min(max(c1, 0), W)
    r1 = min(max(r1, 0), H)
    if c1 <= c0:
        c1 = min(W, c0 + 1)
    if r1 <= r0:
        r1 = min(H, r0 + 1)

    window_origin = (ox + c0 * resolution, oy + r0 * resolution)
    return r0, r1, c0, c1, window_origin
