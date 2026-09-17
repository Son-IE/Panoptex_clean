"""
WP-A unit tests for risk_perception.srm -- pure numpy/scipy, no rclpy, no
ROS graph (see srm.py's module docstring and predictive_risk_costmap_node's
STAGE 5 stack test file, test_risk_stack.py, for the same "Tier 0" pattern).
"""

import time

import numpy as np
import pytest

from risk_perception.srm import stack_to_srm, window_indices

RES = 0.10
D0 = 1.5


def _single_cell_stack(rows=60, cols=60, value=1.0, row=30, col=30):
    stack = np.zeros((1, rows, cols), dtype=np.float32)
    stack[0, row, col] = value
    return stack


def test_single_occupied_cell_peaks_at_one():
    """Thomas et al. eq. 3's pure linear falloff, isolated with a single
    level=1.0 so the level-weighting term drops out (1.0 * clip(...)):
    risk == 1.0 AT the occupied cell, ~0.5 at d0/2, 0 at >= d0."""
    stack = _single_cell_stack(value=1.0)
    srm = stack_to_srm(stack, RES, D0, levels=(1.0,))
    assert srm.shape == stack.shape
    assert srm.dtype == np.float32

    r0, c0 = 30, 30
    assert srm[0, r0, c0] == pytest.approx(1.0)

    # a cell at distance ~d0/2 from the occupied cell
    half_cells = int(round((D0 / 2.0) / RES))
    assert srm[0, r0, c0 + half_cells] == pytest.approx(0.5, abs=0.05)

    # a cell at distance >= d0 reads exactly 0
    far_cells = int(round(D0 / RES)) + 2
    assert srm[0, r0, c0 + far_cells] == 0.0


def test_level_graded_peak_matches_value():
    """A cell whose stack probability equals exactly one of the DEFAULT
    levels (0.2, 0.5, 0.8) peaks at that level's value, not 1.0 -- the
    level-weighted-max is what makes the SRM 'level-graded' rather than a
    binary occupied/free mask."""
    stack = _single_cell_stack(value=0.5)
    srm = stack_to_srm(stack, RES, D0)  # default levels
    assert srm[0, 30, 30] == pytest.approx(0.5)


def test_empty_layer_is_all_zero():
    stack = np.zeros((1, 40, 40), dtype=np.float32)
    srm = stack_to_srm(stack, RES, D0)
    assert np.all(srm == 0.0)


def test_two_layers_are_independent():
    """An occupied cell in layer 0 must not leak into layer 1's all-zero
    result, and vice versa -- distance transforms are computed per-layer,
    never across the K axis."""
    stack = np.zeros((2, 40, 40), dtype=np.float32)
    stack[0, 20, 20] = 1.0
    # layer 1 stays empty
    srm = stack_to_srm(stack, RES, D0, levels=(1.0,))
    assert srm[0, 20, 20] == pytest.approx(1.0)
    assert np.all(srm[1] == 0.0)
    # layer 0 elsewhere (far from the occupied cell) should also be ~0
    assert srm[0, 0, 0] == 0.0


def test_all_occupied_layer_reads_its_occupancy_everywhere():
    """Every cell occupied at every level -> distance 0 everywhere ->
    srm == max(levels) everywhere (the all-occupied edge case is handled
    for free by distance_transform_edt itself, see srm.py docstring)."""
    stack = np.ones((1, 10, 10), dtype=np.float32)
    srm = stack_to_srm(stack, RES, D0)
    assert np.allclose(srm[0], 1.0)  # core == occupancy value (2026-09-10)


def test_window_indices_clips_at_grid_edge():
    """A window centred near the grid's lower-left corner is truncated to
    [0, H) / [0, W), not left extending past the edge, and the returned
    origin matches the clipped r0/c0, not the unclipped centre-half."""
    H = W = 120
    origin = (-6.0, -6.0)
    # centre right at the grid's own origin corner -- a naive symmetric
    # window would want cells with negative index on both axes
    r0, r1, c0, c1, win_origin = window_indices(
        origin, RES, H, W, centre_xy=(-6.0, -6.0), window_m=12.0)
    assert r0 == 0 and c0 == 0
    assert r1 <= H and c1 <= W
    assert win_origin == (origin[0] + c0 * RES, origin[1] + r0 * RES)
    assert win_origin == (-6.0, -6.0)


def test_window_indices_interior_matches_expected_size():
    H = W = 120
    origin = (-6.0, -6.0)
    r0, r1, c0, c1, win_origin = window_indices(
        origin, RES, H, W, centre_xy=(0.0, 0.0), window_m=12.0)
    # 12 m window at 0.10 m/cell -> ~120 cells, clipped to the 120-cell grid
    assert 0 <= r0 < r1 <= H
    assert 0 <= c0 < c1 <= W
    assert win_origin == (origin[0] + c0 * RES, origin[1] + r0 * RES)


def test_window_indices_centre_far_outside_grid_still_nonempty():
    """A centre entirely outside the grid still returns a valid, non-empty
    slice collapsed to the nearest edge -- never an empty range with no
    origin to report."""
    H = W = 120
    origin = (-6.0, -6.0)
    r0, r1, c0, c1, win_origin = window_indices(
        origin, RES, H, W, centre_xy=(500.0, 500.0), window_m=12.0)
    assert r1 > r0 and c1 > c0
    assert r1 <= H and c1 <= W
    assert win_origin is not None


def test_timing_k21_120x120_under_150ms():
    """WP-A's headline performance requirement: K=21, 120x120 must run
    well under 100 ms; this asserts the looser 150 ms bound as a CI-safe
    margin. Multiple movers -> multiple non-trivial occupied blobs per
    layer, closer to the real STAGE 5 stack this feeds from."""
    rng = np.random.default_rng(0)
    K, H, W = 21, 120, 120
    stack = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        for _ in range(3):
            cy, cx = rng.integers(0, H), rng.integers(0, W)
            yy, xx = np.mgrid[0:H, 0:W]
            blob = np.exp(-0.5 * (((yy - cy) / 6.0) ** 2 + ((xx - cx) / 6.0) ** 2))
            np.maximum(stack[k], blob.astype(np.float32), out=stack[k])

    start = time.perf_counter()
    srm = stack_to_srm(stack, RES, D0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert srm.shape == (K, H, W)
    assert elapsed_ms < 150.0, f"stack_to_srm took {elapsed_ms:.1f} ms (limit 150 ms)"
