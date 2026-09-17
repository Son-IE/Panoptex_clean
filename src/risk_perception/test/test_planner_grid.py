"""
WP3 deliverable 4: STAGE 6 planner grid (paint_track_planner()). Pure
numpy, no ROS graph -- same tier-0 style as test_risk_stack.py, targeting
the module-level helper directly rather than PredictiveRiskCostmapNode
(see that node's module docstring for why: it needs rclpy.init just to
exist).

Grid geometry is wider than test_risk_stack.py's 12x12 m default -- the
planner grid's own horizon (planner_horizon_steps=20 default) sweeps a
1.0 m/s track up to 6 m from its start, so a 20x20 m grid keeps every
tested step comfortably inside bounds with no boundary clipping.
"""

import numpy as np
import pytest

from risk_perception.predictive_risk_costmap_node import paint_track, paint_track_planner

RES = 0.10
OX, OY = -10.0, -10.0
ROWS, COLS = 200, 200
PRED_DT = 0.3
VEL_INFL = 0.05
PXX = PYY = 0.04
PVX = PVY = 0.01


def cell_index(x, y):
    """(row, col) of the cell whose center is nearest (x, y), for this
    module's fixed grid geometry."""
    col = round((x - OX) / RES - 0.5)
    row = round((y - OY) / RES - 0.5)
    return row, col


# ------------------------------------------------------------ gamma dominance

def test_planner_grid_gamma_dominates_collapsed_grid_at_far_step():
    """Same track (same pmot/consequence/relbonus), CPA factor pinned to
    1.0 on BOTH sides for a fair comparison (paint_track_planner has no
    factor argument at all -- it is always 1.0): at step 15 (t=4.8s) the
    planner grid's slower decay (planner_gamma=0.97) must read a strictly
    higher value than the collapsed grid's (gamma=0.9) at the exact same
    cell -- 0.97**15 ~= 0.633 vs 0.9**15 ~= 0.206."""
    x, y, vx, vy = 0.05, 0.05, 1.0, 0.0
    pmot, consequence, relbonus = 1.0, 0.90, 0.0
    N = 20
    step = 15
    t = (step + 1) * PRED_DT
    px, py = x + vx * t, y + vy * t
    row, col = cell_index(px, py)

    grid, _stack, _rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.9, VEL_INFL,
        want_stack=False,
    )
    planner = paint_track_planner(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
    )

    assert grid[row, col] > 0.01, "test setup: step-15 cell should be non-trivially painted"
    assert planner[row, col] > grid[row, col]


def test_planner_grid_matches_analytic_gamma_ratio_at_far_step():
    """More precise than the inequality above: with pmot=1 (stationary
    term exactly zero) and no cross-step interference (steps 1 m apart,
    sigma ~0.2 m), the value AT the step-k rollout cell is dominated by
    that step's own peak, so planner/grid should track
    (planner_gamma/gamma)**step to within a few percent."""
    x, y, vx, vy = 0.05, 0.05, 3.0, 0.0   # 0.9 m/step spacing -> well separated
    pmot, consequence, relbonus = 1.0, 0.90, 0.0
    N = 20
    step = 10
    t = (step + 1) * PRED_DT
    px, py = x + vx * t, y + vy * t
    row, col = cell_index(px, py)

    grid, _stack, _rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.9, VEL_INFL,
        want_stack=False,
    )
    planner = paint_track_planner(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
    )

    expected_ratio = (0.97 ** step) / (0.9 ** step)
    got_ratio = planner[row, col] / grid[row, col]
    assert got_ratio == pytest.approx(expected_ratio, rel=0.05)


# ------------------------------------------------------------ flow blend reuse

def test_planner_flow_blend_zero_equals_pure_cv_sweep():
    """planner_flow_blend_weight <= 0 must be a pure pass-through --
    reuses blend_velocity()'s own off-by-default guarantee (see
    test_predictive_costmap_flow.py) -- regardless of what flow data is
    on offer. An aggressive, uniform flow field is deliberately supplied
    here specifically so a leak would be obvious."""
    x, y, vx, vy = 0.05, 0.05, 1.0, 0.0
    pmot, consequence, relbonus = 1.0, 0.9, 0.0
    N = 20
    flow_array = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    flow_array[:, :] = (1.0, 0.0, 5.0)  # s=1, strong "everyone flows north"

    planner_with_flow_offered = paint_track_planner(
        x, y, x, y, vx, vy, pmot, PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        flow_array=flow_array, flow_blend_weight=0.0)
    planner_pure_cv = paint_track_planner(
        x, y, x, y, vx, vy, pmot, PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        flow_array=None, flow_blend_weight=0.0)

    assert np.array_equal(planner_with_flow_offered, planner_pure_cv)


def test_planner_flow_blend_shifts_far_horizon_cell_toward_flow():
    """v=(1,0) (east), F=(0,1) (north), S=1 everywhere: at
    planner_flow_blend_weight=0.5 the far-horizon rollout must lean north
    relative to the pure-CV sweep -- blend_velocity() blends the
    effective velocity toward F, growing with horizon step (paper eq.
    16-17), so the y-mass of the painted grid should shift toward
    positive y. Uses a row-wise mass centroid rather than argmax so the
    assertion is robust to discretization across many overlapping steps."""
    x, y, vx, vy = 0.05, 0.05, 1.0, 0.0
    pmot, consequence, relbonus = 1.0, 0.9, 0.0
    N = 20
    flow_array = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    flow_array[:, :] = (1.0, 0.0, 1.0)  # s=1, F=(0,1) perpendicular to v

    planner_cv = paint_track_planner(
        x, y, x, y, vx, vy, pmot, PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        flow_array=None, flow_blend_weight=0.0)
    planner_blend = paint_track_planner(
        x, y, x, y, vx, vy, pmot, PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        flow_array=flow_array, flow_blend_weight=0.5)

    ys = OY + (np.arange(ROWS) + 0.5) * RES

    def y_centroid(grid):
        row_mass = grid.sum(axis=1)
        assert row_mass.sum() > 0.0, "test setup: grid should be non-trivially painted"
        return float((row_mass * ys).sum() / row_mass.sum())

    assert y_centroid(planner_blend) > y_centroid(planner_cv)


# ------------------------------------------------------------ ego-independence / mixture

def test_planner_grid_stationary_hypothesis_uses_one_minus_pmot():
    """pmot=0 -> the moving hypothesis contributes nothing (weight 0 on
    its very first step), so the grid is painted ONLY by the stationary
    blob at (stat_x, stat_y) with weight (1 - 0) * C."""
    x, y = -3.05, 4.05
    consequence, relbonus = 0.65, 0.0
    N = 20

    planner = paint_track_planner(
        x, y, x, y, 2.0, -1.0, 0.0,
        PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
    )
    row, col = cell_index(x, y)
    assert planner[row, col] == pytest.approx(consequence, rel=1e-6)


# ------------------------------------------------------------ WP2: shared clamps/gate

def test_planner_grid_high_velocity_prior_stays_within_max_sigma():
    """Same 2026-09-10 evening WP2 fix as paint_track's own clamps (see
    test_risk_stack.test_high_velocity_prior_never_exceeds_max_sigma, which
    this mirrors): Pvx=Pvy=1.0 (m/s)^2 -- the Kalman filter's OWN
    track-init prior, not evidence -- must not let the planner grid's
    farthest rollout step (t=6.0 s at N=20/pred_dt=0.3, i.e. the whole
    horizon) inflate its spatial footprint past max_sigma_m (0.6 m
    default). Compared directly against the SAME call with the clamps
    effectively disabled (huge sigma_v_max_mps/max_sigma_m) to prove the
    clamp is actually doing something at this call site too, not just
    structurally present but never reached."""
    x, y, vx, vy = 0.0, 0.0, 0.5, 0.0
    pmot, consequence, relbonus = 1.0, 0.9, 0.0
    N = 20
    x_far = x + vx * N * PRED_DT  # farthest step's rollout x, t = 6.0 s exactly

    def farthest_step_radius(grid, threshold=0.05):
        """Max distance from (x_far, y) among painted cells whose column is
        within one cell of x_far -- restricts the read to the FARTHEST
        step's own blob, since earlier (less inflated) steps along the
        same y=0 line would otherwise dilute the radius measurement."""
        rs, cs = np.nonzero(grid > threshold)
        if rs.size == 0:
            return 0.0
        xs = OX + (cs.astype(np.float64) + 0.5) * RES
        ys = OY + (rs.astype(np.float64) + 0.5) * RES
        near = np.abs(xs - x_far) < RES
        if not near.any():
            return 0.0
        return float(np.hypot(xs[near] - x_far, ys[near] - y).max())

    after = paint_track_planner(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, 1.0, 1.0, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        sigma_v_max_mps=0.5, max_sigma_m=0.6,
    )
    before = paint_track_planner(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, 1.0, 1.0, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        sigma_v_max_mps=1e3, max_sigma_m=1e3,
    )

    r_after = farthest_step_radius(after)
    r_before = farthest_step_radius(before)
    assert r_after > 0.0, "test setup: the far step should still paint something"
    assert r_after <= 3 * 0.6 + RES + 1e-6
    assert r_before > r_after


def test_planner_grid_moving_allowed_false_collapses_onto_stationary_blob():
    """moving_allowed=False must zero pmot's effect on every weight, same
    as paint_track's own moving_allowed gate (test_risk_stack.
    test_moving_allowed_false_paints_stationary_only) -- the planner grid
    ends up painted ONLY by the stationary blob at (stat_x, stat_y) with
    the FULL weight C, byte-for-byte identical to calling the SAME track
    with pmot=0.0 outright."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    consequence, relbonus = 0.75, 0.0
    N = 20

    blocked = paint_track_planner(
        x, y, x, y, vx, vy, 0.9,
        PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
        moving_allowed=False,
    )
    stationary_only = paint_track_planner(
        x, y, x, y, vx, vy, 0.0,
        PXX, PYY, PVX, PVY, consequence, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, 0.97, VEL_INFL,
    )
    assert np.array_equal(blocked, stationary_only)
    row, col = cell_index(x, y)
    assert blocked[row, col] == pytest.approx(consequence, rel=1e-6)


def test_planner_grid_is_max_combined_across_tracks():
    """Two independent tracks painted into the same array -- max, not
    sum -- same convention as paint_track's grid/stack."""
    a_xy = (1.05, 1.05)
    b_xy = (-2.05, -2.05)

    planner_a = paint_track_planner(
        a_xy[0], a_xy[1], a_xy[0], a_xy[1], 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, 0.5, 0.0,
        RES, OX, OY, ROWS, COLS, 20, PRED_DT, 0.97, VEL_INFL)
    planner_b = paint_track_planner(
        b_xy[0], b_xy[1], b_xy[0], b_xy[1], 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, 0.9, 0.0,
        RES, OX, OY, ROWS, COLS, 20, PRED_DT, 0.97, VEL_INFL)

    combined = paint_track_planner(
        a_xy[0], a_xy[1], a_xy[0], a_xy[1], 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, 0.5, 0.0,
        RES, OX, OY, ROWS, COLS, 20, PRED_DT, 0.97, VEL_INFL)
    combined = paint_track_planner(
        b_xy[0], b_xy[1], b_xy[0], b_xy[1], 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, 0.9, 0.0,
        RES, OX, OY, ROWS, COLS, 20, PRED_DT, 0.97, VEL_INFL,
        grid=combined)

    assert np.array_equal(combined, np.maximum(planner_a, planner_b))
