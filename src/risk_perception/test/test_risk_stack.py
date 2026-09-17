"""
Tier 0 of the STAGE 5 (time-layered risk stack) test plan: pure numpy, no
ROS graph, no rclpy context at all -- unlike test_predictive_costmap_
multiobject.py, which constructs PredictiveRiskCostmapNode directly (that
only needs rclpy.init, no spin). This file can't do that: it targets
paint_track(), the module-level helper predictive_risk_costmap_node.py
factors _predict_and_splat's per-track math into specifically so it is
importable and callable with nothing but plain floats and numpy arrays
(see that function's docstring for why -- Node.__init__ subscribes/
publishes and starts a tf listener + timer, so it needs rclpy.init just to
exist, and this test doesn't want that dependency at all).

Grid geometry mirrors the node's own defaults (res=0.10, origin=(-6,-6))
so that "x=2.05" lands on an exact cell center, same convention
test_predictive_costmap_multiobject.py uses.
"""

import math

import numpy as np
import pytest

from risk_perception.encounter_geometry import cpa_geometry
from risk_perception.predictive_risk_costmap_node import (
    paint_track, resolve_stack_consequence, resolve_agnostic_consequence,
    resolve_extent_half, select_min_extent,
    is_track_mature, speed_cap_for_category, resolve_moving_allowed)

RES = 0.10
OX, OY = -6.0, -6.0
ROWS, COLS = 120, 120
N = 10
PRED_DT = 0.3
GAMMA = 0.9
VEL_INFL = 0.05

CPA_GAIN, CPA_SCALE_M, TTC_SCALE_S, MIN_REL_SPEED = 1.5, 0.6, 3.0, 0.05

# same covariance fixture as test_predictive_costmap_multiobject.make_detection
PXX = PYY = 0.04
PVX = PVY = 0.01


def cell_index(x, y):
    """(row, col) of the cell whose center is nearest (x, y), for this
    module's fixed grid geometry."""
    col = round((x - OX) / RES - 0.5)
    row = round((y - OY) / RES - 0.5)
    return row, col


def paint(x, y, vx, vy, pmot, consequence, factor=1.0, relbonus=0.0,
          move_x=None, move_y=None):
    """One track, fresh grid+stack, no spatial-flow blend -- the common case
    every test below needs."""
    if move_x is None:
        move_x = x
    if move_y is None:
        move_y = y
    grid, stack, _rollout = paint_track(
        x, y, move_x, move_y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, factor, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
    )
    return grid, stack


def test_moving_track_layer_k_argmax_tracks_constant_velocity():
    """pmot=1 -> the stationary hypothesis's weight (1 - pmot) * C is
    exactly zero, so layer 0 (which only the stationary hypothesis paints)
    is untouched and layers 1..N are painted ONLY by the moving hypothesis,
    at x + vx*k*dt. Since pred_dt=0.3 is an exact multiple of the 0.1 m
    grid resolution, that position lands exactly on a cell center."""
    x, y = 2.05, 0.05
    vx, vy = 1.0, 0.0
    _grid, stack = paint(x, y, vx, vy, pmot=1.0, consequence=0.90)

    assert stack.shape == (N + 1, ROWS, COLS)
    # Layer 0 ("now") carries the mover at its CURRENT position even at pmot=1
    # (fixed 2026-09-10; before, a pmot=1 track left "now" empty).
    assert stack[0].max() == pytest.approx(0.9, rel=1e-6)

    for k in range(1, N + 1):
        expected_x = x + vx * k * PRED_DT
        want_row, want_col = cell_index(expected_x, y)
        got_row, got_col = np.unravel_index(np.argmax(stack[k]), stack[k].shape)
        assert (got_row, got_col) == (want_row, want_col), f"layer {k}"
        assert stack[k].max() > 0.0


def test_stationary_track_contributes_equally_to_every_layer():
    """pmot=0 -> the moving hypothesis's weight is exactly zero (both its
    grid and stack contributions), so every stack layer is painted by
    nothing but the one stationary blob at (x, y) -- all N+1 layers must
    come out byte-for-byte identical, including layer 0."""
    x, y = -2.05, 1.05
    _grid, stack = paint(x, y, vx=3.0, vy=-2.0, pmot=0.0, consequence=0.65)

    want_row, want_col = cell_index(x, y)
    for k in range(N + 1):
        assert np.array_equal(stack[k], stack[0]), f"layer {k} != layer 0"
        got_row, got_col = np.unravel_index(np.argmax(stack[k]), stack[k].shape)
        assert (got_row, got_col) == (want_row, want_col), f"layer {k}"
        assert stack[k].max() > 0.0


def test_stack_is_invariant_to_robot_velocity_but_grid_is_not():
    """Both hypotheses' stack weights use C_stack, which pins the CPA/TTC
    factor to 1.0 -- the stack must not change no matter what the robot is
    doing, for ANY pmot. The collapsed grid uses the real factor and DOES
    change once the encounter geometry is closing.

    Same closing scenario test_predictive_costmap_multiobject.
    test_moving_robot_scales_grid_by_encounter_ratio uses: a person at
    (4.05, 0.05) walking at -1.0 m/s in x, robot parked vs. robot driving
    toward it. pmot=0.5 so the stationary term is non-zero and checked too.
    consequence=0.5, not the person class value 0.90 (WP-A, 2026-09-11):
    combine_severity's own 1.0 clip means 0.90 * EITHER factor here already
    saturates to 1.0 (0.90*1.40=1.26, 0.90*1.77=1.59, both > 1), which would
    make grid_parked and grid_driving identical and defeat the very
    difference this test exists to check -- 0.5 keeps both products under
    the clip (0.70/0.89) so the assertion below still tests what it says.
    See test_combine_severity_clips_at_one in test_encounter_geometry.py
    for the clip itself."""
    x, y, vx, vy = 4.05, 0.05, -1.0, 0.0
    robot_xy = (0.05, 0.05)

    factor_parked, _, _ = cpa_geometry(
        x, y, vx, vy, robot_xy[0], robot_xy[1], 0.0, 0.0,
        CPA_GAIN, CPA_SCALE_M, TTC_SCALE_S, MIN_REL_SPEED)
    factor_driving, _, _ = cpa_geometry(
        x, y, vx, vy, robot_xy[0], robot_xy[1], 1.0, 0.0,
        CPA_GAIN, CPA_SCALE_M, TTC_SCALE_S, MIN_REL_SPEED)
    assert factor_driving > factor_parked  # sanity: the scenario IS closing harder

    grid_parked, stack_parked = paint(
        x, y, vx, vy, pmot=0.5, consequence=0.5, factor=factor_parked)
    grid_driving, stack_driving = paint(
        x, y, vx, vy, pmot=0.5, consequence=0.5, factor=factor_driving)

    assert not np.array_equal(grid_parked, grid_driving)
    assert np.array_equal(stack_parked, stack_driving)


def test_stack_undiscounted_versus_grid_gamma_discount():
    """The grid's moving-hypothesis weight carries gamma**step; the
    stack's does not. With pmot=1 (stationary weight zero) the peak value
    at each layer's argmax cell must be the SAME constant across all N
    layers of the stack, while the corresponding weights baked into the
    grid decay geometrically -- i.e. this is what "undiscounted" buys."""
    x, y, vx, vy = 1.05, -3.05, 0.0, 1.0
    consequence = 0.90
    _grid, stack = paint(x, y, vx, vy, pmot=1.0, consequence=consequence)

    peaks = [stack[k].max() for k in range(1, N + 1)]
    assert all(p == pytest.approx(peaks[0], rel=1e-6) for p in peaks)
    # C_stack = consequence here (factor pinned to 1.0, relbonus=0)
    assert peaks[0] == pytest.approx(consequence, rel=1e-6)


def test_want_stack_false_returns_none_and_skips_allocation():
    grid, stack, _rollout = paint_track(
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5,
        PXX, PYY, PVX, PVY, 0.9, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        want_stack=False,
    )
    assert stack is None
    assert grid.shape == (ROWS, COLS)


# ------------------------------------------------------------ two-hypothesis CPA

def test_factor_stat_none_is_byte_identical_to_single_factor():
    """factor_stat's documented default (None -> reuse `factor`) must
    reproduce the pre-two-hypothesis-CPA grid exactly -- every existing
    positional call in this file relies on that, since none of them ever
    pass factor_stat at all."""
    args = (2.05, -3.05, 2.05, -3.05, 0.0, 1.0, 0.5,
            PXX, PYY, PVX, PVY, 0.9, 1.6, 0.1,
            RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL)
    grid_implicit, stack_implicit, _ = paint_track(*args)
    grid_explicit, stack_explicit, _ = paint_track(*args, factor_stat=None)
    assert np.array_equal(grid_implicit, grid_explicit)
    assert np.array_equal(stack_implicit, stack_explicit)


def test_factor_stat_only_changes_the_stationary_weight():
    """A track with pmot=0 paints ONLY its stationary hypothesis (w_move is
    zero every step), so the grid's peak at (stat_x, stat_y) is exactly
    C_stat = combine_severity(consequence, factor_stat, relbonus) --
    _splat_into's own guarantee that a Gaussian's value at its own mean is
    its weight, unaffected by variance. `factor` (the moving-hypothesis
    argument) must have NO effect at all in this case."""
    x, y = 2.05, -3.05
    consequence, relbonus = 0.6, 0.0
    row, col = cell_index(x, y)

    grid_a, _stack, _ = paint_track(
        x, y, x, y, 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, consequence, 1.0, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        want_stack=False, factor_stat=2.0,
    )
    grid_b, _stack, _ = paint_track(
        x, y, x, y, 0.0, 0.0, 0.0,
        PXX, PYY, PVX, PVY, consequence, 99.0, relbonus,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        want_stack=False, factor_stat=2.0,
    )
    expected = min(1.0, consequence * 2.0 + relbonus)
    assert grid_a[row, col] == pytest.approx(expected, rel=1e-5)
    assert grid_a[row, col] == pytest.approx(grid_b[row, col], rel=1e-6)


def test_factor_stat_versus_factor_diverge_the_two_hypotheses():
    """The two-hypothesis CPA scenario the feature exists for: a track
    whose MOVING-hypothesis CPA factor reads inert (factor=1.0, e.g. it
    appears to keep pace with the robot) but whose STATIONARY-hypothesis
    factor is highly amplified (e.g. the robot is actually closing on it
    head-on). With pmot=0.5 both hypotheses paint, and raising factor_stat
    alone must raise the grid's peak even though factor (moving) is
    unchanged."""
    x, y = 0.05, 0.05
    args_common = (x, y, x, y, 1.0, 0.0, 0.5,
                   PXX, PYY, PVX, PVY, 0.9, 1.0, 0.0,
                   RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL)
    grid_inert, _stack, _ = paint_track(*args_common, want_stack=False)
    grid_amplified, _stack, _ = paint_track(
        *args_common, want_stack=False, factor_stat=1.8)

    assert grid_amplified.max() > grid_inert.max()


def test_extrapolated_move_start_shifts_only_the_moving_hypothesis():
    """paint_track itself is agnostic to WHY move_x/move_y differ from
    stat_x/stat_y (the Node computes that offset from track age -- see
    extrapolate_to_now in predictive_risk_costmap_node's module docstring
    and _predict_and_splat); this only checks that paint_track honors the
    two positions independently: the stationary blob stays at (x, y) and
    the moving rollout's FIRST step is offset by the extra distance."""
    x, y = 0.05, 0.05
    vx, vy = 1.0, 0.0
    extra = 0.5  # as if the track were "extra / vx" seconds stale
    grid_fresh, stack_fresh = paint(x, y, vx, vy, pmot=1.0, consequence=0.9)
    grid_stale, stack_stale = paint(
        x, y, vx, vy, pmot=1.0, consequence=0.9, move_x=x + extra, move_y=y)

    row1, col1 = np.unravel_index(np.argmax(stack_fresh[1]), stack_fresh[1].shape)
    row2, col2 = np.unravel_index(np.argmax(stack_stale[1]), stack_stale[1].shape)
    want_row1, want_col1 = cell_index(x + vx * PRED_DT, y)
    want_row2, want_col2 = cell_index(x + extra + vx * PRED_DT, y)
    assert (row1, col1) == (want_row1, want_col1)
    assert (row2, col2) == (want_row2, want_col2)
    assert (row1, col1) != (row2, col2)


def test_stack_consequence_overrides_magnitude_but_not_grid():
    """paint_track(stack_consequence=...) sets the STAGE 5 severity
    independently of the collapsed grid's confidence-scaled consequence."""
    x, y, vx, vy = 1.05, -3.05, 0.0, 1.0
    grid_a, stack_a, _ = paint_track(
        x, y, x, y, vx, vy, 1.0, PXX, PYY, PVX, PVY, 0.2, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL)
    grid_b, stack_b, _ = paint_track(
        x, y, x, y, vx, vy, 1.0, PXX, PYY, PVX, PVY, 0.2, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        stack_consequence=0.75)
    assert np.array_equal(grid_a, grid_b)
    assert stack_a[1].max() == pytest.approx(0.2, rel=1e-6)
    assert stack_b[1].max() == pytest.approx(0.75, rel=1e-6)


# ------------------------------------------------------------ unknown-but-moving
# WP3 deliverable 2: resolve_stack_consequence() is the pure decision function
# the node calls before handing paint_track its stack_consequence -- covered
# directly here (the decision) and via paint_track (the painted outcome),
# same split as the extent-floor tests below.

UNKNOWN_KW = dict(
    stack_paint_unknown_moving=True, unknown_moving_risk=0.60,
    unknown_moving_pmot_min=0.5,
)
STACK_CATEGORIES = {"person", "robot", "wheeled"}


def test_resolve_stack_consequence_known_category_passes_through():
    """A category already in stack_categories is untouched by the new
    unknown-moving logic -- this is the pre-existing person/robot/wheeled
    behaviour, regression-checked."""
    got = resolve_stack_consequence(
        "person", STACK_CATEGORIES, pmot=0.0, base_stack_consequence=0.90,
        **UNKNOWN_KW)
    assert got == pytest.approx(0.90)


def test_resolve_stack_consequence_unknown_moving_substitutes_flat_risk():
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        **UNKNOWN_KW)
    assert got == pytest.approx(0.60)


def test_resolve_stack_consequence_static_unknown_stays_excluded():
    """pmot below unknown_moving_pmot_min -- a genuinely STATIC unlabeled
    track (e.g. static lidar clutter) must not paint the stack at all."""
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=0.0, base_stack_consequence=0.40,
        **UNKNOWN_KW)
    assert got == 0.0


def test_resolve_stack_consequence_flag_off_excludes_even_when_moving():
    kw = dict(UNKNOWN_KW)
    kw["stack_paint_unknown_moving"] = False
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40, **kw)
    assert got == 0.0


def test_resolve_stack_consequence_furniture_excluded_regardless_of_motion():
    """Furniture (or any category that isn't literally 'unknown') stays
    excluded no matter how confidently it reads as moving."""
    got = resolve_stack_consequence(
        "furniture", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.35,
        **UNKNOWN_KW)
    assert got == 0.0


def test_unknown_moving_track_paints_flat_risk_along_velocity():
    """End-to-end: a resolved unknown-moving stack_consequence, handed to
    paint_track with pmot=1.0 (stationary weight exactly zero), must paint
    unknown_moving_risk -- not the base class-derived magnitude -- in
    every layer 1..N, at the constant-velocity rollout position. Mirrors
    test_moving_track_layer_k_argmax_tracks_constant_velocity's shape."""
    x, y = 1.05, 2.05
    vx, vy = 0.0, 1.0
    stack_consequence = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        **UNKNOWN_KW)
    assert stack_consequence == pytest.approx(0.60)

    _grid, stack, _rollout = paint_track(
        x, y, x, y, vx, vy, 1.0,
        PXX, PYY, PVX, PVY, 0.40, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        stack_consequence=stack_consequence,
    )
    assert stack[0].max() == pytest.approx(0.60, rel=1e-6)   # "now" holds the mover too
    for k in range(1, N + 1):
        expected_y = y + vy * k * PRED_DT
        want_row, want_col = cell_index(x, expected_y)
        got_row, got_col = np.unravel_index(np.argmax(stack[k]), stack[k].shape)
        assert (got_row, got_col) == (want_row, want_col), f"layer {k}"
        assert stack[k].max() == pytest.approx(0.60, rel=1e-6)


def test_unknown_static_track_paints_nothing_in_stack():
    """pmot=0.0 -> resolve_stack_consequence excludes it -> paint_track
    must produce an all-zero stack."""
    stack_consequence = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=0.0, base_stack_consequence=0.40,
        **UNKNOWN_KW)
    assert stack_consequence == 0.0

    _grid, stack, _rollout = paint_track(
        1.05, 2.05, 1.05, 2.05, 0.0, 1.0, 0.0,
        PXX, PYY, PVX, PVY, 0.40, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        stack_consequence=stack_consequence,
    )
    assert stack.max() == pytest.approx(0.0)


def test_unknown_moving_track_paints_nothing_when_flag_off():
    kw = dict(UNKNOWN_KW)
    kw["stack_paint_unknown_moving"] = False
    stack_consequence = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40, **kw)
    assert stack_consequence == 0.0

    _grid, stack, _rollout = paint_track(
        1.05, 2.05, 1.05, 2.05, 0.0, 1.0, 1.0,
        PXX, PYY, PVX, PVY, 0.40, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        stack_consequence=stack_consequence,
    )
    assert stack.max() == pytest.approx(0.0)


# ------------------------------------------------------------ semantic keep-out floor
# WP3 deliverable 3: resolve_extent_half() combines the cap with the floor;
# select_min_extent() picks the floor. Both pure functions, tested directly
# per the module docstring's "if floor > cap, floor wins" contract.

MIN_EXTENT_KW = dict(
    min_extent_person=0.50, min_extent_robot=0.45, min_extent_wheeled=0.45,
    min_extent_forklift=0.80, min_extent_other=0.25,
)


def test_extent_floor_widens_a_point_like_person_track():
    # person: cap 0.30 (extent_cap_person_m default), floor 0.50 -> floor wins
    half = resolve_extent_half(bbox_half=0.02, cap=0.30, floor=0.50)
    assert half == pytest.approx(0.50)


def test_extent_cap_still_bounds_an_oversized_bbox():
    # a 3 m bbox (half=1.5) under a category whose floor does NOT exceed
    # its cap (extent_cap_other_m == min_extent_other_m == 0.25 by default)
    # -- the cap must still be the binding constraint.
    half = resolve_extent_half(bbox_half=1.5, cap=0.25, floor=0.25)
    assert half == pytest.approx(0.25)


def test_extent_floor_wins_when_it_exceeds_the_cap():
    # forklift floor 0.80 > wheeled cap 0.50 -- floor wins regardless of
    # bbox size, per the module docstring's documented contract.
    half_point = resolve_extent_half(bbox_half=0.02, cap=0.50, floor=0.80)
    half_big = resolve_extent_half(bbox_half=1.5, cap=0.50, floor=0.80)
    assert half_point == pytest.approx(0.80)
    assert half_big == pytest.approx(0.80)


def test_select_min_extent_forklift_label_overrides_wheeled_category():
    """'forklift' is matched on the raw LABEL, not just its 'wheeled'
    category -- an ordinary cart (also category 'wheeled') must NOT get
    the forklift floor."""
    forklift = select_min_extent("forklift", "wheeled", **MIN_EXTENT_KW)
    cart = select_min_extent("cart", "wheeled", **MIN_EXTENT_KW)
    assert forklift == pytest.approx(0.80)
    assert cart == pytest.approx(0.45)


def test_select_min_extent_by_category():
    assert select_min_extent("person", "person", **MIN_EXTENT_KW) == pytest.approx(0.50)
    assert select_min_extent("mobile robot", "robot", **MIN_EXTENT_KW) == pytest.approx(0.45)
    assert select_min_extent("chair", "furniture", **MIN_EXTENT_KW) == pytest.approx(0.25)


def test_select_min_extent_forklift_substring_case_insensitive():
    assert select_min_extent("Toy Forklift", "wheeled", **MIN_EXTENT_KW) == pytest.approx(0.80)


# ------------------------------------------------------------ WP-A: bounded
# rollout uncertainty, young-track gate, speed plausibility, comet fix
# (2026-09-10 "SRM fills half the map with real perception" finding). See
# predictive_risk_costmap_node.py's module docstring WP-A section and
# paint_track/is_track_mature/speed_cap_for_category's own docstrings.

MAX_SIGMA_M_DEFAULT = 0.6   # paint_track's own default when not overridden


def test_high_velocity_prior_never_exceeds_max_sigma():
    """Pvx=Pvy=1.0 (m/s)^2 -- the Kalman filter's OWN track-init prior
    (object_tracker_node's default, not evidence) -- must never make a
    rollout step's splat wider than max_sigma_m (0.6 m default), however
    large the raw t^2*Pv term would otherwise be. Checked two ways: the
    returned var_x/var_y for the farthest rolled-out step, AND the actual
    spatial extent of that layer's splat (no cell > 0.05 farther than
    ~3*max_sigma from the mean)."""
    x, y = 0.0, 0.0
    vx, vy = 0.5, 0.0
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, 1.0,
        PXX, PYY, 1.0, 1.0, 0.9, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
    )
    assert rollout, "test setup: the moving hypothesis should still be painted"
    step, mx, my, var_x, var_y, _w_move = rollout[-1]
    assert var_x <= MAX_SIGMA_M_DEFAULT ** 2 + 1e-9
    assert var_y <= MAX_SIGMA_M_DEFAULT ** 2 + 1e-9

    layer = stack[step + 1]
    rs, cs = np.nonzero(layer > 0.05)
    assert rs.size > 0, "test setup: the far step should still paint something"
    xs = OX + (cs.astype(np.float64) + 0.5) * RES
    ys = OY + (rs.astype(np.float64) + 0.5) * RES
    dist = np.hypot(xs - mx, ys - my)
    assert dist.max() <= 3 * MAX_SIGMA_M_DEFAULT + RES


def test_moving_allowed_false_paints_stationary_only():
    """moving_allowed=False -- the caller's determination for e.g. a
    track with hits=2 (below min_hits_for_motion) -- must zero pmot's
    effect on EVERY weight: the moving hypothesis paints nothing at all
    (empty rollout, every stack layer identical to a pmot=0 splat), and
    the stationary blob carries the FULL weight C instead of
    (1 - pmot) * C."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    consequence = 0.9
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, 0.9,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=False,
    )
    assert rollout == []
    assert grid.max() == pytest.approx(consequence, rel=1e-6)
    for k in range(N + 1):
        assert np.array_equal(stack[k], stack[0]), f"layer {k} != layer 0"
        assert stack[k].max() == pytest.approx(consequence, rel=1e-6)


def test_is_track_mature_hits_below_threshold_is_young():
    assert is_track_mature(
        {"hits": 2.0, "age": 5.0},
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5) is False


def test_is_track_mature_age_below_threshold_is_young():
    assert is_track_mature(
        {"hits": 10.0, "age": 0.3},
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5) is False


def test_is_track_mature_both_satisfied():
    assert is_track_mature(
        {"hits": 10.0, "age": 5.0},
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5) is True


def test_is_track_mature_missing_keys_reads_as_mature():
    """Neither key present (gt_tracks_node's oracle tracks, or any
    publisher predating this feature) -- must read as mature, never as
    young, so perception:=oracle keeps working unchanged."""
    assert is_track_mature({}, min_track_age_for_motion_sec=1.0, min_hits_for_motion=5) is True
    assert is_track_mature(
        {"pmot": 0.9, "vx": 1.0, "vy": 0.0},
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5) is True


ROBOT_SPEED_CAPS = {"person": 2.0, "robot": 1.5, "wheeled": 2.0}


def test_speed_cap_for_category_known_category():
    assert speed_cap_for_category(
        "robot", ROBOT_SPEED_CAPS, default_cap=1.2) == pytest.approx(1.5)


def test_speed_cap_for_category_unknown_falls_back_to_default():
    assert speed_cap_for_category(
        "furniture", ROBOT_SPEED_CAPS, default_cap=1.2) == pytest.approx(1.2)


def test_speed_above_category_cap_paints_stationary_only():
    """A |v|=3.0 m/s 'robot'-category track exceeds max_speed_robot_mps
    (1.5 default) -- the node's moving_allowed determination (here
    computed the same way _resolve_moving_allowed does, without needing
    the rclpy Node) must come out False, and paint_track must then paint
    stationary-only, same shape as
    test_moving_allowed_false_paints_stationary_only."""
    vx, vy = 3.0, 0.0
    speed = math.hypot(vx, vy)
    cap = speed_cap_for_category("robot", ROBOT_SPEED_CAPS, default_cap=1.2)
    assert speed > cap
    moving_allowed = is_track_mature({"hits": 10.0, "age": 5.0}, 1.0, 5) and speed <= cap
    assert moving_allowed is False

    x, y = 1.05, 0.05
    consequence = 0.75
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, 0.9,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=moving_allowed,
    )
    assert rollout == []
    assert grid.max() == pytest.approx(consequence, rel=1e-6)


def test_oracle_track_without_hits_age_keys_still_paints_comet():
    """A class_id with neither 'hits' nor 'age' (gt_tracks_node's oracle
    tracks under perception:=oracle) must read as mature -- moving_allowed
    stays True and the moving hypothesis rolls out exactly as before this
    gate existed (mirrors test_moving_track_layer_k_argmax_tracks_
    constant_velocity's shape)."""
    kv = {"pmot": 0.9, "vx": 1.0, "vy": 0.0}  # no hits/age keys at all
    mature = is_track_mature(kv, min_track_age_for_motion_sec=1.0, min_hits_for_motion=5)
    assert mature is True

    x, y = 2.05, 0.05
    vx, vy = 1.0, 0.0
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, 0.9,
        PXX, PYY, PVX, PVY, 0.9, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=mature,
    )
    assert len(rollout) > 0
    for k in range(1, N + 1):
        expected_x = x + vx * k * PRED_DT
        want_row, want_col = cell_index(expected_x, y)
        got_row, got_col = np.unravel_index(np.argmax(stack[k]), stack[k].shape)
        assert (got_row, got_col) == (want_row, want_col), f"layer {k}"
        assert stack[k].max() > 0.0


# ------------------------------------------------------------ unknown-moving:
# stricter gates (hits/age/speed) added 2026-09-10 alongside the raised
# unknown_moving_pmot_min default (0.5 -> 0.8). All four new keyword
# arguments default to "no additional restriction" so every pre-2026-09-10
# call above (UNKNOWN_KW, no new kwargs) is unaffected by these gates.

def test_resolve_stack_consequence_unknown_moving_requires_hits():
    kw = dict(UNKNOWN_KW)
    kw["unknown_moving_pmot_min"] = 0.8
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        unknown_moving_min_hits=8, unknown_moving_min_age_sec=2.0,
        hits=3, age=5.0, speed=0.0, max_speed_unknown_mps=1.2, **kw)
    assert got == 0.0


def test_resolve_stack_consequence_unknown_moving_requires_age():
    kw = dict(UNKNOWN_KW)
    kw["unknown_moving_pmot_min"] = 0.8
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        unknown_moving_min_hits=8, unknown_moving_min_age_sec=2.0,
        hits=10, age=0.5, speed=0.0, max_speed_unknown_mps=1.2, **kw)
    assert got == 0.0


def test_resolve_stack_consequence_unknown_moving_requires_speed_cap():
    kw = dict(UNKNOWN_KW)
    kw["unknown_moving_pmot_min"] = 0.8
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        unknown_moving_min_hits=8, unknown_moving_min_age_sec=2.0,
        hits=10, age=5.0, speed=5.0, max_speed_unknown_mps=1.2, **kw)
    assert got == 0.0


def test_resolve_stack_consequence_unknown_moving_passes_all_gates():
    kw = dict(UNKNOWN_KW)
    kw["unknown_moving_pmot_min"] = 0.8
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        unknown_moving_min_hits=8, unknown_moving_min_age_sec=2.0,
        hits=10, age=5.0, speed=0.5, max_speed_unknown_mps=1.2, **kw)
    assert got == pytest.approx(0.60)


# ------------------------------------------------------------ WP2: resolve_moving_allowed
# Pure extraction of _resolve_moving_allowed's decision -- see that
# function's own docstring for the "immature" / "over_cap" /
# "below_min_speed" / "ok" precedence.

MIN_SPEED_MPS = 0.15


def test_resolve_moving_allowed_immature_track():
    """Below min_hits_for_motion -- fails is_track_mature before the speed
    checks are even consulted."""
    kv = {"hits": 2.0, "age": 5.0}
    allowed, reason = resolve_moving_allowed(
        kv, speed=1.0, category="robot",
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5,
        speed_caps=ROBOT_SPEED_CAPS, default_cap=1.2, min_speed_mps=MIN_SPEED_MPS)
    assert (allowed, reason) == (False, "immature")


def test_resolve_moving_allowed_over_cap():
    """Mature, but speed exceeds the category's plausibility cap."""
    kv = {"hits": 10.0, "age": 5.0}
    allowed, reason = resolve_moving_allowed(
        kv, speed=3.0, category="robot",
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5,
        speed_caps=ROBOT_SPEED_CAPS, default_cap=1.2, min_speed_mps=MIN_SPEED_MPS)
    assert (allowed, reason) == (False, "over_cap")


def test_resolve_moving_allowed_below_min_speed():
    """Mature, within cap, but below min_speed_mps -- the new 2026-09-10
    evening guard: camera jitter reading a few cm/s of "motion"."""
    kv = {"hits": 10.0, "age": 5.0}
    allowed, reason = resolve_moving_allowed(
        kv, speed=0.05, category="robot",
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5,
        speed_caps=ROBOT_SPEED_CAPS, default_cap=1.2, min_speed_mps=MIN_SPEED_MPS)
    assert (allowed, reason) == (False, "below_min_speed")


def test_resolve_moving_allowed_ok():
    """Mature, within cap, at or above min_speed_mps -- the moving
    hypothesis is trusted."""
    kv = {"hits": 10.0, "age": 5.0}
    allowed, reason = resolve_moving_allowed(
        kv, speed=0.6, category="robot",
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5,
        speed_caps=ROBOT_SPEED_CAPS, default_cap=1.2, min_speed_mps=MIN_SPEED_MPS)
    assert (allowed, reason) == (True, "ok")


def test_resolve_moving_allowed_oracle_kv_no_hits_age_at_speed_ok():
    """An oracle-style class_id (gt_tracks_node -- no 'hits'/'age' keys at
    all) at a genuine 0.6 m/s must read (True, "ok"): is_track_mature
    treats a missing hits/age pair as "no evidence against maturity" (see
    its own docstring), the speed is within the default robot cap, and
    0.6 m/s clears min_speed_mps -- so oracle mode is unaffected by this
    WP2 guard end to end, not just at the is_track_mature layer."""
    kv = {"pmot": 0.9, "vx": 0.6, "vy": 0.0}  # no hits/age keys
    allowed, reason = resolve_moving_allowed(
        kv, speed=0.6, category="robot",
        min_track_age_for_motion_sec=1.0, min_hits_for_motion=5,
        speed_caps=ROBOT_SPEED_CAPS, default_cap=1.2, min_speed_mps=MIN_SPEED_MPS)
    assert (allowed, reason) == (True, "ok")


def test_resolve_stack_consequence_missing_hits_age_bypasses_those_gates():
    """hits=None/age=None (a track/publisher that doesn't carry either
    field) must not be blocked by the hits/age gates -- only speed and
    pmot still apply. Mirrors is_track_mature's own "missing key means no
    evidence against it" convention."""
    kw = dict(UNKNOWN_KW)
    kw["unknown_moving_pmot_min"] = 0.8
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=1.0, base_stack_consequence=0.40,
        unknown_moving_min_hits=8, unknown_moving_min_age_sec=2.0,
        hits=None, age=None, speed=0.5, max_speed_unknown_mps=1.2, **kw)
    assert got == pytest.approx(0.60)


# ------------------------------------------------------------ WP-A: class-
# agnostic (motion-first) consequence, 2026-09-11. See predictive_risk_
# costmap_node.py's module docstring "WP-A CLASS-AGNOSTIC CONSEQUENCE"
# section and resolve_agnostic_consequence()/resolve_stack_consequence()'s
# own docstrings.

AGNOSTIC_VALUE = 0.75


def test_resolve_agnostic_consequence_no_modifier_ignores_class_value():
    """semantic_modifier off (the default): agnostic_value outright,
    whatever the class-derived base_consequence says -- furniture (whose
    class value would ordinarily be low) and person (whose class value
    would ordinarily be high) both read exactly the flat agnostic value."""
    assert resolve_agnostic_consequence(
        base_consequence=0.35, agnostic_value=AGNOSTIC_VALUE,
        semantic_modifier=False) == pytest.approx(0.75)
    assert resolve_agnostic_consequence(
        base_consequence=0.90, agnostic_value=AGNOSTIC_VALUE,
        semantic_modifier=False) == pytest.approx(0.75)


def test_resolve_agnostic_consequence_modifier_only_pushes_up():
    """semantic_modifier on: max(agnostic, class) -- a person's 0.90 class
    value beats the flat 0.75; furniture's lower class value does not pull
    the reading below 0.75 (a modifier, never a veto)."""
    assert resolve_agnostic_consequence(
        base_consequence=0.90, agnostic_value=AGNOSTIC_VALUE,
        semantic_modifier=True) == pytest.approx(0.90)
    assert resolve_agnostic_consequence(
        base_consequence=0.35, agnostic_value=AGNOSTIC_VALUE,
        semantic_modifier=True) == pytest.approx(0.75)


MOTION_FIRST_KW = dict(
    stack_paint_unknown_moving=False, unknown_moving_risk=0.60,
    unknown_moving_pmot_min=0.8, motion_first=True,
    agnostic_value=AGNOSTIC_VALUE, semantic_modifier=False,
)


def test_resolve_stack_consequence_motion_first_furniture_paints_agnostic():
    """A 'furniture' category -- excluded outright under the pre-WP-A
    category gate (test_resolve_stack_consequence_furniture_excluded_
    regardless_of_motion) -- paints the flat agnostic value under
    motion_first: no stack_categories membership required at all."""
    got = resolve_stack_consequence(
        "furniture", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.35,
        **MOTION_FIRST_KW)
    assert got == pytest.approx(0.75)


def test_resolve_stack_consequence_motion_first_person_without_modifier():
    got = resolve_stack_consequence(
        "person", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.90,
        **MOTION_FIRST_KW)
    assert got == pytest.approx(0.75)


def test_resolve_stack_consequence_motion_first_person_with_modifier():
    kw = dict(MOTION_FIRST_KW)
    kw["semantic_modifier"] = True
    got = resolve_stack_consequence(
        "person", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.90,
        **kw)
    assert got == pytest.approx(0.90)


def test_resolve_stack_consequence_motion_first_unknown_paints_agnostic():
    got = resolve_stack_consequence(
        "unknown", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.0,
        **MOTION_FIRST_KW)
    assert got == pytest.approx(0.75)


def test_resolve_stack_consequence_motion_first_low_score_still_vetoes():
    """stack_min_track_score still vetoes low-confidence tracks under
    motion_first -- the category gate is removed, the score veto is not."""
    got = resolve_stack_consequence(
        "wheeled", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.65,
        score=0.05, stack_min_track_score=0.1, **MOTION_FIRST_KW)
    assert got == 0.0


def test_resolve_stack_consequence_motion_first_score_at_or_above_floor_passes():
    got = resolve_stack_consequence(
        "wheeled", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.65,
        score=0.5, stack_min_track_score=0.1, **MOTION_FIRST_KW)
    assert got == pytest.approx(0.75)


def test_resolve_stack_consequence_motion_first_missing_score_bypasses_veto():
    """score/stack_min_track_score default to None -- pre-2026-09-11
    callers that never pass them are unaffected (no veto applied)."""
    got = resolve_stack_consequence(
        "wheeled", STACK_CATEGORIES, pmot=0.9, base_stack_consequence=0.65,
        **MOTION_FIRST_KW)
    assert got == pytest.approx(0.75)


def test_paint_track_pmot_below_stack_moving_pmot_min_paints_stationary_only():
    """WP-A: moving_allowed = allowed and pmot >= stack_moving_pmot_min --
    a mature, plausible-speed track (e.g. a promoted 'table') whose OWN
    pmot (0.3) is below stack_moving_pmot_min (0.5) still paints (agnostic
    consequence), but ONLY its stationary blob: empty rollout, every stack
    layer identical to layer 0 -- same shape as
    test_moving_allowed_false_paints_stationary_only."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    pmot = 0.3
    stack_moving_pmot_min = 0.5
    allowed = True  # is_track_mature/speed_cap/min_speed all clear
    moving_allowed = allowed and pmot >= stack_moving_pmot_min
    assert moving_allowed is False

    consequence = resolve_agnostic_consequence(0.35, AGNOSTIC_VALUE, False)
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=moving_allowed,
    )
    assert rollout == []
    assert grid.max() == pytest.approx(consequence, rel=1e-6)
    for k in range(N + 1):
        assert np.array_equal(stack[k], stack[0]), f"layer {k} != layer 0"
        assert stack[k].max() == pytest.approx(consequence, rel=1e-6)


def test_paint_track_pmot_at_or_above_stack_moving_pmot_min_paints_moving():
    """Same setup, pmot now at the floor -- the moving hypothesis rolls
    out normally."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    pmot = 0.5
    stack_moving_pmot_min = 0.5
    allowed = True
    moving_allowed = allowed and pmot >= stack_moving_pmot_min
    assert moving_allowed is True

    consequence = resolve_agnostic_consequence(0.35, AGNOSTIC_VALUE, False)
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=moving_allowed,
    )
    assert len(rollout) > 0


def test_paint_track_pmov_below_stack_moving_pmov_min_paints_stationary_only():
    """Same shape as test_paint_track_pmot_below_stack_moving_pmot_min_...,
    but gating on pmov (p_movable) instead of pmot: a track whose OWN
    movability belief (0.2, e.g. parked shelving momentarily misread as
    moving) is below stack_moving_pmov_min (0.4) paints agnostic
    consequence at its stationary blob only, even though pmot alone would
    have allowed the moving hypothesis."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    pmot = 0.9  # clears stack_moving_pmot_min on its own
    pmov = 0.2
    stack_moving_pmov_min = 0.4
    allowed = True  # is_track_mature/speed_cap/min_speed all clear
    moving_allowed = allowed and pmov >= stack_moving_pmov_min
    assert moving_allowed is False

    consequence = resolve_agnostic_consequence(0.35, AGNOSTIC_VALUE, False)
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=moving_allowed,
    )
    assert rollout == []
    assert grid.max() == pytest.approx(consequence, rel=1e-6)
    for k in range(N + 1):
        assert np.array_equal(stack[k], stack[0]), f"layer {k} != layer 0"
        assert stack[k].max() == pytest.approx(consequence, rel=1e-6)


def test_paint_track_pmov_at_or_above_stack_moving_pmov_min_paints_moving():
    """Same setup, pmov now at the floor -- the moving hypothesis rolls
    out normally. Also covers the default (stack_moving_pmov_min=0.0):
    pmov is always >= 0.0, so this gate is a no-op unless raised."""
    x, y = 1.05, 0.05
    vx, vy = 1.0, 0.0
    pmot = 0.9
    pmov = 0.4
    stack_moving_pmov_min = 0.4
    allowed = True
    moving_allowed = allowed and pmov >= stack_moving_pmov_min
    assert moving_allowed is True

    consequence = resolve_agnostic_consequence(0.35, AGNOSTIC_VALUE, False)
    grid, stack, rollout = paint_track(
        x, y, x, y, vx, vy, pmot,
        PXX, PYY, PVX, PVY, consequence, 1.0, 0.0,
        RES, OX, OY, ROWS, COLS, N, PRED_DT, GAMMA, VEL_INFL,
        moving_allowed=moving_allowed,
    )
    assert len(rollout) > 0
