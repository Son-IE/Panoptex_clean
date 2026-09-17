"""
WP-C's activity channel (A), at the same tier as test_spatial_prior_grid.py
and test_spatial_prior_headway.py: pure functions over numpy, no ROS, no
rclpy context, no simulator.

A is a class-agnostic, 5-min-half-life channel -- see
risk_perception/spatial_prior_node.py's module docstring "ACTIVITY (WP-C)"
section for the full design rationale (why it exists alongside the 2 h
lifelong S, why it is never gated by `frozen`/`learn_rate`, why it is never
persisted). The properties tested here are exactly the ones that section
promises:

  * the deposit ignores p_movable_min and category -- unlike S, a "table"
    label or a low p_movable never blocks it (that gating lives in
    spatial_prior_node._update's caller code, not in
    deposit_activity_into_grid itself, which this test exercises directly
    by simply never passing category/p_movable through at all -- there is
    no parameter for either);
  * the channel halves in value after one half-life's worth of decay
    (300 s default);
  * decay continues under `frozen=True` (the node's `is_frozen` state)
    while S does not change at all -- reproducing the node's actual
    _update() sequencing at the pure-function tier;
  * publish packs [0, 1] floats to 0-100 int8, same convention as S's
    OccupancyGrid.
"""

import numpy as np

from risk_perception.spatial_prior_node import (
    CategoryFlowGrid,
    decay_factor,
    deposit_activity_into_grid,
    deposit_into_grid,
    ema_deposit_clipped,
    is_frozen,
    occupancy_grid_data,
)

RES = 0.10
OX, OY = -5.0, -5.0
ROWS, COLS = 100, 100  # 10 m x 10 m


def cell_of(cx, cy):
    col = int((cx - OX) / RES)
    row = int((cy - OY) / RES)
    return row, col


def new_activity():
    return np.zeros((ROWS, COLS), dtype=np.float32)


# ------------------------------------------------------------------ deposit_activity_into_grid

def test_deposit_raises_activity_toward_one_at_the_center_cell():
    a = new_activity()
    deposit_activity_into_grid(a, cx=0.0, cy=0.0, radius=0.3, k=0.5,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert abs(a[r, c] - 0.5) < 1e-6  # A = 0 + k*(1-0) = k, same as S


def test_deposit_ignores_p_movable_min_and_category_by_construction():
    """deposit_activity_into_grid has no p_movable/category parameter at
    all -- unlike deposit_into_grid's caller in spatial_prior_node._update,
    which gates on p_movable_min BEFORE ever computing k for S. A "moving
    table" (p_movable well under any sane p_movable_min, category
    "furniture") deposits into A exactly the same as a "moving person"
    would, given the same k -- there is no gate to bypass because the
    function only ever sees pmot/confidence-derived k, never the label or
    p_movable at all. This mirrors how spatial_prior_node._update computes
    k_a before the p_movable_min `continue` that gates S/F."""
    furniture_a = new_activity()
    person_a = new_activity()
    # Same k for both -- as spatial_prior_node._update would compute for a
    # "table" track (p_movable=0.1, ignored) and a "person" track
    # (p_movable=0.9), both with pmot=0.8, confidence=1.0.
    k = 0.5 * 0.2 * 0.8 * 1.0  # activity_learn_rate * dt * pmot * conf
    deposit_activity_into_grid(furniture_a, cx=0.0, cy=0.0, radius=0.3, k=k,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    deposit_activity_into_grid(person_a, cx=0.0, cy=0.0, radius=0.3, k=k,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    assert np.array_equal(furniture_a, person_a)


def test_zero_or_negative_k_is_a_noop():
    a = new_activity()
    deposit_activity_into_grid(a, cx=0.0, cy=0.0, radius=0.3, k=0.0,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    assert np.all(a == 0.0)


def test_cells_outside_radius_are_untouched():
    a = new_activity()
    deposit_activity_into_grid(a, cx=0.0, cy=0.0, radius=0.2, k=0.9,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    far_r, far_c = cell_of(3.0, 3.0)
    assert a[far_r, far_c] == 0.0


def test_activity_never_exceeds_one_even_with_k_above_one():
    a = new_activity()
    deposit_activity_into_grid(a, cx=0.0, cy=0.0, radius=0.3, k=5.0,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert a[r, c] == 1.0


def test_deposit_far_outside_grid_bounds_does_not_crash():
    a = new_activity()
    deposit_activity_into_grid(a, cx=1000.0, cy=1000.0, radius=0.3, k=0.5,
                               resolution=RES, origin_x=OX, origin_y=OY,
                               rows=ROWS, cols=COLS)
    assert np.all(a == 0.0)


def test_repeated_deposits_saturate_toward_one_same_as_s():
    """A's EMA-toward-1 maths is byte-for-byte the same formula as S's
    (ema_deposit_clipped, shared by both) -- confirm the two channels
    converge identically for the same k, deposited the same number of
    times."""
    a = new_activity()
    g = CategoryFlowGrid(ROWS, COLS)
    for _ in range(50):
        deposit_activity_into_grid(a, cx=0.0, cy=0.0, radius=0.3, k=0.2,
                                   resolution=RES, origin_x=OX, origin_y=OY,
                                   rows=ROWS, cols=COLS)
        deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.2, vx=0.0, vy=0.0,
                          resolution=RES, origin_x=OX, origin_y=OY,
                          rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert abs(a[r, c] - g.s[r, c]) < 1e-9
    assert a[r, c] > 0.99


# ------------------------------------------------------------------ ema_deposit_clipped

def test_ema_deposit_clipped_matches_the_original_s_formula():
    region = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    inside = np.array([True, True, False])
    ema_deposit_clipped(region, inside, k=0.5)
    # cell 0: 0 + 0.5*(1-0) = 0.5 ; cell 1: 0.5 + 0.5*(1-0.5) = 0.75 ;
    # cell 2 untouched (inside=False)
    assert abs(region[0] - 0.5) < 1e-6
    assert abs(region[1] - 0.75) < 1e-6
    assert region[2] == 1.0


# ------------------------------------------------------------------ 300 s half-life decay

def test_activity_halves_after_one_half_life_of_decay():
    """300 s default half-life: a single decay_factor call for exactly
    300 s must halve A, independent of is_frozen (see the frozen test
    below for that half)."""
    a = new_activity()
    a[:] = 0.8
    factor = decay_factor(dt=300.0, half_life_s=300.0, frozen=False)
    a *= factor
    assert abs(float(a[0, 0]) - 0.4) < 1e-6


def test_activity_decay_over_many_short_ticks_matches_one_big_tick():
    """Repeated small-dt decay ticks (as spatial_prior_node._update applies
    them every timer period) must compound to the same half-life curve as
    one large tick -- decay_factor is exponential, so 30 ticks of 10 s each
    should land at the same value as one 300 s tick."""
    a_ticked = new_activity()
    a_ticked[:] = 0.8
    for _ in range(30):
        a_ticked *= decay_factor(dt=10.0, half_life_s=300.0, frozen=False)

    a_single = new_activity()
    a_single[:] = 0.8
    a_single *= decay_factor(dt=300.0, half_life_s=300.0, frozen=False)

    assert abs(float(a_ticked[0, 0]) - float(a_single[0, 0])) < 1e-5
    assert abs(float(a_ticked[0, 0]) - 0.4) < 1e-5


# ------------------------------------------------------------------ decay continues when frozen

def test_activity_decay_continues_when_node_is_frozen_while_s_does_not_change():
    """Reproduces spatial_prior_node._update's actual per-tick sequence: S's
    decay_factor call passes the node's `frozen` state (freezes when
    learn_rate<=0, per is_frozen); the activity channel's decay_factor call
    hard-codes frozen=False regardless -- see the module docstring's
    ACTIVITY section and _update's own comment on this. A study run
    (learn_rate=0.0, the every-study-run knob) must decay A on schedule
    while leaving S bit-for-bit frozen, the whole reason the two channels
    exist separately."""
    s_grid = CategoryFlowGrid(ROWS, COLS)
    deposit_into_grid(s_grid, cx=0.0, cy=0.0, radius=0.3, k=0.6, vx=0.0, vy=0.0,
                      resolution=RES, origin_x=OX, origin_y=OY,
                      rows=ROWS, cols=COLS)
    before_s = s_grid.s.copy()

    a = new_activity()
    a[:] = 0.8

    learn_rate, frozen_param = 0.0, False   # the study-run knob
    node_frozen = is_frozen(learn_rate, frozen_param)
    assert node_frozen is True

    for _ in range(30):
        dt = 10.0
        # S: gated by the node's frozen state (no-op every tick).
        s_factor = decay_factor(dt, half_life_s=7200.0, frozen=node_frozen)
        if s_factor != 1.0:
            s_grid.decay(s_factor)
        # A: NEVER gated by node_frozen -- always frozen=False here.
        a_factor = decay_factor(dt, half_life_s=300.0, frozen=False)
        if a_factor != 1.0:
            a *= a_factor

    assert np.array_equal(s_grid.s, before_s)          # S untouched
    assert float(a[0, 0]) < 0.8                          # A decayed
    assert abs(float(a[0, 0]) - 0.4) < 1e-4               # ~one half-life (300 s)


def test_activity_decay_factor_is_not_a_noop_even_though_is_frozen_would_say_so():
    """Sanity check on the exact bug this closes for A: calling
    decay_factor with the node's `frozen` flag (as S does) would make this
    a no-op; calling it with frozen=False (as A must) does not."""
    frozen_via_s_path = decay_factor(dt=300.0, half_life_s=300.0, frozen=True)
    frozen_via_a_path = decay_factor(dt=300.0, half_life_s=300.0, frozen=False)
    assert frozen_via_s_path == 1.0
    assert abs(frozen_via_a_path - 0.5) < 1e-9


# ------------------------------------------------------------------ occupancy_grid_data

def test_occupancy_grid_data_packs_zero_to_one_into_zero_to_hundred():
    values = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.999]], dtype=np.float32)
    packed = occupancy_grid_data(values)
    assert packed.dtype == np.int8
    assert packed.tolist() == [[0, 50, 100], [25, 75, 100]]


def test_occupancy_grid_data_clips_out_of_range_inputs():
    values = np.array([-0.5, 1.5], dtype=np.float32)
    packed = occupancy_grid_data(values)
    assert packed.tolist() == [0, 100]


def test_occupancy_grid_data_matches_the_original_s_publish_formula():
    """_publish() used to compute
    np.clip(np.round(combined*100.0),0,100).astype(np.int8) inline; confirm
    the factored-out helper reproduces that exactly for a representative
    S-like array."""
    combined = np.array([0.0, 0.05, 0.5, 0.999, 1.0], dtype=np.float32)
    expected = np.clip(np.round(combined * 100.0), 0, 100).astype(np.int8)
    assert np.array_equal(occupancy_grid_data(combined), expected)
