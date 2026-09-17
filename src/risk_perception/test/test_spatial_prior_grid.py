"""
Tier 0 of the Spatial-Flow prior: pure geometry, no ROS, no camera, no GPU,
no rclpy context. Runs in a second with plain pytest -- see
risk_perception/spatial_prior_node.py's CategoryFlowGrid and
deposit_into_grid for why this logic is a module-level function rather
than a method on the (rclpy Node) SpatialPriorNode, same rationale as
relation_matching.py / encounter_geometry.py / mask_relation.py.
"""

import os

import numpy as np
import pytest

from risk_perception.spatial_prior_node import (
    CategoryFlowGrid,
    decay_factor,
    deposit_into_grid,
    is_frozen,
    merge_group_flow,
    parse_flow_groups,
    record_pass_into_grid,
    reload_decay_factor,
    save_prior,
    track_id_int,
)

RES = 0.10
OX, OY = -5.0, -5.0
ROWS, COLS = 100, 100  # 10 m x 10 m


def new_grid():
    return CategoryFlowGrid(ROWS, COLS)


def cell_of(cx, cy):
    """World (x,y) -> (row, col) for asserting on a specific cell."""
    col = int((cx - OX) / RES)
    row = int((cy - OY) / RES)
    return row, col


# ------------------------------------------------------------------ CategoryFlowGrid

def test_new_grid_starts_at_zero():
    g = new_grid()
    assert g.s.shape == (ROWS, COLS)
    assert g.fx.shape == (ROWS, COLS)
    assert g.fy.shape == (ROWS, COLS)
    assert np.all(g.s == 0.0)
    assert np.all(g.fx == 0.0)
    assert np.all(g.fy == 0.0)


def test_decay_scales_all_three_channels():
    g = new_grid()
    g.s[:] = 1.0
    g.fx[:] = 2.0
    g.fy[:] = -3.0
    g.decay(0.5)
    assert np.all(g.s == 0.5)
    assert np.all(g.fx == 1.0)
    assert np.all(g.fy == -1.5)


# ------------------------------------------------------------------ deposit_into_grid

def test_deposit_raises_s_toward_one_at_the_center_cell():
    g = new_grid()
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.5, vx=1.0, vy=0.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert 0.0 < g.s[r, c] <= 1.0
    assert abs(g.s[r, c] - 0.5) < 1e-6  # S = 0 + k*(1-0) = k


def test_repeated_deposits_saturate_s_toward_one_not_past_it():
    g = new_grid()
    prev = 0.0
    for _ in range(50):
        deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.2, vx=0.0, vy=0.0,
                          resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
        r, c = cell_of(0.0, 0.0)
        assert g.s[r, c] >= prev  # monotonically non-decreasing
        assert g.s[r, c] <= 1.0
        prev = g.s[r, c]
    assert prev > 0.99  # converged close to 1


def test_f_converges_toward_observed_velocity_not_clipped_to_unit_range():
    g = new_grid()
    for _ in range(200):
        deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.1, vx=2.5, vy=-1.5,
                          resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert abs(g.fx[r, c] - 2.5) < 0.05
    assert abs(g.fy[r, c] - (-1.5)) < 0.05


def test_cells_outside_radius_are_untouched():
    g = new_grid()
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.2, k=0.9, vx=1.0, vy=1.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    far_r, far_c = cell_of(3.0, 3.0)
    assert g.s[far_r, far_c] == 0.0
    assert g.fx[far_r, far_c] == 0.0


def test_zero_or_negative_k_is_a_noop():
    g = new_grid()
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.0, vx=5.0, vy=5.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    assert np.all(g.s == 0.0)
    assert np.all(g.fx == 0.0)


def test_deposit_far_outside_grid_bounds_does_not_crash():
    g = new_grid()
    deposit_into_grid(g, cx=1000.0, cy=1000.0, radius=0.3, k=0.5, vx=1.0, vy=1.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    assert np.all(g.s == 0.0)  # entirely off-grid: no effect, no exception


def test_large_velocity_is_clamped_defensively():
    """One deposit at k=1.0 (the theoretical max) with a corrupted/huge
    velocity reading should not leave the grid holding an unbounded value."""
    g = new_grid()
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=1.0, vx=500.0, vy=-500.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert g.fx[r, c] <= 10.0
    assert g.fy[r, c] >= -10.0


def test_s_never_exceeds_one_even_with_k_above_one():
    g = new_grid()
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=5.0, vx=0.0, vy=0.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    r, c = cell_of(0.0, 0.0)
    assert g.s[r, c] == 1.0


# ------------------------------------------------------------------ is_frozen / decay_factor
#
# The frozen-prior contract (README section 8 / CLAUDE.md's "the spatial
# prior must be frozen during a study run" gotcha): learn_rate<=0 OR the
# explicit `frozen` param must stop decay, deposits, headway samples, AND
# saving. Deposits/headway were already gated on k/learn_rate<=0; decay was
# NOT, which is the bug being closed here.

def test_is_frozen_true_when_learn_rate_non_positive():
    assert is_frozen(0.0, False) is True
    assert is_frozen(-0.1, False) is True


def test_is_frozen_false_when_learning_normally():
    assert is_frozen(0.20, False) is False


def test_is_frozen_true_when_explicit_param_forces_it_despite_positive_learn_rate():
    """The explicit `frozen` param must win even if learn_rate itself is
    still configured to learn -- a caller forcing frozen behaviour for a
    reason other than the learn_rate knob (e.g. a bench harness) must get
    the full contract, not just whatever learn_rate<=0 alone would gate."""
    assert is_frozen(0.20, True) is True


def test_decay_factor_is_a_noop_when_frozen_even_with_a_half_life_configured():
    assert decay_factor(dt=3600.0, half_life_s=7200.0, frozen=True) == 1.0


def test_decay_factor_is_a_noop_when_half_life_is_zero():
    assert decay_factor(dt=3600.0, half_life_s=0.0, frozen=False) == 1.0


def test_decay_factor_applies_the_half_life_when_not_frozen():
    factor = decay_factor(dt=7200.0, half_life_s=7200.0, frozen=False)
    assert abs(factor - 0.5) < 1e-9


def test_frozen_leaves_s_and_headway_unchanged_across_many_updates_with_elapsed_time():
    """Simulates spatial_prior_node._update()'s per-tick sequence (decay
    gated by decay_factor, deposit k / record_pass learn_rate gated by the
    same effective_learn_rate the node computes) across many ticks and a
    large elapsed time -- long enough that an ungated forget_half_life_s
    decay would visibly fade S. Frozen must leave S bit-for-bit identical
    and record zero headway samples, reproducing the bug this closes
    (S dropping from 172 to 67 strong cells across one study run)."""
    g = new_grid()
    # Seed some evidence as if a warm-up run had already learned it.
    deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=0.6, vx=1.0, vy=0.0,
                      resolution=RES, origin_x=OX, origin_y=OY, rows=ROWS, cols=COLS)
    before_s = g.s.copy()
    before_fx = g.fx.copy()

    learn_rate, frozen_param = 0.0, False   # the study-run knob
    assert is_frozen(learn_rate, frozen_param) is True
    total_samples = 0
    now = 0.0
    for tick in range(50):
        dt = 3600.0    # 1 h per tick -- many half-lives across the loop
        now += dt
        factor = decay_factor(dt, half_life_s=7200.0,
                              frozen=is_frozen(learn_rate, frozen_param))
        if factor != 1.0:
            g.decay(factor)
        effective_learn_rate = (0.0 if is_frozen(learn_rate, frozen_param)
                                else learn_rate)
        k = effective_learn_rate * dt * 1.0 * 1.0
        deposit_into_grid(g, cx=0.0, cy=0.0, radius=0.3, k=k, vx=1.0, vy=0.0,
                          resolution=RES, origin_x=OX, origin_y=OY,
                          rows=ROWS, cols=COLS)
        total_samples += record_pass_into_grid(
            g, cx=0.0, cy=0.0, radius=0.3,
            track_id=track_id_int(str(tick)), now=now,
            learn_rate=effective_learn_rate, headway_min_s=2.0,
            headway_alpha=0.2, resolution=RES, origin_x=OX, origin_y=OY,
            rows=ROWS, cols=COLS)

    assert np.array_equal(g.s, before_s)
    assert np.array_equal(g.fx, before_fx)
    assert total_samples == 0
    assert np.all(g.headway_count == 0)


# ------------------------------------------------------------------ parse_flow_groups

def test_parse_flow_groups_default_splits_into_group_name_and_members():
    assert parse_flow_groups(["robot,wheeled"]) == [("robot", ["robot", "wheeled"])]


def test_parse_flow_groups_strips_whitespace_and_skips_blank_specs():
    assert parse_flow_groups([" robot , wheeled ", "", "  ", "person"]) == [
        ("robot", ["robot", "wheeled"]),
        ("person", ["person"]),
    ]


# ------------------------------------------------------------------ merge_group_flow

def test_merge_group_flow_picks_max_s_and_the_matching_f_per_cell():
    a = new_grid()
    b = new_grid()
    r, c = cell_of(0.0, 0.0)
    a.s[r, c] = 0.3
    a.fx[r, c], a.fy[r, c] = 1.0, 0.0
    b.s[r, c] = 0.8
    b.fx[r, c], b.fy[r, c] = -2.0, 3.0

    s, fx, fy = merge_group_flow([a, b], ROWS, COLS)
    assert s[r, c] == np.float32(0.8)
    assert fx[r, c] == -2.0        # b's heading, not a's, not an average
    assert fy[r, c] == 3.0


def test_merge_group_flow_with_one_empty_member_equals_the_other_member():
    populated = new_grid()
    r, c = cell_of(1.0, 1.0)
    populated.s[r, c] = 0.55
    populated.fx[r, c], populated.fy[r, c] = 0.4, -0.2
    empty = new_grid()

    s, fx, fy = merge_group_flow([populated, empty], ROWS, COLS)
    assert np.array_equal(s, populated.s)
    assert np.array_equal(fx, populated.fx)
    assert np.array_equal(fy, populated.fy)

    # Order must not matter.
    s2, fx2, fy2 = merge_group_flow([empty, populated], ROWS, COLS)
    assert np.array_equal(s2, populated.s)
    assert np.array_equal(fx2, populated.fx)


def test_merge_group_flow_all_members_none_is_all_zero():
    s, fx, fy = merge_group_flow([None, None], ROWS, COLS)
    assert np.all(s == 0.0)
    assert np.all(fx == 0.0)
    assert np.all(fy == 0.0)


# ------------------------------------------------------------------ save_prior

def test_save_prior_frozen_is_a_noop_returns_false_and_writes_nothing(tmp_path):
    path = str(tmp_path / "prior.npz")
    g = new_grid()
    g.s[0, 0] = 0.9
    wrote, msg = save_prior(
        path, {"robot": g}, RES, OX, OY, ROWS, COLS,
        deposits=5, headway_samples=3, frozen=True, persist_blocked=False)
    assert wrote is False
    assert msg is not None and "FROZEN" in msg
    assert not os.path.exists(path)


def test_save_prior_persist_blocked_is_a_silent_noop(tmp_path):
    path = str(tmp_path / "prior.npz")
    wrote, msg = save_prior(
        path, {}, RES, OX, OY, ROWS, COLS,
        deposits=0, headway_samples=0, frozen=False, persist_blocked=True)
    assert wrote is False
    assert msg is None
    assert not os.path.exists(path)


def test_save_prior_writes_the_file_and_returns_true_when_not_frozen(tmp_path):
    path = str(tmp_path / "prior.npz")
    g = new_grid()
    g.s[0, 0] = 0.42
    wrote, msg = save_prior(
        path, {"robot": g}, RES, OX, OY, ROWS, COLS,
        deposits=5, headway_samples=3, frozen=False, persist_blocked=False)
    assert wrote is True
    assert msg is not None and "saved prior" in msg
    assert os.path.exists(path)


# ------------------------------------------------- reload_decay_factor (2026-09-13)
def test_reload_decay_is_a_noop_when_frozen_however_long_the_gap():
    """The frozen contract is stronger on reload than during a run: a study
    sweep loads ONE warm-up prior into arm after arm over several hours, so
    aging it on load would hand the last arm a weaker prior than the first
    and the arms would stop being comparable."""
    factor, reason = reload_decay_factor(
        saved_at=0.0, now=30 * 24 * 3600.0, half_life_s=1800.0, frozen=True)
    assert factor == 1.0
    assert "frozen" in reason


def test_reload_decay_applies_the_half_life_over_the_wall_clock_gap():
    factor, reason = reload_decay_factor(
        saved_at=1000.0, now=1000.0 + 1800.0, half_life_s=1800.0, frozen=False)
    assert factor == pytest.approx(0.5)
    assert "0.50 h" in reason


def test_reload_decay_compounds_over_several_half_lives():
    factor, _ = reload_decay_factor(
        saved_at=0.0, now=3 * 1800.0, half_life_s=1800.0, frozen=False)
    assert factor == pytest.approx(0.125)


def test_reload_decay_is_a_noop_without_a_saved_timestamp():
    """A pre-2026-09-13 .npz has no saved_at; _load falls back to the file
    mtime, but if even that is unavailable the prior must load verbatim
    rather than be aged by a guessed interval."""
    factor, _ = reload_decay_factor(
        saved_at=None, now=9999.0, half_life_s=1800.0, frozen=False)
    assert factor == 1.0


def test_reload_decay_refuses_to_amplify_on_a_backwards_clock():
    """A negative elapsed (clock skew, or a file from a machine ahead of
    this one) would make 0.5 ** negative a value ABOVE 1.0 and invent
    traffic that was never observed."""
    factor, _ = reload_decay_factor(
        saved_at=5000.0, now=1000.0, half_life_s=1800.0, frozen=False)
    assert factor == 1.0


def test_reload_decay_is_a_noop_when_no_half_life_is_configured():
    factor, _ = reload_decay_factor(
        saved_at=0.0, now=9999.0, half_life_s=0.0, frozen=False)
    assert factor == 1.0
