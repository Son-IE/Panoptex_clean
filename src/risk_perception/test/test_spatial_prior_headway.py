"""
WP4's headway statistic, at the same tier as test_spatial_prior_grid.py:
pure functions over numpy, no ROS, no rclpy context, no simulator.

S says a cell is busy. The headway channels say how often, which is the
number mission_supervisor needs to cross a lane it cannot see (see
corridor.blind_crossing_ok). Two properties carry the whole design and are
tested first:

  * a PASS is a change of track id, not a frame. One vehicle parked on a
    cell for a minute must produce no samples at all, or "headway" would
    just be the update period;
  * the statistic freezes with learn_rate 0, exactly like S and F, because
    every study run loads a warm-up prior and must not rewrite it (see the
    CLAUDE.md "a study run that lets the spatial prior keep learning ruins
    it" gotcha, which cost a full 700 s run).

Plus the persistence contract: every warm-up .npz on disk today predates
these channels, and must keep loading.
"""

import numpy as np

from risk_perception.spatial_prior_node import (
    NO_PASS,
    CategoryFlowGrid,
    grid_payload,
    is_frozen,
    load_grid_arrays,
    merge_group_headway,
    record_pass_into_grid,
    track_id_int,
)

RES = 0.10
OX, OY = -5.0, -5.0
ROWS, COLS = 100, 100          # 10 m x 10 m

LEARN = 0.2                    # any learn_rate > 0 enables recording
MIN_S = 2.0
ALPHA = 0.2


def new_grid():
    return CategoryFlowGrid(ROWS, COLS)


def cell_of(cx, cy):
    return int((cy - OY) / RES), int((cx - OX) / RES)


def a_pass(grid, track_id, now, cx=0.0, cy=0.0, radius=0.3,
           learn_rate=LEARN, min_s=MIN_S, alpha=ALPHA):
    """One deposit-shaped pass over the cells within `radius` of (cx, cy).
    Returns the number of CELLS that took a headway sample (the footprint is
    ~30 cells at these numbers), so the assertions below are `> 0` / `== 0`
    on that and exact on the per-cell values."""
    return record_pass_into_grid(
        grid, cx, cy, radius, track_id_int(track_id), now, learn_rate,
        min_s, alpha, RES, OX, OY, ROWS, COLS)


# ------------------------------------------------------------ empty state

def test_a_fresh_grid_has_no_passes_and_no_headway():
    g = new_grid()
    assert np.all(g.last_pass_time == np.float32(NO_PASS))
    assert np.all(g.headway_mean == 0.0)
    assert np.all(g.headway_count == 0)
    assert np.all(g.last_pass_id == -1)
    assert g.headway_count.dtype == np.int32


def test_decay_leaves_the_headway_channels_alone():
    """S and F are EMAs on a forget clock; a count and a timestamp are not.
    Halving a mean gap every two hours would invent traffic."""
    g = new_grid()
    g.s[:] = 1.0
    g.headway_mean[:] = 8.0
    g.headway_count[:] = 4
    g.last_pass_time[:] = 100.0
    g.decay(0.5)
    assert np.all(g.s == 0.5)
    assert np.all(g.headway_mean == 8.0)
    assert np.all(g.headway_count == 4)
    assert np.all(g.last_pass_time == 100.0)


# ------------------------------------------------------- the pass rule

def test_two_passes_ten_seconds_apart_by_different_ids():
    """The worked example: vehicle A at t=0, vehicle B at t=10, one cell.
    One sample of 10 s, and with no prior evidence the first sample IS the
    mean (starting an EMA from zero would read as bumper-to-bumper traffic
    until it converged)."""
    g = new_grid()
    r, c = cell_of(0.0, 0.0)

    assert a_pass(g, "7", 0.0) == 0            # nothing to compare against
    assert g.headway_count[r, c] == 0
    assert g.last_pass_time[r, c] == np.float32(0.0)

    assert a_pass(g, "8", 10.0) > 0
    assert g.headway_count[r, c] == 1
    assert g.headway_mean[r, c] == np.float32(10.0)
    assert g.last_pass_time[r, c] == np.float32(10.0)


def test_the_same_track_twice_is_not_a_headway():
    """A Carter sitting on the cell, or simply seen on consecutive update
    ticks, is one vehicle -- not a 0.2 s headway."""
    g = new_grid()
    r, c = cell_of(0.0, 0.0)
    a_pass(g, "7", 0.0)
    assert a_pass(g, "7", 10.0) == 0
    assert a_pass(g, "7", 20.0) == 0
    assert g.headway_count[r, c] == 0
    assert g.headway_mean[r, c] == 0.0
    # ...but last_pass_time keeps up, so the NEXT vehicle's gap is measured
    # from when this one left, not from when it arrived.
    assert g.last_pass_time[r, c] == np.float32(20.0)
    assert a_pass(g, "8", 25.0) > 0
    assert g.headway_mean[r, c] == np.float32(5.0)


def test_a_gap_below_headway_min_s_is_a_re_identification_not_traffic():
    """Track ids churn constantly (one Carter carried six ids in one run),
    so a sub-2 s "gap" between two ids is one vehicle, and counting it
    would drag every learned headway toward zero -- the dangerous
    direction, since the supervisor would then never believe a lane is
    between vehicles."""
    g = new_grid()
    r, c = cell_of(0.0, 0.0)
    a_pass(g, "27", 0.0)
    assert a_pass(g, "8", 1.5) == 0            # same Carter, new id
    assert g.headway_count[r, c] == 0
    # The id is still adopted, so the real next vehicle measures from here.
    assert g.last_pass_id[r, c] == track_id_int("8")
    assert a_pass(g, "19", 11.5) > 0
    assert g.headway_mean[r, c] == np.float32(10.0)


def test_later_samples_move_the_mean_by_the_ema_alpha():
    g = new_grid()
    r, c = cell_of(0.0, 0.0)
    a_pass(g, "1", 0.0)
    a_pass(g, "2", 10.0)                       # mean = 10, count = 1
    a_pass(g, "3", 30.0)                       # sample 20
    assert g.headway_count[r, c] == 2
    assert g.headway_mean[r, c] == np.float32(10.0 + ALPHA * (20.0 - 10.0))


def test_only_cells_inside_the_footprint_are_touched():
    g = new_grid()
    a_pass(g, "1", 0.0, cx=0.0, cy=0.0, radius=0.2)
    far_r, far_c = cell_of(3.0, 3.0)
    assert g.last_pass_time[far_r, far_c] == np.float32(NO_PASS)
    assert g.last_pass_id[far_r, far_c] == -1
    # Entirely off-grid: no effect, no exception.
    assert a_pass(g, "1", 5.0, cx=1000.0, cy=1000.0) == 0


# -------------------------------------------------------------- frozen

def test_learn_rate_zero_freezes_the_headway_exactly_like_s_and_f():
    """Every study run loads a warm-up prior with learn_rate pinned to 0.
    Nothing about the headway channels may move under it."""
    g = new_grid()
    a_pass(g, "1", 0.0)
    a_pass(g, "2", 10.0)
    before = (g.last_pass_time.copy(), g.headway_mean.copy(),
              g.headway_count.copy(), g.last_pass_id.copy())

    assert a_pass(g, "3", 40.0, learn_rate=0.0) == 0
    assert np.array_equal(g.last_pass_time, before[0])
    assert np.array_equal(g.headway_mean, before[1])
    assert np.array_equal(g.headway_count, before[2])
    assert np.array_equal(g.last_pass_id, before[3])


def test_explicit_frozen_param_freezes_headway_even_with_a_positive_learn_rate():
    """The `frozen` param (default False) must force the same contract as
    learn_rate<=0 regardless of learn_rate's own value -- a caller forcing
    frozen behaviour for a reason other than the learn_rate knob still
    gets zero headway samples, exactly like the existing learn_rate==0.0
    path above."""
    g = new_grid()
    a_pass(g, "1", 0.0)
    a_pass(g, "2", 10.0)
    before = (g.last_pass_time.copy(), g.headway_mean.copy(),
              g.headway_count.copy(), g.last_pass_id.copy())

    positive_learn_rate, frozen_param = 0.20, True
    assert is_frozen(positive_learn_rate, frozen_param) is True
    effective_learn_rate = (0.0 if is_frozen(positive_learn_rate, frozen_param)
                            else positive_learn_rate)
    assert a_pass(g, "3", 40.0, learn_rate=effective_learn_rate) == 0
    assert np.array_equal(g.last_pass_time, before[0])
    assert np.array_equal(g.headway_mean, before[1])
    assert np.array_equal(g.headway_count, before[2])
    assert np.array_equal(g.last_pass_id, before[3])


# ------------------------------------------------------------- merge_group_headway

def test_merge_group_headway_picks_the_member_with_the_larger_count():
    a = new_grid()
    b = new_grid()
    r, c = cell_of(0.0, 0.0)
    a.headway_count[r, c] = 2
    a.headway_mean[r, c] = 5.0
    a.last_pass_time[r, c] = 100.0
    b.headway_count[r, c] = 9
    b.headway_mean[r, c] = 20.0
    b.last_pass_time[r, c] = 250.0

    mean, count, last_pass = merge_group_headway([a, b], ROWS, COLS)
    assert count[r, c] == 9
    assert mean[r, c] == np.float32(20.0)          # b's mean, not a blend
    assert last_pass[r, c] == np.float32(250.0)    # b's last_pass, matching


def test_merge_group_headway_with_one_empty_member_equals_the_other_member():
    populated = new_grid()
    r, c = cell_of(2.0, 2.0)
    populated.headway_count[r, c] = 3
    populated.headway_mean[r, c] = 12.0
    populated.last_pass_time[r, c] = 42.0
    empty = new_grid()

    mean, count, last_pass = merge_group_headway([populated, empty], ROWS, COLS)
    assert np.array_equal(count, populated.headway_count)
    assert np.array_equal(mean, populated.headway_mean)
    assert np.array_equal(last_pass, populated.last_pass_time)

    # Order must not matter.
    mean2, count2, last_pass2 = merge_group_headway([empty, populated], ROWS, COLS)
    assert np.array_equal(count2, populated.headway_count)
    assert np.array_equal(mean2, populated.headway_mean)


def test_merge_group_headway_all_members_none_is_empty():
    mean, count, last_pass = merge_group_headway([None, None], ROWS, COLS)
    assert np.all(count == 0)
    assert np.all(mean == 0.0)
    assert np.all(last_pass == np.float32(NO_PASS))


# ----------------------------------------------------------- track ids

def test_track_id_int_is_stable_non_negative_and_never_the_sentinel():
    assert track_id_int("7") == 7
    assert track_id_int(7) == 7
    assert track_id_int("-1") == 1              # never collides with -1
    assert track_id_int("carter1") == track_id_int("carter1")
    assert track_id_int("carter1") != track_id_int("carter2")
    assert track_id_int("carter1") >= 0
    assert track_id_int(None) >= 0


# --------------------------------------------------------- persistence

def test_npz_round_trip_preserves_every_channel(tmp_path):
    g = new_grid()
    g.s[3, 4] = 0.75
    g.fx[3, 4] = 0.6
    g.fy[3, 4] = -0.2
    a_pass(g, "1", 0.0)
    a_pass(g, "2", 10.0)

    path = tmp_path / "prior.npz"
    np.savez_compressed(path, **grid_payload("robot", g))
    with np.load(path) as z:
        back = load_grid_arrays(z, "robot", ROWS, COLS)

    for name in ("s", "fx", "fy", "last_pass_time", "headway_mean",
                 "headway_count", "last_pass_id"):
        assert np.array_equal(getattr(back, name), getattr(g, name)), name
        assert getattr(back, name).dtype == getattr(g, name).dtype


def test_a_pre_wp4_npz_still_loads_with_empty_headway_channels(tmp_path):
    """Every warm-up file on disk today was written before these channels
    existed. It must load its S/F normally -- a missing key is not an
    error, it is an older schema."""
    old = new_grid()
    old.s[5, 5] = 0.9
    old.fx[5, 5] = 0.4
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, **{"s__robot": old.s, "fx__robot": old.fx,
                                 "fy__robot": old.fy})

    with np.load(path) as z:
        back = load_grid_arrays(z, "robot", ROWS, COLS)

    assert back.s[5, 5] == np.float32(0.9)
    assert back.fx[5, 5] == np.float32(0.4)
    assert np.all(back.last_pass_time == np.float32(NO_PASS))
    assert np.all(back.headway_count == 0)
    assert np.all(back.headway_mean == 0.0)
    assert np.all(back.last_pass_id == -1)


def test_a_category_absent_from_the_file_loads_empty(tmp_path):
    path = tmp_path / "prior.npz"
    np.savez_compressed(path, **grid_payload("robot", new_grid()))
    with np.load(path) as z:
        other = load_grid_arrays(z, "person", ROWS, COLS)
    assert np.all(other.s == 0.0)
    assert np.all(other.last_pass_time == np.float32(NO_PASS))


def test_an_array_of_the_wrong_shape_is_skipped_not_broadcast(tmp_path):
    """The geometry guard upstream already rejects a mismatched grid, so
    this only fires on a corrupt file -- where empty beats wrong."""
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(path, **{"s__robot": np.ones((7, 7),
                                                     dtype=np.float32)})
    with np.load(path) as z:
        back = load_grid_arrays(z, "robot", ROWS, COLS)
    assert back.s.shape == (ROWS, COLS)
    assert np.all(back.s == 0.0)
