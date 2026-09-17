"""
Tests for risk_perception/scan_clustering.py (WP1 -- lidar dynamic-cluster
detector). Pure numpy/scipy, no ROS/rclpy -- see that module's docstring for
the pipeline these functions implement.
"""
import math

import numpy as np
import pytest

from risk_perception.scan_clustering import (
    build_static_distance_field, ClusterPersistence, cluster_points,
    cluster_summaries, push_centroid_from_sensor, subtract_static)


# ---------------------------------------------------------------------------
# subtract_static / build_static_distance_field
# ---------------------------------------------------------------------------

def _wall_grid():
    """10x10 cells @ 0.1 m resolution, origin (0, 0), a wall at row 5
    (y in [0.5, 0.6)), free everywhere else."""
    grid = np.zeros((10, 10), dtype=np.int16)
    grid[5, :] = 100
    occ_grid_info = {"resolution": 0.1, "origin_x": 0.0, "origin_y": 0.0,
                     "width": 10, "height": 10}
    return grid, occ_grid_info


def test_static_subtraction_drops_points_near_a_wall():
    grid, occ_grid_info = _wall_grid()
    dist = build_static_distance_field(grid, occ_grid_info)

    points = np.array([
        [0.25, 0.45],   # 0.05 m from the wall row -- inside static_margin_m
        [0.25, 0.10],   # far from the wall -- should survive
    ])
    keep = subtract_static(points, occ_grid_info, dist, static_margin_m=0.15)
    assert list(keep) == [False, True]


def test_static_subtraction_drops_points_on_unknown_cells():
    grid = np.full((10, 10), -1, dtype=np.int16)
    grid[0:5, :] = 0   # only the bottom half of the map is known/free
    occ_grid_info = {"resolution": 0.1, "origin_x": 0.0, "origin_y": 0.0,
                     "width": 10, "height": 10}
    dist = build_static_distance_field(grid, occ_grid_info)

    points = np.array([
        [0.25, 0.75],   # unknown cell (row 7)
        [0.25, 0.25],   # known free cell (row 2), far from the unknown band
    ])
    keep = subtract_static(points, occ_grid_info, dist, static_margin_m=0.15)
    assert list(keep) == [False, True]


def test_static_subtraction_drops_out_of_bounds_points():
    grid, occ_grid_info = _wall_grid()
    dist = build_static_distance_field(grid, occ_grid_info)
    points = np.array([[50.0, 50.0]])  # way outside the 1x1 m map
    keep = subtract_static(points, occ_grid_info, dist, static_margin_m=0.15)
    assert list(keep) == [False]


def test_static_subtraction_empty_input():
    grid, occ_grid_info = _wall_grid()
    dist = build_static_distance_field(grid, occ_grid_info)
    keep = subtract_static(np.zeros((0, 2)), occ_grid_info, dist, static_margin_m=0.15)
    assert keep.shape == (0,)


# ---------------------------------------------------------------------------
# cluster_points -- range-adaptive gap
# ---------------------------------------------------------------------------

def test_range_adaptive_gap_splits_two_objects_half_metre_apart_at_3m():
    # gap threshold at r=3: 0.25 + 0.02*3 = 0.31 m < the 0.5 m separation.
    points = np.array([
        [3.0, 0.00], [3.0, 0.02], [3.0, 0.04],     # object A
        [3.0, 0.54], [3.0, 0.56], [3.0, 0.58],     # object B, 0.5 m away
    ])
    ranges = np.full(6, 3.0)
    clusters = cluster_points(points, gap_base_m=0.25, gap_per_m=0.02, ranges=ranges)
    assert len(clusters) == 2
    assert list(clusters[0]) == [0, 1, 2]
    assert list(clusters[1]) == [3, 4, 5]


def test_range_adaptive_gap_merges_two_sides_of_a_wide_object_far_away():
    # gap threshold at r=10: 0.25 + 0.02*10 = 0.45 m >= the 0.4 m separation
    # between the object's near-left and near-right visible surfaces (the
    # object's own back is occluded, so there is a real point gap in the
    # middle, but it must still read as ONE cluster at this range).
    points = np.array([
        [10.0, 0.00], [10.0, 0.02], [10.0, 0.04],   # left side
        [10.0, 0.44], [10.0, 0.46], [10.0, 0.48],   # right side, 0.4 m away
    ])
    ranges = np.full(6, 10.0)
    clusters = cluster_points(points, gap_base_m=0.25, gap_per_m=0.02, ranges=ranges)
    assert len(clusters) == 1
    assert list(clusters[0]) == [0, 1, 2, 3, 4, 5]


def test_cluster_points_wraparound_across_beam_zero():
    # Beam order 0..4 around a circle. Cluster A = beams {0, 1}; cluster
    # B = beams {2, 3} (unrelated, far from everything else); beam 4 sits
    # right next to beam 0 in SPACE (wraparound: angle_max meets angle_min)
    # even though it is far from beam 3 in the walk. Expect beam 4 to merge
    # with cluster A, and cluster B to stay untouched.
    points = np.array([
        [-2.00, -0.02],   # 0 -- cluster A
        [-2.00, 0.02],    # 1 -- cluster A
        [0.00, 2.00],     # 2 -- cluster B
        [0.02, 2.02],     # 3 -- cluster B
        [-2.05, -0.05],   # 4 -- far from 3, close to 0 (wraparound)
    ])
    ranges = np.full(5, 2.0)
    clusters = cluster_points(points, gap_base_m=0.25, gap_per_m=0.0, ranges=ranges)
    assert len(clusters) == 2
    merged = {int(i) for i in clusters[0]}
    other = {int(i) for i in clusters[1]}
    assert merged == {4, 0, 1}
    assert other == {2, 3}


def test_cluster_points_empty_and_single_point():
    assert cluster_points(np.zeros((0, 2)), 0.25, 0.02, []) == []
    clusters = cluster_points(np.array([[1.0, 1.0]]), 0.25, 0.02, [1.4])
    assert len(clusters) == 1
    assert list(clusters[0]) == [0]


# ---------------------------------------------------------------------------
# cluster_summaries -- centroid/extent/n_points/mean_range + filters
# ---------------------------------------------------------------------------

def test_cluster_summaries_basic_stats():
    points = np.array([[1.0, 1.0], [1.2, 1.0], [1.0, 1.4]])
    ranges = np.array([2.0, 2.2, 2.4])
    clusters = [np.array([0, 1, 2])]
    out = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=10.0)
    assert len(out) == 1
    s = out[0]
    assert s["n_points"] == 3
    assert s["centroid"] == pytest.approx((1.0667, 1.1333), abs=1e-3)
    assert s["extent"] == pytest.approx((0.2, 0.4))
    assert s["mean_range"] == pytest.approx(2.2)


def test_cluster_summaries_min_points_filter():
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    ranges = np.array([1.0, 1.0])
    clusters = [np.array([0, 1])]
    out = cluster_summaries(points, ranges, clusters, min_points=3, max_extent_m=10.0)
    assert out == []


def test_cluster_summaries_max_extent_filter():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    ranges = np.array([1.0, 2.0, 2.0])
    clusters = [np.array([0, 1, 2])]
    out = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=1.0)
    assert out == []
    out_ok = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=5.0)
    assert len(out_ok) == 1


def test_cluster_summaries_self_radius_filter():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    ranges = np.array([0.3, 0.3, 0.3])
    clusters = [np.array([0, 1, 2])]
    robot_xy = np.array([0.05, 0.05])
    out = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=5.0,
                            self_radius_m=0.25, robot_xy=robot_xy)
    assert out == []
    out_far = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=5.0,
                                self_radius_m=0.25, robot_xy=np.array([5.0, 5.0]))
    assert len(out_far) == 1


def test_cluster_summaries_self_radius_disabled_without_robot_xy():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    ranges = np.array([0.3, 0.3, 0.3])
    clusters = [np.array([0, 1, 2])]
    out = cluster_summaries(points, ranges, clusters, min_points=1, max_extent_m=5.0,
                            self_radius_m=0.25, robot_xy=None)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# push_centroid_from_sensor -- near-face centroid correction (lidar-fusion
# probe follow-up, 2026-09-09)
# ---------------------------------------------------------------------------

def test_push_centroid_from_sensor_basic():
    # 0.5 m wide cluster centred at (3.0, 0.0), sensor at the origin --
    # push = min(0.5 * 0.5, 0.30) = 0.25 m along the +x sensor->centroid
    # bearing.
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(3.0, 0.0), extent=(0.5, 0.1), sensor_xy=np.array([0.0, 0.0]),
        push_factor=0.5, push_max_m=0.30)
    assert (cx, cy) == pytest.approx((3.25, 0.0))


def test_push_centroid_from_sensor_factor_zero_disables():
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(3.0, 0.0), extent=(0.5, 0.1), sensor_xy=np.array([0.0, 0.0]),
        push_factor=0.0, push_max_m=0.30)
    assert (cx, cy) == pytest.approx((3.0, 0.0))


def test_push_centroid_from_sensor_max_zero_disables():
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(3.0, 0.0), extent=(0.5, 0.1), sensor_xy=np.array([0.0, 0.0]),
        push_factor=0.5, push_max_m=0.0)
    assert (cx, cy) == pytest.approx((3.0, 0.0))


def test_push_centroid_from_sensor_capped_at_max():
    # 1.2 m extent -> uncapped push would be 0.6 m; capped at push_max_m.
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(3.0, 0.0), extent=(1.2, 0.1), sensor_xy=np.array([0.0, 0.0]),
        push_factor=0.5, push_max_m=0.30)
    assert (cx, cy) == pytest.approx((3.30, 0.0))


def test_push_centroid_from_sensor_none_sensor_disables():
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(3.0, 0.0), extent=(1.2, 0.1), sensor_xy=None,
        push_factor=0.5, push_max_m=0.30)
    assert (cx, cy) == pytest.approx((3.0, 0.0))


def test_push_centroid_from_sensor_follows_actual_bearing():
    # Sensor not at the origin, and not axis-aligned with the centroid --
    # the push must follow the real sensor->centroid unit vector.
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(1.0, 1.0), extent=(0.4, 0.4), sensor_xy=np.array([0.0, 0.0]),
        push_factor=0.5, push_max_m=0.30)
    push = 0.5 * 0.4  # 0.2, under the 0.30 cap
    expected = push / math.sqrt(2)
    assert (cx, cy) == pytest.approx((1.0 + expected, 1.0 + expected), abs=1e-6)


def test_push_centroid_from_sensor_degenerate_sensor_on_centroid():
    # Sensor sitting exactly on the raw centroid -- no bearing to push
    # along, must return the centroid unchanged rather than dividing by 0.
    cx, cy = push_centroid_from_sensor(
        centroid_xy=(2.0, 2.0), extent=(0.5, 0.5), sensor_xy=np.array([2.0, 2.0]),
        push_factor=0.5, push_max_m=0.30)
    assert (cx, cy) == pytest.approx((2.0, 2.0))


def test_cluster_summaries_applies_centroid_push_before_self_exclusion():
    """The push has to land BEFORE the self_radius_m check: a cluster whose
    RAW centroid sits inside self_radius_m but whose PUSHED centroid does
    not must survive; with the push disabled, the same cluster is dropped."""
    points = np.array([[2.0, 0.0], [2.5, 0.0], [2.0, 0.2]])  # w=0.5, h=0.2
    ranges = np.array([2.0, 2.5, 2.0])
    clusters = [np.array([0, 1, 2])]
    sensor_xy = np.array([0.0, 0.0])

    out = cluster_summaries(
        points, ranges, clusters, min_points=1, max_extent_m=10.0,
        self_radius_m=2.3, robot_xy=sensor_xy, sensor_xy=sensor_xy,
        centroid_push_factor=0.5, centroid_push_max_m=0.30)
    assert len(out) == 1
    assert out[0]["centroid"][0] > 2.3          # pushed past the self-exclusion radius
    assert out[0]["extent"] == pytest.approx((0.5, 0.2))  # extent unchanged by the push

    out_no_push = cluster_summaries(
        points, ranges, clusters, min_points=1, max_extent_m=10.0,
        self_radius_m=2.3, robot_xy=sensor_xy, sensor_xy=sensor_xy,
        centroid_push_factor=0.0, centroid_push_max_m=0.30)
    assert out_no_push == []


# ---------------------------------------------------------------------------
# ClusterPersistence (WP-B, 2026-09-10) -- see the module's WHY-comment
# above its imports for the shelf-edge-flicker finding this class fixes.
# ---------------------------------------------------------------------------

def test_persistence_confirms_only_after_min_consecutive_matches():
    p = ClusterPersistence(min_consecutive=3, match_m=0.3)
    # Scan 1: first sighting -- streak 1, not confirmed.
    assert p.update([(1.0, 1.0)], stamp=0.0) == []
    # Scan 2: matches within match_m -- streak 2, still not confirmed.
    assert p.update([(1.05, 1.0)], stamp=0.1) == []
    # Scan 3: third consecutive match -- streak 3 == min_consecutive.
    confirmed = p.update([(1.02, 1.0)], stamp=0.2)
    assert confirmed == [(1.02, 1.0)]
    # Scan 4: stays confirmed on every subsequent matching scan too.
    confirmed_again = p.update([(1.03, 1.0)], stamp=0.3)
    assert len(confirmed_again) == 1


def test_persistence_one_scan_flicker_never_confirmed():
    p = ClusterPersistence(min_consecutive=3, match_m=0.3)
    assert p.update([(5.0, 5.0)], stamp=0.0) == []       # appears once
    # Gone next scan entirely -- the candidate's streak ends, it is
    # dropped rather than merely paused.
    assert p.update([], stamp=0.1) == []
    # Reappearing nearby afterwards starts a BRAND NEW candidate (streak
    # restarts at 1), so two more matching scans are still required.
    assert p.update([(5.02, 5.0)], stamp=0.2) == []
    assert p.update([(5.01, 5.0)], stamp=0.3) == []
    confirmed = p.update([(5.03, 5.0)], stamp=0.4)
    assert len(confirmed) == 1


def test_persistence_flicker_among_other_confirmed_clusters_is_dropped():
    """A cluster that appears for exactly one scan among clusters that keep
    matching must never itself be confirmed, while the persistent ones are
    unaffected."""
    p = ClusterPersistence(min_consecutive=3, match_m=0.3)
    p.update([(0.0, 0.0)], stamp=0.0)
    p.update([(0.0, 0.0)], stamp=0.1)
    # Scan 3: the persistent cluster confirms; a brand-new one-off flicker
    # appears alongside it.
    confirmed = p.update([(0.0, 0.0), (9.0, 9.0)], stamp=0.2)
    assert confirmed == [(0.0, 0.0)]
    # Scan 4: the flicker is gone; the persistent cluster stays confirmed.
    confirmed2 = p.update([(0.0, 0.0)], stamp=0.3)
    assert confirmed2 == [(0.0, 0.0)]


def test_persistence_mover_displacing_02m_per_scan_stays_confirmed():
    p = ClusterPersistence(min_consecutive=3, match_m=0.3)
    x = 0.0
    confirmed = []
    for i in range(3):
        confirmed = p.update([(x, 0.0)], stamp=float(i) * 0.1)
        x += 0.2
    assert len(confirmed) == 1   # confirmed on the 3rd consecutive match

    # Keeps being confirmed as it keeps moving 0.2 m/scan (well inside
    # match_m=0.3), scan after scan.
    for i in range(3, 8):
        confirmed = p.update([(x, 0.0)], stamp=float(i) * 0.1)
        x += 0.2
        assert len(confirmed) == 1


def test_persistence_match_too_far_is_treated_as_a_new_candidate():
    p = ClusterPersistence(min_consecutive=3, match_m=0.3)
    p.update([(0.0, 0.0)], stamp=0.0)
    p.update([(0.0, 0.0)], stamp=0.1)
    # 1.0 m jump, far past match_m -- this is a fresh candidate, not the
    # same one continuing; the old one also lapses (unmatched this scan).
    confirmed = p.update([(1.0, 0.0)], stamp=0.2)
    assert confirmed == []
