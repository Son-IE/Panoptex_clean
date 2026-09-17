"""
Tier 0 of the evaluation harness: pure computation, no ROS, no rosbag, no
GPU. Runs in a second with plain pytest -- see
risk_perception/evaluation_metrics.py for why the bag-reading half
(tools/evaluate_run.py) is a thin, separately-verified wrapper around this.
"""

import math

import pytest

from risk_perception.encounter_geometry import combine_severity
from risk_perception.risk_visualization import risk_score_from_label
from risk_perception.evaluation_metrics import (
    RunReport,
    clearance_distribution_stats,
    clearance_encounters,
    detection_to_costmap_latency,
    parse_class_id,
    path_length,
    realtime_factor,
    reference_severity_at_point,
    replan_count,
    risk_exposure,
    risk_exposure_per_distance,
    stop_events,
    time_to_goal,
    velocity_smoothness,
    yaw_from_quat,
)


# ------------------------------------------------------------------ efficiency

def test_path_length_straight_line():
    xy = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]  # 3 + 4 = 7
    assert path_length(xy) == pytest.approx(7.0)


def test_path_length_needs_at_least_two_points():
    assert path_length([(0.0, 0.0)]) == 0.0
    assert path_length([]) == 0.0


def test_time_to_goal_is_a_plain_difference():
    assert time_to_goal(10.0, 42.5) == pytest.approx(32.5)


def test_stop_events_counts_only_sustained_stops():
    # stopped 0.0-2.5s (three samples below threshold, then the sample at
    # 2.5s confirms movement resumed -- duration is measured up to THAT
    # sample, not the last stopped one, since that's the earliest evidence
    # available that the stop actually ended), moving 2.5-3.0s, one-sample
    # dip at 3.0-3.1s (too short, does not count), moving after.
    times = [0.0, 1.0, 2.0, 2.5, 3.0, 3.1, 4.0]
    speed = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    count, stopped_time = stop_events(speed, times, stop_speed_mps=0.05,
                                      min_stop_duration_s=0.5)
    assert count == 1
    assert stopped_time == pytest.approx(2.5)


def test_stop_events_empty_input():
    assert stop_events([], []) == (0, 0.0)


def test_velocity_smoothness_is_zero_for_constant_speed():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    speed = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert velocity_smoothness(speed, times) == pytest.approx(0.0)


def test_velocity_smoothness_is_positive_for_jerky_motion():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    speed = [0.0, 2.0, 0.0, 2.0, 0.0]
    assert velocity_smoothness(speed, times) > 0.0


def test_replan_count_is_length():
    assert replan_count([1.0, 2.0, 3.5]) == 3
    assert replan_count([]) == 0


# ------------------------------------------------------------------ safety / risk exposure

def test_risk_exposure_constant_risk_matches_hand_calc():
    # risk=0.5 for 10 seconds total (dt=1 each) -> total=5.0, rate=0.5
    risk = [0.5] * 10
    dt = [1.0] * 10
    total, rate = risk_exposure(risk, dt)
    assert total == pytest.approx(5.0)
    assert rate == pytest.approx(0.5)


def test_risk_exposure_is_robust_to_uneven_sampling():
    # one long low-risk stretch + one short high-risk spike
    risk = [0.1, 0.9]
    dt = [10.0, 0.5]
    total, rate = risk_exposure(risk, dt)
    expected_total = 0.1 * 10.0 + 0.9 * 0.5
    assert total == pytest.approx(expected_total)
    assert rate == pytest.approx(expected_total / 10.5)


def test_risk_exposure_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        risk_exposure([0.1, 0.2], [1.0])


def test_risk_exposure_zero_duration_rate_is_zero_not_nan():
    total, rate = risk_exposure([], [])
    assert total == 0.0
    assert rate == 0.0


def test_risk_exposure_per_distance():
    assert risk_exposure_per_distance(10.0, 5.0) == pytest.approx(2.0)
    assert risk_exposure_per_distance(10.0, 0.0) == 0.0  # no divide-by-zero


# ------------------------------------------------------------------ clearance

def test_clearance_encounters_groups_contiguous_close_range():
    # engaged 0-2 (min 0.8), clear 2-3, engaged 3-4 (min 0.3)
    times = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
    dist = [0.8, 1.5, 1.9, 3.0, 1.2, 0.3, 1.0]
    minima = clearance_encounters(dist, times, engagement_threshold_m=2.0)
    assert minima == pytest.approx([0.8, 0.3])


def test_clearance_encounters_no_engagement_is_empty():
    dist = [5.0, 6.0, 7.0]
    assert clearance_encounters(dist, [0, 1, 2], engagement_threshold_m=2.0) == []


def test_clearance_distribution_stats_percentiles():
    minima = [0.5, 1.0, 1.5, 2.0, 0.2]
    stats = clearance_distribution_stats(minima)
    assert stats["n_encounters"] == 5
    assert stats["min"] == pytest.approx(0.2)
    assert stats["median"] == pytest.approx(1.0)
    assert stats["p5"] <= stats["p25"] <= stats["median"]


def test_clearance_distribution_stats_empty():
    assert clearance_distribution_stats([]) == {"n_encounters": 0}


# ------------------------------------------------------------------ latency / compute

def test_detection_to_costmap_latency_uses_nearest_earlier_detection():
    det_stamps = [0.0, 1.0, 2.0]
    costmap_stamps = [0.5, 1.9, 2.1]
    latencies = detection_to_costmap_latency(det_stamps, costmap_stamps)
    assert latencies == pytest.approx([0.5, 0.9, 0.1])


def test_detection_to_costmap_latency_no_detections_yet_is_skipped():
    latencies = detection_to_costmap_latency([5.0], [1.0, 2.0, 6.0])
    assert latencies == pytest.approx([1.0])  # only the 6.0 costmap has an earlier detection


def test_realtime_factor_matches_configured_period():
    stamps = [0.0, 0.2, 0.4, 0.6]  # perfectly on a 0.2s period
    factors = realtime_factor(stamps, configured_period_s=0.2)
    assert factors == pytest.approx([1.0, 1.0, 1.0])


def test_realtime_factor_flags_falling_behind():
    stamps = [0.0, 0.2, 0.6]  # second interval is 3x the configured period
    factors = realtime_factor(stamps, configured_period_s=0.2)
    assert factors[0] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(2.0)


def test_realtime_factor_needs_at_least_two_stamps():
    assert realtime_factor([1.0], 0.2) == []
    assert realtime_factor([], 0.2) == []


# ------------------------------------------------------------------ raw (unclipped) severity

def test_parse_class_id_round_trip():
    label, kv = parse_class_id("person|pmov=0.90|pmot=0.05|vx=0.12|vy=-0.03|relbonus=0.40")
    assert label == "person"
    assert kv == pytest.approx({"pmov": 0.90, "pmot": 0.05, "vx": 0.12,
                                "vy": -0.03, "relbonus": 0.40})


def test_parse_class_id_bare_label_no_tags():
    label, kv = parse_class_id("chair")
    assert label == "chair"
    assert kv == {}


def test_yaw_from_quat_identity_is_zero():
    assert yaw_from_quat(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_yaw_from_quat_ninety_degrees():
    half = math.sin(math.pi / 4)
    assert yaw_from_quat(0.0, 0.0, half, half) == pytest.approx(math.pi / 2)


def _track(**overrides):
    base = {"label": "person", "score": 0.9, "x": 0.0, "y": 0.0,
           "pmot": 0.0, "vx": 0.0, "vy": 0.0, "relbonus": 0.0,
           "Pxx": 0.04, "Pyy": 0.04, "Pvx": 0.0, "Pvy": 0.0}
    base.update(overrides)
    return base


def test_reference_severity_no_tracks_is_zero():
    assert reference_severity_at_point([], 0.0, 0.0, 0.0, 0.0) == 0.0


def test_reference_severity_stationary_track_at_robot_position():
    """Robot exactly at the track's position, both stationary -> Gaussian
    term is exactly 1.0 and CPA/TTC is exactly inert (no relative motion),
    so this isolates and locks down the consequence/combine_severity wiring."""
    track = _track(label="person", score=0.9)
    result = reference_severity_at_point([track], 0.0, 0.0, 0.0, 0.0)
    expected = combine_severity(risk_score_from_label("person", 0.9), 1.0, 0.0)
    assert result == pytest.approx(expected)


def test_reference_severity_far_track_contributes_near_zero():
    track = _track(x=50.0, y=50.0)
    result = reference_severity_at_point([track], 0.0, 0.0, 0.0, 0.0)
    assert result < 1e-6


def test_reference_severity_use_class_consequence_false_forces_one():
    track = _track(label="chair", score=0.9)  # chair's own base risk is low
    with_class = reference_severity_at_point(
        [track], 0.0, 0.0, 0.0, 0.0, use_class_consequence=True)
    without_class = reference_severity_at_point(
        [track], 0.0, 0.0, 0.0, 0.0, use_class_consequence=False)
    assert without_class > with_class
    assert without_class == pytest.approx(combine_severity(1.0, 1.0, 0.0))


def test_reference_severity_enable_relative_motion_false_forces_factor_one():
    # closing encounter: track approaching the query point from +x
    track = _track(x=4.0, y=0.0, vx=-1.0, vy=0.0)
    with_rel = reference_severity_at_point(
        [track], 0.0, 0.0, 1.0, 0.0, enable_relative_motion=True)
    without_rel = reference_severity_at_point(
        [track], 0.0, 0.0, 1.0, 0.0, enable_relative_motion=False)
    # same query point, same Gaussian falloff -- only the CPA factor differs,
    # so without_rel must be <= with_rel (factor pinned to the inert 1.0)
    assert without_rel <= with_rel


def test_reference_severity_use_motion_mixture_false_forces_pmot_zero():
    """With the moving hypothesis on, a fast-moving track's peak severity
    lands away from its CURRENT position (out along its rolled-out path),
    not stacked entirely onto the stationary splat at (x, y)."""
    track = _track(x=0.0, y=0.0, pmot=1.0, vx=2.0, vy=0.0)
    at_current_pos_mixture_on = reference_severity_at_point(
        [track], 0.0, 0.0, 0.0, 0.0, use_motion_mixture=True)
    at_current_pos_mixture_off = reference_severity_at_point(
        [track], 0.0, 0.0, 0.0, 0.0, use_motion_mixture=False)
    # mixture off => 100% stationary hypothesis at (x, y) => strictly higher
    # severity AT that exact point than with pmot=1 splitting weight away to
    # the moving hypothesis instead
    assert at_current_pos_mixture_off > at_current_pos_mixture_on


def test_reference_severity_is_not_clamped_to_one():
    """The entire point of this function -- see the module's 'Raw
    (unclipped) severity' docstring. A high-consequence, closing, relation-
    tagged, fully-moving (pmot=1) track's rolled-out path crosses close to
    the query point around t~1s -- see the worked numbers in this test's
    history for why (x=3, vx=-2 reaches x=1 at t=1s, matching the query
    point) -- pushing the raw value well past what the published
    (clipped-to-100) grid would ever show."""
    track = _track(label="forklift", score=1.0, x=3.0, y=0.0,
                   vx=-2.0, vy=0.0, pmot=1.0, relbonus=0.6,
                   Pxx=0.09, Pyy=0.09)
    result = reference_severity_at_point(
        [track], 1.0, 0.0, 2.0, 0.0)  # query point in the path of the rollout
    assert result > 1.0


def test_reference_severity_multiple_tracks_take_the_max_not_sum():
    near_low = _track(label="chair", score=0.5, x=0.0, y=0.0)
    far_high = _track(label="forklift", score=1.0, x=50.0, y=50.0)
    combined = reference_severity_at_point([near_low, far_high], 0.0, 0.0, 0.0, 0.0)
    solo = reference_severity_at_point([near_low], 0.0, 0.0, 0.0, 0.0)
    assert combined == pytest.approx(solo)  # the far track contributes ~0


# ------------------------------------------------------------------ RunReport

def test_run_report_summary_handles_empty_lists_without_nan_crash():
    report = RunReport(label="baseline")
    summary = report.summary()
    assert summary["label"] == "baseline"
    assert math.isnan(summary["latency_mean_s"])
    assert math.isnan(summary["realtime_factor_mean"])


def test_run_report_summary_collapses_lists_to_scalars():
    report = RunReport(
        label="full_system",
        detection_to_costmap_latency_s=[0.1, 0.2, 0.3, 0.4],
        realtime_factor=[1.0, 1.1, 0.9],
    )
    summary = report.summary()
    assert summary["latency_mean_s"] == pytest.approx(0.25)
    assert summary["realtime_factor_mean"] == pytest.approx(1.0)
