"""
Tier 0 of the CPA/TTC test plan: pure geometry, no ROS graph, no camera, no
GPU. Runs in a second with plain pytest -- see
risk_perception/encounter_geometry.py for why this logic lives outside
predictive_risk_costmap_node.py (a rclpy Node, needs a live context to
instantiate at all).

Expected numbers below are worked out by hand from the same formulas
scenario_publisher.py's docstring describes for --scenario head_on /
crossing / parallel / robot_forward -- this file is the analytical version
of the checks that script tells you to eyeball in RViz.
"""

import math

import pytest

from risk_perception.encounter_geometry import cpa_geometry, combine_severity

CPA_GAIN = 1.5
CPA_SCALE_M = 0.6
TTC_SCALE_S = 3.0
MIN_REL_SPEED = 0.05


def geo(ox, oy, ovx, ovy, rx=0.0, ry=0.0, rvx=0.0, rvy=0.0):
    return cpa_geometry(
        ox, oy, ovx, ovy, rx, ry, rvx, rvy,
        CPA_GAIN, CPA_SCALE_M, TTC_SCALE_S, MIN_REL_SPEED)


def test_no_relative_motion_is_exactly_inert():
    """--scenario parallel: same velocity as the robot -> factor == 1.0
    forever, regardless of how close the object is."""
    factor, t_cpa, d_cpa = geo(ox=0.5, oy=2.0, ovx=1.0, ovy=0.0,
                               rx=0.0, ry=0.0, rvx=1.0, rvy=0.0)
    assert factor == 1.0
    assert t_cpa == float("inf")
    assert d_cpa == math.hypot(0.5, 2.0)  # falls back to current distance


def test_diverging_track_is_exactly_inert():
    """Moving apart (t_cpa <= 0) must not amplify, even at high relative speed."""
    factor, t_cpa, d_cpa = geo(ox=1.0, oy=0.0, ovx=5.0, ovy=0.0)  # already past, opening
    assert factor == 1.0
    assert t_cpa <= 0.0


def test_head_on_matches_hand_worked_value():
    """--scenario head_on: person at (4,0) closing at 1 m/s, robot stationary.
    t_cpa = 4/1 = 4s, d_cpa = 0 (exact collision course)."""
    factor, t_cpa, d_cpa = geo(ox=4.0, oy=0.0, ovx=-1.0, ovy=0.0)
    assert abs(t_cpa - 4.0) < 1e-9
    assert abs(d_cpa - 0.0) < 1e-9
    expected = 1.0 + CPA_GAIN * math.exp(0.0) * math.exp(-4.0 / TTC_SCALE_S)
    assert abs(factor - expected) < 1e-9
    assert factor > 1.0


def test_crossing_d_cpa_equals_the_offset():
    """--scenario crossing's own documented claim: 'here d_cpa IS the
    offset' -- person at (offset, y) moving +y at speed v past a
    stationary robot at the origin. Exact, not approximate: for constant
    velocities the CPA distance collapses to the perpendicular offset."""
    for offset in (0.2, 0.5, 1.0, 2.0):
        factor, t_cpa, d_cpa = geo(ox=offset, oy=-3.0, ovx=0.0, ovy=1.0)
        assert abs(d_cpa - offset) < 1e-9
        assert t_cpa > 0.0  # still approaching from y=-3


def test_factor_decreases_as_offset_grows():
    """Sweeping --offset should trace a monotonically decreasing factor(d_cpa)
    curve -- the Stage 4b figure scenario_publisher's docstring describes."""
    factors = []
    for offset in (0.2, 0.5, 1.0, 1.5, 2.0):
        factor, _, _ = geo(ox=offset, oy=-3.0, ovx=0.0, ovy=1.0)
        factors.append(factor)
    assert factors == sorted(factors, reverse=True)
    assert factors[0] > 1.0
    # far enough away, amplification is negligible
    assert factors[-1] < 1.05


def test_factor_decreases_as_ttc_grows():
    """Same miss distance, farther out in time -> less amplification."""
    near, _, d_near = geo(ox=1.0, oy=0.0, ovx=-1.0, ovy=0.0)   # t_cpa = 1s
    far, _, d_far = geo(ox=4.0, oy=0.0, ovx=-1.0, ovy=0.0)     # t_cpa = 4s
    assert abs(d_near - d_far) < 1e-9  # same d_cpa (both exactly head-on)
    assert near > far


def test_robot_forward_static_object_still_gets_amplified():
    """--scenario robot_forward: object never moves (pmot ~ 0), robot drives
    straight at it -- risk is a relation, not a property of the object."""
    factor, t_cpa, d_cpa = geo(ox=3.0, oy=0.0, ovx=0.0, ovy=0.0,
                               rx=0.0, ry=0.0, rvx=1.0, rvy=0.0)
    assert factor > 1.0
    assert abs(d_cpa - 0.0) < 1e-9
    expected = 1.0 + CPA_GAIN * math.exp(0.0) * math.exp(-3.0 / TTC_SCALE_S)
    assert abs(factor - expected) < 1e-9


def test_below_min_rel_speed_is_inert_even_if_technically_closing():
    factor, t_cpa, d_cpa = geo(ox=1.0, oy=0.0, ovx=-0.01, ovy=0.0)  # below 0.05 m/s
    assert factor == 1.0
    assert t_cpa == float("inf")


# ------------------------------------------------------------------ fusion

def test_combine_severity_is_multiply_then_add():
    # 0.3 * 2.0 + 0.3 = 0.9, safely under the 1.0 clip so this exercises the
    # ordering, not the ceiling (see test_combine_severity_clips_at_one).
    assert combine_severity(consequence=0.3, encounter_factor=2.0,
                            relation_bonus=0.3) == pytest.approx(0.9)


def test_relation_bonus_is_not_amplified_by_encounter_factor():
    """The documented modeling choice: relbonus is a flat additive bump,
    outside the CPA multiplier -- (c*factor)+b, never (c+b)*factor. Values
    chosen to stay under the 1.0 clip so this checks the ORDER, not the
    ceiling."""
    c, factor, b = 0.3, 2.0, 0.2
    correct = combine_severity(c, factor, b)
    wrong_if_reordered = (c + b) * factor
    assert abs(correct - (c * factor + b)) < 1e-9
    assert correct != wrong_if_reordered


def test_combine_severity_inert_encounter_is_pass_through_plus_bonus():
    assert combine_severity(consequence=0.65, encounter_factor=1.0,
                            relation_bonus=0.0) == 0.65


def test_combine_severity_clips_at_one():
    """(WP-A, 2026-09-11) consequence * encounter_factor + relation_bonus
    can exceed 1.0 on a close collision course (a high-consequence track,
    cpa_gain's own amplification, plus a relation bonus on top) -- the
    result must saturate at 1.0, not read as 'more than fully occupied' to
    the SRM/critic downstream. See combine_severity's own docstring."""
    assert combine_severity(consequence=0.9, encounter_factor=2.5,
                            relation_bonus=0.3) == 1.0
    # sanity: the unclipped arithmetic really would have exceeded 1.0 here.
    assert 0.9 * 2.5 + 0.3 > 1.0
