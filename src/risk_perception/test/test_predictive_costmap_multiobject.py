"""
Direct math coverage of predictive_risk_costmap_node's multi-object and
moving-robot risk math -- no RViz, no ROS spin, no tf.

The node is constructed normally (rclpy.init only; __init__ subscribes to
topics and starts a tf listener + timer, but nothing is ever published to
those topics and the timer callback (_tick) is never invoked here). Robot
state is injected directly by setting node.robot_xy / node.robot_v, which is
exactly what _update_robot_state would have produced from a real tf lookup
and /odom message. Detections are built to object_tracker_node's exact wire
contract -- the same one tools/scenario_publisher.py emulates for its
head_on/crossing/parallel/robot_forward scenarios -- and fed straight into
_predict_and_splat / _encounter, with assertions on the resulting numpy grid
and hand-computed CPA/TTC constants.
"""

import math

import numpy as np
import pytest
import rclpy
from vision_msgs.msg import Detection3D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray

from risk_perception.predictive_risk_costmap_node import PredictiveRiskCostmapNode


def make_detection(det_id, label, x, y, vx, vy, score=0.9):
    """Match object_tracker_node's world_objects wire contract exactly."""
    moving = math.hypot(vx, vy) > 1e-3
    class_id = (
        f"{label}|pmov=0.90|pmot={0.95 if moving else 0.02:.2f}"
        f"|vx={vx:.3f}|vy={vy:.3f}")
    d = Detection3D()
    d.id = det_id
    d.bbox.center.position.x = float(x)
    d.bbox.center.position.y = float(y)
    d.bbox.center.orientation.w = 1.0
    d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = 0.5
    r = ObjectHypothesisWithPose()
    r.hypothesis.class_id = class_id
    r.hypothesis.score = score
    r.pose.pose = d.bbox.center
    cov = [0.0] * 36
    cov[0] = cov[7] = 0.04      # Pxx, Pyy
    cov[21] = cov[28] = 0.01    # Pvxvx, Pvyvy
    r.pose.covariance = cov
    d.results.append(r)
    return d


def splat(node, detections):
    """Run _predict_and_splat for each detection into a fresh zero grid."""
    grid = np.zeros((node.rows, node.cols), dtype=np.float32)
    markers = MarkerArray()
    for det in detections:
        node._predict_and_splat(det, grid, markers)
    return grid, markers


def cell_index(node, x, y):
    """(row, col) of the cell whose center is nearest (x, y)."""
    col = round((x - node.ox) / node.res - 0.5)
    row = round((y - node.oy) / node.res - 0.5)
    return row, col


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    yield
    if owns_context and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def node():
    n = PredictiveRiskCostmapNode()
    # The grid-peak tests place objects at .X5 coordinates, which are exact
    # cell centers only on the default -6-origin / 0.1-resolution grid.
    assert (n.ox, n.oy, n.res) == (-6.0, -6.0, 0.1), (
        "node grid defaults changed -- update the cell-center coordinates "
        "used by the grid-peak tests")
    yield n
    n.destroy_node()


def test_encounter_head_on_parked_robot(node):
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)
    factor, t_cpa, d_cpa = node._encounter(4.0, 0.0, -1.0, 0.0)
    assert t_cpa == pytest.approx(4.0)
    assert d_cpa == pytest.approx(0.0, abs=1e-12)
    assert factor == pytest.approx(1.0 + 1.5 * math.exp(-4.0 / 3.0), abs=1e-9)


def test_same_label_different_motion_different_factors(node):
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)

    factor_walker, t_cpa, d_cpa = node._encounter(2.0, -3.0, 0.0, 1.0)
    assert t_cpa == pytest.approx(3.0)
    assert d_cpa == pytest.approx(2.0)
    expected_walker = 1.0 + 1.5 * math.exp(-2.0 / 0.6) * math.exp(-1.0)
    assert factor_walker == pytest.approx(expected_walker, abs=1e-9)

    factor_stander, _, _ = node._encounter(-2.0, 1.0, 0.0, 0.0)
    assert factor_stander == 1.0  # below min_rel_speed: literal return


def test_multi_object_grid_is_max_of_singles(node):
    # WP-A, 2026-09-11: stack_motion_first defaults to True, which would
    # override BOTH detections' consequence to the flat
    # stack_consequence_agnostic (0.75) regardless of class -- exactly what
    # this test exists to check is NOT happening (person 0.81 vs chair
    # 0.315, a class-graded difference). Disabled here so this test keeps
    # exercising the class-weighted consequence path directly; the
    # agnostic-override path itself is covered in test_risk_stack.py's WP-A
    # tests and by test_grid_uses_agnostic_consequence_under_motion_first
    # below.
    node.stack_motion_first = False
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)

    person = make_detection("1", "person", 2.05, 0.05, 0.0, 0.0)
    chair = make_detection("2", "chair", -2.05, 2.05, 0.0, 0.0)

    gp, _ = splat(node, [person])
    gc, _ = splat(node, [chair])
    gb, _ = splat(node, [person, chair])

    assert np.array_equal(gb, np.maximum(gp, gc))
    # Both detections are parked (vx=vy=0.0, speed 0.0 m/s) -- below WP2's
    # min_speed_for_motion_mps (0.15 default), so _resolve_moving_allowed
    # now reads moving_allowed=False for both and paint_track treats pmot
    # as exactly 0 (not the raw 0.02 make_detection's class_id carries),
    # so w_stat is the FULL consequence C, not (1 - 0.02) * C. See
    # test_risk_stack.py's resolve_moving_allowed tests for the same gate
    # in isolation.
    assert gp.max() == pytest.approx(0.81, rel=1e-5)
    assert gc.max() == pytest.approx(0.315, rel=1e-5)
    assert gp.max() > gc.max()

    pr, pc = cell_index(node, 2.05, 0.05)
    cr, cc = cell_index(node, -2.05, 2.05)
    assert np.unravel_index(np.argmax(gp), gp.shape) == (pr, pc)
    assert np.unravel_index(np.argmax(gc), gc.shape) == (cr, cc)


def test_moving_robot_scales_grid_by_encounter_ratio(node):
    # WP-A, 2026-09-11: disabled for the same reason as
    # test_multi_object_grid_is_max_of_singles -- this test's `person`
    # detection uses score=0.5 (not make_detection's 0.9 default) so
    # neither combine_severity(...) product below saturates at its 1.0
    # clip (0.9 class value * 0.9 default score * factor_b(~1.77) would
    # exceed 1.0 and clip BOTH grid_a/grid_b to the same value, defeating
    # the ratio check this test exists to make -- see
    # test_combine_severity_clips_at_one in test_encounter_geometry.py for
    # the clip itself); stack_motion_first would otherwise also override
    # consequence to the flat agnostic value, which happens to leave the
    # ratio math intact but no longer tests the class-weighted path this
    # test's hand-worked comments describe.
    node.stack_motion_first = False
    person = make_detection("1", "person", 4.05, 0.05, -1.0, 0.0, score=0.5)

    node.robot_xy = (0.05, 0.05)
    node.robot_v = (0.0, 0.0)
    factor_a, t_cpa_a, d_cpa_a = node._encounter(4.05, 0.05, -1.0, 0.0)
    grid_a, _ = splat(node, [person])

    node.robot_xy = (0.05, 0.05)
    node.robot_v = (1.0, 0.0)
    factor_b, t_cpa_b, d_cpa_b = node._encounter(4.05, 0.05, -1.0, 0.0)
    grid_b, _ = splat(node, [person])

    assert t_cpa_a == pytest.approx(4.0)
    assert d_cpa_a == pytest.approx(0.0, abs=1e-9)
    assert factor_a == pytest.approx(1.0 + 1.5 * math.exp(-4.0 / 3.0), abs=1e-9)

    assert t_cpa_b == pytest.approx(2.0)
    assert d_cpa_b == pytest.approx(0.0, abs=1e-9)
    assert factor_b == pytest.approx(1.0 + 1.5 * math.exp(-2.0 / 3.0), abs=1e-9)

    assert factor_b > factor_a
    ratio = factor_b / factor_a
    assert np.allclose(grid_b, grid_a * ratio, rtol=1e-5, atol=1e-8)


def test_parallel_motion_is_inert(node):
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (1.0, 0.0)
    factor, _, _ = node._encounter(0.05, 2.05, 1.0, 0.0)
    assert factor == 1.0  # relative speed is exactly zero -> min_rel_speed path

    walker = make_detection("1", "person", 0.05, 2.05, 1.0, 0.0)
    grid_with_robot, _ = splat(node, [walker])

    node.robot_xy = None
    node.robot_v = (0.0, 0.0)
    grid_no_robot, _ = splat(node, [walker])

    assert np.array_equal(grid_with_robot, grid_no_robot)


def test_encounter_stationary_uses_zero_object_velocity(node):
    """_encounter_stationary always evaluates v_obj = 0 -- the STATIONARY
    half of two-hypothesis CPA (see predictive_risk_costmap_node's module
    docstring, STAGE 4 / two_hypothesis_cpa). Same track as
    test_parallel_motion_is_inert (moving in lockstep with the robot):
    _encounter (the MOVING hypothesis) reads exactly inert, but
    _encounter_stationary sees a robot closing head-on on a track it does
    not itself assume is parked."""
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (1.0, 0.0)

    factor_mov, _, _ = node._encounter(4.0, 0.0, 1.0, 0.0)
    assert factor_mov == 1.0  # zero relative speed -> inert, as above

    factor_stat, t_cpa_stat, d_cpa_stat = node._encounter_stationary(4.0, 0.0)
    assert t_cpa_stat == pytest.approx(4.0)
    assert d_cpa_stat == pytest.approx(0.0, abs=1e-9)
    expected = 1.0 + 1.5 * math.exp(0.0) * math.exp(-4.0 / 3.0)
    assert factor_stat == pytest.approx(expected, abs=1e-9)
    assert factor_stat > factor_mov  # exactly the discrepancy the flag exists to expose


def test_two_hypothesis_cpa_exposes_parked_collision_the_parallel_reading_hides(node):
    """Same scenario as test_parallel_motion_is_inert, but the behavioral
    prior does NOT believe this track is really moving (pmot forced to
    0) -- its "keeps pace with the robot" velocity estimate may just be
    noise. two_hypothesis_cpa off (default) reuses the single moving-
    hypothesis factor (inert) for the stationary blob too, so the grid is
    byte-identical to the no-robot case, exactly like
    test_parallel_motion_is_inert. Turning it on scores the stationary
    hypothesis against v_obj=0 instead -- a robot closing head-on on a
    track it does not believe is moving -- and the grid must read
    strictly higher."""
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (1.0, 0.0)

    walker = make_detection("1", "person", 4.05, 0.05, 1.0, 0.0)
    walker.results[0].hypothesis.class_id = (
        "person|pmov=0.90|pmot=0.00|vx=1.000|vy=0.000")

    assert node.two_hypothesis_cpa is False
    grid_off, _ = splat(node, [walker])

    node.robot_xy = None
    node.robot_v = (0.0, 0.0)
    grid_no_robot, _ = splat(node, [walker])
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (1.0, 0.0)
    assert np.array_equal(grid_off, grid_no_robot)  # historical: inert either way

    node.two_hypothesis_cpa = True
    grid_on, _ = splat(node, [walker])

    assert grid_on.max() > grid_off.max()


def test_diverging_after_pass_is_one(node):
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)
    factor, t_cpa, _ = node._encounter(-1.0, 0.0, -1.0, 0.0)
    assert t_cpa < 0.0
    assert factor == 1.0


def test_grid_and_planner_use_agnostic_consequence_under_motion_first(node):
    """WP-A, 2026-09-11 default (stack_motion_first: true, unmodified on
    this fixture): a 'table' (CLASS_BASE_RISK 0.40, well under the 0.75
    agnostic value) reads the SAME flat stack_consequence_agnostic on the
    collapsed grid (/risk_costmap_predictive) AND the planner grid
    (/risk_costmap_planner) -- resolve_agnostic_consequence() is a shared
    helper, not two independent overrides. See test_risk_stack.py's WP-A
    tests for the pure-function coverage this exercises end to end."""
    assert node.stack_motion_first is True
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)

    table = make_detection("1", "table", 2.05, 0.05, 0.0, 0.0)
    grid = np.zeros((node.rows, node.cols), dtype=np.float32)
    planner_grid = np.zeros((node.rows, node.cols), dtype=np.float32)
    markers = MarkerArray()
    node._predict_and_splat(table, grid, markers, planner_grid=planner_grid)

    assert grid.max() == pytest.approx(node.stack_consequence_agnostic, rel=1e-5)
    assert planner_grid.max() == pytest.approx(node.stack_consequence_agnostic, rel=1e-5)


def test_two_detections_get_distinct_marker_ids(node):
    node.robot_xy = (0.0, 0.0)
    node.robot_v = (0.0, 0.0)

    walker_a = make_detection("1", "person", 2.05, 0.05, -0.5, 0.0)
    walker_b = make_detection("2", "person", -2.05, 1.05, 0.5, 0.0)

    _, markers = splat(node, [walker_a, walker_b])

    keys = [(m.ns, m.id) for m in markers.markers]
    assert len(keys) == len(set(keys))
    assert len(keys) > 0
