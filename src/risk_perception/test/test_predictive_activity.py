"""
WP-C: class-agnostic recent-activity floor (predictive_risk_costmap_node's
consumption of spatial_prior_node's `/risk_perception/activity_prior`) --
direct math coverage, no RViz, no ROS spin, no tf. Same rationale/pattern as
test_predictive_costmap_multiobject.py (see that file's own docstring): the
node is constructed normally (rclpy.init only; nothing is ever published to
its subscribed topics and the timer callback is invoked directly, never via
spin), and the resample/seed logic under test
(_resample_occupancy_grid/_activity_cb/_tick) is exercised on the real node
object, not a re-implementation of its arithmetic.

See predictive_risk_costmap_node.py's module docstring "WP-C -- CLASS-
AGNOSTIC RECENT-ACTIVITY FLOOR" section for the design this checks: seeded
by MAX (never addition), before any track is painted, independently
weighted per consumer (activity_weight for the stack, activity_weight_
planner for the planner grid, activity_weight_grid for the collapsed grid).
"""

import math

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from risk_perception.predictive_risk_costmap_node import PredictiveRiskCostmapNode


def make_detection(det_id, label, x, y, vx, vy, score=0.9):
    """Match object_tracker_node's world_objects wire contract -- same
    helper as test_predictive_costmap_multiobject.make_detection."""
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
    cov[0] = cov[7] = 0.04
    cov[21] = cov[28] = 0.01
    r.pose.covariance = cov
    d.results.append(r)
    return d


def make_activity_grid(node, value: float, width=None, height=None,
                       resolution=None, origin_x=None, origin_y=None
                       ) -> OccupancyGrid:
    """A uniform OccupancyGrid at `value` (0-1), matching spatial_prior_
    node's activity_prior wire contract (int8 0-100). Defaults to node's
    OWN geometry; pass the override kwargs to build a mismatched-geometry
    message for the resample test."""
    msg = OccupancyGrid()
    msg.info.resolution = node.res if resolution is None else resolution
    msg.info.width = node.cols if width is None else width
    msg.info.height = node.rows if height is None else height
    msg.info.origin.position.x = node.ox if origin_x is None else origin_x
    msg.info.origin.position.y = node.oy if origin_y is None else origin_y
    msg.info.origin.orientation.w = 1.0
    cell = int(round(max(0.0, min(1.0, value)) * 100))
    msg.data = [cell] * (msg.info.width * msg.info.height)
    return msg


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
    assert (n.ox, n.oy, n.res) == (-6.0, -6.0, 0.1), (
        "node grid defaults changed -- update the cell coordinates used "
        "by the activity-floor tests")
    yield n
    n.destroy_node()


def _patch_publishers(node):
    """Capture the raw numpy arrays _tick() would have published, without
    needing a subscriber or a spin -- same pattern as monkeypatching a
    callback directly. Returns the dict later filled in place."""
    captured = {}

    def capture_grid(grid, publisher=None, stamp=None):
        key = "grid" if publisher is node.grid_pub else "planner"
        captured[key] = grid.copy()

    def capture_stack(stack, stamp=None):
        captured["stack"] = stack.copy()

    node._publish_grid = capture_grid
    node._publish_stack = capture_stack
    node._publish_srm = lambda *a, **kw: None
    return captured


# ------------------------------------------------------ _resample_occupancy_grid

def test_activity_cb_same_geometry_is_a_straight_copy(node):
    node._activity_cb(make_activity_grid(node, 0.6))
    assert node.activity_baseline is not None
    assert node.activity_baseline.shape == (node.rows, node.cols)
    assert np.allclose(node.activity_baseline, 0.6, atol=1e-3)


def test_activity_cb_resamples_mismatched_geometry(node):
    """A message on a coarser grid, half this node's extent, anchored at
    the same origin: cells inside its footprint resample to its value;
    cells outside it (never covered by the source message at all) default
    to 0.0 -- 'never observed at that geometry' must not read as risk."""
    half_cols, half_rows = node.cols // 2, node.rows // 2
    src_res = node.res  # same resolution, half the EXTENT (fewer cells)
    msg = make_activity_grid(
        node, 0.8, width=half_cols, height=half_rows, resolution=src_res,
        origin_x=node.ox, origin_y=node.oy)
    node._activity_cb(msg)

    assert node.activity_baseline.shape == (node.rows, node.cols)

    # Well inside the source grid's footprint.
    near_col, near_row = 5, 5
    assert node.activity_baseline[near_row, near_col] == pytest.approx(0.8, abs=1e-3)

    # The far corner of THIS node's (larger) grid, genuinely outside the
    # smaller source grid's covered area.
    far_col, far_row = node.cols - 1, node.rows - 1
    src_max_x = node.ox + half_cols * src_res
    src_max_y = node.oy + half_rows * src_res
    far_wx = node.ox + (far_col + 0.5) * node.res
    far_wy = node.oy + (far_row + 0.5) * node.res
    assert far_wx > src_max_x or far_wy > src_max_y  # sanity: genuinely outside
    assert node.activity_baseline[far_row, far_col] == pytest.approx(0.0)


def test_activity_baseline_is_unweighted_unlike_prior_baseline(node):
    """_resample_occupancy_grid is shared with _prior_cb, but ONLY
    _prior_cb bakes its own weight in at ingestion -- _activity_cb must
    cache the raw [0, 1] resampled value, since activity's three consumers
    each apply their OWN weight later, in _tick."""
    node._activity_cb(make_activity_grid(node, 0.4))
    assert node.activity_baseline.max() == pytest.approx(0.4, abs=1e-3)


# --------------------------------------------------------------------- _tick

def test_tick_activity_seed_leaves_painted_core_unchanged(node):
    """A track's own painted peak (0.75, the WP-A agnostic default) is far
    above any of the activity weights' floor (<= 0.25 * activity) -- the
    MAX-seed must not touch it."""
    node.activity_baseline = np.full((node.rows, node.cols), 0.5, dtype=np.float32)
    captured = _patch_publishers(node)

    det = make_detection("1", "person", 2.05, 0.05, 0.0, 0.0)
    node.latest = Detection3DArray(detections=[det])

    node._tick()

    row = round((0.05 - node.oy) / node.res - 0.5)
    col = round((2.05 - node.ox) / node.res - 0.5)
    assert captured["grid"][row, col] == pytest.approx(
        node.stack_consequence_agnostic, rel=1e-5)
    assert captured["stack"][0, row, col] == pytest.approx(
        node.stack_consequence_agnostic, rel=1e-5)


def test_tick_activity_seed_halos_untouched_cells_at_exactly_weight_times_a(node):
    """Far from any track, an empty cell reads EXACTLY weight * A on the
    collapsed grid, every stack layer, and the planner grid -- each with
    ITS OWN weight, not a shared one."""
    a_value = 0.5
    node.activity_baseline = np.full((node.rows, node.cols), a_value, dtype=np.float32)
    captured = _patch_publishers(node)

    det = make_detection("1", "person", 2.05, 0.05, 0.0, 0.0)
    node.latest = Detection3DArray(detections=[det])

    node._tick()

    # Far corner of the grid -- outside any track's splat footprint.
    far_row, far_col = node.rows - 1, node.cols - 1

    assert captured["grid"][far_row, far_col] == pytest.approx(
        node.activity_weight_grid * a_value, rel=1e-5)
    assert captured["planner"][far_row, far_col] == pytest.approx(
        node.activity_weight_planner * a_value, rel=1e-5)
    for k in range(captured["stack"].shape[0]):
        assert captured["stack"][k, far_row, far_col] == pytest.approx(
            node.activity_weight * a_value, rel=1e-5), f"stack layer {k}"


def test_tick_activity_weight_zero_is_a_no_op(node):
    """Every weight at 0.0 -- the halo cell reads exactly 0.0, i.e. the
    activity floor contributes nothing at all, matching pre-WP-C
    behaviour bit for bit."""
    node.activity_baseline = np.full((node.rows, node.cols), 0.9, dtype=np.float32)
    node.activity_weight = 0.0
    node.activity_weight_planner = 0.0
    node.activity_weight_grid = 0.0
    captured = _patch_publishers(node)

    det = make_detection("1", "person", 2.05, 0.05, 0.0, 0.0)
    node.latest = Detection3DArray(detections=[det])

    node._tick()

    far_row, far_col = node.rows - 1, node.cols - 1
    assert captured["grid"][far_row, far_col] == pytest.approx(0.0)
    assert captured["planner"][far_row, far_col] == pytest.approx(0.0)
    assert captured["stack"][:, far_row, far_col].max() == pytest.approx(0.0)


def test_tick_no_activity_message_is_a_no_op(node):
    """activity_baseline stays None until the first message arrives --
    _tick's seeding block must be skipped outright, not crash on None."""
    assert node.activity_baseline is None
    captured = _patch_publishers(node)

    node.latest = Detection3DArray(detections=[])

    node._tick()  # must not raise

    assert captured["grid"].max() == pytest.approx(0.0)
