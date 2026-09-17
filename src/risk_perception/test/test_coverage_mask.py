"""
Tier 0 of the WP4 analytic exposure map: pure geometry, no ROS, no camera,
no Isaac, no rclpy context. Same rationale (and the same speed) as
test_spatial_prior_grid.py -- coverage_mask_node.py keeps its projection and
ray-casting maths in module-level functions precisely so the claim "this
grid says where we can see" is checkable against hand-derived numbers rather
than against a screenshot of RViz.

The two things worth being sure of:
  * a camera's floor footprint really is the patch under its frustum (and
    the grazing-range guard really removes the far corners rather than
    silently clamping them inward, which would over-claim coverage);
  * a lidar ray stops AT a wall and leaves everything behind it uncovered,
    with unknown map territory treated as opaque -- "nobody surveyed this"
    is the one thing an exposure map must never call visible.
"""

import numpy as np

from risk_perception.coverage_mask_node import (
    camera_floor_polygon,
    combine_sources,
    lidar_los_mask,
    quaternion_matrix,
    rasterise_polygon,
)

RES = 0.10
OX, OY = -5.0, -5.0
ROWS, COLS = 100, 100                      # 10 m x 10 m, x/y in [-5, 5)

GRID = {"resolution": RES, "origin_x": OX, "origin_y": OY,
        "rows": ROWS, "cols": COLS}
MAP_INFO = {"resolution": RES, "origin_x": OX, "origin_y": OY}

# A camera 3 m up looking straight down: optical +z (forward) -> map -z,
# optical +x (image right) -> map +x, optical +y (image down) -> map -y.
# Columns of R are where the optical axes land in the map frame.
STRAIGHT_DOWN = np.array([[1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0]])
# fx = fy = 100 px over a 100 x 100 image => a half-angle of atan(0.5), so
# from 3 m up the footprint is the square |x| <= 1.5, |y| <= 1.5.
K = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]


def cell_of(x, y):
    return int((y - OY) / RES), int((x - OX) / RES)


def covered(mask, x, y):
    r, c = cell_of(x, y)
    return bool(mask[r, c])


# ------------------------------------------------------------ quaternions

def test_quaternion_matrix_is_orthonormal_and_handles_identity():
    assert np.allclose(quaternion_matrix(0.0, 0.0, 0.0, 1.0), np.eye(3))
    # 90 deg about +z: x -> y.
    rot = quaternion_matrix(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0],
                       atol=1e-9)
    # Un-normalised input is normalised rather than scaling every ray.
    scaled = quaternion_matrix(0.0, 0.0, 0.0, 2.0)
    assert np.allclose(scaled, np.eye(3))
    # A zero quaternion cannot rotate anything; identity beats NaN.
    assert np.allclose(quaternion_matrix(0.0, 0.0, 0.0, 0.0), np.eye(3))


# -------------------------------------------------- camera floor footprint

def test_camera_polygon_is_the_square_under_a_downward_frustum():
    poly = camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (0.0, 0.0, 3.0),
                                max_range_m=15.0, samples_per_edge=16)
    assert poly is not None
    # Hand-derived extent: (0 - 50)/100 * 3 = -1.5 on both axes.
    assert poly[:, 0].min() == np.float64(-1.5)
    assert poly[:, 0].max() < 1.5 and poly[:, 0].max() > 1.4
    assert poly[:, 1].max() == np.float64(1.5)
    assert poly[:, 1].min() < -1.4


def test_camera_footprint_covers_the_cells_under_the_frustum():
    poly = camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (0.0, 0.0, 3.0),
                                max_range_m=15.0, samples_per_edge=16)
    mask = rasterise_polygon(poly, GRID)

    assert covered(mask, 0.0, 0.0)          # straight below the camera
    assert covered(mask, 1.0, 1.0)          # inside the square
    assert covered(mask, -1.4, 0.0)
    assert not covered(mask, 2.0, 0.0)      # outside it
    assert not covered(mask, 0.0, 2.0)
    assert not covered(mask, -3.0, -3.0)
    # ~3 m x 3 m at 0.1 m cells is ~900-1000 cells; the boundary rounds.
    assert 850 <= int(mask.sum()) <= 1000


def test_camera_footprint_is_offset_with_the_camera():
    """Same camera moved 2 m east: the same patch, 2 m east. (The projector
    chain's extrinsics come from the USD prim, so an error here would show
    up as coverage claimed under the wrong shelf.)"""
    poly = camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (2.0, 0.0, 3.0),
                                max_range_m=15.0, samples_per_edge=16)
    mask = rasterise_polygon(poly, GRID)
    assert covered(mask, 2.0, 0.0)
    assert covered(mask, 3.0, 1.0)
    assert not covered(mask, 0.0, 0.0)


def test_max_range_drops_the_far_corners_instead_of_clamping_them():
    """The grazing-view guard. A corner ray of this frustum reaches the
    floor at 3*sqrt(1.5) = 3.67 m; an edge-midpoint ray at 3.35 m. A cap
    between the two must keep the middle of each edge and lose the corners
    -- and it must SHRINK the footprint, never pull the corner inward and
    keep claiming it."""
    full = rasterise_polygon(
        camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (0.0, 0.0, 3.0),
                             max_range_m=15.0, samples_per_edge=16), GRID)
    clipped = rasterise_polygon(
        camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (0.0, 0.0, 3.0),
                             max_range_m=3.5, samples_per_edge=16), GRID)

    assert int(clipped.sum()) < int(full.sum())
    assert not (clipped & ~full).any()          # strictly a subset
    assert covered(clipped, 0.0, 0.0)           # the centre survives
    assert covered(clipped, 1.4, 0.0)           # ...and the edge midpoints
    assert covered(full, 1.4, 1.4)
    assert not covered(clipped, 1.4, 1.4)       # the corner is gone


def test_camera_pointing_away_from_the_floor_covers_nothing():
    """Optical +z along map +z (an extrinsic that got flipped, or a camera
    aimed at the ceiling): no border ray meets z = 0, so the honest answer
    is None -- not a polygon stretching to infinity, and not a silent
    all-zero mask that looks like a working camera seeing nowhere."""
    upward = np.eye(3)
    poly = camera_floor_polygon(K, 100, 100, upward, (0.0, 0.0, 3.0),
                                max_range_m=15.0, samples_per_edge=16)
    assert poly is None
    assert not rasterise_polygon(poly, GRID).any()
    # A camera at floor level is refused too.
    assert camera_floor_polygon(K, 100, 100, STRAIGHT_DOWN, (0.0, 0.0, 0.0),
                                max_range_m=15.0) is None


# ---------------------------------------------------- lidar line of sight

def _free_map():
    return np.zeros((ROWS, COLS), dtype=bool)


def test_lidar_line_of_sight_stops_at_a_wall():
    """One 10 cm wall column at x = 2.0. Everything up to it is visible;
    everything behind it, along the same bearing, is not -- while the same
    range in every other direction still is."""
    blocked = _free_map()
    wall_col = int((2.0 - OX) / RES)
    blocked[:, wall_col] = True

    mask = lidar_los_mask(blocked, MAP_INFO, (0.0, 0.0), GRID,
                          n_rays=360, max_range_m=12.0)

    assert covered(mask, 0.0, 0.0)          # the robot's own cell
    assert covered(mask, 1.0, 0.0)          # short of the wall
    assert covered(mask, 1.9, 0.0)
    assert not covered(mask, 2.0, 0.0)      # the wall cell itself stops it
    assert not covered(mask, 3.0, 0.0)      # the shadow behind it
    assert not covered(mask, 4.5, 0.0)
    assert covered(mask, 0.0, 3.0)          # nothing that way
    assert covered(mask, -3.0, 0.0)


def test_lidar_line_of_sight_treats_unknown_as_opaque():
    """`blocked` is built from "occupied OR unknown" by the node; an
    unsurveyed patch must cast a shadow, because the whole point of this
    grid is to distinguish "looked and saw nothing" from "never looked"."""
    blocked = _free_map()
    unknown_col = int((1.5 - OX) / RES)
    blocked[:, unknown_col] = True
    mask = lidar_los_mask(blocked, MAP_INFO, (0.0, 0.0), GRID,
                          n_rays=360, max_range_m=12.0)
    assert covered(mask, 1.0, 0.0)
    assert not covered(mask, 2.5, 0.0)


def test_lidar_range_bounds_the_disc():
    free = _free_map()
    near = lidar_los_mask(free, MAP_INFO, (0.0, 0.0), GRID,
                          n_rays=360, max_range_m=2.0)
    assert covered(near, 1.5, 0.0)
    assert not covered(near, 2.5, 0.0)
    assert not covered(near, 0.0, 3.0)

    far = lidar_los_mask(free, MAP_INFO, (0.0, 0.0), GRID,
                         n_rays=360, max_range_m=12.0)
    assert int(far.sum()) > int(near.sum())


def test_lidar_off_the_edge_of_the_static_map_is_blocked():
    """A 2 m x 2 m map fragment around the origin: beyond its edge there is
    no evidence of anything, which is not the same as clear floor."""
    small_info = {"resolution": RES, "origin_x": -1.0, "origin_y": -1.0}
    small = np.zeros((20, 20), dtype=bool)
    mask = lidar_los_mask(small, small_info, (0.0, 0.0), GRID,
                          n_rays=360, max_range_m=12.0)
    assert covered(mask, 0.5, 0.0)
    assert not covered(mask, 1.5, 0.0)


def test_lidar_pose_on_an_occupied_cell_still_yields_a_disc():
    """AMCL can put the robot inside an inflated wall cell. Blocking at
    range zero would return an empty mask and silently claim the lidar sees
    nothing at all, so the first sample is never a stopper."""
    blocked = _free_map()
    r, c = cell_of(0.0, 0.0)
    blocked[r, c] = True
    mask = lidar_los_mask(blocked, MAP_INFO, (0.0, 0.0), GRID,
                          n_rays=360, max_range_m=5.0)
    assert int(mask.sum()) > 0
    assert covered(mask, 1.0, 0.0)


# --------------------------------------------------------- source combine

def test_combine_sources_ors_and_counts():
    a = np.zeros((ROWS, COLS), dtype=bool)
    b = np.zeros((ROWS, COLS), dtype=bool)
    a[10:20, 10:20] = True
    b[15:25, 15:25] = True

    any_mask, count = combine_sources([a, b], ROWS, COLS)
    assert any_mask[12, 12] and count[12, 12] == 1
    assert any_mask[17, 17] and count[17, 17] == 2
    assert not any_mask[50, 50] and count[50, 50] == 0

    # No sources at all is a valid answer: nowhere is covered.
    empty_mask, empty_count = combine_sources([], ROWS, COLS)
    assert not empty_mask.any() and not empty_count.any()

    # A mask on the wrong geometry is dropped, not broadcast into a lie.
    wrong = np.ones((7, 7), dtype=bool)
    only_a, count_a = combine_sources([a, wrong], ROWS, COLS)
    assert int(count_a.max()) == 1
    assert (only_a == a).all()


def test_rasterise_polygon_uses_the_occupancygrid_row_convention():
    """row 0 at origin_y, col 0 at origin_x -- the same convention as
    OccupancyGrid.data, so the mask flattens into a message with no flip."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]])
    mask = rasterise_polygon(square, GRID)
    assert covered(mask, 0.5, 1.0)
    assert not covered(mask, -0.5, 1.0)
    assert not covered(mask, 0.5, 2.5)
    # Degenerate input is empty, not an exception.
    assert not rasterise_polygon(np.array([[0.0, 0.0], [1.0, 1.0]]),
                                 GRID).any()
    assert not rasterise_polygon(None, GRID).any()
