"""
Pure geometry coverage for global_cam_projector_node.apply_extent_offset
(WP-A#2): the mask's bottom-band pixel is the object's floor-contact point
NEAREST the overhead camera, not its centre, so the projector pushes it
further along the camera->point horizontal ray by a per-category
half-extent. No rclpy.init(), no tf, no image -- apply_extent_offset is a
plain numpy function with no node state, importable the same way the rest
of this file's module is (rclpy/cv_bridge/tf2_ros/vision_msgs must be on
the path, same as every other node-importing test in this directory).
"""

import numpy as np
import pytest

from risk_perception.global_cam_projector_node import apply_extent_offset


def test_offset_moves_point_exactly_extent_further_from_camera():
    cam_xy = np.array([0.0, 0.0])
    point_xy = np.array([3.0, 4.0])  # 5 m from camera, direction (0.6, 0.8)

    result = apply_extent_offset(point_xy, cam_xy, extent=1.0)

    # Same direction, distance grown by exactly `extent`.
    original_distance = float(np.linalg.norm(point_xy - cam_xy))
    new_distance = float(np.linalg.norm(result - cam_xy))
    assert new_distance == pytest.approx(original_distance + 1.0)

    direction = (point_xy - cam_xy) / original_distance
    expected = point_xy + 1.0 * direction
    assert np.allclose(result, expected)


def test_zero_extent_is_identity():
    cam_xy = np.array([1.0, 2.0])
    point_xy = np.array([5.0, -3.0])

    result = apply_extent_offset(point_xy, cam_xy, extent=0.0)

    assert np.allclose(result, point_xy)


def test_degenerate_point_equals_camera_returns_input():
    cam_xy = np.array([2.5, -1.5])
    point_xy = np.array([2.5, -1.5])

    result = apply_extent_offset(point_xy, cam_xy, extent=0.35)

    assert np.allclose(result, point_xy)


def test_offset_direction_points_away_from_camera_on_arbitrary_axis():
    cam_xy = np.array([10.0, 10.0])
    point_xy = np.array([10.0, 6.0])  # straight below the camera in y

    result = apply_extent_offset(point_xy, cam_xy, extent=0.5)

    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(5.5)  # pushed further from cam (y=10)


def test_returned_array_does_not_alias_input():
    cam_xy = np.array([0.0, 0.0])
    point_xy = np.array([1.0, 0.0])

    result = apply_extent_offset(point_xy, cam_xy, extent=0.0)
    result[0] = 999.0

    assert point_xy[0] == pytest.approx(1.0)
