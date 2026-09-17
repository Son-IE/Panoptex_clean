#!/usr/bin/env python3
"""
geometry_utils.py

Small, dependency-free rotation/quaternion helpers shared by the global-cam
nodes (global_cam_calibrator_node, global_cam_projector_node,
global_cam_localizer_node). Kept separate from map_frame_projector_node.py's
own local copies of similar math -- that node predates this one and works;
duplicating a few functions there was less risky than refactoring a working
node to import from here.
"""

import math

import numpy as np


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a normalized quaternion (x, y, z, w) into a 3x3 rotation matrix."""

    norm = np.sqrt(x * x + y * y + z * z + w * w)

    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> normalized (x, y, z, w) quaternion."""

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm > 1e-12:
        q /= norm

    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def yaw_to_rotation_z(yaw: float) -> np.ndarray:
    """Rotation matrix for a rotation of `yaw` radians about the z axis."""

    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def tag_object_points(tag_size: float) -> np.ndarray:
    """The 4 corners of a tag in its OWN frame, centered at the origin.

    Corner order is bottom-left, bottom-right, top-right, top-left --
    counter-clockwise starting at the tag's local (-x, -y), which is the
    standard AprilTag library convention.

    This is the single source of truth for corner ordering: both
    global_cam_survey_node (per-tag solvePnP, and deriving each tag's world
    yaw from it) and tag_corners_world (reconstructing world corners from a
    surveyed yaw) build on it. That matters -- as long as BOTH use this same
    ordering, a mismatch against the installed apriltag_ros version's actual
    convention cancels out: the survey absorbs the offset into the derived
    yaw, and the reconstruction puts it back. Only a handedness (clockwise
    vs. counter-clockwise) error would fail to cancel, and that shows up
    immediately as a wildly wrong solve.
    """

    s = tag_size / 2.0
    return np.array(
        [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
        dtype=np.float64,
    )


def tag_corners_world(pose: dict, tag_size: float) -> np.ndarray:
    """4 corner positions (world/map frame) for a tag surveyed at `pose`.

    `pose` is {x, y, yaw} (and optionally z, default 0) describing the
    tag's CENTER, flat on the floor. See tag_object_points for the corner
    ordering convention.
    """

    local = tag_object_points(tag_size)

    R = yaw_to_rotation_z(float(pose["yaw"]))
    t = np.array(
        [float(pose["x"]), float(pose["y"]), float(pose.get("z", 0.0))],
        dtype=np.float64,
    )

    return (R @ local.T).T + t


def kabsch(points_from: np.ndarray, points_to: np.ndarray):
    """Rigid transform (R, t) minimizing ||points_to - (R @ points_from + t)||.

    Classic Kabsch/Umeyama via SVD, with the determinant correction that
    forbids a reflection (so the result is always a proper rotation). Both
    inputs are (N, 3). With 3 non-collinear point pairs the solution is
    exact and unique.
    """

    points_from = np.asarray(points_from, dtype=np.float64)
    points_to = np.asarray(points_to, dtype=np.float64)

    centroid_from = points_from.mean(axis=0)
    centroid_to = points_to.mean(axis=0)

    centered_from = points_from - centroid_from
    centered_to = points_to - centroid_to

    H = centered_from.T @ centered_to
    U, _, Vt = np.linalg.svd(H)

    # Without this, a mirrored point set would yield a reflection matrix.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    t = centroid_to - R @ centroid_from

    return R, t


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Snap an (averaged, hence slightly non-orthogonal) matrix back to SO(3)."""

    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = U @ Vt

    if np.linalg.det(out) < 0:
        U[:, -1] *= -1.0
        out = U @ Vt

    return out


def triangle_from_distances(d12: float, d13: float, d23: float) -> np.ndarray:
    """Place 3 tags in a world frame given only their pairwise distances.

    Convention: tag 1 at the origin, tag 2 on the +x axis, tag 3 at +y.
    Returns a (3, 3) array of [tag1, tag2, tag3] positions with z = 0.

    The +y choice for tag 3 fixes a handedness that may or may not match
    reality -- the caller resolves that by checking the solved camera ends
    up ABOVE the floor (z > 0) and mirroring if not.
    """

    for a, b, c in ((d12, d13, d23), (d13, d12, d23), (d23, d12, d13)):
        if a >= b + c:
            raise ValueError(
                f"distances {d12}, {d13}, {d23} violate the triangle "
                "inequality -- re-check the measurements"
            )

    x3 = (d12 * d12 + d13 * d13 - d23 * d23) / (2.0 * d12)
    y3 = np.sqrt(max(d13 * d13 - x3 * x3, 0.0))

    return np.array(
        [[0.0, 0.0, 0.0], [d12, 0.0, 0.0], [x3, y3, 0.0]],
        dtype=np.float64,
    )
