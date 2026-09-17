#!/usr/bin/env python3
"""
scan_clustering.py  --  pure math for the lidar dynamic-cluster detector

Separate from scan_cluster_detector_node.py (a rclpy Node) so this logic can
be unit-tested with plain pytest, no ROS graph -- same rationale as
encounter_geometry.py being split out of predictive_risk_costmap_node.py; see
that module's docstring.

Pipeline (see scan_cluster_detector_node.py for the ROS glue):

  1. build_static_distance_field(): once per incoming `map` message, a
     distance-transform field over "occupied or unknown" cells.
  2. subtract_static(): per scan, drop any map-frame point too close to that
     field's zero set (an occupied cell OR unknown territory -- both count
     as "not obviously dynamic").
  3. cluster_points(): range-adaptive Euclidean clustering of the surviving
     points, walked in beam order, with wraparound handling for a full-
     circle scan.
  4. cluster_summaries(): per-cluster centroid/extent/n_points/mean-range,
     with the min_points / max_extent_m / self_radius_m filters applied.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

# WHY (2026-09-10 finding): a lidar_cluster with only static_margin_m as its
# filter can sprout for a single scan off a shelf-edge artifact (AMCL error
# + map quantisation letting the margin through), feed straight into
# object_tracker's label-agnostic pass, and have its one-scan jump read as a
# multi-m/s velocity by the Kalman filter -- the "comet" bug this WP fixes.
# ClusterPersistence (below) requires a cluster to keep matching across
# several consecutive scans before it is ever published, so a flicker like
# that never reaches the tracker at all.

# Occupancy value (0-100 scale, matching nav_msgs/OccupancyGrid) at or above
# which a cell counts as occupied. Not exposed as a node parameter -- the
# saved warehouse map is effectively binary (free/occupied/unknown), and
# this only matters for however a grayscale-derived probability map might
# be thresholded upstream of this node.
OCCUPIED_THRESHOLD = 50


def build_static_distance_field(
    occupancy: np.ndarray,
    occ_grid_info: Dict[str, float],
    occupied_threshold: int = OCCUPIED_THRESHOLD,
) -> np.ndarray:
    """OccupancyGrid.data reshaped to (height, width) -> distance-to-nearest-
    occupied-or-unknown-cell field, in METRES, same shape.

    Built once per incoming `map` message (see scan_cluster_detector_node.
    _map_cb) -- the static map does not change within a run. Unknown cells
    (-1) count as "occupied" for this purpose: a beam landing on unknown
    territory gets distance 0 in the returned field, so subtract_static's
    single `< static_margin_m` test also implements "drop points that land
    on unknown cells" without a second flag.
    """
    occupancy = np.asarray(occupancy)
    resolution = float(occ_grid_info["resolution"])
    occupied_or_unknown = (occupancy < 0) | (occupancy >= occupied_threshold)
    # distance_transform_edt gives the distance (in CELLS) from each cell to
    # the nearest False (zero) cell in its input -- invert the mask so the
    # zero set is exactly "occupied or unknown".
    dist_cells = distance_transform_edt(~occupied_or_unknown)
    return dist_cells * resolution


def subtract_static(
    points_xy: np.ndarray,
    occ_grid_info: Dict[str, float],
    dist_transform_m: np.ndarray,
    static_margin_m: float,
) -> np.ndarray:
    """Boolean mask (True = keep / plausibly dynamic) for map-frame points.

    A point is dropped (False) if it falls within static_margin_m of an
    occupied cell, lands on an unknown cell (dist_transform_m is exactly 0
    there by construction -- see build_static_distance_field), or falls
    outside the map's bounds entirely (no static information there, so it
    cannot be told apart from a wall -- treated conservatively as static).
    """
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = points_xy.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    resolution = float(occ_grid_info["resolution"])
    origin_x = float(occ_grid_info["origin_x"])
    origin_y = float(occ_grid_info["origin_y"])
    height, width = dist_transform_m.shape

    cols = np.floor((points_xy[:, 0] - origin_x) / resolution).astype(np.int64)
    rows = np.floor((points_xy[:, 1] - origin_y) / resolution).astype(np.int64)

    in_bounds = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)

    keep = np.zeros(n, dtype=bool)
    dist = np.zeros(n, dtype=np.float64)
    dist[in_bounds] = dist_transform_m[rows[in_bounds], cols[in_bounds]]
    keep[in_bounds] = dist[in_bounds] >= static_margin_m
    return keep


def cluster_points(
    points_xy: np.ndarray,
    gap_base_m: float,
    gap_per_m: float,
    ranges: Sequence[float],
) -> List[np.ndarray]:
    """Consecutive-beam Euclidean clustering with a range-adaptive gap.

    `points_xy`/`ranges` are arrays of length N, in BEAM ORDER -- gaps from
    beams already dropped upstream (inf/nan/out-of-range/static-subtracted)
    are fine, since the gap test below is on the Euclidean distance between
    successive SURVIVING points, not on beam-index continuity.

    Two consecutive points belong to the same cluster iff their distance is
    <= gap_base_m + gap_per_m * r, where r is the mean of the two points'
    ranges -- a beam gap that would be tight up close is loose far away,
    since a fixed angular resolution spans a wider arc length at range.

    Wraparound: if the scan covers a full circle, the LAST surviving point
    and the FIRST are also beam-adjacent (angle_max wraps to angle_min) --
    if that pair satisfies the same gap test, the last and first clusters
    are merged into one (any clusters strictly between them are untouched).

    Returns a list of clusters, each a 1-D int array of indices into
    points_xy/ranges, in original beam order within the cluster.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    n = points_xy.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [np.array([0])]

    def _gap(i: int, j: int) -> float:
        r = 0.5 * (ranges[i] + ranges[j])
        return gap_base_m + gap_per_m * max(0.0, r)

    def _dist(i: int, j: int) -> float:
        return float(np.hypot(points_xy[i, 0] - points_xy[j, 0],
                              points_xy[i, 1] - points_xy[j, 1]))

    clusters: List[List[int]] = []
    current = [0]
    for i in range(1, n):
        if _dist(i, i - 1) <= _gap(i, i - 1):
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    if len(clusters) > 1:
        first, last = clusters[0], clusters[-1]
        i0, i1 = last[-1], first[0]
        if _dist(i0, i1) <= _gap(i0, i1):
            merged = last + first
            clusters = [merged] + clusters[1:-1]

    return [np.array(c) for c in clusters]


def push_centroid_from_sensor(
    centroid_xy: Tuple[float, float],
    extent: Tuple[float, float],
    sensor_xy: Optional[np.ndarray],
    push_factor: float,
    push_max_m: float,
) -> Tuple[float, float]:
    """Push a cluster's reported centroid away from the sensor.

    A lidar beam only ever returns off the NEAR face of a solid object, so
    the raw mean-of-points centroid sits biased toward the sensor by
    roughly half the object's extent along the sensor->centroid bearing --
    a person or cart is reported ~0.2-0.3 m closer to the robot than their
    true center (see the lidar-fusion probe's 0.40 m median tracker
    offset). Correct for this by nudging the centroid outward along that
    bearing by `min(push_factor * max(extent), push_max_m)`.

    `sensor_xy` is the LIDAR's own map-frame position (not the robot's
    footprint/base_frame -- see scan_cluster_detector_node._scan_cb's
    `origin`), since the bearing has to originate at the sensor, not
    wherever else on the robot self_radius_m is measured from.

    Returns `centroid_xy` unchanged if `sensor_xy` is None, `push_factor <=
    0`, `push_max_m <= 0`, the computed push distance is 0, or the sensor
    sits exactly on the centroid (degenerate bearing). The bbox extent
    itself is never touched by this -- only the reported centroid moves.
    """
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    if sensor_xy is None or push_factor <= 0.0 or push_max_m <= 0.0:
        return (cx, cy)

    push = min(push_factor * max(float(extent[0]), float(extent[1])), push_max_m)
    if push <= 0.0:
        return (cx, cy)

    dx = cx - float(sensor_xy[0])
    dy = cy - float(sensor_xy[1])
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return (cx, cy)

    ux, uy = dx / dist, dy / dist
    return (cx + ux * push, cy + uy * push)


def cluster_summaries(
    points_xy: np.ndarray,
    ranges: Sequence[float],
    clusters: List[np.ndarray],
    min_points: int,
    max_extent_m: float,
    self_radius_m: float = 0.0,
    robot_xy: Optional[np.ndarray] = None,
    sensor_xy: Optional[np.ndarray] = None,
    centroid_push_factor: float = 0.0,
    centroid_push_max_m: float = 0.0,
) -> List[Dict]:
    """Per-cluster centroid/extent/n_points/mean-range, with filters.

    Drops a cluster if it has fewer than `min_points` points, its bbox
    extent (max of width/height) exceeds `max_extent_m` (almost always a
    static-subtraction miss on a large flat surface), or its centroid falls
    within `self_radius_m` of `robot_xy` -- the lidar's return off the
    robot's own mount/body (robot_xy=None disables this last filter, e.g.
    when TF is not yet available).

    The near-face centroid bias is corrected BEFORE the self-exclusion
    check above (and before the returned centroid is used for anything
    else) via `push_centroid_from_sensor` -- see that function's
    docstring. `centroid_push_factor`/`centroid_push_max_m` <= 0 (or
    `sensor_xy=None`) disables this and leaves the raw mean-of-points
    centroid untouched. The bbox extent returned is always the raw,
    un-pushed extent.

    Returns a list of dicts: {"centroid": (x, y), "extent": (w, h),
    "n_points": int, "mean_range": float}, one per surviving cluster.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)

    summaries = []
    for idx in clusters:
        idx = np.asarray(idx)
        n_points = int(idx.shape[0])
        if n_points < min_points:
            continue

        pts = points_xy[idx]
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        if max(w, h) > max_extent_m:
            continue

        cx, cy = push_centroid_from_sensor(
            (cx, cy), (w, h), sensor_xy, centroid_push_factor, centroid_push_max_m)

        if robot_xy is not None and self_radius_m > 0.0:
            d = float(np.hypot(cx - robot_xy[0], cy - robot_xy[1]))
            if d <= self_radius_m:
                continue

        summaries.append({
            "centroid": (cx, cy),
            "extent": (w, h),
            "n_points": n_points,
            "mean_range": float(ranges[idx].mean()),
        })

    return summaries


class ClusterPersistence:
    """Cross-scan cluster confirmation: only publish a cluster once it has
    matched a cluster in EACH of the last `min_consecutive` scans, within
    `match_m` of its previous match each time.

    See the WHY-comment above this module's imports for the 2026-09-10
    finding this exists for. Stateful (it has to remember candidates
    between calls) but otherwise pure/ROS-free, same rationale as the rest
    of this module (scan_cluster_detector_node.py owns one instance and
    calls `update()` once per scan -- see test/test_scan_clustering.py for
    unit tests with no ROS graph involved).

    Matching is greedy nearest-neighbour, one candidate at a time in
    insertion order (oldest candidates first), each candidate claiming the
    closest still-unclaimed input centroid within `match_m` -- deliberately
    simple (this module already has a real Hungarian/greedy split in
    object_tracker_node.py for the tracker's own association; duplicating
    that machinery here for a same-scan, same-sensor match is not worth
    it). A candidate that fails to match ANY input centroid this scan is
    dropped outright -- `min_consecutive` means "the last N scans", not
    "N scans total with gaps allowed", so a single missed scan resets that
    candidate's streak to zero rather than merely pausing it.
    """

    def __init__(self, min_consecutive: int = 3, match_m: float = 0.3):
        self.min_consecutive = int(min_consecutive)
        self.match_m = float(match_m)
        # One dict per live candidate: {"xy": (x, y), "streak": int,
        # "last_stamp": float}. `streak` is consecutive matched scans
        # ending at (and including) this candidate's most recent match.
        self._candidates: List[Dict] = []

    def update(self, centroids: Sequence[Tuple[float, float]], stamp: float) -> List[Tuple[float, float]]:
        """Feed one scan's cluster centroids in; get back the subset that
        are CONFIRMED (streak >= min_consecutive) after this scan --
        confirmed centroids are exactly the matching entries of the input
        `centroids` (same values, not recomputed), so a caller can filter
        anything keyed on the same centroid tuples (e.g. the full summary
        dicts cluster_summaries() produced them from).

        A confirmed candidate stays confirmed on every subsequent matching
        scan too, not just the one scan its streak first crosses
        min_consecutive -- a persistently-tracked object must keep being
        published for as long as it keeps matching.
        """
        centroids = [(float(c[0]), float(c[1])) for c in centroids]
        matched_input = set()
        new_candidates: List[Dict] = []
        confirmed: List[Tuple[float, float]] = []

        for cand in self._candidates:
            cx, cy = cand["xy"]
            best_j, best_d = None, None
            for j, (x, y) in enumerate(centroids):
                if j in matched_input:
                    continue
                d = math.hypot(cx - x, cy - y)
                if d <= self.match_m and (best_d is None or d < best_d):
                    best_j, best_d = j, d
            if best_j is None:
                # No match this scan -- candidate's streak ends here, not
                # carried forward (see class docstring).
                continue
            matched_input.add(best_j)
            xy = centroids[best_j]
            streak = cand["streak"] + 1
            new_candidates.append({"xy": xy, "streak": streak, "last_stamp": stamp})
            if streak >= self.min_consecutive:
                confirmed.append(xy)

        for j, xy in enumerate(centroids):
            if j not in matched_input:
                new_candidates.append({"xy": xy, "streak": 1, "last_stamp": stamp})
                # min_consecutive == 1 is a degenerate but valid config
                # (every cluster confirms on first sight) -- handle it here
                # too, not just for the matched-existing branch above.
                if self.min_consecutive <= 1:
                    confirmed.append(xy)

        self._candidates = new_candidates
        return confirmed
