#!/usr/bin/env python3
"""
coverage_mask_node.py  --  WP4: the ANALYTIC EXPOSURE MAP.

Every other grid in this package answers "what is here?". This one answers
the prior question the paper's future-work item 1 ("visibility-normalized
place statistics") is about: **where can this system see at all?**

    /risk_perception/spatial_prior   an OBSERVED-motion histogram. A cell at
                                     0 means "nothing was ever seen moving
                                     here" -- which is either "nothing ever
                                     goes here" or "no sensor has ever
                                     looked here", and the grid itself
                                     cannot tell the two apart (see
                                     spatial_prior_node.py's "Honest
                                     limitations").
    /risk_perception/coverage        which of the two it is, computed
                                     analytically from geometry rather than
                                     from detections.

Two source kinds, unioned:

  (a) OVERHEAD CAMERA FOOTPRINT (per camera in `camera_names`). A pinhole
      camera with known intrinsics (its own /<cam>/camera_info) and known
      extrinsics (the static TF map -> <cam>_optical_frame that
      sim_global_cams.launch.py publishes off the USD prim) sees exactly the
      floor polygon bounded by the four image edges' rays, intersected with
      z = 0. We march the image BORDER (not the whole image), intersect each
      border ray with the floor plane, drop the rays that miss the floor or
      land beyond `max_range_m` (the grazing-view guard -- the sim mounts sit
      ~24 deg below horizontal, so the top image rows project tens of metres
      out where the ground-plane back-projection is worthless; same number
      and same reason as global_cams_sim.yaml's `projector_max_range_m`,
      which is what global_cam_projector_node applies to its detections),
      and rasterise the surviving polygon. Static camera + static TF =>
      computed ONCE per camera and cached.

  (b) LIDAR LINE OF SIGHT. From the robot's current map-frame pose, cast
      `lidar_rays` rays out to `lidar_range_m` against the STATIC map and
      mark the cells each ray reaches before it is stopped. Occupied
      (>= `occupied_min`) and UNKNOWN cells both stop a ray -- unknown
      territory is not "empty floor you can see through", it is exactly the
      part of the map nobody has ever surveyed. This is a static-map
      visibility argument, not a live scan: the question is "could the lidar
      see a Carter there if one were there", which a live /scan (whose
      returns are wherever things happen to be right now) does not answer.

Published (both RELIABLE + TRANSIENT_LOCAL + KeepLast(1), like every other
grid in this package, so a late subscriber -- e.g. a mission_supervisor
started after the perception stack -- still gets the latest one):

  /risk_perception/coverage        OccupancyGrid, 100 where ANY source
                                   covers the cell, 0 elsewhere. This is the
                                   one mission_supervisor's crossing policy
                                   reads (corridor.zone_coverage thresholds
                                   it at 50).
  /risk_perception/coverage_count  OccupancyGrid, 25 * (number of covering
                                   sources), capped at 100 -- i.e. how
                                   REDUNDANTLY a cell is covered, for
                                   analysis and RViz. Nothing consumes it.

FAIL-SOFT, everywhere. A camera whose CameraInfo or TF never arrives is
logged once and then simply contributes nothing; a missing static map turns
the lidar source off the same way; an empty coverage grid is a perfectly
meaningful answer ("we can see nowhere"), and the supervisor's policy
degrades to its blind-crossing branch rather than to an exception. The one
thing this node must never do is claim coverage it cannot justify.

The geometry parameters are the same five as risk_costmap_node /
predictive_risk_costmap_node / spatial_prior_node, and
panoptex_sim.launch.py spawns this node inside `_grid_nodes` so it inherits
the warehouse floor's extent from global_cams_sim.yaml's `costmap:` block
along with the other three.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener

try:  # cv2 is a hard dependency of this package's camera chain, but the
    # pure helpers below are unit-tested on machines that may not have it --
    # see _fill_poly_numpy for the (identical-in-intent) fallback.
    import cv2 as _cv2
except ImportError:  # pragma: no cover - exercised only without OpenCV
    _cv2 = None

_EPS = 1e-9


# --------------------------------------------------------------- geometry

def quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix (no tf_transformations
    dependency; this is the only piece of it this node needs). Normalises
    defensively: a TF that has been through float32 somewhere is not exactly
    unit, and an un-normalised matrix would scale every projected ray."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < _EPS:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def _border_pixels(width: int, height: int, samples_per_edge: int):
    """The image border, walked ONCE round in order (top edge left->right,
    right edge, bottom edge right->left, left edge). Order matters: the
    result is fed straight to a polygon filler, and a shuffled point set
    would rasterise a bow-tie."""
    n = max(2, int(samples_per_edge))
    w, h = float(width - 1), float(height - 1)
    us = np.linspace(0.0, w, n)
    vs = np.linspace(0.0, h, n)
    pts: List[Tuple[float, float]] = []
    pts += [(u, 0.0) for u in us]
    pts += [(w, v) for v in vs[1:]]
    pts += [(u, h) for u in us[::-1][1:]]
    pts += [(0.0, v) for v in vs[::-1][1:-1]]
    return pts


def camera_floor_polygon(k_matrix: Sequence[float], width: int, height: int,
                         rotation: np.ndarray, translation: Sequence[float],
                         max_range_m: float,
                         samples_per_edge: int = 16) -> Optional[np.ndarray]:
    """The floor patch one pinhole camera sees, as a map-frame polygon.

    k_matrix:    CameraInfo.k, row-major 3x3 (fx 0 cx / 0 fy cy / 0 0 1).
    rotation:    3x3 map <- camera-OPTICAL rotation (x right, y down,
                 z forward -- the REP-103 optical convention Isaac and the
                 rest of this package's projectors use).
    translation: camera origin in the map frame.
    max_range_m: drop a border ray whose floor intersection is farther than
                 this FROM THE CAMERA (3D range, not ground range) -- the
                 grazing-view guard, see the module docstring. <= 0 = off.

    Returns an (N, 2) array of map-frame (x, y) in border order, or None
    when fewer than three border rays reach the floor (a camera looking at
    the horizon, or a bad extrinsic). Rays that miss are SKIPPED, not
    clamped: a clamped ray would assert coverage at a range the projector
    itself refuses to back-project a detection at.
    """
    k = np.asarray(k_matrix, dtype=float).reshape(3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if abs(fx) < _EPS or abs(fy) < _EPS:
        return None
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    if t[2] <= 0.0:
        # A camera at or below the floor plane has no downward intersection
        # to speak of; refuse rather than return the mirror image of one.
        return None

    out: List[Tuple[float, float]] = []
    for u, v in _border_pixels(width, height, samples_per_edge):
        d_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d = rot @ d_opt
        if d[2] > -_EPS:          # pointing level or up: never meets z = 0
            continue
        s = -t[2] / d[2]
        if s <= 0.0:
            continue
        rng = s * float(np.linalg.norm(d))
        if max_range_m > 0.0 and rng > max_range_m:
            continue
        p = t + s * d
        out.append((float(p[0]), float(p[1])))
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=float)


def _fill_poly_numpy(poly_cells: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Even-odd fill of one polygon given in CELL coordinates, vectorised
    over the polygon's bounding box. Only used when OpenCV is unavailable;
    cv2.fillPoly is the same operation with the same cell-centre sampling."""
    mask = np.zeros((rows, cols), dtype=bool)
    xs, ys = poly_cells[:, 0], poly_cells[:, 1]
    c0 = max(0, int(math.floor(xs.min())))
    c1 = min(cols - 1, int(math.ceil(xs.max())))
    r0 = max(0, int(math.floor(ys.min())))
    r1 = min(rows - 1, int(math.ceil(ys.max())))
    if c1 < c0 or r1 < r0:
        return mask
    cc = np.arange(c0, c1 + 1, dtype=float).reshape(1, -1) + 0.5
    rr = np.arange(r0, r1 + 1, dtype=float).reshape(-1, 1) + 0.5
    inside = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    n = poly_cells.shape[0]
    for i in range(n):
        x_i, y_i = xs[i], ys[i]
        x_j, y_j = xs[i - 1], ys[i - 1]
        if abs(y_j - y_i) < _EPS:
            continue
        straddles = ((y_i > rr) != (y_j > rr))
        x_cross = x_i + (rr - y_i) * (x_j - x_i) / (y_j - y_i)
        inside ^= straddles & (cc < x_cross)
    mask[r0:r1 + 1, c0:c1 + 1] = inside
    return mask


def rasterise_polygon(poly_xy, grid_info: Dict) -> np.ndarray:
    """Map-frame polygon -> bool mask on this node's grid.

    grid_info: {"resolution", "origin_x", "origin_y", "rows", "cols"} --
    row 0 at the origin, i.e. mask[row, col] with
    col = (x - origin_x)/res, row = (y - origin_y)/res. Same convention as
    OccupancyGrid.data and as corridor.find_refuge's map_grid, so the mask
    can be flattened into a message with no flip.
    """
    rows, cols = int(grid_info["rows"]), int(grid_info["cols"])
    res = float(grid_info["resolution"])
    mask = np.zeros((rows, cols), dtype=bool)
    if poly_xy is None or res < _EPS:
        return mask
    poly = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    if poly.shape[0] < 3:
        return mask
    cells = np.stack([(poly[:, 0] - float(grid_info["origin_x"])) / res,
                      (poly[:, 1] - float(grid_info["origin_y"])) / res], axis=1)
    if _cv2 is None:  # pragma: no cover - exercised only without OpenCV
        return _fill_poly_numpy(cells, rows, cols)
    buf = np.zeros((rows, cols), dtype=np.uint8)
    _cv2.fillPoly(buf, [np.round(cells).astype(np.int32)], 1)
    return buf.astype(bool)


def lidar_los_mask(blocked: np.ndarray, map_info: Dict,
                   robot_xy: Tuple[float, float], grid_info: Dict,
                   n_rays: int = 360, max_range_m: float = 12.0,
                   step_m: float = 0.0) -> np.ndarray:
    """Line-of-sight disc: the cells a lidar at `robot_xy` could see.

    blocked:  bool array over the STATIC map, True = this cell stops a ray
              (occupied OR unknown -- see the module docstring).
    map_info: {"resolution", "origin_x", "origin_y"} of `blocked`, which
              need NOT match grid_info: the march happens in world
              coordinates and each sample is looked up in both grids
              independently, so the static map's 0.05 m cells and this
              node's 0.10 m cells coexist without resampling either.
    step_m:   march step; 0 = half the finer of the two resolutions, which
              is the Nyquist-ish choice that cannot step over a one-cell
              wall.

    Off the edge of the static map counts as BLOCKED, for the same reason
    unknown does. The robot's own sample (range 0) is never blocking, so a
    pose that has drifted onto an occupied cell still yields a disc instead
    of an empty mask.
    """
    rows, cols = int(grid_info["rows"]), int(grid_info["cols"])
    out = np.zeros((rows, cols), dtype=bool)
    blk_map = np.asarray(blocked, dtype=bool)
    if blk_map.ndim != 2 or blk_map.size == 0:
        return out
    g_res = float(grid_info["resolution"])
    m_res = float(map_info["resolution"])
    if g_res < _EPS or m_res < _EPS or max_range_m <= 0.0 or n_rays < 1:
        return out

    step = float(step_m) if step_m > 0.0 else 0.5 * min(g_res, m_res)
    ranges = np.arange(0.0, float(max_range_m) + step, step)
    angles = np.arange(int(n_rays), dtype=float) * (2.0 * math.pi / int(n_rays))
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    wx = rx + np.outer(np.cos(angles), ranges)
    wy = ry + np.outer(np.sin(angles), ranges)

    m_rows, m_cols = blk_map.shape
    m_col = np.floor((wx - float(map_info["origin_x"])) / m_res).astype(np.int64)
    m_row = np.floor((wy - float(map_info["origin_y"])) / m_res).astype(np.int64)
    in_map = ((m_row >= 0) & (m_row < m_rows) & (m_col >= 0) & (m_col < m_cols))
    stops = np.ones(wx.shape, dtype=bool)          # off-map = blocked
    stops[in_map] = blk_map[m_row[in_map], m_col[in_map]]
    # The cell the sensor is standing in never occludes it. AMCL routinely
    # places the robot inside a wall's own cell (or inside unknown space at
    # the edge of the surveyed area); without this the ray-cast would stop
    # at range ~0 on every bearing and the grid would claim the lidar sees
    # nothing at all -- an exposure map that goes blind exactly when
    # localisation is worst.
    r_row = int(math.floor((ry - float(map_info["origin_y"])) / m_res))
    r_col = int(math.floor((rx - float(map_info["origin_x"])) / m_res))
    if 0 <= r_row < m_rows and 0 <= r_col < m_cols:
        stops &= ~((m_row == r_row) & (m_col == r_col))
    stops[:, 0] = False                            # never block at range 0

    first = np.argmax(stops, axis=1)
    first = np.where(stops.any(axis=1), first, stops.shape[1])
    visible = np.arange(stops.shape[1])[None, :] < first[:, None]

    g_col = np.floor((wx - float(grid_info["origin_x"])) / g_res).astype(np.int64)
    g_row = np.floor((wy - float(grid_info["origin_y"])) / g_res).astype(np.int64)
    ok = (visible & (g_row >= 0) & (g_row < rows) & (g_col >= 0) & (g_col < cols))
    out[g_row[ok], g_col[ok]] = True
    return out


def combine_sources(masks: Sequence[np.ndarray], rows: int, cols: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """(any-sensor OR mask, per-cell source count) over a list of bool
    masks. An empty list is a valid input: nowhere is covered."""
    any_mask = np.zeros((rows, cols), dtype=bool)
    count = np.zeros((rows, cols), dtype=np.int16)
    for m in masks:
        arr = np.asarray(m, dtype=bool)
        if arr.shape != (rows, cols):
            continue
        any_mask |= arr
        count += arr.astype(np.int16)
    return any_mask, count


# ------------------------------------------------------------------ node

class CoverageMaskNode(Node):

    def __init__(self) -> None:
        super().__init__("coverage_mask_node")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        # Geometry -- the same five keys as the other grid nodes, so
        # panoptex_sim.launch.py's _GEOMETRY_KEYS override covers this node
        # too. Defaults are risk_perception.yaml's lab grid, not the sim's.
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width_m", 12.0)
        self.declare_parameter("height_m", 12.0)
        self.declare_parameter("origin_x", -6.0)
        self.declare_parameter("origin_y", -6.0)

        self.declare_parameter("coverage_topic", "/risk_perception/coverage")
        self.declare_parameter("count_topic", "/risk_perception/coverage_count")
        self.declare_parameter("publish_rate", 2.0)

        # (a) overhead cameras
        self.declare_parameter("enable_cameras", True)
        self.declare_parameter(
            "camera_names",
            ["global_cam_mid", "global_cam_entry", "global_cam_exit"])
        # Empty = derive "<name>_optical_frame" / "/<name>/camera_info",
        # exactly the derivations sim_global_cams.launch.py makes from the
        # same `name` (keep the two in sync).
        self.declare_parameter("camera_frames", [""])
        self.declare_parameter("camera_info_topics", [""])
        self.declare_parameter("max_range_m", 15.0)
        self.declare_parameter("boundary_samples", 16)

        # (b) lidar line of sight
        self.declare_parameter("enable_lidar", True)
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("lidar_rays", 360)
        self.declare_parameter("lidar_range_m", 12.0)
        self.declare_parameter("occupied_min", 50)

        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("warn_period_s", 30.0)

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.res = float(gp("resolution").value)
        self.ox = float(gp("origin_x").value)
        self.oy = float(gp("origin_y").value)
        self.cols = max(1, int(round(float(gp("width_m").value) / self.res)))
        self.rows = max(1, int(round(float(gp("height_m").value) / self.res)))
        self.coverage_topic = str(gp("coverage_topic").value)
        self.count_topic = str(gp("count_topic").value)
        self.publish_rate = float(gp("publish_rate").value)
        self.enable_cameras = bool(gp("enable_cameras").value)
        self.camera_names = [str(c) for c in gp("camera_names").value if str(c)]
        frames = [str(c) for c in gp("camera_frames").value if str(c)]
        infos = [str(c) for c in gp("camera_info_topics").value if str(c)]
        self.max_range_m = float(gp("max_range_m").value)
        self.boundary_samples = int(gp("boundary_samples").value)
        self.enable_lidar = bool(gp("enable_lidar").value)
        self.map_topic = str(gp("map_topic").value)
        self.lidar_rays = int(gp("lidar_rays").value)
        self.lidar_range_m = float(gp("lidar_range_m").value)
        self.occupied_min = int(gp("occupied_min").value)
        self.tf_timeout_sec = float(gp("tf_timeout_sec").value)
        self.warn_period_s = float(gp("warn_period_s").value)

        self.grid_info = {"resolution": self.res, "origin_x": self.ox,
                          "origin_y": self.oy, "rows": self.rows,
                          "cols": self.cols}

        if frames and len(frames) != len(self.camera_names):
            self.get_logger().warning(
                "camera_frames has a different length than camera_names -- "
                "ignoring it and deriving <name>_optical_frame")
            frames = []
        if infos and len(infos) != len(self.camera_names):
            self.get_logger().warning(
                "camera_info_topics has a different length than camera_names "
                "-- ignoring it and deriving /<name>/camera_info")
            infos = []
        self.cam_frame = {
            name: (frames[i] if frames else f"{name}_optical_frame")
            for i, name in enumerate(self.camera_names)}
        self.cam_info_topic = {
            name: (infos[i] if infos else f"/{name}/camera_info")
            for i, name in enumerate(self.camera_names)}

        # name -> CameraInfo (first one wins: these are static cameras) and
        # name -> the cached rasterised footprint. A camera contributes
        # nothing until BOTH its info and its TF have been seen once.
        self._cam_info: Dict[str, CameraInfo] = {}
        self._cam_mask: Dict[str, np.ndarray] = {}
        self._cam_warned: Dict[str, bool] = {name: False
                                             for name in self.camera_names}

        self._map_blocked: Optional[np.ndarray] = None
        self._map_info: Optional[Dict] = None
        self._robot_xy: Optional[Tuple[float, float]] = None
        self._warned_map = False
        self._warned_tf = False
        self._logged_ready = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cov_pub = self.create_publisher(
            OccupancyGrid, self.coverage_topic, latched)
        self.count_pub = self.create_publisher(
            OccupancyGrid, self.count_topic, latched)

        if self.enable_cameras:
            for name in self.camera_names:
                self.create_subscription(
                    CameraInfo, self.cam_info_topic[name],
                    lambda msg, n=name: self._info_cb(n, msg), 1)
        if self.enable_lidar:
            self.create_subscription(
                OccupancyGrid, self.map_topic, self._map_cb, latched)

        period = 1.0 / self.publish_rate if self.publish_rate > 0.0 else 0.5
        self.create_timer(period, self._publish)

        lidar_desc = (f"{self.lidar_rays} rays @ {self.lidar_range_m:.1f} m"
                      if self.enable_lidar else "off")
        cam_desc = self.camera_names if self.enable_cameras else "off"
        self.get_logger().info(
            f"coverage_mask {self.rows}x{self.cols} @ {self.res} m/cell, "
            f"origin ({self.ox}, {self.oy}) -> {self.coverage_topic}; "
            f"cameras {cam_desc} (max_range {self.max_range_m:.1f} m), "
            f"lidar {lidar_desc}")

    # ------------------------------------------------------------ inputs

    def _info_cb(self, name: str, msg: CameraInfo) -> None:
        if name not in self._cam_info:
            self._cam_info[name] = msg

    def _map_cb(self, msg: OccupancyGrid) -> None:
        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        # Occupied OR unknown stops a ray -- see the module docstring.
        self._map_blocked = (grid >= self.occupied_min) | (grid < 0)
        self._map_info = {
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
        }

    def _update_robot_pose(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except TransformException as exc:
            if not self._warned_tf:
                self._warned_tf = True
                self.get_logger().warning(
                    f"no {self.map_frame} -> {self.base_frame} tf ({exc}); "
                    "the lidar line-of-sight source contributes nothing "
                    "until it appears")
            return
        self._warned_tf = False
        self._robot_xy = (float(tf.transform.translation.x),
                          float(tf.transform.translation.y))

    # ----------------------------------------------------------- sources

    def _camera_mask(self, name: str) -> Optional[np.ndarray]:
        """One camera's cached floor footprint, computing it the first tick
        both its CameraInfo and its static TF are available. Cached because
        neither input ever changes: the cameras are bolted to the ceiling
        and their TF is /tf_static ground truth off the USD prim."""
        cached = self._cam_mask.get(name)
        if cached is not None:
            return cached
        info = self._cam_info.get(name)
        if info is None:
            self._warn_camera_once(name, "no CameraInfo on "
                                   f"{self.cam_info_topic[name]}")
            return None
        frame = self.cam_frame[name]
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, frame, Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except TransformException as exc:
            self._warn_camera_once(
                name, f"no {self.map_frame} -> {frame} tf ({exc})")
            return None

        rot = quaternion_matrix(tf.transform.rotation.x, tf.transform.rotation.y,
                                tf.transform.rotation.z, tf.transform.rotation.w)
        trans = (tf.transform.translation.x, tf.transform.translation.y,
                 tf.transform.translation.z)
        poly = camera_floor_polygon(info.k, int(info.width), int(info.height),
                                    rot, trans, self.max_range_m,
                                    self.boundary_samples)
        if poly is None:
            self._warn_camera_once(
                name, "fewer than three image-border rays reach the floor "
                      f"within {self.max_range_m:.1f} m (extrinsic wrong, or "
                      "the camera looks at the horizon)")
            return None
        mask = rasterise_polygon(poly, self.grid_info)
        self._cam_mask[name] = mask
        self.get_logger().info(
            f"{name}: floor footprint {int(mask.sum())} cells "
            f"({100.0 * mask.mean():.1f} % of the grid) from a "
            f"{int(info.width)}x{int(info.height)} view at "
            f"({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})")
        return mask

    def _warn_camera_once(self, name: str, why: str) -> None:
        """Exactly once per camera, then silence -- a camera that never
        comes up must not fill the log at publish_rate, and 'not covering'
        is a legitimate steady state, not an error to keep shouting about."""
        if self._cam_warned.get(name):
            return
        self._cam_warned[name] = True
        self.get_logger().warning(
            f"camera {name} contributes NO coverage: {why}. Its part of the "
            "floor will read as unobserved, which makes the supervisor's "
            "crossing policy more conservative there, not less.")

    def _lidar_mask(self) -> Optional[np.ndarray]:
        if self._map_blocked is None or self._map_info is None:
            if not self._warned_map:
                self._warned_map = True
                self.get_logger().warning(
                    f"nothing on {self.map_topic} yet -- the lidar "
                    "line-of-sight source contributes nothing until the "
                    "static map arrives")
            return None
        if self._robot_xy is None:
            return None
        return lidar_los_mask(self._map_blocked, self._map_info,
                              self._robot_xy, self.grid_info,
                              n_rays=self.lidar_rays,
                              max_range_m=self.lidar_range_m)

    # ------------------------------------------------------------ output

    def _grid_msg(self, values: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        info = MapMetaData()
        info.resolution = self.res
        info.width = self.cols
        info.height = self.rows
        info.origin.position.x = self.ox
        info.origin.position.y = self.oy
        info.origin.orientation.w = 1.0
        msg.info = info
        msg.data = np.clip(values, 0, 100).astype(np.int8).flatten(
            order="C").tolist()
        return msg

    def _publish(self) -> None:
        self._update_robot_pose()

        masks: List[np.ndarray] = []
        if self.enable_cameras:
            for name in self.camera_names:
                mask = self._camera_mask(name)
                if mask is not None:
                    masks.append(mask)
        if self.enable_lidar:
            mask = self._lidar_mask()
            if mask is not None:
                masks.append(mask)

        any_mask, count = combine_sources(masks, self.rows, self.cols)
        self.cov_pub.publish(self._grid_msg(any_mask.astype(np.int16) * 100))
        self.count_pub.publish(self._grid_msg(
            np.minimum(count.astype(np.int16) * 25, 100)))

        if masks and not self._logged_ready:
            self._logged_ready = True
            self.get_logger().info(
                f"coverage live: {len(masks)} source(s), "
                f"{100.0 * float(any_mask.mean()):.1f} % of the grid covered")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageMaskNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
