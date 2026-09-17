#!/usr/bin/env python3
"""
motion_evidence.py -- WP1 pure helpers for the tracker's displacement-based
motion evidence (2026-09-10 "phantom velocity" fix).

`object_tracker_node.Track`'s Kalman velocity state is the finite difference
of jittery overhead-camera measurements (~0.3 m inter-camera ground-plane
offset); on a genuinely parked object that jitter alone reads as
sigma_v ~ 0.5-0.9 m/s, so a purely Mahalanobis/speed-threshold motion test
(the pre-2026-09-10 `Track.update_beliefs`) latches p_motion=1 on noise. The
fix is the same idea `panoptex_nav/gt_tracks_node.py` already uses for its
oracle Carter tracks: does the object's OWN measurement history show real
net displacement over a window, not just an instantaneous velocity
estimate? `displacement_velocity`/`ema_update` below are the same ~10-line
helpers duplicated from `gt_tracks_node.py` (risk_perception must not import
panoptex_nav -- the two packages have no dependency edge between them, and
adding one just to share ten lines is not worth it); `displacement_evidence`
is new, straightness-gated maths that `gt_tracks_node.py` doesn't need
(an oracle has no camera zig-zag to reject).

Pure functions only -- no ROS, no rclpy -- so `tools/test_retrack_bag.py`
and `test/test_object_tracker_displacement.py` can unit-test the maths on
synthetic measurement sequences without a ROS graph.
"""

import math
import statistics
from typing import List, Optional, Sequence, Tuple


def displacement_velocity(
    times: List[float], xs: List[float], ys: List[float],
) -> Tuple[float, float]:
    """(vx, vy) from the straight-line displacement between the OLDEST and
    NEWEST of >= 2 samples, divided by the elapsed time between them -- a
    window-baseline estimate, not a per-sample finite difference, so camera
    jitter doesn't dominate. Returns (0.0, 0.0) if fewer than 2 samples, or
    the elapsed time is non-positive (a degenerate or out-of-order window).
    Identical to `panoptex_nav.gt_tracks_node.displacement_velocity`."""
    if len(times) < 2:
        return 0.0, 0.0
    dt = times[-1] - times[0]
    if dt <= 0.0:
        return 0.0, 0.0
    return (xs[-1] - xs[0]) / dt, (ys[-1] - ys[0]) / dt


def ema_update(
    prev: Optional[Tuple[float, float]],
    raw: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """First-order EMA smoothing of a raw (vx, vy) sample. `prev is None`
    seeds directly to `raw` (nothing to smooth against yet). `alpha` is
    clamped to [0, 1]; 1.0 means "no smoothing, snap straight to raw".
    Identical to `panoptex_nav.gt_tracks_node.ema_update`."""
    if prev is None:
        return raw
    a = max(0.0, min(1.0, alpha))
    return (prev[0] + a * (raw[0] - prev[0]),
            prev[1] + a * (raw[1] - prev[1]))


def displacement_evidence(
    meas: Sequence[Tuple[float, float, float, float]],
    window_s: float,
    min_disp_m: float,
    k_sigma: float,
    min_samples: int,
    straightness_min: float,
    method: str = "half_median",
) -> Tuple[float, float, float, bool]:
    """Is this track's own measurement history consistent with real motion?

    `meas` is a sequence of `(stamp, x, y, cov)` raw measurement tuples,
    oldest first (not required to already be window-pruned -- this function
    re-filters to the newest `window_s` seconds itself, so a caller that
    seeds a Track's `meas` deque directly, as the WP5 displacement tests
    do, doesn't have to replicate `Track.update`'s own pruning).

    `method` selects the statistic:

    - "half_median" (default, 2026-09-10 follow-up finding): a track fused
      from TWO measurement sources with different systematic ground-plane
      offsets (a 10 Hz lidar-cluster stream and a slower camera stream,
      each individually precise but persistently ~0.3 m apart from the
      other) reads as a zigzag when treated as one endpoint-to-endpoint
      displacement -- see `_displacement_evidence_endpoints`'s docstring.
      This splits the window in half BY TIME (midpoint of [oldest,
      newest], not by sample count, so it doesn't care which source
      contributed how many points to each half) and compares the
      coordinate-wise MEDIAN position of each half -- a per-source offset
      that is roughly constant within each half cancels out of the
      median-to-median difference instead of alternating into inflated path
      length. See `_displacement_evidence_half_median`.
    - "endpoints" (the original WP1 design; kept as an explicit fallback,
      `motion_disp_method: endpoints`, not just dead code): newest-vs-
      oldest displacement gated by a straightness (path-length) ratio.

    Returns (net_disp_m, disp_vx, disp_vy, moving). `meas` empty -> all
    zero / not moving."""
    if not meas:
        return 0.0, 0.0, 0.0, False
    newest_t = meas[-1][0]
    # Same 1e-6 s boundary tolerance as Track.update()'s own pruning -- a
    # sample landing exactly on the window edge must not flicker in/out
    # from float accumulation error alone.
    window = [m for m in meas if (newest_t - m[0]) <= window_s + 1e-6]
    n = len(window)
    if n == 0:
        return 0.0, 0.0, 0.0, False

    if method == "endpoints":
        return _displacement_evidence_endpoints(
            window, min_disp_m, k_sigma, min_samples, straightness_min)
    return _displacement_evidence_half_median(window, min_disp_m, k_sigma, min_samples)


def _displacement_evidence_endpoints(
    window: Sequence[Tuple[float, float, float, float]],
    min_disp_m: float, k_sigma: float, min_samples: int, straightness_min: float,
) -> Tuple[float, float, float, bool]:
    """The original WP1 statistic (2026-09-10 morning): net_disp_m =
    |newest - oldest| straight-line displacement over the window.
    thr = max(min_disp_m, k_sigma * sqrt(cov_oldest + cov_newest)) -- the
    noise floor a genuine displacement has to clear, scaled by how
    uncertain the two endpoint measurements were.

    straightness = net_disp_m / sum(|z_i - z_(i-1)|) -- path length, not
    just endpoint distance: a real SINGLE-source mover's path is close to
    a straight line over a couple of seconds, while overhead-camera
    hand-over jitter zigs and zags without going anywhere, so its path
    length grows much faster than its net displacement (straightness <<
    1). At <= 3 samples there are too few path segments for this ratio to
    be meaningful, so straightness is treated as 1.0 (never blocks a
    sparsely-observed track) -- likewise when the path length itself is
    ~0 (a single repeated point).

    2026-09-10 follow-up finding (see `displacement_evidence`'s own
    docstring): this statistic reads a genuinely moving, TWO-source-fused
    track as near-zero straightness, because a persistent (not jittery)
    inter-source offset alternates into the path sum every time
    association hands the window a point from the other source. Kept
    here, selectable via `motion_disp_method: endpoints`, as the pre-fix
    behaviour / an explicit fallback -- not the default.

    moving = (n >= min_samples) AND (net_disp_m >= thr) AND
             (straightness >= straightness_min).

    disp_vx, disp_vy = (newest - oldest) / dt (dt <= 0 -> (0.0, 0.0))."""
    n = len(window)
    t0, x0, y0, cov0 = window[0]
    t1, x1, y1, cov1 = window[-1]
    net_disp_m = math.hypot(x1 - x0, y1 - y0)
    dt = t1 - t0
    if dt > 0.0:
        disp_vx, disp_vy = (x1 - x0) / dt, (y1 - y0) / dt
    else:
        disp_vx, disp_vy = 0.0, 0.0

    if n <= 3:
        straightness = 1.0
    else:
        path = 0.0
        for i in range(1, n):
            path += math.hypot(
                window[i][1] - window[i - 1][1], window[i][2] - window[i - 1][2])
        straightness = 1.0 if path <= 1e-9 else min(1.0, net_disp_m / path)

    thr = max(min_disp_m, k_sigma * math.sqrt(max(0.0, cov0) + max(0.0, cov1)))
    moving = (n >= min_samples) and (net_disp_m >= thr) and (straightness >= straightness_min)
    return net_disp_m, disp_vx, disp_vy, moving


def _displacement_evidence_half_median(
    window: Sequence[Tuple[float, float, float, float]],
    min_disp_m: float, k_sigma: float, min_samples: int,
) -> Tuple[float, float, float, bool]:
    """2026-09-10 follow-up fix for `_displacement_evidence_endpoints`'s
    two-source zigzag failure (see `displacement_evidence`'s docstring).

    Split `window` into two halves BY TIME at the midpoint of
    [oldest_stamp, newest_stamp] (not by sample count -- a 10 Hz lidar
    stream and a ~1-2 Hz camera stream contribute very different sample
    counts to any fixed time span, and splitting by count would put most
    of one source entirely in one half). half1 = stamp <= mid, half2 =
    stamp > mid.

    Each half's POSITION is the coordinate-wise MEDIAN of its own
    measurements -- not an endpoint, not a mean (a single mid-half
    outlier from either source can't drag it) -- so a persistent, roughly
    constant inter-source offset within a half cancels out of the
    median-to-median comparison instead of alternating into an inflated
    "path length" the way `_displacement_evidence_endpoints`'s
    endpoint-to-endpoint statistic does. net = |median2 - median1|.
    dt = mean(stamp in half2) - mean(stamp in half1); disp_v =
    (median2 - median1) / dt. cov_i = median measurement covariance of
    half i; thr = max(min_disp_m, k_sigma * sqrt(cov1 + cov2)), same
    noise-floor idea as the endpoints statistic. No straightness term --
    two medians have no "path" to zigzag along.

    Requires EACH half to have >= 1 measurement and the total sample
    count >= min_samples; either failing reads as not moving (net_disp_m
    is still reported as 0.0 in that case -- there is no valid two-point
    comparison to report)."""
    n = len(window)
    t_oldest = window[0][0]
    t_newest = window[-1][0]
    mid = 0.5 * (t_oldest + t_newest)
    half1 = [m for m in window if m[0] <= mid]
    half2 = [m for m in window if m[0] > mid]
    if not half1 or not half2 or n < min_samples:
        return 0.0, 0.0, 0.0, False

    x1 = statistics.median(m[1] for m in half1)
    y1 = statistics.median(m[2] for m in half1)
    x2 = statistics.median(m[1] for m in half2)
    y2 = statistics.median(m[2] for m in half2)
    t1_mean = sum(m[0] for m in half1) / len(half1)
    t2_mean = sum(m[0] for m in half2) / len(half2)
    dt = t2_mean - t1_mean

    net_disp_m = math.hypot(x2 - x1, y2 - y1)
    if dt > 0.0:
        disp_vx, disp_vy = (x2 - x1) / dt, (y2 - y1) / dt
    else:
        disp_vx, disp_vy = 0.0, 0.0

    cov1 = statistics.median(m[3] for m in half1)
    cov2 = statistics.median(m[3] for m in half2)
    thr = max(min_disp_m, k_sigma * math.sqrt(max(0.0, cov1) + max(0.0, cov2)))
    moving = net_disp_m >= thr
    return net_disp_m, disp_vx, disp_vy, moving
