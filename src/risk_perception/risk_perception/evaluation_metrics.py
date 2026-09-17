#!/usr/bin/env python3
"""
evaluation_metrics.py  --  pure computation for the four evaluation
                            categories: Efficiency, Safety, Latency, Clearance

No ROS, no rosbag, no Node -- every function here takes plain numpy-friendly
arrays (already extracted from a bag or a live run) and returns numbers or
small dicts. Same rationale as relation_matching.py / encounter_geometry.py /
mask_relation.py / spatial_prior_node.deposit_into_grid: keep the actual
computation unit-testable with plain pytest, and put ONLY the ROS/rosbag
plumbing (tools/evaluate_run.py) in a thin, untested-by-necessity wrapper
around it.

-------------------------------------------------------------------------
Risk exposure -- the one metric worth deriving carefully (see
tools/evaluate_run.py's module docstring for the full "why"):

    E       = sum_i risk(x_r(t_i), y_r(t_i)) * dt_i          (total)
    E_rate  = E / sum_i dt_i                                  (time-normalized)

sampled from a risk field that is IDENTICAL across every condition being
compared (typically the full-system reference costmap -- see
launch/evaluation_reference.launch.py), NOT from whichever costmap actually
drove Nav2 during that run. Scoring a run on the costmap it was itself
planning against makes an ablated ("weaker") condition self-report lower
exposure merely because it computed less risk, not because the robot was
actually safer -- see risk_score_reference_note below.

Clearance is reported per ENCOUNTER (one minimum per contiguous close-range
stretch), not per timestep, so a single slow pass close to an object does
not contribute dozens of highly-correlated samples to the distribution.

-------------------------------------------------------------------------
Raw (unclipped) severity -- risk_exposure() above, when fed samples read
back from a published OccupancyGrid (nav_msgs/OccupancyGrid.data is int8,
0-100), inherits that grid's saturation: predictive_risk_costmap_node's own
severity C = consequence * encounter_factor + relbonus is DELIBERATELY
unbounded above 1.0 (a collision-course track should widen its keep-out
region, not just paint a redder single peak -- see that node's docstring),
so a near-miss and a near-collision can both saturate to grid value 100 and
become indistinguishable once read back through the grid.

reference_severity_at_point() recomputes that SAME node's per-track math
directly from world_objects (continuous-valued, never clipped) instead of
sampling the rasterized, quantized, clipped grid -- reusing
encounter_geometry.cpa_geometry and risk_visualization.
risk_score_from_label, the exact functions predictive_risk_costmap_node
itself calls, so this is a faithful point-sample of that node's field, not a
second implementation that can silently drift out of sync with it. The one
deliberate divergence is the severity combination itself: the live node's
encounter_geometry.combine_severity() clips to 1.0 (WP-A, 2026-09-11) so the
SRM never sees a "more than fully occupied" cell, while this scorer keeps
the raw sum -- see reference_severity_at_point().
-------------------------------------------------------------------------
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from risk_perception.encounter_geometry import cpa_geometry
from risk_perception.risk_visualization import risk_score_from_label

risk_score_reference_note = (
    "Risk exposure must be scored against a FIXED reference risk field "
    "(the full-system costmap, all priors and CPA/TTC on) for every "
    "condition being compared in an ablation -- not against whichever "
    "costmap actually drove Nav2 that run. See evaluate_run.py.")


# ------------------------------------------------------------------ efficiency

def path_length(xy: Sequence[Sequence[float]]) -> float:
    """Total arc length of a (N, 2) polyline of robot positions."""
    arr = np.asarray(xy, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def time_to_goal(start_stamp_s: float, goal_reached_stamp_s: float) -> float:
    return float(goal_reached_stamp_s - start_stamp_s)


def stop_events(speed_mps: Sequence[float], times_s: Sequence[float],
                stop_speed_mps: float = 0.05,
                min_stop_duration_s: float = 0.5) -> Tuple[int, float]:
    """Count discrete stop events -- contiguous stretches with speed under
    stop_speed_mps lasting at least min_stop_duration_s (filters out a
    single low-speed sample from counting as a "stop"). Returns
    (count, total_stopped_time_s)."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if len(speed) == 0:
        return 0, 0.0
    below = speed < stop_speed_mps
    count = 0
    total_stopped = 0.0
    i, n = 0, len(speed)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            end_idx = min(j, n - 1)
            duration = float(times[end_idx] - times[i])
            if duration >= min_stop_duration_s:
                count += 1
                total_stopped += duration
            i = j
        else:
            i += 1
    return count, total_stopped


def velocity_smoothness(speed_mps: Sequence[float], times_s: Sequence[float]) -> float:
    """Std. dev. of jerk (d(accel)/dt) as a smoothness proxy -- lower is
    smoother. Complements stop_events, which is a coarse threshold count;
    this is continuous and does not depend on picking a stop-speed cutoff."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if len(speed) < 3:
        return 0.0
    dt = np.diff(times)
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.diff(speed) / dt
        jerk = np.diff(accel) / dt[:-1]
    jerk = jerk[np.isfinite(jerk)]
    return float(np.std(jerk)) if len(jerk) else 0.0


def replan_count(replan_event_stamps_s: Sequence[float]) -> int:
    """Count GLOBAL replan events only -- a local/DWB-style controller
    effectively "replans" every control cycle, which would make this
    metric trivially large and uninformative if counted. Callers must
    extract global-planner replan events specifically (e.g. Nav2's
    ComputePathToPose being re-invoked), not local trajectory updates."""
    return len(replan_event_stamps_s)


# ------------------------------------------------------------------ safety / risk exposure

def risk_exposure(risk_samples: Sequence[float], dt_s: Sequence[float]
                  ) -> Tuple[float, float]:
    """(total, rate). risk_samples: the reference risk field's value at the
    robot's actual position at each sample; dt_s: elapsed time since the
    previous sample. A Riemann sum, not a plain sum -- robust to uneven
    sampling intervals (a dropped message, a bag with variable rate)."""
    risk = np.asarray(risk_samples, dtype=np.float64)
    dt = np.asarray(dt_s, dtype=np.float64)
    if risk.shape != dt.shape:
        raise ValueError("risk_samples and dt_s must be the same length")
    total_time = float(dt.sum())
    total = float(np.sum(risk * dt))
    rate = total / total_time if total_time > 0 else 0.0
    return total, rate


def risk_exposure_per_distance(total_exposure: float, path_length_m: float) -> float:
    """Exposure normalized by distance traveled, complementing the
    time-normalized rate above -- useful when comparing policies that take
    different PATHS (not just different speeds) to the same goal."""
    return total_exposure / path_length_m if path_length_m > 0 else 0.0


def parse_class_id(class_id: str) -> Tuple[str, Dict[str, float]]:
    """'label|pmov=..|pmot=..|vx=..|vy=..|relbonus=..' -> (label, dict).
    Same contract as object_tracker_node's Track output / every costmap
    node's own parse_class_id -- kept here (not imported from a node file)
    so this module stays import-clean of anything ROS."""
    parts = str(class_id).split("|")
    label = parts[0] if parts else "object"
    kv: Dict[str, float] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                kv[k] = float(v)
            except ValueError:
                pass
    return label, kv


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Same formula as predictive_risk_costmap_node.yaw_from_quat, taking
    plain floats instead of a geometry_msgs/Quaternion so this stays a pure
    function -- callers extract .x/.y/.z/.w themselves."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _gaussian_peak(dx: float, dy: float, var_x: float, var_y: float,
                   peak: float) -> float:
    """Value of an axis-aligned 2D Gaussian with the given peak amplitude,
    sampled at offset (dx, dy) from its mean -- same shape as
    predictive_risk_costmap_node._splat's rasterized Gaussian, evaluated at
    ONE point instead of over a whole grid."""
    if peak <= 0.0:
        return 0.0
    return peak * math.exp(-0.5 * (dx * dx / var_x + dy * dy / var_y))


def reference_severity_at_point(
    tracks: Sequence[Dict[str, float]],
    robot_x: float, robot_y: float, robot_vx: float, robot_vy: float,
    horizon_steps: int = 20, pred_dt: float = 0.3, gamma: float = 0.9,
    vel_inflation: float = 0.05, min_sigma_m: float = 0.25,
    cpa_gain: float = 0.8, cpa_scale_m: float = 0.6, ttc_scale_s: float = 3.0,
    min_rel_speed: float = 0.05,
    use_class_consequence: bool = True, enable_relative_motion: bool = True,
    use_motion_mixture: bool = True,
) -> float:
    """The UNCLIPPED severity predictive_risk_costmap_node's reference field
    would have at (robot_x, robot_y) right now -- see this module's
    docstring for why sampling this instead of the published grid matters
    for exposure scoring. Defaults match risk_perception.yaml's
    predictive_risk_costmap_node block AND evaluation_reference.launch.py's
    forced full-system settings; pass the ablation-under-test's own values
    here instead if you want the RAW-severity metric to track a specific
    condition rather than the fixed full-system reference.

    tracks: dicts with x, y, label, score, and the same keys
    parse_class_id() returns (pmot, vx, vy, relbonus) plus Pxx, Pyy, Pvx,
    Pvy from that detection's pose.covariance[0, 7, 21, 28] -- i.e. exactly
    what a world_objects Detection3D carries, already unpacked.
    """
    min_var = min_sigma_m ** 2
    peak = 0.0
    for tr in tracks:
        consequence = (
            risk_score_from_label(tr["label"], max(0.0, min(1.0, tr["score"])))
            if use_class_consequence else 1.0)
        if enable_relative_motion:
            factor, _t_cpa, _d_cpa = cpa_geometry(
                tr["x"], tr["y"], tr.get("vx", 0.0), tr.get("vy", 0.0),
                robot_x, robot_y, robot_vx, robot_vy,
                cpa_gain, cpa_scale_m, ttc_scale_s, min_rel_speed)
        else:
            factor = 1.0
        # Deliberately NOT encounter_geometry.combine_severity(): that
        # function gained a min(1.0, ...) clip on user-b/sandbox (WP-A,
        # 2026-09-11) so the live field never hands the SRM a "more than
        # fully occupied" cell. This scorer wants the RAW value -- that is
        # its entire reason to exist (see this module's "Raw (unclipped)
        # severity" section and test_reference_severity_is_not_clamped_to_
        # one): once clipped, a near-miss and a near-collision become
        # indistinguishable again, which is exactly the grid saturation
        # this function was written to get behind. Same arithmetic as
        # combine_severity, minus the clip -- keep the two in step if the
        # relation_bonus term's placement ever changes there.
        severity = consequence * factor + tr.get("relbonus", 0.0)

        pmot = tr.get("pmot", 0.0) if use_motion_mixture else 0.0
        Pxx = max(tr.get("Pxx", min_var), min_var)
        Pyy = max(tr.get("Pyy", min_var), min_var)
        Pvx, Pvy = tr.get("Pvx", 0.0), tr.get("Pvy", 0.0)

        # stationary hypothesis
        w_stat = (1.0 - pmot) * severity
        peak = max(peak, _gaussian_peak(
            robot_x - tr["x"], robot_y - tr["y"], Pxx, Pyy, w_stat))

        # moving hypothesis, rolled out over the same horizon the live node uses
        vx, vy = tr.get("vx", 0.0), tr.get("vy", 0.0)
        for step in range(horizon_steps):
            t = (step + 1) * pred_dt
            w_move = pmot * (gamma ** step) * severity
            if w_move <= 0.001:
                break
            mx, my = tr["x"] + vx * t, tr["y"] + vy * t
            var_x = Pxx + (t * t) * Pvx + vel_inflation * t
            var_y = Pyy + (t * t) * Pvy + vel_inflation * t
            peak = max(peak, _gaussian_peak(
                robot_x - mx, robot_y - my, var_x, var_y, w_move))
    return peak


# ------------------------------------------------------------------ clearance

def clearance_encounters(distance_to_nearest_m: Sequence[float],
                         times_s: Sequence[float],
                         engagement_threshold_m: float = 2.0) -> List[float]:
    """Collapse a time series of "distance to nearest dynamic object" into
    one MINIMUM value per encounter (contiguous stretch under
    engagement_threshold_m). Reporting per-encounter minima rather than
    every timestep avoids one slow close pass contributing dozens of
    highly-correlated samples to the resulting distribution."""
    distance = np.asarray(distance_to_nearest_m, dtype=np.float64)
    if len(distance) == 0:
        return []
    engaged = distance < engagement_threshold_m
    minima: List[float] = []
    i, n = 0, len(distance)
    while i < n:
        if engaged[i]:
            j = i
            local_min = distance[i]
            while j < n and engaged[j]:
                local_min = min(local_min, distance[j])
                j += 1
            minima.append(float(local_min))
            i = j
        else:
            i += 1
    return minima


def clearance_distribution_stats(encounter_minima_m: Sequence[float]) -> Dict[str, float]:
    """Distribution summary over per-ENCOUNTER minima (see
    clearance_encounters) -- percentiles, not just mean/median, since a
    safety claim rests on the tail, not the average case."""
    if not encounter_minima_m:
        return {"n_encounters": 0}
    arr = np.asarray(encounter_minima_m, dtype=np.float64)
    return {
        "n_encounters": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
    }


# ------------------------------------------------------------------ latency / compute cost

def detection_to_costmap_latency(detection_stamps_s: Sequence[float],
                                 costmap_stamps_s: Sequence[float]) -> List[float]:
    """For each costmap message, elapsed time since the most recent
    detections message it could plausibly reflect (nearest EARLIER
    detection stamp) -- a perception-to-costmap latency proxy computed
    purely from message header timestamps already in any recorded bag, no
    node instrumentation required."""
    det = np.asarray(sorted(detection_stamps_s), dtype=np.float64)
    if len(det) == 0:
        return []
    latencies = []
    for cs in sorted(costmap_stamps_s):
        idx = int(np.searchsorted(det, cs, side="right")) - 1
        if idx >= 0:
            latencies.append(float(cs - det[idx]))
    return latencies


def realtime_factor(costmap_stamps_s: Sequence[float],
                    configured_period_s: float) -> List[float]:
    """Ratio of ACTUAL inter-publish interval to the node's CONFIGURED tick
    period (1/publish_rate). > 1.0 means the node fell behind its target
    rate for that interval -- a "does it keep up" indicator that needs no
    wall-clock instrumentation inside the node, only message timestamps."""
    stamps = np.asarray(sorted(costmap_stamps_s), dtype=np.float64)
    if len(stamps) < 2 or configured_period_s <= 0:
        return []
    intervals = np.diff(stamps)
    return (intervals / configured_period_s).tolist()


# ------------------------------------------------------------------ report container

@dataclass
class RunReport:
    """One evaluation run's full metric set -- what tools/evaluate_run.py
    fills in and prints/serializes. Fields left at their default are simply
    unavailable for that run (e.g. no replan topic recorded)."""
    label: str = ""
    # efficiency
    time_to_goal_s: float = 0.0
    path_length_m: float = 0.0
    stop_count: int = 0
    stopped_time_s: float = 0.0
    velocity_smoothness: float = 0.0
    replan_count: int = 0
    # safety
    risk_exposure_total: float = 0.0
    risk_exposure_rate: float = 0.0
    risk_exposure_per_m: float = 0.0
    # Unclipped counterpart of the three above -- see this module's
    # "Raw (unclipped) severity" docstring section. Left at 0.0 (not None)
    # when unavailable, same convention as every other numeric field here,
    # but a run that never called reference_severity_at_point() should be
    # read as "not computed," not "zero risk" -- see evaluate_run.py.
    risk_exposure_raw_total: float = 0.0
    risk_exposure_raw_rate: float = 0.0
    risk_exposure_raw_per_m: float = 0.0
    # clearance
    clearance: Dict[str, float] = field(default_factory=dict)
    # latency / compute
    detection_to_costmap_latency_s: List[float] = field(default_factory=list)
    realtime_factor: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Flat dict for a CSV row / table column -- lists collapsed to
        mean + p95 rather than dumped raw."""
        lat = np.asarray(self.detection_to_costmap_latency_s, dtype=np.float64)
        rtf = np.asarray(self.realtime_factor, dtype=np.float64)
        return {
            "label": self.label,
            "time_to_goal_s": self.time_to_goal_s,
            "path_length_m": self.path_length_m,
            "stop_count": self.stop_count,
            "stopped_time_s": self.stopped_time_s,
            "velocity_smoothness": self.velocity_smoothness,
            "replan_count": self.replan_count,
            "risk_exposure_total": self.risk_exposure_total,
            "risk_exposure_rate": self.risk_exposure_rate,
            "risk_exposure_per_m": self.risk_exposure_per_m,
            "risk_exposure_raw_total": self.risk_exposure_raw_total,
            "risk_exposure_raw_rate": self.risk_exposure_raw_rate,
            "risk_exposure_raw_per_m": self.risk_exposure_raw_per_m,
            "clearance_n_encounters": self.clearance.get("n_encounters", 0),
            "clearance_p5_m": self.clearance.get("p5", float("nan")),
            "clearance_median_m": self.clearance.get("median", float("nan")),
            "clearance_min_m": self.clearance.get("min", float("nan")),
            "latency_mean_s": float(lat.mean()) if lat.size else float("nan"),
            "latency_p95_s": float(np.percentile(lat, 95)) if lat.size else float("nan"),
            "realtime_factor_mean": float(rtf.mean()) if rtf.size else float("nan"),
            "realtime_factor_max": float(rtf.max()) if rtf.size else float("nan"),
        }
