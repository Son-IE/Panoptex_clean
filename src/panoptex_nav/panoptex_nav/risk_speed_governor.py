#!/usr/bin/env python3
"""
risk_speed_governor.py  --  WP3.2: Panoptex's semantic priors as a hard speed
cap for Nav2, with ZERO Nav2 modification.

Nav2's `controller_server` already subscribes `nav2_msgs/SpeedLimit` on
whatever topic `speed_limit_topic` names (we point it at `speed_limit`), and
DWB honours it via `setSpeedLimit()` -- see
`nav2_controller/controller_server.cpp`. This node is the ONLY thing that
publishes on that topic: it turns Panoptex's map-frame world model
(`/risk_perception/world_objects`, the same tracker output
`predictive_risk_costmap_node.py` consumes) into a single percentage cap,
independent of and complementary to the `PredictedRisk` DWB critic in this
same package -- the critic reshapes the *trajectory scoring*, this node
puts a hard ceiling on *how fast any trajectory is allowed to go*.

Semantics of `nav2_msgs/SpeedLimit` (read the .msg before touching this
file): `percentage: true` means `speed_limit` is 0-100, a percent of the
robot's configured max speed. **`speed_limit == 0.0` means "no limit" --
it is never used to mean "stop"** (see `SpeedLimit.msg`: "When no-limit it
is set to 0.0"). A hard stop is out of scope here; that is costmap/lidar
territory (`nav2_costmap_2d` obstacle layers, `Oscillation`/`BaseObstacle`
critics), not a percentage governor.

Policy (all thresholds are ROS parameters -- see `DEFAULT_PARAMS` below and
`config/risk_speed_governor.yaml`):

  person            distance-only. People change direction far faster than
                    any CPA model can track, so this category never looks
                    at velocity or CPA -- only the plain distance to the
                    robot.
                        d <= person_crawl_radius_m  -> person_crawl_pct
                        d <= person_slow_radius_m   -> person_slow_pct

  any mover        (act_on_any_mover, WP-A, 2026-09-11, default True) --
  (except person)   CPA-gated exactly like the old robot/wheeled branch,
                    but the category test that used to gate entry into it
                    is gone: ANY track (except person, which keeps its own
                    distance-only branch above) whose p_motion clears
                    moving_pmot_min enters the SAME closing test. This is
                    the class-agnostic policy User B asked for -- a mover is
                    a mover, and a "table" that the tracker's motion-
                    overrides-class promotion has correctly identified as
                    moving (object_tracker_node.Track.update_promotion,
                    which relabels it category "wheeled" -- see
                    object_tracker_node.py) must not evade this cap just
                    because its RAW label was never "robot" or "wheeled".
                        p_motion >= moving_pmot_min AND
                        0 < t_cpa <= robot_ttc_s AND d_cpa <= robot_cpa_m
                            -> robot_closing_pct
                        p_motion <  moving_pmot_min AND
                        category in ("robot", "wheeled") AND
                        d <= static_obstacle_slow_radius_m
                            -> static_slow_pct

                    static_slow stays restricted to category robot/wheeled
                    (which already includes a promoted mover -- promotion
                    relabels the category itself) even though the moving
                    branch above no longer is: an ordinary STATIC chair or
                    table is Nav2's own lidar-based obstacle layer's job
                    (see furniture/unknown below), and duplicating that
                    cap for every piece of static furniture in view would
                    just fight the local planner for no safety benefit --
                    only a genuinely wheeled/robot-category track (or one
                    the tracker has promoted into that category) gets the
                    static-clutter cap.

                    act_on_any_mover: false restores the pre-2026-09-11
                    behaviour: only category in ("robot", "wheeled") ever
                    enters this branch at all (furniture/unknown are
                    dropped outright, same as below) -- a regression/
                    ablation knob, not the shipped default.

  furniture/unknown ignored entirely when NOT convincingly moving (and
                    always, under act_on_any_mover: false) -- Nav2's own
                    lidar-based costmap layers already handle static
                    clutter; duplicating that here would just fight the
                    local planner.

A track is dropped before any of the above if its detection score is below
`min_track_score` or its header stamp is older than `max_track_age_sec`
(measured against this node's own clock, so it behaves under
`use_sim_time`).

The final limit is the MINIMUM percentage over every track that currently
applies (most restrictive wins), floored at `min_pct` so the robot is never
capped to a crawl slower than that floor. If no track applies, the
published value is `0.0` -- "no limit", per the semantics above, NOT a
request to stop.

Hysteresis: a cap is applied the INSTANT some track satisfies it (no delay
going more restrictive), but is only RELEASED -- allowed to drop back to
"no limit" -- after `release_hold_s` seconds with no applicable track at
all. Without this, a track flickering in and out of its threshold radius
(sensor noise, a person pausing exactly at 2.0 m) would chatter the
robot's speed limit every control cycle.

Robot pose comes from TF (`map_frame` -> `base_frame`); robot body-twist
comes from `odom_topic` (`nav_msgs/Odometry`, in `base_frame`) and is
rotated into `map_frame` by the TF yaw before it is handed to
`cpa_geometry` -- copy of the pattern in
`risk_perception.predictive_risk_costmap_node.PredictiveRiskCostmapNode.
_update_robot_state`; see that method's docstring for why (CPA/TTC is a
relation between the robot's OWN motion and the object's, both must be
expressed in the same frame).

The policy core is the free function `compute_cap()` below -- no rclpy, no
Node, pure data in / data out, unit-tested directly in
`test/test_risk_speed_governor.py` with no ROS graph needed. The Node
class is a thin ROS shell around it: gather tracks + robot state, call it,
apply hysteresis, publish.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from tf2_ros import Buffer, TransformListener, TransformException

from risk_perception.encounter_geometry import cpa_geometry
from risk_perception.risk_visualization import parse_class_id, label_category


def yaw_from_quat(q) -> float:
    """Same formula as risk_perception.risk_visualization.yaw_from_quat --
    duplicated (not imported) to keep this package's runtime dependency on
    risk_perception limited to the two pure-geometry helpers it actually
    needs (cpa_geometry, parse_class_id, label_category)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# cpa_geometry()'s signature also takes cpa_gain / cpa_scale_m / ttc_scale_s,
# but those three ONLY shape its `factor` return value -- which this
# governor never reads (it reads t_cpa and d_cpa directly and applies its
# own robot_ttc_s / robot_cpa_m thresholds to them). Fixed placeholders
# below keep that fact obvious at the call site instead of wiring three
# unused ROS parameters through for no behavioural effect.
_CPA_GAIN_UNUSED = 1.0
_CPA_SCALE_UNUSED = 1.0
_TTC_SCALE_UNUSED = 1.0

# Categories this node acts on when act_on_any_mover is False (the
# pre-2026-09-11 behaviour); "furniture" / "unknown" fall through to Nav2's
# own costmap layers (see module docstring). Under the default
# act_on_any_mover: true this constant no longer gates entry to the
# mover branch at all -- only the static_slow sub-branch still consults it
# (see compute_cap).
_ACTIONABLE_CATEGORIES = ("person", "robot", "wheeled")

# Categories eligible for static_slow (compute_cap) regardless of
# act_on_any_mover -- an ordinary static chair/table is Nav2's own
# obstacle-layer's job; only a genuinely wheeled/robot-category track (or
# one the tracker has promoted into "wheeled" -- see module docstring)
# gets this cap.
_STATIC_SLOW_CATEGORIES = ("robot", "wheeled")

# One place for every policy default, shared by the Node's declare_parameter
# calls and by config/risk_speed_governor.yaml -- keeps the two from
# silently drifting apart.
DEFAULT_PARAMS: Dict[str, float] = {
    "min_track_score": 0.15,
    "max_track_age_sec": 3.0,

    "person_slow_radius_m": 2.0,
    "person_slow_pct": 40.0,
    "person_crawl_radius_m": 1.0,
    "person_crawl_pct": 15.0,

    "moving_pmot_min": 0.5,
    "robot_ttc_s": 3.0,
    "robot_cpa_m": 1.0,
    "robot_closing_pct": 30.0,
    "static_obstacle_slow_radius_m": 1.0,
    "static_slow_pct": 50.0,
    # WP-A, 2026-09-11 (default 1.0 = true; see DEFAULT_PARAMS/params are
    # Dict[str, float] throughout this module, same convention as
    # panoptex_nav.corridor.CORRIDOR_DEFAULTS): any category except person
    # enters the CPA-gated closing branch once p_motion clears
    # moving_pmot_min -- a mover is a mover, the class label never vetoes
    # it. 0.0 restores the pre-2026-09-11 behaviour (only robot/wheeled).
    # See module docstring and compute_cap().
    "act_on_any_mover": 1.0,
    # Below this relative speed, cpa_geometry treats geometry as noise and
    # returns t_cpa = inf -- which then always fails our `<= robot_ttc_s`
    # window, so this only needs to be small and positive (avoids a
    # near-zero-relative-speed division inside cpa_geometry), not itself a
    # tuned policy threshold the way robot_ttc_s / robot_cpa_m are.
    "min_rel_speed_mps": 0.02,

    "min_pct": 15.0,
}


def compute_cap(
    tracks: List[Dict],
    robot_xy: Tuple[float, float],
    robot_v_map: Tuple[float, float],
    params: Dict,
) -> Tuple[float, Optional[Dict]]:
    """Pure policy core -- no ROS, no Node, unit-testable directly.

    tracks: list of dicts, one per world-model track already past whatever
        upstream fusion/tracking happened, each with keys:
            id (str), label (str), category (str: person/robot/wheeled/
            furniture/unknown), score (float 0-1), age_sec (float, >=0),
            x, y (float, map-frame position), vx, vy (float, map-frame
            velocity), pmot (float 0-1, P(moving) from the tracker's motion
            mixture).
        Extra keys are ignored; missing optional keys (vx/vy/pmot) default
        to 0.0.
    robot_xy, robot_v_map: robot position and body-twist, BOTH already in
        the map frame (the caller rotates the odom twist into map by the
        TF yaw -- this function does no frame math of its own).
    params: dict with every key in DEFAULT_PARAMS (see that dict for the
        policy meaning of each one).

    Returns (pct, reason):
        pct    0.0 means "no limit" (nav2_msgs/SpeedLimit semantics -- see
               module docstring; NEVER interpret 0.0 here as "stop").
               Otherwise the most-restrictive percentage across every track
               that currently applies, floored at params["min_pct"].
        reason None when pct is 0.0 (no track applied), else a dict
               describing the single track that produced the returned pct:
               id, label, category, a short reason tag ("person_crawl",
               "person_slow", "closing", "static_slow"), the pct itself,
               and the geometry that triggered it (distance_m, or
               t_cpa_s/d_cpa_m for the CPA-gated branch).
    """
    rx, ry = robot_xy
    rvx, rvy = robot_v_map
    min_rel_speed = float(params.get("min_rel_speed_mps",
                                     DEFAULT_PARAMS["min_rel_speed_mps"]))
    act_on_any_mover = bool(params.get("act_on_any_mover",
                                       DEFAULT_PARAMS["act_on_any_mover"]))

    candidates: List[Tuple[float, Dict]] = []

    for tr in tracks:
        if float(tr.get("score", 0.0)) < float(params["min_track_score"]):
            continue
        if float(tr.get("age_sec", 0.0)) > float(params["max_track_age_sec"]):
            continue

        category = tr.get("category", "unknown")
        if not act_on_any_mover and category not in _ACTIONABLE_CATEGORIES:
            continue

        x, y = float(tr["x"]), float(tr["y"])
        distance = math.hypot(x - rx, y - ry)
        base = {"id": tr.get("id"), "label": tr.get("label"),
                "category": category}

        if category == "person":
            # Distance-only, nearest (most restrictive) radius first.
            if distance <= float(params["person_crawl_radius_m"]):
                candidates.append((float(params["person_crawl_pct"]), {
                    **base, "reason": "person_crawl", "distance_m": distance,
                }))
            elif distance <= float(params["person_slow_radius_m"]):
                candidates.append((float(params["person_slow_pct"]), {
                    **base, "reason": "person_slow", "distance_m": distance,
                }))
            continue

        # WP-A: category in ("robot", "wheeled") under act_on_any_mover:
        # false (the _ACTIONABLE_CATEGORIES filter above already enforced
        # that); ANY OTHER category too under the default act_on_any_mover:
        # true -- see module docstring's "any mover" policy section.
        vx, vy = float(tr.get("vx", 0.0)), float(tr.get("vy", 0.0))
        pmot = float(tr.get("pmot", 0.0))

        if pmot >= float(params["moving_pmot_min"]):
            _factor, t_cpa, d_cpa = cpa_geometry(
                x, y, vx, vy, rx, ry, rvx, rvy,
                _CPA_GAIN_UNUSED, _CPA_SCALE_UNUSED, _TTC_SCALE_UNUSED,
                min_rel_speed)
            if 0.0 < t_cpa <= float(params["robot_ttc_s"]) \
                    and d_cpa <= float(params["robot_cpa_m"]):
                candidates.append((float(params["robot_closing_pct"]), {
                    **base, "reason": "closing",
                    "t_cpa_s": t_cpa, "d_cpa_m": d_cpa,
                }))
            # else: diverging, or closing outside the TTC/CPA window -- no cap.
        elif category in _STATIC_SLOW_CATEGORIES \
                and distance <= float(params["static_obstacle_slow_radius_m"]):
            # static_slow stays restricted to robot/wheeled (which already
            # covers a promoted mover -- promotion relabels the category
            # itself) even under act_on_any_mover: an ordinary static
            # chair/table is Nav2's own obstacle layer's job, not this
            # governor's -- see module docstring.
            candidates.append((float(params["static_slow_pct"]), {
                **base, "reason": "static_slow", "distance_m": distance,
            }))

    if not candidates:
        return 0.0, None

    pct, reason = min(candidates, key=lambda c: c[0])
    pct = max(float(params["min_pct"]), pct)
    reason["pct"] = pct
    return pct, reason


class RiskSpeedGovernor(Node):
    """Thin ROS shell around compute_cap(): gather tracks + robot state
    each tick, call the pure policy, apply release hysteresis, publish."""

    def __init__(self):
        super().__init__("risk_speed_governor")

        self.declare_parameter("input_topic", "/risk_perception/world_objects")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("speed_limit_topic", "speed_limit")
        self.declare_parameter("debug_topic", "~/active_cap")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("release_hold_s", 0.5)

        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)

        gp = self.get_parameter
        self.input_topic = str(gp("input_topic").value)
        self.odom_topic = str(gp("odom_topic").value)
        self.speed_limit_topic = str(gp("speed_limit_topic").value)
        self.debug_topic = str(gp("debug_topic").value)
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.tf_timeout = float(gp("tf_timeout_sec").value)
        self.odom_timeout = float(gp("odom_timeout_sec").value)
        self.publish_rate = float(gp("publish_rate_hz").value)
        self.release_hold_s = float(gp("release_hold_s").value)

        self.params: Dict[str, float] = {
            name: float(gp(name).value) for name in DEFAULT_PARAMS
        }

        self.latest_detections: Optional[Detection3DArray] = None
        self.last_odom: Optional[Odometry] = None
        self.last_odom_t: float = -1.0
        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_v_map: Tuple[float, float] = (0.0, 0.0)
        self._warned_tf = False

        # Hysteresis state.
        self._active_pct: float = 0.0
        self._active_reason: Optional[Dict] = None
        self._last_applicable_time = self.get_clock().now()
        self._last_logged_key = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Detection3DArray, self.input_topic, self._objects_cb, 10)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, 10)

        self.speed_pub = self.create_publisher(
            SpeedLimit, self.speed_limit_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        period = 1.0 / self.publish_rate if self.publish_rate > 0.0 else 0.1
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"risk_speed_governor up: {self.input_topic} + {self.odom_topic} "
            f"({self.map_frame} -> {self.base_frame}) -> "
            f"{self.speed_limit_topic} @ {self.publish_rate:.1f} Hz")

    # ------------------------------------------------------------- callbacks

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _objects_cb(self, msg: Detection3DArray) -> None:
        self.latest_detections = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
        self.last_odom_t = self._now()

    # --------------------------------------------------------- robot state

    def _update_robot_state(self) -> None:
        """map->base pose via TF; body twist from odom rotated into map by
        the TF yaw. Same pattern as PredictiveRiskCostmapNode.
        _update_robot_state -- see that method's docstring for why the
        rotation is needed (CPA/TTC compares the robot's own motion against
        the object's, both must be in one common frame)."""
        self.robot_xy = None
        self.robot_v_map = (0.0, 0.0)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time(),
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException as exc:
            if not self._warned_tf:
                self.get_logger().warning(
                    f"no {self.map_frame} -> {self.base_frame} tf ({exc}); "
                    "speed governor publishes no-limit until it appears",
                    throttle_duration_sec=5.0)
                self._warned_tf = True
            return
        self._warned_tf = False

        self.robot_xy = (float(tf.transform.translation.x),
                        float(tf.transform.translation.y))

        if self.last_odom is None:
            return
        if self._now() - self.last_odom_t > self.odom_timeout:
            return  # stale odom: pose is still good, treat the robot as stopped
        yaw = yaw_from_quat(tf.transform.rotation)
        bx = float(self.last_odom.twist.twist.linear.x)
        by = float(self.last_odom.twist.twist.linear.y)
        self.robot_v_map = (bx * math.cos(yaw) - by * math.sin(yaw),
                            bx * math.sin(yaw) + by * math.cos(yaw))

    # -------------------------------------------------------------- tracks

    def _build_tracks(self) -> List[Dict]:
        tracks: List[Dict] = []
        if self.latest_detections is None:
            return tracks
        now = self._now()
        for det in self.latest_detections.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            label, kv = parse_class_id(str(hyp.class_id))
            stamp = det.header.stamp
            t = stamp.sec + stamp.nanosec * 1e-9
            age = 0.0 if t <= 0.0 else max(0.0, now - t)
            tracks.append({
                "id": str(det.id),
                "label": label,
                "category": label_category(label),
                "score": float(hyp.score),
                "age_sec": age,
                "x": float(det.bbox.center.position.x),
                "y": float(det.bbox.center.position.y),
                "vx": kv.get("vx", 0.0),
                "vy": kv.get("vy", 0.0),
                "pmot": kv.get("pmot", 0.0),
            })
        return tracks

    # -------------------------------------------------------------- tick

    def _tick(self) -> None:
        self._update_robot_state()
        if self.robot_xy is None:
            # Fail-soft, same rationale as PredictedRiskCritic: no TF yet
            # means "cannot cap against anything," not "stop." The
            # currently-active cap (if any) still decays through the normal
            # release hold below rather than vanishing/latching instantly.
            raw_pct, raw_reason = 0.0, None
        else:
            tracks = self._build_tracks()
            raw_pct, raw_reason = compute_cap(
                tracks, self.robot_xy, self.robot_v_map, self.params)

        now = self.get_clock().now()
        if raw_pct > 0.0:
            self._active_pct = raw_pct
            self._active_reason = raw_reason
            self._last_applicable_time = now
        elif self._active_pct > 0.0:
            elapsed = (now - self._last_applicable_time).nanoseconds * 1e-9
            if elapsed >= self.release_hold_s:
                self._active_pct = 0.0
                self._active_reason = None

        self._publish(now)

    # ------------------------------------------------------------- publish

    def _publish(self, now) -> None:
        msg = SpeedLimit()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.base_frame
        msg.percentage = True
        msg.speed_limit = self._active_pct
        self.speed_pub.publish(msg)

        debug = {"active_pct": self._active_pct,
                 "no_limit": self._active_pct == 0.0}
        if self._active_reason:
            debug.update(self._active_reason)
        dbg_msg = String()
        dbg_msg.data = json.dumps(debug)
        self.debug_pub.publish(dbg_msg)

        log_key = (self._active_pct,
                  json.dumps(self._active_reason, sort_keys=True)
                  if self._active_reason else None)
        if log_key != self._last_logged_key:
            self._last_logged_key = log_key
            if self._active_pct > 0.0 and self._active_reason:
                r = self._active_reason
                extra = {k: v for k, v in r.items()
                        if k not in ("id", "label", "category", "reason", "pct")}
                self.get_logger().info(
                    f"speed cap {self._active_pct:.0f}% -- track {r.get('id')} "
                    f"({r.get('label')}/{r.get('category')}) "
                    f"reason={r.get('reason')} {extra}",
                    throttle_duration_sec=2.0)
            else:
                self.get_logger().info(
                    "speed cap released -- no limit", throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = RiskSpeedGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # This node never self-terminates, so there is no callback-context
        # rclpy.shutdown() deadlock risk here (see house rule) -- this only
        # ever runs after spin() returns, i.e. from main()'s own stack.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
