#!/usr/bin/env python3
"""
bench_mppi_arm.py -- bench-test harness for the panoptex_mppi arm's
nav2_mppi_controller + panoptex_nav::PredictedRiskMppiCritic, with NO Isaac
Sim: this node IS the robot. It supplies everything a bare `controller_server`
process needs to activate and run a FollowPath goal --

  * static TF  map -> odom (identity), base_footprint -> laser,
    base_footprint -> base_link, base_link -> camera_depth_optical_frame
  * dynamic TF odom -> base_footprint + /odom, both driven by a simple
    Euler integrator over the controller's own /cmd_vel output (so this
    script closes the loop around whatever controller_server publishes --
    no separate robot/sim process involved)
  * /scan (LaserScan, frame "laser", 360 beams, constant 8.0 m) so the local
    costmap's obstacle_layer has something to subscribe to and never reports
    obstacles near the robot
  * /risk_stack (panoptex_msgs/RiskStack), a synthetic time-layered
    predictive risk field modelling ONE hazard crossing the robot's path
    (or, with --no-hazard, an all-zero stack -- the "no hazard" control
    case), republished every 0.2 s with the hazard's predicted position
    advanced per layer and in wall time

then sends a single nav2_msgs/action/FollowPath goal (a straight path along
+x, poses every path-step metres) and logs (t, vx, vy, wz, integrated pose,
hazard true position, robot-hazard distance) to a CSV until the action
finishes or --duration elapses.

This script does NOT start controller_server or a lifecycle_manager --
that's a separate step (see panoptex_nav/README.md's PredictedRiskMppiCritic
section and the bench runbook this was written for). It assumes
controller_server is already ACTIVE and serving the `follow_path` action
before the warm-up period ends; it starts publishing TF/scan/risk_stack
immediately on construction so an already-running controller_server's
costmap can activate against it if the two are started close together.

Geometry (deliberately NOT the sim risk_perception grid -- any geometry
that covers the path is fine per the bench design; this is 0.10 m
resolution, 20x20 m centred on the origin, 21 layers, dt 0.3 s,
horizon_start 0, matching the /risk_stack contract in
panoptex_msgs/msg/RiskStack.msg and README.md Sec 8):

  robot path   straight line (0,0) -> (path_len, 0), frame "map"
  hazard       linear motion from (--hazard-x0, --hazard-y0) at
               (--hazard-vx, --hazard-vy) m/s, starting at scenario t=0
               (this node's own start, NOT goal-send time). Painted as a
               disc of radius --hazard-radius and value --hazard-value in
               whichever layer covers that instant; every layer gets its
               own disc position (the layer for "now" gets the hazard's
               current position, the layer 6 s out gets its position 6 s
               from now), so nothing needs an explicit "before/after"
               clear step -- a layer whose disc
               falls outside the grid is simply left at 0 there.

--field {disc,gaussian,srm} (WP-B, 2026-09-10) picks WHAT is painted:

  disc      (default, the pre-WP-B behaviour) a hard disc of radius
            --hazard-radius and value --hazard-value.
  gaussian  the same peak value, falling off as exp(-d^2 / 2 sigma^2) with
            sigma = --hazard-radius, truncated at 3 sigma. The "old field"
            alternative -- still a class/confidence-shaped brush.
  srm       a SPATIOTEMPORAL RISK MAP (Thomas et al., 2021): the hazard is
            painted as an OCCUPANCY disc of radius 0.3 m and value 1 into
            each layer at that layer's predicted position, then every layer
            is converted by risk_perception.srm.stack_to_srm(stack,
            resolution, d0_m, levels=(0.2, 0.5, 0.8)) -- an EDT per risk
            level, srm = max_L L * clip(1 - d/d0, 0, 1). If risk_perception
            is not importable the same formula is used inline. --d0 (1.5 m)
            is the distance at which the field reaches 0.

  With --field srm the stack is published on /risk_stack_srm (matching
  config/nav2_x3_panoptex_mppi.yaml's PredictedRiskCritic.topic); the other
  two publish on /risk_stack. --stack-topic overrides either way.

Alongside the per-sample CSV the run writes <csv>.summary.csv (and logs one
BENCH_SUMMARY line) with the three numbers the WP-B bench is actually
comparing: hazard-robot min distance, time-to-goal, and a `passed` label --
at the moment the robot crosses the hazard's PATH LINE, was the hazard
already past that point ("behind", the robot yielded) or not yet there
("ahead", the robot cut in front)? "collinear" when the two paths never
cross (the head-on case), "never" when the robot never reaches the line.

Recipe (the bench runs on ROS_DOMAIN_ID=199 only, no Isaac, no /clock, so
every node runs on the system clock):

    export ROS_DOMAIN_ID=199
    ros2 run nav2_controller controller_server --ros-args \
      --params-file src/panoptex_nav/config/nav2_x3_panoptex_mppi.yaml \
      -p use_sim_time:=false -p local_costmap.local_costmap.use_sim_time:=false \
      -p global_costmap.global_costmap.use_sim_time:=false &
    python3 tools/bench_mppi_arm.py --field srm --csv /tmp/case.csv &
    ros2 run nav2_util lifecycle_bringup controller_server

Usage:
    python3 tools/bench_mppi_arm.py --hazard --csv /tmp/case_a.csv
    python3 tools/bench_mppi_arm.py --no-hazard --csv /tmp/case_b.csv
    python3 tools/bench_mppi_arm.py --field srm --csv /tmp/case_c.csv
"""

import argparse
import csv
import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry, Path
from panoptex_msgs.msg import RiskStack
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


SRM_LEVELS = (0.2, 0.5, 0.8)


def _stack_to_srm_fallback(stack, resolution, d0_m, levels=SRM_LEVELS):
    """Inline copy of risk_perception.srm.stack_to_srm, used when that module
    is not importable (WP-A may land after this script runs).

    stack: float array (steps, H, W) with values in [0, 1].
    For every risk level L: EDT to the cells at or above L, then
    srm = max_L L * clip(1 - d/d0, 0, 1). Same formula, same levels."""
    from scipy.ndimage import distance_transform_edt

    stack = np.asarray(stack, dtype=np.float32)
    out = np.zeros_like(stack)
    for level in levels:
        for k in range(stack.shape[0]):
            occupied = stack[k] >= level
            if not occupied.any():
                continue
            dist = distance_transform_edt(~occupied, sampling=resolution)
            out[k] = np.maximum(
                out[k], level * np.clip(1.0 - dist / d0_m, 0.0, 1.0))
    return out


try:  # WP-A owns the canonical implementation.
    from risk_perception.srm import stack_to_srm
    _SRM_SOURCE = "risk_perception.srm"
except ImportError:  # pragma: no cover - depends on the workspace state
    stack_to_srm = _stack_to_srm_fallback
    _SRM_SOURCE = "inline fallback"


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class BenchMppiArm(Node):
    def __init__(self, args):
        super().__init__("bench_mppi_arm")
        self.args = args
        self.start_time = time.monotonic()

        # Robot state, integrated in the odom frame -- numerically identical
        # to map, since map -> odom is published as identity below.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_cmd = Twist()
        self._last_integ_t = self.start_time
        self.rows = []

        self.cmd_sub = self.create_subscription(Twist, "cmd_vel", self._cmd_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)

        # RELIABLE + TRANSIENT_LOCAL + KeepLast(1): matches both critics'
        # subscription QoS (README Sec 8 "RiskStack contract" /
        # panoptex_nav/README.md's PredictedRiskMppiCritic.topic doc) and
        # predictive_risk_costmap_node's own publisher QoS.
        stack_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # --field srm publishes the Spatiotemporal Risk Map on its own topic,
        # matching the yaml's PredictedRiskCritic.topic; disc/gaussian keep
        # the historical /risk_stack.
        self.stack_topic = args.stack_topic or (
            "/risk_stack_srm" if args.field == "srm" else "/risk_stack")
        self.stack_pub = self.create_publisher(RiskStack, self.stack_topic, stack_qos)
        self.get_logger().info(
            f"field={args.field} d0={args.d0} stack topic={self.stack_topic} "
            f"srm impl={_SRM_SOURCE}")

        # local_costmap's static_layer (map_subscribe_transient_local: True,
        # per the panoptex_mppi/baseline_mppi yamls) subscribes /map with
        # RELIABLE + TRANSIENT_LOCAL and otherwise blocks the costmap update
        # forever ("Can't update static costmap layer, no map received"),
        # which silently zeroes every cmd_vel the controller ever produces --
        # found the hard way running case A before this existed. A plain
        # all-free grid, big enough to cover the path + hazard sweep, is
        # enough; nothing here is under test.
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self._publish_map()
        self.create_timer(3.0, self._publish_map)

        self.tf_bc = TransformBroadcaster(self)
        self.static_bc = StaticTransformBroadcaster(self)
        self._send_static_tf()

        # /risk_stack geometry -- any geometry covering the path is fine.
        self.res = 0.10
        self.size_m = 20.0
        self.W = int(round(self.size_m / self.res))
        self.H = self.W
        self.origin_x = -self.size_m / 2.0
        self.origin_y = -self.size_m / 2.0
        self.steps = 21
        self.dt = 0.3
        self.horizon_start = 0.0
        self.hazard_r_m = args.hazard_radius
        self.hazard_value = args.hazard_value
        self._logged_srm_scale = False

        self.create_timer(1.0 / 20.0, self._integrate_and_publish)
        self.create_timer(1.0 / 10.0, self._publish_scan)
        self.create_timer(0.2, self._publish_stack)

        self.action_client = ActionClient(self, FollowPath, "follow_path")
        self.goal_send_elapsed = None
        self.goal_result_elapsed = None

    # ---- helpers ---------------------------------------------------

    def _elapsed(self):
        return time.monotonic() - self.start_time

    def _hazard_true_pos(self, t):
        """Ground-truth hazard position at scenario time t (seconds since
        this node started), used both to paint /risk_stack layers (future
        t) and, in the CSV, as the ground truth to compare the robot's
        integrated position against (current t). Linear motion from
        (--hazard-x0, --hazard-y0) at (--hazard-vx, --hazard-vy) m/s --
        general enough for both a crossing mover (vx=0) and a head-on
        mover (vy=0, vx<0)."""
        a = self.args
        return a.hazard_x0 + a.hazard_vx * t, a.hazard_y0 + a.hazard_vy * t

    def _send_static_tf(self):
        now = self.get_clock().now().to_msg()

        def make(parent, child, x=0.0, y=0.0, z=0.0, yaw=0.0):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            qx, qy, qz, qw = yaw_quat(yaw)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            return t

        tfs = [
            make("map", "odom"),
            make("base_footprint", "laser", z=0.12),
            make("base_footprint", "base_link"),
            make("base_link", "camera_depth_optical_frame", x=0.08, z=0.15),
        ]
        self.static_bc.sendTransform(tfs)

    # ---- periodic publishers ----------------------------------------

    def _cmd_cb(self, msg):
        self.last_cmd = msg

    def _integrate_and_publish(self):
        now_t = time.monotonic()
        dt = now_t - self._last_integ_t
        self._last_integ_t = now_t

        vx = self.last_cmd.linear.x
        vy = self.last_cmd.linear.y
        wz = self.last_cmd.angular.z

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * cy - vy * sy) * dt
        self.y += (vx * sy + vy * cy) * dt
        self.yaw += wz * dt

        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_quat(self.yaw)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_bc.sendTransform(tf)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        elapsed = self._elapsed()
        hx, hy = self._hazard_true_pos(elapsed)
        dist = math.hypot(self.x - hx, self.y - hy)
        self.rows.append((elapsed, vx, vy, wz, self.x, self.y, self.yaw, hx, hy, dist))

    def _publish_map(self):
        # Free (0) everywhere, big enough for the path (0..path_len in x)
        # plus the hazard sweep (y from -2.5 up) plus the rolling local
        # costmap's window around the robot.
        res = 0.05
        width_m, height_m = 24.0, 20.0
        origin_x, origin_y = -8.0, -10.0
        w = int(round(width_m / res))
        h = int(round(height_m / res))

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"
        grid.info = MapMetaData()
        grid.info.resolution = res
        grid.info.width = w
        grid.info.height = h
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (w * h)
        self.map_pub.publish(grid)

    def _publish_scan(self):
        n = 360
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "laser"
        scan.angle_min = -math.pi
        scan.angle_max = math.pi - (2.0 * math.pi / n)
        scan.angle_increment = 2.0 * math.pi / n
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.05
        scan.range_max = 12.0
        scan.ranges = [8.0] * n
        self.scan_pub.publish(scan)

    def _paint_layer(self, layer, hx, hy, radius_m, peak, gaussian):
        """Paint one hazard blob at (hx, hy) into `layer` (a float H x W view
        holding values in [0, 1]), keeping the max with whatever is there."""
        reach_m = 3.0 * radius_m if gaussian else radius_m
        r_cells = max(1, int(round(reach_m / self.res)))
        cx = int(round((hx - self.origin_x) / self.res))
        cy = int(round((hy - self.origin_y) / self.res))
        x0, x1 = max(0, cx - r_cells), min(self.W, cx + r_cells + 1)
        y0, y1 = max(0, cy - r_cells), min(self.H, cy + r_cells + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) * (self.res ** 2)
        if gaussian:
            blob = peak * np.exp(-d2 / (2.0 * radius_m ** 2))
            blob[d2 > reach_m ** 2] = 0.0
        else:
            blob = np.where(d2 <= radius_m ** 2, peak, 0.0)
        layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], blob)

    def _build_field(self, elapsed):
        """The whole (steps, H, W) field in [0, 1], per --field."""
        field = np.zeros((self.steps, self.H, self.W), dtype=np.float32)
        if not self.args.hazard:
            return field

        srm_mode = self.args.field == "srm"
        # SRM: the hazard is an OCCUPANCY core (radius 0.3 m, value 1), and
        # the distance falloff comes from stack_to_srm, not from the brush.
        radius_m = self.args.srm_core_radius if srm_mode else self.hazard_r_m
        peak = 1.0 if srm_mode else (self.hazard_value / 100.0)
        gaussian = self.args.field == "gaussian"

        for k in range(self.steps):
            tk = elapsed + self.horizon_start + k * self.dt
            hx, hy = self._hazard_true_pos(tk)
            self._paint_layer(field[k], hx, hy, radius_m, peak, gaussian)

        if srm_mode:
            occupied = field >= 1.0
            field = np.asarray(
                stack_to_srm(field, self.res, self.args.d0, levels=SRM_LEVELS),
                dtype=np.float32)
            # The /risk_stack_srm contract is "1 at occupied cells, falling
            # linearly to 0 at d0". stack_to_srm's max_L L * clip(...) form
            # tops out at max(levels) = 0.8 with these levels, which would put
            # the whole field below the critic's collision_threshold (0.9), so
            # renormalise by whatever the field actually reads ON an occupied
            # cell. A stack_to_srm that already honours the contract leaves
            # this a no-op.
            if occupied.any():
                peak_on_core = float(field[occupied].max())
                if 0.0 < peak_on_core < 0.999:
                    field = np.clip(field / peak_on_core, 0.0, 1.0)
                    if not self._logged_srm_scale:
                        self._logged_srm_scale = True
                        self.get_logger().warn(
                            f"stack_to_srm peaks at {peak_on_core:.3f} on an occupied "
                            f"cell; renormalising to honour the '1 at occupied cells' "
                            f"/risk_stack_srm contract")
        return field

    def _publish_stack(self):
        elapsed = self._elapsed()
        field = self._build_field(elapsed)
        data = np.clip(np.rint(field * 100.0), 0, 100).astype(np.int8)

        msg = RiskStack()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.res
        msg.info.width = self.W
        msg.info.height = self.H
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.dt = self.dt
        msg.steps = self.steps
        msg.horizon_start = self.horizon_start
        msg.data = data.reshape(-1).tolist()
        self.stack_pub.publish(msg)

    # ---- FollowPath goal ---------------------------------------------

    def _build_path(self):
        n_steps = int(round(self.args.path_len / self.args.path_step))
        stamp = self.get_clock().now().to_msg()
        poses = []
        for i in range(n_steps + 1):
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.header.stamp = stamp
            ps.pose.position.x = i * self.args.path_step
            ps.pose.position.y = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = stamp
        path.poses = poses
        return path

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f"distance_to_goal={fb.distance_to_goal:.3f} speed={fb.speed:.3f}")

    def send_goal_and_wait(self, timeout_s):
        goal = FollowPath.Goal()
        goal.path = self._build_path()
        goal.controller_id = self.args.controller_id
        goal.goal_checker_id = ""

        self.get_logger().info("waiting for follow_path action server...")
        if not self.action_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("follow_path action server never appeared")
            return "no_server"

        self.goal_send_elapsed = self._elapsed()
        send_future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("FollowPath goal rejected or send timed out")
            return "rejected"

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not result_future.done():
            self.get_logger().warn(
                f"FollowPath did not finish within {timeout_s:.1f}s; canceling")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            self.goal_result_elapsed = self._elapsed()
            return "timeout"

        result = result_future.result()
        self.goal_result_elapsed = self._elapsed()
        return {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_ABORTED: "aborted",
            GoalStatus.STATUS_CANCELED: "canceled",
        }.get(result.status, f"status_{result.status}")

    # ---- output --------------------------------------------------

    def summarize(self, status):
        """The three WP-B bench numbers: hazard-robot min distance,
        time-to-goal, and the `passed` label.

        `passed` answers: at the moment the robot crosses the hazard's PATH
        LINE (the infinite line through the hazard's start along its velocity),
        is the hazard already past that point or not yet there? Both are read
        off the same pair of positions, projected onto the hazard's own
        direction: s_hazard > s_robot means the hazard has already swept
        through the crossing, i.e. the robot went BEHIND it.

          behind     the robot yielded and crossed after the hazard
          ahead      the robot cut in front of the hazard
          collinear  the two paths never cross (the head-on case)
          never      the robot never reached the hazard's path line
          no_hazard  --no-hazard control run
        """
        min_dist = min((r[9] for r in self.rows), default=float("nan"))
        ttg = None
        if self.goal_send_elapsed is not None and self.goal_result_elapsed is not None:
            ttg = self.goal_result_elapsed - self.goal_send_elapsed

        a = self.args
        if not a.hazard:
            return min_dist, ttg, "no_hazard"

        speed = math.hypot(a.hazard_vx, a.hazard_vy)
        if speed < 1e-6:
            return min_dist, ttg, "collinear"
        ux, uy = a.hazard_vx / speed, a.hazard_vy / speed
        # The nominal robot path is +x (see _build_path). A hazard travelling
        # along that same line (the head-on case) never gets crossed, so there
        # is no ahead/behind to read -- and a robot that sidesteps and comes
        # back would otherwise produce a spurious label off its own detour.
        if abs(uy) < 0.1:
            return min_dist, ttg, "collinear"
        nx, ny = -uy, ux          # unit normal to the hazard's path line

        def signed(row):
            return (row[4] - a.hazard_x0) * nx + (row[5] - a.hazard_y0) * ny

        offsets = [signed(r) for r in self.rows]
        if not offsets:
            return min_dist, ttg, "never"

        cross_i = None
        for i in range(1, len(offsets)):
            if offsets[i - 1] == 0.0 or offsets[i] * offsets[i - 1] < 0.0:
                cross_i = i
                break
        if cross_i is None:
            return min_dist, ttg, "never"

        row = self.rows[cross_i]
        s_robot = (row[4] - a.hazard_x0) * ux + (row[5] - a.hazard_y0) * uy
        s_hazard = speed * row[0]
        return min_dist, ttg, ("behind" if s_hazard > s_robot else "ahead")

    def write_csv(self, path, status="not_run"):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "vx", "vy", "wz", "x", "y", "yaw",
                        "hazard_x", "hazard_y", "dist_robot_hazard"])
            for row in self.rows:
                w.writerow([f"{v:.4f}" for v in row])
        self.get_logger().info(f"wrote {len(self.rows)} rows to {path}")

        min_dist, ttg, passed = self.summarize(status)
        summary_path = f"{path}.summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["field", "hazard", "cost_weight_note", "status",
                        "min_dist_m", "time_to_goal_s", "passed",
                        "hazard_x0", "hazard_y0", "hazard_vx", "hazard_vy", "d0"])
            w.writerow([
                self.args.field, int(bool(self.args.hazard)), self.args.label,
                status,
                f"{min_dist:.4f}",
                "" if ttg is None else f"{ttg:.3f}",
                passed,
                self.args.hazard_x0, self.args.hazard_y0,
                self.args.hazard_vx, self.args.hazard_vy, self.args.d0])
        self.get_logger().info(
            "BENCH_SUMMARY "
            f"label={self.args.label} field={self.args.field} "
            f"hazard={int(bool(self.args.hazard))} status={status} "
            f"min_dist={min_dist:.3f} "
            f"time_to_goal={'nan' if ttg is None else f'{ttg:.2f}'} "
            f"passed={passed}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hazard", dest="hazard", action="store_true", default=True,
                    help="paint the crossing hazard into /risk_stack (default: on)")
    ap.add_argument("--no-hazard", dest="hazard", action="store_false",
                    help="publish an all-zero /risk_stack (the no-hazard control case)")
    ap.add_argument("--csv", required=True, help="output CSV path")
    ap.add_argument("--duration", type=float, default=40.0,
                    help="max seconds to wait for FollowPath to finish (default 40)")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds to publish TF/scan/risk_stack before sending "
                         "the goal, so the critic never sees 'no stack' (default 2)")
    ap.add_argument("--path-len", type=float, default=6.0,
                    help="straight path length in +x, metres (default 6.0)")
    ap.add_argument("--path-step", type=float, default=0.1,
                    help="spacing between path poses, metres (default 0.1)")
    ap.add_argument("--controller-id", default="FollowPath",
                    help="FollowPath.controller_id (default 'FollowPath', matching "
                         "controller_plugins in both study-arm yamls)")
    ap.add_argument("--hazard-x0", type=float, default=3.0,
                    help="hazard start x, metres, at scenario t=0 (default 3.0)")
    ap.add_argument("--hazard-y0", type=float, default=-2.5,
                    help="hazard start y, metres, at scenario t=0 (default -2.5)")
    ap.add_argument("--hazard-vx", type=float, default=0.0,
                    help="hazard x velocity, m/s (default 0.0 -- a pure crossing mover)")
    ap.add_argument("--hazard-vy", type=float, default=0.6,
                    help="hazard y velocity, m/s (default 0.6 -- a pure crossing mover; "
                         "use --hazard-vy 0 and a negative --hazard-vx for a head-on mover)")
    ap.add_argument("--hazard-radius", type=float, default=0.5,
                    help="hazard disc radius, metres (default 0.5)")
    ap.add_argument("--hazard-value", type=int, default=75,
                    help="hazard disc/gaussian peak risk value, 0-100 (default 75); "
                         "ignored with --field srm, whose core is occupancy 1")
    ap.add_argument("--field", choices=("disc", "gaussian", "srm"), default="disc",
                    help="what to paint into the stack (default 'disc' -- the "
                         "pre-WP-B behaviour). See this script's docstring.")
    ap.add_argument("--d0", type=float, default=1.5,
                    help="--field srm: distance at which the SRM reaches 0, "
                         "metres (default 1.5)")
    ap.add_argument("--srm-core-radius", type=float, default=0.3,
                    help="--field srm: radius of the occupancy core painted "
                         "before the SRM conversion, metres (default 0.3)")
    ap.add_argument("--stack-topic", default=None,
                    help="override the RiskStack topic (default: /risk_stack_srm "
                         "for --field srm, /risk_stack otherwise)")
    ap.add_argument("--label", default="",
                    help="free-text run label, copied into the summary CSV")
    args, rest = ap.parse_known_args()
    if rest and rest[0] != "--ros-args":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")

    rclpy.init(args=rest)
    node = BenchMppiArm(args)
    status = "not_run"
    try:
        deadline = time.monotonic() + args.warmup
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        status = node.send_goal_and_wait(args.duration)
        node.get_logger().info(f"FollowPath result: {status}")
        if node.goal_send_elapsed is not None and node.goal_result_elapsed is not None:
            node.get_logger().info(
                "path time: "
                f"{node.goal_result_elapsed - node.goal_send_elapsed:.2f} s")
    finally:
        node.write_csv(args.csv, status)
        node.get_logger().info(f"FINAL_STATUS={status}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
