#!/usr/bin/env python3
"""
scenario_publisher.py  --  synthetic world model, for testing Stage 2/3/4 with
no camera, no robot and no bag.

Publishes exactly what object_tracker_node publishes
(/risk_perception/world_objects, same class_id and covariance contract) plus a
robot state (/odom + tf map -> base_footprint), so
predictive_risk_costmap_node and spatial_prior_node can be driven through
known geometry and their output checked against a number you worked out by
hand. The perception chain is the part most likely to be broken on any given
day; this lets the risk stages be debugged while it is.

    python3 tools/scenario_publisher.py --scenario head_on

Scenarios (robot at the origin facing +x unless stated):
  head_on       person at (4,0) walking -x at 1.0 m/s straight into the robot.
                CPA -> 0, TTC counts down 4s -> 0. Max amplification.
  crossing      person crossing the robot's front at x=--offset (default 2.0),
                moving +y at 1.0 m/s. Closes, then opens: amplification peaks
                as it passes and drops to exactly 1.0 afterwards. Here d_cpa
                IS the offset, so sweeping --offset 0.2 0.4 ... 2.0 and reading
                the factor off the "encounter" marker traces
                factor(d_cpa) directly -- that curve is the Stage 4b figure.
  parallel      robot drives +x at --speed with a person 2 m to its left
                moving identically: separation pinned at 2 m, relative speed
                exactly 0 -> factor exactly 1.0 the whole run. This is the
                "does Stage 4b stay out of the way" control. Both actors
                leave the costmap's default +-6 m extent after ~6 s at
                1 m/s; use a smaller --speed to watch it longer.
  robot_forward static chair at (3,0), robot driving +x at --speed. The object
                never moves, yet risk must rise -- risk is a relation. Run it
                with --label chair to see consequence weighting too: the same
                geometry must peak lower for a chair (0.35) than a person.
  corridor      person shuttling along y=0 between x=-3 and x=3, robot parked
                off to the side. Feeds spatial_prior_node; the y=0 band should
                brighten over ~a minute and survive a restart.
  circular      person walking a closed loop of radius --radius (default 2 m)
                centred on the origin, robot parked outside it. This is the
                "walk in a circle, then ask the robot to cross the middle"
                evaluation scenario: F should read back as a smooth
                TANGENTIAL heading all the way around, unlike corridor's two
                opposite directions sharing one straight band.
  zigzag        person shuttling in x exactly like corridor, PLUS a faster
                triangle-wave oscillation in y (--zigzag-amplitude,
                --zigzag-period) -- the path crosses itself repeatedly, so
                nearby cells see opposing headings. Exercises
                spatial_prior_node's documented honest limitation directly:
                "at a location genuinely used by traffic going both
                directions, opposing velocities partially cancel in the EMA
                rather than being represented as two distinct headings" --
                watch F's magnitude (not just direction) wash out where the
                zigzag's own legs cross.
  relation_test everything stationary: robot at the origin, object at (3,0),
                zero velocity throughout. Isolates the relation prior's
                additive term (see --rel-pulse below) from motion and CPA/TTC
                entirely -- pmot stays ~0 and the CPA factor stays exactly
                1.0 (no relative motion at all), so the only thing that can
                move the costmap peak is relbonus. This is Half A of the
                relation-prior test plan; it validates the tracker/costmap
                math only, not GDINO's own grounding (that needs real or
                simulated imagery -- see Half B).
  multi         person head-on from (4,0) at --speed plus a static chair at
                (2,-2); the person's blob amplifies (factor 1 + 1.5*exp(-4/3)
                ~= 1.40 at t=0, v=1, rising to 2.5, saturating the clipped
                cost at 100 for the whole run) while the chair holds a
                constant factor-1.0 blob at cost ~29 (analytic 31, sampled
                0.05 m off the cell centre) -- two objects at once, plus the
                costmap's max-combination. To read the raw consequence
                ordering (person 0.81 vs chair 0.315) off the grid, run
                --speed 0: both freeze unclipped at ~75 vs ~29.
  twins         TWO objects with the SAME label but different motions --
                person crossing at x=--offset from (offset,-3) moving +y,
                plus a second person standing at (-2,1). Walker's factor
                peaks ~1.05 at the default offset 2 then snaps to exactly
                1.0; stander stays factor 1.0 at cost ~75 (analytic 79,
                sampled off-centre). Ids "1"/"2" keep their RViz markers
                (prediction/encounter) distinct. Keep --offset away from -2
                (the walker would plow through the stander) and inside the
                +-5 m grid.
  mutual        robot and object BOTH moving -- robot starts at (0,-2)
                heading +y (yaw pi/2) driving at --speed, person at (0,3)
                walking -y at --speed. Closing speed 2v, TTC starts at
                2.5/v s and counts to 0 (factor 1 + 1.5*exp(-2.5/3) ~= 1.65
                at t=0, v=1), snapping to exactly 1.0 after they pass.
                The yaw pi/2 exercises predictive_risk_costmap_node's odom
                body-twist -> map rotation for the first time. Everything
                stays inside the deployed +-5 m grid through the interesting
                phase.
  crowd         --num-people (1-5) independent pedestrians in a
                --space x --space room (default 5 m half-extent, i.e. 10x10 m
                -- deliberately bigger than the other scenarios' +-3..6 m so
                spatial_prior_node gets real floor coverage). Each is a
                RANDOM-WAYPOINT walker (see the Walker class): pick a random
                point in the room, walk straight at it, pick a new one on
                arrival -- the standard simple "purposeful but not
                perfectly straight" pedestrian model, not a full social-force
                sim. The "not perfectly straight" part is an
                Ornstein-Uhlenbeck (mean-reverting Brownian) wobble on TOP of
                the straight-line goal heading, --wobble std-dev rad/sqrt(s):
                0 is a robotic straight-line beeline between corners, ~0.8
                (default) reads as a person, >1.5 starts looking drunk. Fully
                reproducible from --seed (each person gets seed+i). Unlike
                every other scenario this one is NOT a pure function of t --
                Walker carries its own position/goal/noise state, stepped
                once per tick -- because a random-waypoint path has no
                closed-form position(t). Robot parked fixed outside the room.

For multi-object scenarios, --rel-pulse tags object 1 only. --label renames
every object that has no scenario-fixed label: both of twins' persons (the
point is the SAME label, whatever it is), but not multi's chair. --offset
moves the crossing walker in both crossing and twins.

--rel-pulse cycles a synthetic relbonus=<value> tag on the object's class_id
on and off every --rel-pulse-period seconds (default 6s: 3s on, 3s off) --
for watching predictive_risk_costmap_node's additive relation-prior fusion
(C += relbonus) directly in RViz or ros2 topic echo, without GDINO, a
camera, or object_tracker ever needing to run. Note the scope: this script
publishes STRAIGHT to /risk_perception/world_objects, i.e. it emulates
object_tracker's OUTPUT and bypasses it entirely (same as every other field
here) -- so --rel-pulse exercises the fusion step in
predictive_risk_costmap_node only. It does NOT exercise object_tracker's
own rise/decay smoothing (Track.update_relation), since that code never
runs in this data path at all. Test that separately and directly -- it's
pure arithmetic on a Track object, no ROS graph needed, see
src/risk_perception/test/test_object_tracker_relation.py.

WARNING: with --tf (the default) this publishes map -> base_footprint. Do not
run it while AMCL, slam_toolbox or the robot's EKF are up -- two owners of one
transform is exactly the corruption the README warns about. Use --no-tf to
publish only /odom and world_objects.
"""

import argparse
import math
import random

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

SCENARIOS = ("head_on", "crossing", "parallel", "robot_forward", "corridor",
            "circular", "zigzag", "crowd", "relation_test", "multi", "twins",
            "mutual")


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class Walker:
    """One random-waypoint pedestrian, deliberately ROS-free (like
    encounter_geometry.py / mask_relation.py elsewhere in this codebase) so
    the walk itself is testable and tunable without spinning up rclpy.

    Model: walk in a straight line toward a random point in the room; on
    arrival, pick a new random point. The only deviation from a dead-
    straight beeline is an Ornstein-Uhlenbeck (mean-reverting) random
    offset added to that goal heading -- literally a small Brownian motion
    on the HEADING, not on position, which is what keeps it wobbly without
    ever drifting into loops or wandering off-course entirely (a plain
    random walk on position does that, and reads as drunk, not distracted).
    wobble=0 is a perfect corner-to-corner beeline; ~0.8 reads as a person;
    past ~1.5 it starts looking erratic. Tuned by eye, see the walker
    prototype plots in this change's discussion, not derived from any real
    pedestrian dataset.
    """

    def __init__(self, seed, bounds, speed, wobble, arrive_radius=0.3,
                noise_decay=1.5):
        self.rng = random.Random(seed)
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        # +-20% per-person speed spread so a multi-person scene isn't
        # visibly lock-step identical.
        self.speed = speed * self.rng.uniform(0.8, 1.2)
        self.wobble = wobble
        self.noise_decay = noise_decay
        self.arrive_radius = arrive_radius
        self.x = self.rng.uniform(self.xmin, self.xmax)
        self.y = self.rng.uniform(self.ymin, self.ymax)
        self.noise = 0.0
        self._pick_goal()

    def _pick_goal(self):
        self.gx = self.rng.uniform(self.xmin, self.xmax)
        self.gy = self.rng.uniform(self.ymin, self.ymax)

    def step(self, dt):
        """Advance by dt seconds; -> (x, y, vx, vy)."""
        dx, dy = self.gx - self.x, self.gy - self.y
        if math.hypot(dx, dy) < self.arrive_radius:
            self._pick_goal()
            dx, dy = self.gx - self.x, self.gy - self.y
        goal_heading = math.atan2(dy, dx)
        self.noise += (-self.noise_decay * self.noise * dt
                       + self.rng.gauss(0.0, self.wobble) * math.sqrt(dt))
        heading = goal_heading + self.noise
        vx = self.speed * math.cos(heading)
        vy = self.speed * math.sin(heading)
        self.x = min(max(self.x + vx * dt, self.xmin), self.xmax)
        self.y = min(max(self.y + vy * dt, self.ymin), self.ymax)
        return self.x, self.y, vx, vy


class ScenarioPublisher(Node):
    def __init__(self, scenario, rate, speed, publish_tf, label, offset,
                rel_pulse, rel_pulse_period, rel_conf, radius,
                zigzag_amplitude, zigzag_period, num_people, space, wobble,
                seed):
        super().__init__("scenario_publisher")
        self.scenario = scenario
        self.speed = speed
        self.label = label
        self.offset = offset
        self.publish_tf = publish_tf
        self.rel_pulse = rel_pulse
        self.rel_pulse_period = max(0.5, rel_pulse_period)
        self.rel_conf = rel_conf
        self.radius = radius
        self.zigzag_amplitude = zigzag_amplitude
        self.zigzag_period = zigzag_period
        self.space = space
        self.t0 = None
        self.last_tick_t = None

        if scenario == "crowd":
            bounds = (-space, space, -space, space)
            self.walkers = [Walker(seed + i, bounds, speed, wobble)
                            for i in range(num_people)]

        self.obj_pub = self.create_publisher(
            Detection3DArray, "/risk_perception/world_objects", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_bc = TransformBroadcaster(self) if publish_tf else None
        self.create_timer(1.0 / rate, self._tick)
        rel_msg = (f", rel-pulse {rel_pulse_period:.1f}s period @ conf {rel_conf:.2f}"
                  if rel_pulse else "")
        crowd_msg = (f", {num_people} people in a {2*space:.0f}x{2*space:.0f} m "
                    f"room, wobble {wobble}, seed {seed}"
                    if scenario == "crowd" else "")
        self.get_logger().info(
            f"scenario '{scenario}' at {rate} Hz, actor speed {speed} m/s, "
            f"tf {'on' if publish_tf else 'off'}{rel_msg}{crowd_msg}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _state(self, t):
        """-> ((robot_x, robot_y, robot_yaw, robot_vx_body), [obj, ...]).

        Each obj is dict(x=..., y=..., vx=..., vy=...) plus an optional
        "label" key (defaults to --label at publish time). Object i in the
        list publishes as detection id str(i + 1).
        """
        v = self.speed
        if self.scenario == "head_on":
            return (0, 0, 0.0, 0.0), [dict(x=4.0 - v * t, y=0.0, vx=-v, vy=0.0)]
        if self.scenario == "crossing":
            return (0, 0, 0.0, 0.0), [dict(x=self.offset, y=-3.0 + v * t, vx=0.0, vy=v)]
        if self.scenario == "parallel":
            return (v * t, 0.0, 0.0, v), [dict(x=v * t, y=2.0, vx=v, vy=0.0)]
        if self.scenario == "robot_forward":
            return (v * t, 0.0, 0.0, v), [dict(x=3.0, y=0.0, vx=0.0, vy=0.0)]
        if self.scenario == "relation_test":
            return (0, 0, 0.0, 0.0), [dict(x=3.0, y=0.0, vx=0.0, vy=0.0)]
        if self.scenario == "multi":
            return (0, 0, 0.0, 0.0), [
                dict(x=4.0 - v * t, y=0.0, vx=-v, vy=0.0),
                dict(x=2.0, y=-2.0, vx=0.0, vy=0.0, label="chair"),
            ]
        if self.scenario == "twins":
            return (0, 0, 0.0, 0.0), [
                dict(x=self.offset, y=-3.0 + v * t, vx=0.0, vy=v),
                dict(x=-2.0, y=1.0, vx=0.0, vy=0.0),
            ]
        if self.scenario == "mutual":
            return (0.0, -2.0 + v * t, math.pi / 2.0, v), [
                dict(x=0.0, y=3.0 - v * t, vx=0.0, vy=-v),
            ]
        if self.scenario == "circular":
            r = self.radius
            omega = v / r
            theta = omega * t
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            vx = -v * math.sin(theta)
            vy = v * math.cos(theta)
            return (0.0, -r - 2.0, 0.0, 0.0), [dict(x=x, y=y, vx=vx, vy=vy)]
        if self.scenario == "zigzag":
            # same x-shuttle as corridor...
            span = 6.0
            period_x = 2.0 * span / v
            phase_x = (t % period_x) / period_x
            if phase_x < 0.5:
                x = -3.0 + span * (phase_x * 2.0)
                vx = v
            else:
                x = 3.0 - span * ((phase_x - 0.5) * 2.0)
                vx = -v
            # ...plus a faster triangle-wave in y, so the path crosses itself.
            amp, period_y = self.zigzag_amplitude, self.zigzag_period
            v_lat = 4.0 * amp / period_y
            phase_y = (t % period_y) / period_y
            if phase_y < 0.5:
                y = -amp + 2.0 * amp * (phase_y * 2.0)
                vy = v_lat
            else:
                y = amp - 2.0 * amp * ((phase_y - 0.5) * 2.0)
                vy = -v_lat
            return (-1.0, -4.0, 0.0, 0.0), [dict(x=x, y=y, vx=vx, vy=vy)]
        # corridor: shuttle between x=-3 and x=3 at --speed
        span = 6.0
        period = 2.0 * span / v          # out and back
        phase = (t % period) / period
        if phase < 0.5:
            x = -3.0 + span * (phase * 2.0)
            vx = v
        else:
            x = 3.0 - span * ((phase - 0.5) * 2.0)
            vx = -v
        return (-1.0, -2.5, 0.0, 0.0), [dict(x=x, y=0.0, vx=vx, vy=0.0)]

    def _tick(self):
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        if self.scenario == "crowd":
            # The only scenario that is NOT a pure function of t -- see the
            # Walker docstring. dt clamped against a clock jump / long pause
            # the same way spatial_prior_node._update() guards its own dt.
            dt = 0.0 if self.last_tick_t is None else now - self.last_tick_t
            dt = max(0.0, min(dt, 1.0))
            self.last_tick_t = now
            objs = [dict(zip(("x", "y", "vx", "vy"), w.step(dt)))
                    for w in self.walkers]
            robot = (0.0, -self.space - 1.5, 0.0, 0.0)
        else:
            robot, objs = self._state(t)
        rx, ry, ryaw, rv = robot
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_quat(ryaw)

        if self.tf_bc is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = "map"
            tf.child_frame_id = "base_footprint"
            tf.transform.translation.x = float(rx)
            tf.transform.translation.y = float(ry)
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_bc.sendTransform(tf)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = float(rx)
        odom.pose.pose.position.y = float(ry)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(rv)     # body frame, as a real base reports
        self.odom_pub.publish(odom)

        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        for i, obj in enumerate(objs):
            d = Detection3D()
            d.header = msg.header
            d.id = str(i + 1)
            d.bbox.center.position.x = float(obj["x"])
            d.bbox.center.position.y = float(obj["y"])
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = 0.5
            ovx, ovy = obj["vx"], obj["vy"]
            moving = math.hypot(ovx, ovy) > 1e-3
            label = obj.get("label", self.label)
            class_id = (
                f"{label}|pmov=0.90|pmot={0.95 if moving else 0.02:.2f}"
                f"|vx={ovx:.3f}|vy={ovy:.3f}")
            if self.rel_pulse and i == 0:
                # This message emulates object_tracker's OUTPUT (world_objects),
                # not gdino_detector's input -- so the key is relbonus (what
                # predictive_risk_costmap_node.parse_class_id reads), not
                # relconf (what object_tracker reads from detections_3d_map).
                # First half of each period "on" (tag present), second half
                # "off" (tag omitted entirely).
                on = (t % self.rel_pulse_period) < (self.rel_pulse_period / 2.0)
                if on:
                    class_id += f"|relbonus={self.rel_conf:.2f}"
            r = ObjectHypothesisWithPose()
            r.hypothesis.class_id = class_id
            r.hypothesis.score = 0.9
            r.pose.pose = d.bbox.center
            cov = [0.0] * 36
            cov[0] = cov[7] = 0.04      # Pxx, Pyy
            cov[21] = cov[28] = 0.01    # Pvxvx, Pvyvy
            r.pose.covariance = cov
            d.results.append(r)
            msg.detections.append(d)
        self.obj_pub.publish(msg)


def positive_float(s):
    v = float(s)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return v


def finite_float(s):
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError("must be finite")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="head_on", choices=SCENARIOS)
    ap.add_argument("--rate", type=positive_float, default=10.0)
    ap.add_argument("--speed", type=finite_float, default=1.0,
                    help="actor speed, m/s. 0 freezes the moving actor (a static-object "
                         "variant of any scenario); corridor requires > 0")
    ap.add_argument("--label", default="person")
    ap.add_argument("--offset", type=float, default=2.0,
                    help="crossing/twins: how far in front of the robot the walker "
                         "passes, i.e. the miss distance d_cpa")
    ap.add_argument("--no-tf", dest="tf", action="store_false",
                    help="do not publish map -> base_footprint (use with a real robot up)")
    ap.add_argument("--rel-pulse", action="store_true",
                    help="cycle a synthetic relbonus tag on/off -- see --rel-pulse-period. "
                         "Pairs with --scenario relation_test to isolate "
                         "predictive_risk_costmap_node's relation-prior fusion from "
                         "motion/CPA-TTC entirely. Does NOT exercise object_tracker's "
                         "own rise/decay smoothing -- see test_object_tracker_relation.py.")
    ap.add_argument("--rel-pulse-period", type=float, default=6.0,
                    help="seconds per on/off cycle, half on half off (default 6.0)")
    ap.add_argument("--rel-conf", type=float, default=0.85,
                    help="relbonus value published during the 'on' half of each cycle")
    ap.add_argument("--radius", type=positive_float, default=2.0,
                    help="circular: loop radius in metres (default 2.0)")
    ap.add_argument("--zigzag-amplitude", type=positive_float, default=1.2,
                    help="zigzag: lateral half-width in metres (default 1.2)")
    ap.add_argument("--zigzag-period", type=positive_float, default=3.0,
                    help="zigzag: seconds per lateral out-and-back (default 3.0)")
    ap.add_argument("--num-people", type=int, default=3,
                    help="crowd: how many independent walkers, 1-5 (default 3)")
    ap.add_argument("--space", type=positive_float, default=5.0,
                    help="crowd: room half-extent in metres -- room spans "
                         "+-space in x and y (default 5.0, i.e. 10x10 m)")
    ap.add_argument("--wobble", type=float, default=0.8,
                    help="crowd: heading-noise std-dev, rad/sqrt(s) (default "
                         "0.8; 0 = perfectly straight beelines between "
                         "waypoints, >1.5 starts looking erratic)")
    ap.add_argument("--seed", type=int, default=42,
                    help="crowd: RNG seed, for a reproducible crowd -- "
                         "person i uses seed+i (default 42)")
    args, rest = ap.parse_known_args()
    # Leftovers must be ROS args ("--ros-args -p use_sim_time:=true ..."), which
    # get handed to rclpy; anything else is a typo and should fail loudly.
    if rest and rest[0] != "--ros-args":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.scenario in ("corridor", "circular", "zigzag", "crowd") and args.speed <= 0:
        ap.error(f"--scenario {args.scenario} requires --speed > 0")
    if args.scenario == "crowd" and not 1 <= args.num_people <= 5:
        ap.error("--scenario crowd requires 1 <= --num-people <= 5")

    rclpy.init(args=rest)
    node = ScenarioPublisher(args.scenario, args.rate, args.speed, args.tf,
                             args.label, args.offset,
                             args.rel_pulse, args.rel_pulse_period, args.rel_conf,
                             args.radius, args.zigzag_amplitude, args.zigzag_period,
                             args.num_people, args.space, args.wobble, args.seed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
