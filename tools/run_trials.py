#!/usr/bin/env python3
"""
run_trials.py -- automates "run this waypoint tour N times, once per label
in a trial config file," pairing each run with evaluation_node.py's live
/evaluation/trial_control -> /evaluation/trial_result protocol.

If --isaac-launch-cmd is given, this script ALSO owns the Isaac Sim
process lifecycle, one instance per trial: launch it, wait for its ROS 2
bridge to actually publish (polling --isaac-ready-topic, not just "the
process started"), run the trial, then terminate the whole process group
(SIGTERM, escalating to SIGKILL after --isaac-shutdown-grace-s) before the
next trial launches a fresh one. This is "close the simulation stage after
each trial" made literal -- the simplest reset strategy, chosen over an
in-place scene reset specifically because it needs no knowledge of the
scene's prim layout or spawn API, only a shell command.

Nav2 and the risk_perception stack (including evaluation_reference) are
assumed to already be running continuously ACROSS trials -- only Isaac Sim
itself cycles. If you use AMCL rather than ground-truth tf from the sim,
you likely need to (re-)publish an initial pose after each relaunch before
the first waypoint of the next trial; this script does not do that for
you, since it depends on your localization setup.

DOMAIN: every process in this pipeline -- Isaac Sim's ROS 2 bridge, the
risk_perception stack, evaluation_reference, evaluation_node, and this
script -- must share one ROS_DOMAIN_ID. ROS 2 domains are a hard partition,
not a filter: a node on the wrong domain sees nothing and raises no error.
Pass --ros-domain-id to force it for this process and the Isaac Sim
process it launches; export ROS_DOMAIN_ID in the shells running the other
launch files.

Trial config (JSON):
  {"trials": [
    {"label": "order1", "waypoints": [[x, y, yaw], [x, y, yaw], ...]},
    {"label": "order2", "waypoints": [...]}
  ]}
waypoints are map-frame (x, y, yaw_rad); a trial's waypoints are visited
in order as one tour, matching the "fixed waypoint set, multiple visiting
orders" experiment design -- decorrelating "which corridor is risky" from
"when the robot happened to be there."

Usage:
  python3 tools/run_trials.py trials.json --condition-prefix full_system \\
      --out-dir results/full_system --ros-domain-id 62 \\
      --isaac-launch-cmd "~/isaacsim/python.sh scenario.py --stage warehouse.usd --headless"
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

_OUTCOME_NAMES = {
    TaskResult.SUCCEEDED: "SUCCEEDED",
    TaskResult.CANCELED: "CANCELED",
    TaskResult.FAILED: "FAILED",
}


class TrialResultListener(Node):
    """Side node dedicated to the trial_control/trial_result exchange (and
    to polling the graph for Isaac Sim's readiness topic) -- kept separate
    from BasicNavigator's own node so the two never contend over which one
    is spinning when."""

    def __init__(self) -> None:
        super().__init__("run_trials_listener")
        self.control_pub = self.create_publisher(String, "/evaluation/trial_control", 10)
        self._latest: Optional[str] = None
        self.create_subscription(String, "/evaluation/trial_result", self._cb, 10)

    def _cb(self, msg: String) -> None:
        self._latest = msg.data

    def start_trial(self, label: str) -> None:
        self._latest = None
        self.control_pub.publish(String(data=f"start {label}"))

    def end_trial_and_wait(self, timeout_s: float = 10.0) -> Optional[dict]:
        self.control_pub.publish(String(data="end"))
        deadline = time.time() + timeout_s
        while self._latest is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest is None:
            return None
        result = json.loads(self._latest)
        self._latest = None
        return result


def launch_isaac(cmd: str) -> subprocess.Popen:
    """Start Isaac Sim in its own process group so it (and anything it
    spawns) can be torn down as a unit -- see terminate_process_group."""
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)


def terminate_process_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the whole process group, escalating to SIGKILL if it has
    not exited after grace_s. A plain proc.terminate() only signals the
    shell Popen spawned, not Isaac Sim's actual child processes underneath
    it -- this is why launch_isaac uses preexec_fn=os.setsid."""
    if proc.poll() is not None:
        return  # already exited
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=5.0)


def wait_for_topic(node: Node, topic_name: str, timeout_s: float) -> bool:
    """Poll the ROS graph for topic_name to appear -- confirms Isaac Sim's
    ROS 2 bridge is actually publishing, not just that the process
    started. Domain-aware for free: this only sees topics on whatever
    ROS_DOMAIN_ID this process itself is running under."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        names = [name for name, _ in node.get_topic_names_and_types()]
        if topic_name in names:
            return True
        rclpy.spin_once(node, timeout_sec=0.5)
    return False


def make_pose(nav: BasicNavigator, x: float, y: float, yaw: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    half = float(yaw) / 2.0
    pose.pose.orientation.z = math.sin(half)
    pose.pose.orientation.w = math.cos(half)
    return pose


def run_one_leg(nav: BasicNavigator, pose: PoseStamped, leg_timeout_s: float) -> str:
    """leg_timeout_s is a SIM-time budget, not wall-clock -- nav's own clock
    (use_sim_time forced True at construction, see main()) tracks the same
    /clock every Nav2 node here paces itself off. Isaac Sim's own docs/
    config note it does not run at 1x (nav2_sim.yaml: ~0.55-0.9x, measured
    as low as 0.36x on the Complex_Scene_ForkliftNavMesh.usd stage) -- a
    wall-clock deadline would cancel legs Nav2 was still genuinely executing,
    misreporting a slow scene as a navigation failure. See run_ablation_repeats.sh's
    /clock-vs-wall-time notes for the same class of bug elsewhere.

    Wall-clock deadline is a SECOND, independent backstop -- 2026-09-13, a
    rep hung for 2+ hours on the first leg (robot never left spawn, /odom
    frozen at (0,0)) with /clock advancing normally the whole time, because
    the sim-time deadline above never fires if Nav2 itself is stuck (the
    same intermittent lidar/costmap bug run_ablation_repeats.sh's
    check_scan_valid retries against) -- sim time ticking is not proof Nav2
    is making progress. 10x the sim budget at a pessimistic 0.2x realtime
    floor is generous for a genuinely slow-but-working leg, short enough to
    not burn the whole night on one stuck leg."""
    nav.goToPose(pose)
    deadline = nav.get_clock().now() + Duration(seconds=leg_timeout_s)
    wall_deadline = time.time() + max(leg_timeout_s / 0.2, leg_timeout_s * 2)
    while not nav.isTaskComplete():
        if nav.get_clock().now() > deadline:
            nav.cancelTask()
            return "TIMEOUT"
        if time.time() > wall_deadline:
            nav.cancelTask()
            return "TIMEOUT_WALLCLOCK_STUCK"
        time.sleep(0.2)
    return _OUTCOME_NAMES.get(nav.getResult(), "UNKNOWN")


def wait_until_nav2_active_bounded(
    nav: BasicNavigator, timeout_s: float,
    navigator: str = "bt_navigator", localizer: str = "amcl",
) -> bool:
    """Bounded reimplementation of BasicNavigator.waitUntilNav2Active().

    The stock method has NO deadline anywhere in it:
    _waitForNodeToActivate loops on state_client.wait_for_service(1.0)
    forever if the service never appears, and _waitForInitialPose loops on
    self.initial_pose_received forever if AMCL never processes a pose. On
    2026-09-14 this hung a rep for 18+ minutes straight on
    "amcl/get_state service not available, waiting..." with nothing else
    to show for it -- the leg-level timeouts in run_one_leg() don't help
    here since no leg had even started yet. Returns True once
    navigator+localizer are both active, False if timeout_s elapses first
    -- the caller should abort this trial run, not retry in place;
    run_ablation_repeats.sh already treats a nonzero exit here as "this
    rep failed, clean up and move to the next one."
    """
    deadline = time.time() + timeout_s

    def wait_for_active(node_name: str) -> bool:
        service = f"{node_name}/get_state"
        client = nav.create_client(GetState, service)
        while not client.wait_for_service(timeout_sec=1.0):
            if time.time() > deadline:
                print(f"  TIMEOUT waiting for {service} to appear "
                      f"({timeout_s:.0f}s budget exhausted)")
                return False
            nav.get_logger().info(f"{service} service not available, waiting...")
        req = GetState.Request()
        while True:
            if time.time() > deadline:
                print(f"  TIMEOUT waiting for {node_name} to reach 'active' "
                      f"({timeout_s:.0f}s budget exhausted)")
                return False
            future = client.call_async(req)
            rclpy.spin_until_future_complete(nav, future, timeout_sec=2.0)
            if future.result() is not None and future.result().current_state.label == "active":
                return True
            time.sleep(0.5)

    if not wait_for_active(localizer):
        return False
    if localizer == "amcl":
        while not nav.initial_pose_received:
            if time.time() > deadline:
                print(f"  TIMEOUT waiting for initial_pose_received "
                      f"({timeout_s:.0f}s budget exhausted)")
                return False
            rclpy.spin_once(nav, timeout_sec=1.0)
    if not wait_for_active(navigator):
        return False
    nav.info("Nav2 is ready for use!")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_config", help="JSON file, see module docstring for the schema")
    ap.add_argument("--condition-prefix", required=True,
                    help="prepended to every trial's label -- keep distinct per "
                         "ablation condition so output files never collide")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--leg-timeout-s", type=float, default=180.0,
                    help="per-waypoint timeout before canceling and moving on")
    ap.add_argument("--ros-domain-id", type=int, default=None,
                    help="force ROS_DOMAIN_ID for this process and the Isaac Sim "
                         "process it launches (e.g. 55). Leave unset to use "
                         "whatever is already exported in this shell.")
    ap.add_argument("--isaac-launch-cmd", default=None,
                    help="shell command that starts Isaac Sim for ONE trial (e.g. "
                         "'~/isaacsim/python.sh scenario.py --stage warehouse.usd "
                         "--headless'). If given, this script launches it before "
                         "each trial and closes it after -- see module docstring.")
    ap.add_argument("--isaac-ready-topic", default="/tf",
                    help="topic polled before the first Nav2 goal of a trial, to "
                         "confirm Isaac Sim's bridge is actually up")
    ap.add_argument("--isaac-ready-timeout-s", type=float, default=60.0)
    ap.add_argument("--isaac-shutdown-grace-s", type=float, default=15.0)
    ap.add_argument("--reset-cmd", default=None,
                    help="extra shell command run AFTER Isaac Sim is closed (or "
                         "after each trial if --isaac-launch-cmd is not used) -- "
                         "for anything else you need between trials")
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="pause after teardown/reset before the next trial starts")
    ap.add_argument("--nav2-active-timeout-s", type=float, default=240.0,
                    help="give up on this whole rep (exit nonzero, no trials run) "
                         "if Nav2+AMCL don't reach 'active' within this many seconds "
                         "-- covers the AMCL-never-comes-up hang that leg-level "
                         "timeouts can't see since it happens before any leg starts")
    args = ap.parse_args()

    if args.ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    with open(args.trial_config) as f:
        config = json.load(f)

    # 2026-09-13: complex_forklift_no_relation/no_flow/no_encounter's WHOLE
    # dataset silently ran the wrong tour -- whatever process wrote
    # trials_x3_tour.json at the exact moment THIS script's json.load above
    # ran had 3 waypoints each appearing twice instead of 6 distinct ones
    # (trials_x3_tour.json itself, checked before and after, was always
    # correct -- something else touched it mid-run, most likely a second
    # session sharing this checkout). Never silently run a corrupted trial
    # list again: fail loudly the moment it's loaded, before Isaac/Nav2/bag
    # recording spend even one second on it.
    labels = [t["label"] for t in config["trials"]]
    if len(labels) != len(set(labels)):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise SystemExit(
            f"FATAL: {args.trial_config} has duplicate trial labels {dupes} "
            f"(full label list: {labels}) -- refusing to run. This exact "
            f"corruption silently produced a wrong-tour dataset on "
            f"2026-09-13. If something is concurrently editing this repo, "
            f"resolve that first; if duplicates are genuinely intended, "
            f"remove this check.")

    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    nav = BasicNavigator()
    # Force sim time AFTER construction (BasicNavigator's __init__ doesn't
    # expose parameter_overrides) -- rclpy's TimeSource watches this
    # parameter and subscribes to /clock the moment it flips true, so
    # nav.get_clock().now() in run_one_leg tracks the same clock every
    # nav2_sim.yaml node (use_sim_time: True everywhere) already does.
    # Without this, run_one_leg's deadline silently fell back to wall time
    # even though it looked like it was using nav's own clock.
    nav.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    listener = TrialResultListener()
    if not wait_until_nav2_active_bounded(nav, args.nav2_active_timeout_s):
        raise SystemExit(
            f"FATAL: Nav2/AMCL never reached 'active' within "
            f"{args.nav2_active_timeout_s:.0f}s -- aborting this rep with no "
            f"trials run. Not a data-corruption case: nothing in "
            f"{args.out_dir} was written, so the caller (run_ablation_repeats.sh) "
            f"sees this rep's tour_finished check fail and moves on to the next "
            f"repeat, same as any other real failure.")

    all_summaries = []
    isaac_proc: Optional[subprocess.Popen] = None
    try:
        for trial in config["trials"]:
            label = f"{args.condition_prefix}__{trial['label']}"
            print(f"\n=== trial: {label} ===")

            if args.isaac_launch_cmd:
                print(f"  launching Isaac Sim: {args.isaac_launch_cmd}")
                isaac_proc = launch_isaac(args.isaac_launch_cmd)
                if not wait_for_topic(listener, args.isaac_ready_topic,
                                      args.isaac_ready_timeout_s):
                    print(f"  WARNING: {args.isaac_ready_topic} never appeared within "
                          f"{args.isaac_ready_timeout_s}s -- running the trial anyway, "
                          "it will likely fail")

            listener.start_trial(label)

            outcomes = []
            for wp in trial["waypoints"]:
                pose = make_pose(nav, wp[0], wp[1], wp[2] if len(wp) > 2 else 0.0)
                outcome = run_one_leg(nav, pose, args.leg_timeout_s)
                outcomes.append(outcome)
                print(f"  leg -> {outcome}")
                if outcome != "SUCCEEDED":
                    print(f"  leg failed ({outcome}) -- ending trial early")
                    break

            summary = listener.end_trial_and_wait()
            if summary is None:
                print(f"  WARNING: no trial_result received for {label} within timeout")
                summary = {"label": label, "error": "no_result"}
            summary["leg_outcomes"] = outcomes
            all_summaries.append(summary)

            out_path = os.path.join(args.out_dir, f"{label}.json")
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  wrote {out_path}")

            if isaac_proc is not None:
                print("  closing Isaac Sim")
                terminate_process_group(isaac_proc, args.isaac_shutdown_grace_s)
                isaac_proc = None

            if args.reset_cmd:
                print(f"  extra reset step: {args.reset_cmd}")
                subprocess.run(args.reset_cmd, shell=True, check=False)
            time.sleep(args.settle_s)
    finally:
        if isaac_proc is not None:
            terminate_process_group(isaac_proc, args.isaac_shutdown_grace_s)
        combined_path = os.path.join(args.out_dir, "_all_trials.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n{len(all_summaries)} trial(s) done -> {combined_path}")
        listener.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
