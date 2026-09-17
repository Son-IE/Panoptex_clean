#!/usr/bin/env python3
"""
check_sim_tracks.py -- decisive, no-driving accuracy check for the sim
overhead-camera + robot RGB-D fusion pipeline (panoptex_sim.launch.py).

The five warehouse people and the two Nova Carters are (at spawn / while
static) at known stage positions. This subscribes to
/risk_perception/world_objects (vision_msgs/Detection3DArray, map frame --
object_tracker_node's fused, confirmed output, not raw per-camera
detections) for `--seconds`, then reports, per ground-truth object, the
nearest track's label/score/distance and whether it is within `--tol`.

Track identity: object_tracker_node.py stamps each Detection3D.id with a
stable per-track string (`d.id = str(tr.id)`, see its Track class), so
"distinct tracks" below counts unique `.id` values seen across the whole
window, not unique per-message detections -- one object seen by three
cameras and fused into one track must count once.

Usage:
    conda activate panoptex
    ros2 run ...  # not installed as a console_script -- run directly:
    python3 tools/isaac_sim/check_sim_tracks.py          # no use_sim_time: the window is wall time
    python3 tools/isaac_sim/check_sim_tracks.py --seconds 30 \\
        --extra carter1:1.265:1.4315 --extra carter2:6.5625:5.0 \\
        --robot-xy 0.38:0.07

Exit 0 if every required ground-truth object (default: all of them) has a
track within `--tol` metres; exit 1 otherwise. Unwinds out of rclpy.spin()
via the `_StopRunner` exception pattern from warehouse/nav/carter_loop.py --
calling rclpy.shutdown() from inside a timer callback deadlocks on this
machine (verified there).
"""
import argparse
import json
import os
import sys

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

DEFAULT_GT = os.path.expanduser(
    "~/workspace/warehouse/carters/tests/people_positions.json")


class _StopRunner(Exception):
    """Raised from the deadline timer to unwind rclpy.spin() cleanly."""
    pass


def _parse_xy(spec, what):
    parts = spec.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{what} must be X:Y, got {spec!r}")
    return float(parts[0]), float(parts[1])


def _parse_extra(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--extra must be NAME:X:Y, got {spec!r}")
    return parts[0], (float(parts[1]), float(parts[2]))


class CheckSimTracks(Node):
    def __init__(self, args):
        super().__init__("check_sim_tracks")
        self.args = args
        self.tracks = {}  # id -> (x, y, label, score)

        with open(args.gt) as f:
            gt = {name: tuple(xy) for name, xy in json.load(f).items()}
        gt.update(dict(args.extra))
        self.gt = gt
        self.require = args.require if args.require else list(gt.keys())
        for name in self.require:
            if name not in gt:
                self.get_logger().error(f"--require {name!r} not in ground truth set")
                raise SystemExit(1)

        self.create_subscription(
            Detection3DArray, "/risk_perception/world_objects", self._cb, 10)
        self.get_logger().info(
            f"Watching /risk_perception/world_objects for {args.seconds:.0f}s "
            f"against {len(gt)} ground-truth object(s), tol={args.tol}m")
        # WALL-clock timer on purpose: under use_sim_time the node clock starts
        # at 0 and jumps to the running sim time on the first /clock message,
        # which fires a sim-clock timer immediately ("0 tracks seen" after 2 s).
        # The observation window is a wall-time budget, not a sim quantity.
        self.create_timer(args.seconds, self._finish,
                          clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        self.exit_code = 1

    def _cb(self, msg: Detection3DArray) -> None:
        for d in msg.detections:
            label, score = "object", 0.0
            if d.results:
                h = d.results[0].hypothesis
                label, score = str(h.class_id).split("|", 1)[0], float(h.score)
            p = d.bbox.center.position
            self.tracks[str(d.id)] = (p.x, p.y, label, score)

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _finish(self) -> None:
        print(f"\n{len(self.tracks)} distinct track(s) seen")

        all_ok = True
        matched_ids = set()
        for name, xy in self.gt.items():
            best_id, best = None, None
            for tid, (tx, ty, label, score) in self.tracks.items():
                d = self._dist(xy, (tx, ty))
                if best is None or d < best[0]:
                    best, best_id = (d, label, score), tid

            required = name in self.require

            if best is None:
                print(f"  {'NO TRACK':8s} {name:12s} -> no track at all")
                ok = False
            else:
                dist, label, score = best
                ok = dist <= self.args.tol
                status = "PASS" if ok else "FAIL"
                print(f"  {status:8s} {name:12s} -> {label} (score {score:.2f}) "
                      f"dist={dist:.3f}m [track {best_id}]")
                if ok:
                    matched_ids.add(best_id)

            if required:
                all_ok = all_ok and ok

        unmatched = [tid for tid in self.tracks if tid not in matched_ids]
        print(f"{len(unmatched)} track(s) with no ground-truth object within "
              f"{self.args.tol}m: {unmatched}")

        if self.args.robot_xy is not None:
            near_robot = [
                tid for tid, (tx, ty, *_r) in self.tracks.items()
                if self._dist(self.args.robot_xy, (tx, ty)) <= 0.4
            ]
            print(f"track(s) within 0.4m of the robot "
                  f"{self.args.robot_xy}: {near_robot}")

        self.exit_code = 0 if all_ok else 1
        raise _StopRunner()


def main() -> None:
    rclpy.init()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--extra", type=_parse_extra, action="append", default=[])
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--require", nargs="*", default=None)
    parser.add_argument("--robot-xy", type=lambda s: _parse_xy(s, "--robot-xy"),
                         default=None)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    node = CheckSimTracks(args)
    try:
        rclpy.spin(node)
    except _StopRunner:
        pass
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
