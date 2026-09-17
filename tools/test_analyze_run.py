#!/usr/bin/env python3
"""WP6 validation: no real bag survives under results/ (bags are kept in the
session scratchpad only -- see results/*/SUMMARY.md and
`find results -name '*.db3' -o -name '*.mcap'`), so this builds a small
synthetic rosbag2 bag with known ground truth and asserts analyze_run.py's
metrics against hand-computed expected values.

Run (ROS sourced): `pytest tools/test_analyze_run.py -v`
"""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_run as ar  # noqa: E402

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Twist, TransformStamped
from nav2_msgs.msg import CollisionMonitorState
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from panoptex_msgs.msg import RiskStack

SPAWN = ar.DEFAULT_SPAWN


def _time(t: float) -> TimeMsg:
    sec = int(t)
    nanosec = int(round((t - sec) * 1e9))
    return TimeMsg(sec=sec, nanosec=nanosec)


class BagBuilder:
    """Thin wrapper around rosbag2_py's writer -- create_topic once per
    topic, then write(topic, msg, t_ns)."""

    def __init__(self, path: Path):
        self.writer = SequentialWriter()
        self.writer.open(StorageOptions(uri=str(path), storage_id="sqlite3"),
                          ConverterOptions("", ""))
        self._known = set()

    def write(self, topic: str, type_str: str, msg, t_ns: int):
        if topic not in self._known:
            self.writer.create_topic(TopicMetadata(
                name=topic, type=type_str, serialization_format="cdr"))
            self._known.add(topic)
        self.writer.write(topic, serialize_message(msg), t_ns)


@pytest.fixture(scope="module")
def bag_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("wp6_bag")
    bag_path = root / "bag"
    b = BagBuilder(bag_path)

    n = 40
    dt = 0.1
    x3_map_x, x3_map_y = 1.0, 0.0  # X3 stays put; carter1 sweeps through it

    for i in range(n):
        t = i * dt
        stamp = _time(t)

        # X3 odom: constant map position x3_map -> stored as offset from spawn
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.pose.pose.position.x = x3_map_x - SPAWN[0]
        odom.pose.pose.position.y = x3_map_y - SPAWN[1]
        b.write("/odom", "nav_msgs/msg/Odometry", odom, i * 100_000_000)

        # carter1: sweeps y from 3.0 -> 0.0 -> 3.0 through the X3's position,
        # closest approach (gap 0) exactly at t=2.0s (i=20).
        if t <= 2.0:
            c1y = 3.0 - 1.5 * t
        else:
            c1y = 1.5 * (t - 2.0)
        c2y = 10.0  # carter2 stays far away the whole run

        tfm = TFMessage()
        for name, y in (("carter1", c1y), ("carter2", c2y)):
            tr = TransformStamped()
            tr.header.stamp = stamp
            tr.header.frame_id = "world"
            tr.child_frame_id = name
            tr.transform.translation.x = x3_map_x
            tr.transform.translation.y = y
            tr.transform.rotation.w = 1.0
            tfm.transforms.append(tr)
        b.write("/gt_tf", "tf2_msgs/msg/TFMessage", tfm, i * 100_000_000)

        # world_objects: a "cart" (category wheeled) track 0.05 m off
        # carter1's true position, plus a "table" (furniture, filtered out)
        # sitting exactly ON carter1's true position -- if the category
        # filter were broken the offset stats would come out ~0 instead of
        # ~0.05.
        wo = Detection3DArray()
        wo.header.stamp = stamp
        wo.header.frame_id = "map"
        cart_det = Detection3D()
        cart_det.id = "carter1_track"
        cart_det.bbox.center.position.x = x3_map_x
        cart_det.bbox.center.position.y = c1y + 0.05
        r = ObjectHypothesisWithPose()
        r.hypothesis.class_id = "cart|pmov=0.80|pmot=0.90|vx=0.0|vy=-1.50|relbonus=0.00"
        r.hypothesis.score = 0.9
        cart_det.results.append(r)
        wo.detections.append(cart_det)

        table_det = Detection3D()
        table_det.id = "furniture_track"
        table_det.bbox.center.position.x = x3_map_x
        table_det.bbox.center.position.y = c1y
        rt = ObjectHypothesisWithPose()
        rt.hypothesis.class_id = "table|pmov=0.10|pmot=0.00|vx=0.0|vy=0.0|relbonus=0.00"
        rt.hypothesis.score = 0.7
        table_det.results.append(rt)
        wo.detections.append(table_det)

        # A "lidar_cluster" (unknown category) track 0.1 m off carter2's
        # true position (carter2 stays put at c2y=10.0 the whole run, never
        # claimed by any robot/wheeled detection) -- must count toward
        # tracked_incl_lidar_pct but NOT toward the strict tracked_pct.
        lidar_det = Detection3D()
        lidar_det.id = "carter2_lidar_track"
        lidar_det.bbox.center.position.x = x3_map_x
        lidar_det.bbox.center.position.y = c2y + 0.1
        rl = ObjectHypothesisWithPose()
        rl.hypothesis.class_id = "lidar_cluster|pmov=0.30|pmot=0.10|vx=0.0|vy=0.0|relbonus=0.00"
        rl.hypothesis.score = 0.5
        lidar_det.results.append(rl)
        wo.detections.append(lidar_det)

        b.write("/risk_perception/world_objects", "vision_msgs/msg/Detection3DArray",
                wo, i * 100_000_000)

    # mission/state: 2 completed laps of 20 s each, one 1.5 s hold, one 1.5 s refuge
    mission_rows = [
        (0.0, "navigating", 0, 0),
        (5.0, "holding", 0, 1),
        (6.5, "holding", 0, 1),
        (8.0, "resuming", 0, 1),
        (8.1, "navigating", 0, 1),
        (20.0, "navigating", 1, 1),
        (25.0, "refuge", 1, 2),
        (26.5, "refuge", 1, 2),
        (27.0, "resuming", 1, 2),
        (27.1, "navigating", 1, 2),
        (40.0, "navigating", 2, 2),
    ]
    for i, (t, state, lap, yc) in enumerate(mission_rows):
        msg = String()
        msg.data = json.dumps({"t": t, "waypoint": 0, "name": "wp_001", "lap": lap,
                                "state": state, "yield_count": yc, "yield": None,
                                "goal_xy": [0.0, 0.0]})
        b.write("/mission/state", "std_msgs/msg/String", msg, i * 1_000_000_000)

    # cmd_vel: 1.0 s moving, 1.5 s stopped (one stop episode), 1.0 s moving
    speeds = [0.3] * 10 + [0.0] * 15 + [0.3] * 10
    for i, s in enumerate(speeds):
        tw = Twist()
        tw.linear.x = s
        b.write("/cmd_vel", "geometry_msgs/msg/Twist", tw, i * 100_000_000)

    # /clock: two samples giving an exact RTF of 2.0 (sim runs 2x wall)
    c0 = Clock(); c0.clock = TimeMsg(sec=1000, nanosec=0)
    b.write("/clock", "rosgraph_msgs/msg/Clock", c0, 0)
    c1 = Clock(); c1.clock = TimeMsg(sec=1020, nanosec=0)
    b.write("/clock", "rosgraph_msgs/msg/Clock", c1, 10_000_000_000)

    # /risk_stack: one 5x5 grid, 6/25 cells >= 45 -> lethal-area% = 24.0 exactly
    rs = RiskStack()
    rs.header.stamp = _time(0.0)
    rs.header.frame_id = "map"
    rs.info.resolution = 1.0
    rs.info.width = 5
    rs.info.height = 5
    rs.dt = 0.3
    rs.steps = 1
    data = [0] * 25
    for idx in (0, 1, 2, 3, 4, 5):
        data[idx] = 50
    rs.data = data
    b.write("/risk_stack", "panoptex_msgs/msg/RiskStack", rs, 0)

    # /risk_stack_srm (WP-C): two 2-layer 5x5 messages, exercising both the
    # layer0-vs-last-layer split and p50/p90 aggregation across messages.
    #   msg1: layer0 10/25 >=45 (40%), layer1(last) 5/25 >=45 (20%)
    #   msg2: layer0 15/25 >=45 (60%), layer1(last) 0/25 >=45 (0%)
    # -> layer0 p50/p90 = 50.0/58.0, last-layer p50/p90 = 10.0/18.0
    # (percentile() linear-interpolates over the 2-message sorted series --
    # see test_srm_area_metrics's hand computation below).
    def _srm_msg(t, n_layer0, n_layer1):
        m = RiskStack()
        m.header.stamp = _time(t)
        m.header.frame_id = "map"
        m.info.resolution = 1.0
        m.info.width = 5
        m.info.height = 5
        m.dt = 0.3
        m.steps = 2
        layer0 = [0] * 25
        for idx in range(n_layer0):
            layer0[idx] = 50
        layer1 = [0] * 25
        for idx in range(n_layer1):
            layer1[idx] = 50
        m.data = layer0 + layer1
        return m

    b.write("/risk_stack_srm", "panoptex_msgs/msg/RiskStack", _srm_msg(0.0, 10, 5), 0)
    b.write("/risk_stack_srm", "panoptex_msgs/msg/RiskStack", _srm_msg(1.0, 15, 0),
            1_000_000_000)

    # /collision_monitor_state: 2 STOP events + 1 DO_NOTHING (must not count)
    for action, poly, t_ns in ((CollisionMonitorState.DO_NOTHING, "", 0),
                                (CollisionMonitorState.STOP, "stop_zone", 1_000_000_000),
                                (CollisionMonitorState.STOP, "stop_zone", 2_000_000_000)):
        cm = CollisionMonitorState()
        cm.action_type = action
        cm.polygon_name = poly
        b.write("/collision_monitor_state", "nav2_msgs/msg/CollisionMonitorState", cm, t_ns)

    del b.writer  # flush/close
    return root


@pytest.fixture(scope="module")
def run_dir(bag_dir):
    log_text = (
        "[controller_server-1] [ERROR] [1.0] [DWBLocalPlanner]: No valid trajectories out of 5! \n"
        "[controller_server-1] [ERROR] [1.1] [DWBLocalPlanner]: No valid trajectories out of 5! \n"
        "[controller_server-1] [ERROR] [1.2] [DWBLocalPlanner]: No valid trajectories out of 5! \n"
        "[planner_server-1] [ERROR] [1.3]: failed to create plan\n"
        "[controller_server-1] [ERROR] [1.4]: MPPI critic reported a failure\n"
        "[controller_server-1] [ERROR] [1.5]: MPPI rollout failed\n"
        "[mission_supervisor-1] [WARN] [1.6]: waypoint 0 (wp_001) aborted (status 6) -- "
        "counted as missed, continuing\n"
        "[controller_server-1] [ERROR] [1.7]: Failed to compute a path\n"
    )
    (bag_dir / "x3_nav_excerpt.log").write_text(log_text)
    return bag_dir


def _args(run_dir: Path, **overrides) -> argparse.Namespace:
    base = ar.parse_args([str(run_dir)])
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_bag_discovery_and_full_run(run_dir):
    metrics = ar.build_metrics(_args(run_dir), run_dir)

    assert metrics["rtf"] == pytest.approx(2.0, abs=1e-6)
    assert metrics["rtf_source"] == "bag /clock"

    c1 = metrics["per_carter"]["carter1"]
    assert c1["min_gt_gap_m"] == pytest.approx(0.0, abs=0.02)
    assert c1["n_contact_episodes"] == 1
    assert c1["n_collision_episodes"] == 1
    assert 0.3 < c1["contact_time_s"] < 0.9
    assert c1["collision_episodes"][0]["dur_s"] >= c1["contact_episodes"][0]["dur_s"]

    c2 = metrics["per_carter"]["carter2"]
    assert c2["min_gt_gap_m"] == pytest.approx(10.0, abs=0.05)
    assert c2["n_contact_episodes"] == 0
    assert c2["n_collision_episodes"] == 0

    ta1 = c1["track_accuracy"]
    assert ta1["n_gt_samples"] == 40
    # the "table" (furniture) detection sits exactly on carter1's GT position
    # (offset 0) but must be excluded by the robot/wheeled category filter --
    # if it leaked in, offset_p50 would read ~0 instead of ~0.05.
    assert ta1["offset_p50_m"] == pytest.approx(0.05, abs=1e-3)
    assert ta1["offset_p90_m"] == pytest.approx(0.05, abs=1e-3)
    assert ta1["tracked_pct"] == pytest.approx(100.0, abs=0.1)
    assert ta1["distinct_track_ids"] == 1
    assert ta1["pmot_gt0.5_while_moving_pct"] >= 90.0
    # the synthetic lidar_cluster track sits near carter2, nowhere near
    # carter1 -- carter1's inclusive metric must equal its strict one.
    assert ta1["tracked_incl_lidar_pct"] == pytest.approx(100.0, abs=0.1)
    assert ta1["distinct_track_ids_incl_lidar"] == 1

    ta2 = c2["track_accuracy"]
    assert ta2["tracked_pct"] == 0.0
    assert ta2["offset_p50_m"] is None
    assert ta2["distinct_track_ids"] == 0
    # carter2 has no robot/wheeled detection nearby (strict tracked_pct
    # stays 0), but the synthetic "lidar_cluster" track 0.1 m off its GT
    # position every frame must count toward the inclusive metric only.
    assert ta2["tracked_incl_lidar_pct"] == pytest.approx(100.0, abs=0.1)
    assert ta2["distinct_track_ids_incl_lidar"] == 1

    mission = metrics["mission"]
    assert mission["source"] == "/mission/state"
    assert mission["laps_completed"] == 2
    assert mission["per_lap_time_s"] == [
        {"lap": 0, "dur_s": 20.0}, {"lap": 1, "dur_s": 20.0}]
    assert mission["yield_count_final"] == 2
    assert mission["hold_episodes"] == 1
    assert mission["hold_time_s"] == pytest.approx(1.5, abs=1e-6)
    assert mission["refuge_episodes"] == 1
    assert mission["refuge_time_s"] == pytest.approx(1.5, abs=1e-6)

    cv = metrics["cmd_vel"]
    assert cv["topic_used"] == "/cmd_vel"
    assert cv["n_msgs"] == 35
    assert cv["activity_ratio"] == pytest.approx(20 / 35, abs=1e-3)
    assert cv["mean_speed_moving_mps"] == pytest.approx(0.3, abs=1e-6)
    # 15 stopped samples spaced 0.1 s apart span 1.4 s wall; RTF=2.0 -> 2.8 s sim
    assert cv["stop_time_s"] == pytest.approx(2.8, abs=0.05)
    assert cv["stop_episodes"] == 1

    rs = metrics["risk_stack"]
    assert rs["n_msgs"] == 1
    assert rs["lethal_area_pct_p50"] == pytest.approx(24.0, abs=1e-6)
    assert rs["lethal_area_pct_p90"] == pytest.approx(24.0, abs=1e-6)

    # WP-C: SRM area% metrics, from the real /risk_stack_srm topic (not the
    # /risk_stack fallback -- see the bag fixture's _srm_msg block and its
    # hand-computed 40/60% layer0, 20/0% last-layer area fractions).
    srm = metrics["srm"]
    assert srm["topic_used"] == "/risk_stack_srm"
    assert srm["n_msgs"] == 2
    assert srm["layer0_area_pct_p50"] == pytest.approx(50.0, abs=1e-6)
    assert srm["layer0_area_pct_p90"] == pytest.approx(58.0, abs=1e-6)
    assert srm["last_layer_area_pct_p50"] == pytest.approx(10.0, abs=1e-6)
    assert srm["last_layer_area_pct_p90"] == pytest.approx(18.0, abs=1e-6)

    nav = metrics["nav_log_health"]
    assert nav["dwb_no_valid_trajectories"] == 3
    assert nav["failed_to_create_plan"] == 1
    assert nav["mppi_fail_lines"] == 2
    assert nav["mission_supervisor_waypoint_aborts"] == 1
    assert nav["controller_server_error_lines"] >= 1

    cmon = metrics["collision_monitor"]
    assert cmon["n_msgs"] == 3
    assert cmon["n_stop_or_limit_events"] == 2
    assert cmon["by_polygon"] == {"stop_zone": 2}

    md = ar.render_markdown(metrics)
    assert "carter1" in md and "carter2" in md
    # carter2 legitimately has no nearby track (n/a offset/pmot columns);
    # every run-level metric, however, has data in this synthetic bag.
    assert "| laps completed | n/a |" not in md
    assert "| RiskStack lethal-area" in md and "n/a" not in md.split("carter2")[0]
    assert "SRM area%" in md and "/risk_stack_srm" in md


def test_missing_bag_raises(tmp_path):
    empty_dir = tmp_path / "no_bag_here"
    empty_dir.mkdir()
    with pytest.raises(SystemExit):
        ar.build_metrics(_args(empty_dir), empty_dir)


def test_mission_log_fallback():
    text = "blah\nmission complete: 4 lap(s), 12 yield(s), 1 missed waypoint(s)\nblah\n"
    m = ar.mission_metrics_from_log(text)
    assert m["laps_completed"] == 4
    assert m["per_lap_time_s"] == "n/a (log fallback)"


def test_nav_log_health_missing():
    assert ar.nav_log_health(None) == {"source": "n/a (no log file found)"}


def test_percentile_basic():
    assert ar.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert ar.percentile([], 0.5) is None


# ---------------------------------------------- WP-C SRM area% metric fallback

def test_srm_metric_falls_back_to_risk_stack_when_no_srm_topic(tmp_path):
    """A bag with no /risk_stack_srm topic at all -- the SRM area% metric
    must fall back to reading /risk_stack itself (same message the
    existing RiskStack lethal-area% row already reads), not go n/a."""
    bag_path = tmp_path / "bag"
    b = BagBuilder(bag_path)
    rs = RiskStack()
    rs.header.stamp = _time(0.0)
    rs.header.frame_id = "map"
    rs.info.resolution = 1.0
    rs.info.width = 5
    rs.info.height = 5
    rs.dt = 0.3
    rs.steps = 1
    data = [0] * 25
    for idx in range(6):
        data[idx] = 50
    rs.data = data
    b.write("/risk_stack", "panoptex_msgs/msg/RiskStack", rs, 0)
    del b.writer

    metrics = ar.build_metrics(_args(tmp_path), tmp_path)
    srm = metrics["srm"]
    assert srm["topic_used"] == "/risk_stack"
    assert srm["n_msgs"] == 1
    assert srm["layer0_area_pct_p50"] == pytest.approx(24.0, abs=1e-6)
    # steps == 1 -> layer0 and the last layer are the same layer.
    assert srm["last_layer_area_pct_p50"] == pytest.approx(24.0, abs=1e-6)


def test_srm_metric_na_when_neither_topic_present(tmp_path):
    bag_path = tmp_path / "bag"
    b = BagBuilder(bag_path)
    odom = Odometry()
    odom.header.stamp = _time(0.0)
    b.write("/odom", "nav_msgs/msg/Odometry", odom, 0)
    del b.writer

    metrics = ar.build_metrics(_args(tmp_path), tmp_path)
    srm = metrics["srm"]
    assert srm["topic_used"] == "n/a"
    assert srm["n_msgs"] == 0
    assert srm["layer0_area_pct_p50"] is None
    assert srm["last_layer_area_pct_p50"] is None


# ------------------------------------------------------ WP-C crossing labels
#
# lane_crossings() is a pure function over (t, x, y) series -- no bag, no
# rosbag2_py -- so these build synthetic X3/Carter trajectories directly
# instead of a bag fixture (see that function's own docstring for the
# label semantics: contact > waited > behind/ahead in priority, "behind" =
# the Carter already passed the crossing point in its own direction of
# travel by > behind_m).

LANE_X = 1.32
HALF_W = 0.55  # band = [0.77, 1.87]


def _linspace_series(t0, t1, dt, x_of_t, y_of_t):
    n = int(round((t1 - t0) / dt)) + 1
    return [(t0 + i * dt, x_of_t(t0 + i * dt), y_of_t(t0 + i * dt)) for i in range(n)]


def test_lane_crossings_ahead():
    # X3 sweeps x from 0 -> 2 at 0.5 m/s, y fixed at 0.0 -- crosses the
    # lane centre (x=1.32) at t=2.64, band entry/exit at t=1.54/3.74.
    x3 = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)
    # carter1 moving north (vy=+1) starting well south of the crossing --
    # at t_mid=2.64 it is still at y = -5 + 2.64 = -2.36, far short of
    # crossing_y=0.0 -- "not yet at the crossing".
    carter = _linspace_series(0.0, 4.0, 0.1, lambda t: LANE_X, lambda t: -5.0 + 1.0 * t)

    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W)
    assert len(eps) == 1
    e = eps[0]
    assert e["label"] == "ahead"
    assert e["t_mid_s"] == pytest.approx(2.64, abs=0.05)
    assert e["carter_y_offset_m"] < -0.6
    assert e["carter_vy_mps"] == pytest.approx(1.0, abs=1e-6)
    assert e["min_gt_gap_m"] is not None and e["min_gt_gap_m"] > 0.45


def test_lane_crossings_behind():
    x3 = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)
    # carter1 moving north (vy=+1), but starting well NORTH of y=0 already
    # -- by t_mid=2.64 it has travelled well past the crossing point in
    # its own (northbound) direction of travel.
    carter = _linspace_series(0.0, 4.0, 0.1, lambda t: LANE_X, lambda t: 2.0 + 1.0 * t)

    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W)
    assert len(eps) == 1
    e = eps[0]
    assert e["label"] == "behind"
    assert e["carter_y_offset_m"] > 0.6
    assert e["carter_vy_mps"] == pytest.approx(1.0, abs=1e-6)


def test_lane_crossings_behind_southbound_carter():
    """Same 'already past' situation but with a SOUTHBOUND Carter (vy < 0)
    -- the sign-normalisation in lane_crossings() must still call this
    'behind', not 'ahead', since the Carter has still cleared the crossing
    in ITS OWN direction of travel."""
    x3 = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)
    # southbound: starts north, moving toward -y, already south of the
    # crossing (y=0) well before t_mid=2.64. Kept well off the X3's own
    # x range (x=10, not near lane_x) purely so its y=0 passage doesn't
    # incidentally register as contact -- this test isolates the
    # behind/ahead sign logic, not the contact check.
    carter = _linspace_series(0.0, 4.0, 0.1, lambda t: 10.0, lambda t: 2.0 - 2.0 * t)

    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W)
    assert len(eps) == 1
    e = eps[0]
    assert e["carter_vy_mps"] < 0.0
    assert e["label"] == "behind"


def test_lane_crossings_waited():
    # X3 approaches to x=0.5 (still OUTSIDE the band -- band starts at
    # x=0.77 -- but within wait_radius_m=1.5 of it), stops there for 2.5s,
    # then resumes and drives on through the band and out the far side.
    approach = _linspace_series(0.0, 0.9, 0.1, lambda t: 0.5 * t, lambda t: 0.0)  # 0 -> 0.45
    held = _linspace_series(1.0, 3.5, 0.1, lambda t: 0.5, lambda t: 0.0)  # 2.5s stopped at x=0.5
    resume = _linspace_series(3.6, 7.0, 0.1, lambda t: 0.5 + 0.5 * (t - 3.5), lambda t: 0.0)
    x3 = approach + held + resume

    # carter1 far away throughout -- no contact, isolates the waited label.
    carter = _linspace_series(0.0, 7.0, 0.1, lambda t: LANE_X, lambda t: 20.0)

    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W,
                            wait_speed_mps=0.05, wait_min_s=2.0, wait_radius_m=1.5)
    assert len(eps) == 1
    assert eps[0]["label"] == "waited"


def test_lane_crossings_contact():
    x3 = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)
    # carter1 sweeps directly through the X3's position at the crossing
    # instant -- gap goes to ~0, well under the 0.45 m contact threshold,
    # even though by timing alone this would otherwise read as "ahead"
    # (carter not yet past) -- contact must win regardless.
    carter = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)

    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W,
                            contact_thresh_m=0.45)
    assert len(eps) == 1
    e = eps[0]
    assert e["label"] == "contact"
    assert e["min_gt_gap_m"] == pytest.approx(0.0, abs=1e-6)


def test_lane_crossings_no_episode_when_never_in_band():
    x3 = _linspace_series(0.0, 2.0, 0.1, lambda t: 5.0, lambda t: 0.0)  # far outside the band
    carter = _linspace_series(0.0, 2.0, 0.1, lambda t: LANE_X, lambda t: 0.0)
    assert ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W) == []


def test_lane_crossings_approach_speed():
    # Constant 0.5 m/s approach -- mean speed over the last 3 m before
    # entry must read back as ~0.5 m/s regardless of the sampling rate.
    x3 = _linspace_series(0.0, 4.0, 0.05, lambda t: 0.5 * t, lambda t: 0.0)
    carter = _linspace_series(0.0, 4.0, 0.05, lambda t: LANE_X, lambda t: 20.0)
    eps = ar.lane_crossings(x3, carter, lane_x=LANE_X, half_width=HALF_W,
                            approach_dist_m=3.0)
    assert len(eps) == 1
    assert eps[0]["x3_mean_speed_approach_mps"] == pytest.approx(0.5, abs=0.02)


def test_lane_crossings_no_carter_data_defaults_to_ahead():
    x3 = _linspace_series(0.0, 4.0, 0.1, lambda t: 0.5 * t, lambda t: 0.0)
    eps = ar.lane_crossings(x3, [], lane_x=LANE_X, half_width=HALF_W)
    assert len(eps) == 1
    e = eps[0]
    assert e["label"] == "ahead"
    assert e["carter_xy"] is None
    assert e["carter_y_offset_m"] is None
    assert e["min_gt_gap_m"] is None


def test_series_velocity_basic():
    series = [(0.0, 0.0, 0.0), (1.0, 1.0, 2.0), (2.0, 2.0, 4.0)]
    v = ar._series_velocity(series, 0.5, tol=0.5)
    assert v == pytest.approx((1.0, 2.0))


def test_series_velocity_none_when_far_from_data():
    series = [(0.0, 0.0, 0.0), (1.0, 1.0, 2.0)]
    assert ar._series_velocity(series, 50.0, tol=0.5) is None


def test_lane_in_band():
    assert ar.lane_in_band(1.32, 1.32, 0.55) is True
    assert ar.lane_in_band(1.87, 1.32, 0.55) is True
    assert ar.lane_in_band(1.88, 1.32, 0.55) is False
    assert ar.lane_in_band(0.76, 1.32, 0.55) is False
