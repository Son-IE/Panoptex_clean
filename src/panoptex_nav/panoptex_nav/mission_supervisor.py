#!/usr/bin/env python3
"""
mission_supervisor.py  --  WP-B: the mission executor for the Panoptex /
                           baseline A-B experiment, with an optional
                           "polite yielding" layer on top.

TWO JOBS, ONE NODE, ONE SWITCH
------------------------------
With `yield_enabled:=false` this is a plain waypoint executor: load a
waypoint YAML, drive the list one `navigate_to_pose` goal at a time, loop.
Nothing else. That is the BASELINE arm of the experiment, and its behaviour
must be bit-for-bit the same code path as the Panoptex arm minus the yield
logic -- which is why both arms are this one node rather than two, and why
every yield branch below is gated behind a single `self.yield_enabled`
check instead of being scattered through the goal handling.

With `yield_enabled:=true` the same executor additionally watches the
Panoptex world model and gets out of other agents' way BEFORE Nav2's
reactive layers have to do anything: see `corridor.py`'s module docstring
for the corridor model, the gap-acceptance rule and the refuge search. This
node is the ROS half -- tracks, plan, map, TF and clock in; a small state
machine and Nav2 goals out.

WHY ONE navigate_to_pose GOAL PER WAYPOINT (not FollowWaypoints)
----------------------------------------------------------------
The yield layer has to be able to *interrupt* the current leg -- send the
robot to a hold point short of a crossing, or to a refuge off the lane --
and then resume the leg it interrupted. `FollowWaypoints` owns the whole
list inside nav2's waypoint_follower and gives no way to divert and return;
`NavigateToPose` per waypoint keeps that control here, at the cost of this
node having to do its own list/lap bookkeeping. `yahboomcar_nav`'s
`waypoint_runner.py` is the FollowWaypoints sibling of this file and the
source of both patterns copied below:

  * raw `rclpy.action.ActionClient`, NOT `nav2_simple_commander`'s
    BasicNavigator -- see waypoint_runner's docstring: waitUntilNav2Active()
    blocks on amcl unconditionally (we localise with amcl OR slam_toolbox
    depending on the arm) and followWaypoints() spins internally, which
    does not compose with a node that has its own timers and subscriptions.

  * the `_StopRunner` sentinel exception for fatal startup errors. Calling
    `rclpy.shutdown()` from inside a callback running under `rclpy.spin()`
    deadlocks on this machine -- the process hangs unkillably, ignoring
    SIGTERM (verified; see waypoint_runner._StopRunner). Shutdown happens in
    main(), after spin() has returned. Note that "mission complete" is NOT
    such an error: the node stays alive publishing `complete` on
    /mission/state so the experiment harness can see it finished, and the
    launch file owns process lifetime.

STATE MACHINE (only the yield states need `yield_enabled`)
----------------------------------------------------------
    paused  --(start_delay, waypoint_pause)-->  navigating
    navigating --(plan crosses a lane and the gap is too small)--> holding
    navigating --(robot already IN a lane, user approaching)--> refuge
    holding | refuge --(user passed, or track lost)--> resuming
    resuming --(the interrupted waypoint's goal is accepted)--> navigating
    navigating --(last waypoint of the last lap succeeded)--> complete

`holding` drives to a stand-off point `hold_back_m` short of the corridor
entry (or simply cancels the goal and stops where it is, if that stand-off
is already behind the robot). `refuge` cancels and drives to the nearest
free cell outside every corridor; if there is none it cancels and stops,
which is the honest answer in a lane too narrow to leave.

CANCEL THEN SEND (2026-09-09, mppi_panoptex_3 / mppi_panoptex_4)
----------------------------------------------------------------
Every one of those goal changes goes through `_transition_goal`, and NOTHING
else in this node may put a NavigateToPose goal on the wire while the goal it
replaces is still terminating. nav2's SimpleActionServer runs one goal at a
time and can take a freshly accepted goal down together with the goal it
replaced: mppi_panoptex_3 logged `refuge goal ended with status 6` seven
milliseconds after `state: navigating -> refuge`, immediately after the
superseded waypoint goal's late result, and the old code answered that with
"staying put until the corridor releases" -- i.e. stood in the lane while
carter1 pushed the X3 five metres north.

`_transition_goal` therefore queues the new goal, cancels the old one, and
releases the queue only when the OLD GOAL'S RESULT ARRIVES -- not when the
CancelGoal *service* replies. mppi_panoptex_4 is why: that reply means only
that nav2 accepted the cancel request, and its SimpleActionServer processes
the cancellation asynchronously in its own work loop. Releasing on it sent
refuge goal #25 into a server that had not finished halting hold goal #24;
21 ms later the pending cancel terminated #25 instead ("Aborting handle."
while halting the BT, status 6). Waiting for the terminal result means the
server is genuinely idle when the replacement arrives. `_tick`'s
`cancel_timeout_s` watchdog still force-sends the queued goal if no result
ever comes, and a result from an unexpected handle releases the queue too --
a supervisor sitting on an unsent refuge is worse than one that races.

The collateral-detection re-send (`_is_collateral`) is kept as a second line
of defence and now covers hold and refuge goals as well as waypoints.

Every transition is logged at INFO with the numbers that caused it (tta,
t_clear, the user's track id), and published immediately on
`state_topic` -- which is also published at `publish_rate_hz` regardless, so
a recorder never has to infer state from the gaps between transitions.

The published JSON (std_msgs/String) is the experiment's ground truth for
"what was the robot trying to do at time t":

    {"t": 1234.5, "waypoint": 2, "name": "wp_003", "lap": 0,
     "state": "holding", "yield_count": 1,
     "yield": {"user": "7", "kind": "hold", "tta": 10.2, "t_clear": 13.0,
               "xy": [1.32, -3.0]},
     "goal_xy": [0.62, -3.0],
     "crossing": {"zone_cov": 0.42, "zone_users": [], "mode": "blind",
                  "wait_s": 6.4, "xy": [1.32, -3.0], "v_lane": 0.6,
                  "pref_xy": [1.32, -3.0]},
     "lanes": 2, "lanes_effective": 2, "waypoint_in_lane": true}

LANE-LINE MEMORY (2026-09-09, mppi_panoptex_2)
----------------------------------------------
Every corridor user deposits the line it has actually been observed driving
along into a session-local `corridor.LaneMemory`; see that module's
"LANE-LINE MEMORY" section for what the instantaneous corridor window could
not do. This node owns the memory (one per node, never reset), feeds it from
`_build_corridors`, publishes it as a LINE_LIST MarkerArray on
`~/observed_lanes` for RViz, and reports `lanes` (how many are remembered)
plus `waypoint_in_lane` (whether the waypoint currently being driven to sits
inside `refuge_lane_clearance()` of one) on the state topic. Those keys, like
`crossing`, exist ONLY in the panoptex arm.

`lanes_effective` (2026-09-09) is how many of those `lanes` actually keep a
refuge out -- `corridor.lane_is_effective`, i.e. observed over at least
`lane_min_length_m` and from at least `lane_min_points` sightings. The two
numbers diverging is the signature of the memory fragmenting, which is what
mppi_panoptex_3 did (32 lanes over two patrol lines, every refuge pushed out
to the 4 m stage); see corridor.py's "LANE HYGIENE" section.

`waypoint_in_lane` is not telemetry alone: waypoint C of the sim triangle
(0.65, 0.5) sits 0.62 m off carter1's patrol line, so ARRIVING there is a
lane crossing. `_waypoint_gate_candidate` holds the approach until no
tracked user will sweep the waypoint inside t_yield_sec, and the
lane-line clause of `corridor.hold_in_place_is_unsafe` stops the robot
parking there once it arrives.

`crossing` (WP4) is the exposure/headway view of the crossing the yield
decision was taken on -- how much of its approach zone any sensor covers,
who is inside it, which branch of corridor.crossing_policy answered
("seen" / "gap" / "blind"), and how long the blind wait has run. It is
present ONLY in the panoptex arm: `yield_enabled:=false` publishes the same
eight keys it always did, byte for byte, because the baseline arm is the
experiment's control and its recorded stream must not change under it.

`t` is this node's clock (sim time under use_sim_time:=true, which the whole
sim stack runs with -- see global_cams_sim.yaml). `goal_xy` is the goal
CURRENTLY being pursued, so during a yield it is the hold/refuge point, not
the waypoint; `waypoint`/`name` always name the leg being executed, which is
what the interrupted goal will resume to.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from risk_perception.risk_visualization import label_category, parse_class_id

from panoptex_nav.corridor import (
    CORRIDOR_DEFAULTS,
    LANE_CLEARANCE_SLACK_M,
    NO_PASS,
    Corridor,
    LaneMemory,
    along,
    corridor_contains,
    crossing_choice,
    crossing_decision,
    crossing_policy,
    danger_corridor,
    find_refuge,
    find_wall_hug,
    hold_in_place_is_unsafe,
    in_lane_band,
    hold_point,
    lane_guards,
    lateral,
    loss_release_guard,
    make_corridor,
    make_lane_band,
    refuge_lane_clearance,
    refuge_recompute_reason,
    select_corridor_users,
    tta,
    user_passed,
)

# The learned-lane heading is a nice-to-have, and the module it lives in
# drags in cv_bridge / visualization_msgs / panoptex_msgs. Import it
# defensively so a bare install without those still runs the supervisor --
# just with heading_source stuck on "velocity" (see _sample_flow).
try:
    from risk_perception.predictive_risk_costmap_node import sample_flow_grid
except ImportError:  # pragma: no cover - only on an incomplete install
    sample_flow_grid = None


# ------------------------------------------------------------------ states
NAVIGATING = "navigating"
HOLDING = "holding"
REFUGE = "refuge"
RESUMING = "resuming"
PAUSED = "paused"
COMPLETE = "complete"

# Goal kinds -- what the currently outstanding NavigateToPose goal is FOR.
# The result of a "waypoint" goal advances the mission; the result of a
# "hold"/"refuge" goal never does.
GOAL_WAYPOINT = "waypoint"
GOAL_HOLD = "hold"
GOAL_REFUGE = "refuge"

# Collateral-cancel rule (see _is_collateral / _waypoint_failed). A goal that
# was sent in the wake of one of our OWN cancels, that nav2 never published a
# single feedback message for, and that ended this soon after being sent,
# never actually ran -- so it is re-sent rather than counted as a missed
# waypoint (or, for a hold/refuge goal, rather than being left parked in the
# lane). Bounded, so a waypoint that genuinely cannot be planned still fails
# the way it always did instead of looping forever.
#
# "In the wake of" is deliberately wider than "while _cancels_in_flight > 0":
# mppi_panoptex_4's refuge #25 was sent from the cancel RESPONSE of hold #24,
# i.e. with our own counter already back at zero, and was then killed by that
# very cancel 21 ms later -- our accounting said no cancel was outstanding
# while nav2's work loop was still acting on one. `_last_cancel_t` is the
# honest test: did we ask nav2 to cancel anything within collateral_window_s
# of sending this goal? The baseline arm never cancels, so it never matches.
#
# This is now the SECOND line of defence only: _transition_goal below makes
# sure a replacement goal is not sent until the cancel of the goal it replaces
# has been answered, so the race should not happen in the first place. It is
# kept because "should not" is not "cannot" -- the watchdog can force a send
# past an unanswered cancel, and nav2's SimpleActionServer is free to preempt
# for reasons of its own.
COLLATERAL_GRACE_SEC = 5.0          # default for the collateral_window_s param
MAX_COLLATERAL_RESENDS = 2
# A hold/refuge goal gets ONE re-send and then falls back to holding in place:
# unlike a waypoint, there is nothing to miss, and standing still is a valid
# (if worse) answer, so there is no reason to spend a second retry on it.
MAX_YIELD_RESENDS = 1
# Default for the cancel_timeout_s param: how long a queued goal waits for
# nav2 to answer the cancel of the goal it is replacing before being sent
# anyway. Long enough for a round trip, short enough that a supervisor never
# sits on a queued refuge while a Carter closes.
CANCEL_TIMEOUT_SEC = 1.0


class _Goal:
    """Bookkeeping for ONE NavigateToPose goal this node sent.

    Every send makes a record; every response/result callback carries the
    record of the goal it belongs to and is matched against `_active` BY
    IDENTITY. That is what makes "cancel the refuge goal and send the resumed
    waypoint in the same breath" safe: the cancelled goal's result arrives
    later, is recognised as belonging to a record that is no longer active,
    and is dropped -- instead of being attributed to whichever goal happens
    to be outstanding by the time it lands (which is how mppi_panoptex_2 lost
    wp_001; see _waypoint_failed).
    """

    __slots__ = ("seq", "kind", "xy", "yaw", "wp_index", "lap", "sent_t",
                 "handle", "cancelled_by_us", "after_self_cancel",
                 "cancel_deferred", "progressed", "done")

    def __init__(self, seq: int, kind: str, xy: List[float], wp_index: int,
                 lap: int, sent_t: float, after_self_cancel: bool,
                 yaw: float = 0.0):
        self.seq = seq
        self.kind = kind
        self.xy = xy
        # Kept so a collateral-killed hold/refuge goal can be re-sent exactly
        # as it was, without recomputing the geometry that chose it.
        self.yaw = float(yaw)
        self.wp_index = wp_index
        self.lap = lap
        self.sent_t = sent_t
        # True while WE have an unanswered cancel out for an earlier goal --
        # see _cancel_goal and COLLATERAL_GRACE_SEC.
        self.after_self_cancel = after_self_cancel
        self.handle = None
        self.cancelled_by_us = False
        # True when we asked to cancel this goal before the server had even
        # accepted it: the cancel is issued from _on_goal_response instead,
        # and _cancels_in_flight counts it from the moment it is asked for so
        # a queued replacement waits for that round trip too.
        self.cancel_deferred = False
        # Set by the action feedback callback: proof that nav2 actually
        # started running this goal (bt_navigator publishes feedback every BT
        # tick), as opposed to terminating it before it ever began.
        self.progressed = False
        self.done = False

    def note_progress(self) -> None:
        self.progressed = True


class _StopRunner(Exception):
    """Raised from a callback to unwind out of rclpy.spin() cleanly --
    copied verbatim from yahboomcar_nav.waypoint_runner; see this module's
    docstring and that file's for why rclpy.shutdown() must never be called
    from inside a callback on this machine."""


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class MissionSupervisor(Node):

    def __init__(self, **kwargs):
        # **kwargs is forwarded straight to rclpy's Node so a test can build
        # this node with parameter_overrides instead of a launch file; the
        # node itself never reads it. See test/test_mission_supervisor_goals.py.
        super().__init__("mission_supervisor", **kwargs)

        # ------------------------------------------------ mission params
        self.declare_parameter("waypoints_file", "")
        self.declare_parameter("loop", True)
        self.declare_parameter("laps", 0)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("start_delay_sec", 0.0)
        self.declare_parameter("server_timeout_sec", 60.0)
        self.declare_parameter("waypoint_pause_sec", 0.2)
        # nav2 rejects NavigateToPose goals while bt_navigator is still
        # configuring/activating; treating that as "missed" burned all six
        # laps in 4 s (unit2_srm_1, 2026-09-10). Retry a rejected WAYPOINT goal
        # for up to reject_retry_sec * max_reject_retries before giving up.
        self.declare_parameter("reject_retry_sec", 2.0)
        self.declare_parameter("max_reject_retries", 30)
        self.declare_parameter("yield_enabled", False)
        self.declare_parameter("state_topic", "/mission/state")

        # ------------------------------------------------- plumbing params
        self.declare_parameter("world_objects_topic",
                               "/risk_perception/world_objects")
        self.declare_parameter("plan_topic", "plan")
        self.declare_parameter("map_topic", "map")
        # The MERGED group channel, not the bare `robot` one: spatial_prior_
        # node publishes /risk_perception/spatial_flow/robot_group as the max
        # over the robot AND wheeled categories (F from the winning member),
        # and the sim's Carters are learned as `wheeled` -- GroundingDINO
        # calls them "cart"/"forklift" as often as "mobile robot", so the
        # `robot` channel is simply empty and every corridor heading fell
        # back to raw velocity. One array serves EVERY corridor user
        # regardless of its own category (see _sample_flow), which is the
        # point of a merged channel: a lane is a lane whichever label the
        # detector put on the thing driving down it. Overridable, and
        # pointing this back at `robot` reproduces the old behaviour.
        self.declare_parameter("flow_topic",
                               "/risk_perception/spatial_flow/robot_group")
        self.declare_parameter("flow_enabled", True)
        # Static lane keep-out from the learned Spatial-Prior occupancy --
        # see corridor.make_lane_band for why live corridors alone are not
        # enough to choose a refuge with.
        self.declare_parameter("lane_band_enabled", True)
        self.declare_parameter("lane_band_topic",
                               "/risk_perception/spatial_prior")
        # WP4 -- the analytic exposure map and the learned headway
        # statistic. Both are OPTIONAL inputs: with neither of them present
        # the crossing decision degrades exactly to the pre-WP4 gap
        # acceptance (see corridor.crossing_policy's coverage_grid=None
        # branch), which is also what crossing_policy_enabled:=false forces.
        self.declare_parameter("crossing_policy_enabled", True)
        self.declare_parameter("coverage_topic", "/risk_perception/coverage")
        # Merged group channel for the same reason as flow_topic above, and
        # it MUST be the same group as flow_topic: lane_speed() sizes the
        # approach zone from the flow sample and blind_crossing_ok() times it
        # from the headway sample at the same crossing cell, so the two
        # reading different populations would be a zone measured against one
        # traffic pattern and waited out against another.
        self.declare_parameter("headway_topic",
                               "/risk_perception/spatial_headway/robot_group")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("tf_timeout_sec", 0.1)
        self.declare_parameter("publish_rate_hz", 5.0)
        # OccupancyGrid values <= this count as free floor for the refuge
        # search. UNKNOWN (-1) is deliberately NOT free -- an unmapped cell
        # is not somewhere to reverse into, and the same "free = mapped and
        # clear" convention is what config/x3_sim_waypoints.yaml's clearance
        # numbers were measured with.
        self.declare_parameter("map_free_max", 25)
        # Spatial-Flow image geometry. sensor_msgs/Image carries no
        # resolution/origin (see predictive_risk_costmap_node._flow_cb), so
        # it has to be told; rows/cols come from the image itself. Defaults
        # are the sim's whole-map extent from
        # risk_perception/config/global_cams_sim.yaml's `costmap:` block.
        self.declare_parameter("flow_resolution", 0.10)
        self.declare_parameter("flow_origin_x", -10.4)
        self.declare_parameter("flow_origin_y", -12.3)
        # Consecutive positive evaluations before a yield fires. The one
        # integer knob of the corridor policy, so it is declared here rather
        # than in CORRIDOR_DEFAULTS (which is all floats).
        self.declare_parameter("confirm_ticks", 3)
        # If the computed hold point is already this close, stopping is
        # better than sending a goal to drive the last few centimetres.
        self.declare_parameter("hold_cancel_radius_m", 0.3)
        # How long a goal queued behind a cancel (see _transition_goal) waits
        # for nav2 to answer that cancel before being sent anyway. The
        # watchdog exists because a supervisor that never sends its refuge
        # because a cancel response was dropped is worse than one that sends
        # it into the same old race.
        self.declare_parameter("cancel_timeout_s", CANCEL_TIMEOUT_SEC)
        # A goal that died this soon after being sent, with no feedback, and
        # with one of our own cancels unanswered, never ran -- see
        # _is_collateral.
        self.declare_parameter("collateral_window_s", COLLATERAL_GRACE_SEC)

        for name, default in CORRIDOR_DEFAULTS.items():
            self.declare_parameter(name, default)

        gp = self.get_parameter
        self.waypoints_file = str(gp("waypoints_file").value)
        self.loop = bool(gp("loop").value)
        self.laps = int(gp("laps").value)
        self.start_index = int(gp("start_index").value)
        self.default_frame_id = str(gp("frame_id").value)
        self.start_delay_sec = float(gp("start_delay_sec").value)
        self.server_timeout_sec = float(gp("server_timeout_sec").value)
        self.waypoint_pause_sec = float(gp("waypoint_pause_sec").value)
        self.yield_enabled = bool(gp("yield_enabled").value)
        self.state_topic = str(gp("state_topic").value)

        self.world_objects_topic = str(gp("world_objects_topic").value)
        self.plan_topic = str(gp("plan_topic").value)
        self.map_topic = str(gp("map_topic").value)
        self.flow_topic = str(gp("flow_topic").value)
        self.flow_enabled = bool(gp("flow_enabled").value)
        self.lane_band_enabled = bool(gp("lane_band_enabled").value)
        self.lane_band_topic = str(gp("lane_band_topic").value)
        self.crossing_policy_enabled = bool(gp("crossing_policy_enabled").value)
        self.coverage_topic = str(gp("coverage_topic").value)
        self.headway_topic = str(gp("headway_topic").value)
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.tf_timeout_sec = float(gp("tf_timeout_sec").value)
        self.publish_rate_hz = float(gp("publish_rate_hz").value)
        self.map_free_max = int(gp("map_free_max").value)
        self.flow_resolution = float(gp("flow_resolution").value)
        self.flow_origin_x = float(gp("flow_origin_x").value)
        self.flow_origin_y = float(gp("flow_origin_y").value)
        self.confirm_ticks = int(gp("confirm_ticks").value)
        self.hold_cancel_radius_m = float(gp("hold_cancel_radius_m").value)
        self.cancel_timeout_s = float(gp("cancel_timeout_s").value)
        self.collateral_window_s = float(gp("collateral_window_s").value)

        self.params: Dict[str, float] = {
            name: float(gp(name).value) for name in CORRIDOR_DEFAULTS
        }

        # ------------------------------------------------------ waypoints
        self.waypoints = self._load_waypoints()
        if self.start_index < 0 or self.start_index >= len(self.waypoints):
            self.get_logger().error(
                f"start_index {self.start_index} out of range for "
                f"{len(self.waypoints)} waypoints")
            raise SystemExit(1)

        # -------------------------------------------------- mission state
        self._wp_index = self.start_index
        self._lap = 0
        self._missed = 0
        self._reject_retries = 0
        self._reject_timer = None
        self.reject_retry_sec = float(self.get_parameter("reject_retry_sec").value)
        self.max_reject_retries = int(self.get_parameter("max_reject_retries").value)
        self._state = PAUSED
        self._started = False
        self._pause_until: Optional[float] = None

        # ------------------------------------------------------ goal state
        # ONE RECORD PER SENT GOAL (_Goal), and `_active` is the record of
        # the goal the mission is actually pursuing. Response/result
        # callbacks close over their own record and are matched to it by
        # identity, so a goal we cancelled can never have its result
        # attributed to the goal that replaced it -- see _Goal's docstring,
        # _cancel_goal and _on_result.
        self._goal_seq = 0                      # ids for the logs only
        self._active: Optional[_Goal] = None
        self._goal_xy: Optional[List[float]] = None
        # Cancels we have asked for and the server has not answered yet. A
        # goal sent while one of these is outstanding can be terminated as
        # collateral damage of it -- see _is_collateral.
        self._cancels_in_flight = 0
        # When we last ASKED nav2 to cancel a goal. nav2 acts on a cancel
        # asynchronously, long after the CancelGoal service has replied, so
        # this -- not _cancels_in_flight -- is what _is_collateral tests
        # against. Never advanced in the baseline arm, which never cancels.
        self._last_cancel_t = -1e9
        # The goal record whose TERMINAL RESULT the queued goal is waiting
        # for (see _transition_goal / _release_queued_goal). The cancel
        # service reply is not enough: nav2 is still halting the BT then.
        self._cancel_wait: Optional[_Goal] = None
        self._resends = 0
        # Re-sends spent on the CURRENT hold/refuge goal (reset every time a
        # yield chooses a new point). Separate budget from _resends, which is
        # per waypoint.
        self._yield_resends = 0
        # The goal that is waiting for an outstanding cancel to be answered
        # before it may be sent -- see _transition_goal. At most one: a newer
        # transition simply replaces it.
        self._pending_goal: Optional[Dict] = None

        # ------------------------------------------------------ yield state
        self._yield_count = 0
        self._yield: Optional[Dict] = None      # last reason dict, or None
        self._yield_ref_xy: Optional[np.ndarray] = None
        self._yield_started_t: float = 0.0
        self._pending_kind: Optional[str] = None
        self._pending_user: Optional[str] = None
        self._pending_info: Optional[Dict] = None
        self._pending_ticks = 0
        self._last_seen: Dict[str, float] = {}
        # Loss-release guard + cooldown (see _eval_yielding / _eval_navigating).
        self._loss_clear_ticks = 0
        self._last_loss_release_t = -1e9
        self._lane_band: Optional[Dict] = None
        self._warned_no_band = False
        # ------------------------------------------ observed-lane memory
        # Session-local, never reset: the lines corridor users have actually
        # been seen driving along. `self.params` is passed by reference so a
        # live parameter change reaches it. See corridor.LaneMemory.
        self._lanes = LaneMemory(self.params)
        # Rate limit for _reevaluate_target -- the whole point of the
        # hysteresis is that a committed refuge is worth something.
        self._last_refuge_recompute_t = -1e9
        # Is the waypoint we are currently driving to itself in a lane?
        self._waypoint_in_lane = False
        self._lanes_pub_t = -1e9
        # ------------------------------------------- crossing policy (WP4)
        # `crossing` block of the published state, or None until the first
        # crossing is evaluated (and always None in the baseline arm -- see
        # _publish_state, which must stay byte-identical there).
        self._crossing: Optional[Dict] = None
        # Crossing cell -> when a corridor user's centre was last seen
        # passing it. LIVE observation, and it beats the persisted
        # last_pass_time channel outright -- see _crossing_last_pass.
        self._last_observed_pass: Dict[Tuple[int, int], float] = {}
        # Crossing cell -> when we started waiting there for a blind
        # crossing (the bounded-wait clock of corridor.blind_crossing_ok).
        self._crossing_wait_since: Dict[Tuple[int, int], float] = {}
        # First tick's clock: the floor for "how long since a pass" when we
        # have never seen one. See _crossing_last_pass.
        self._watch_since: Optional[float] = None
        self._users: List[Dict] = []
        self._coverage: Optional[Dict] = None
        self._prior_grid: Optional[Dict] = None
        self._headway: Optional[np.ndarray] = None
        self._warned_no_coverage = False

        # ------------------------------------------------------- inputs
        self._detections: Optional[Detection3DArray] = None
        self._plan_xy: Optional[np.ndarray] = None
        self._plan_stamp: float = -1.0
        self._map_free: Optional[np.ndarray] = None
        self._map_info: Optional[Dict] = None
        self._flow_array: Optional[np.ndarray] = None
        self._robot_xy: Optional[Tuple[float, float]] = None
        self._warned_tf = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        # Debug/RViz view of the remembered lane lines. Private topic, and
        # only the yield arm ever has anything to put on it.
        self.lanes_pub = self.create_publisher(
            MarkerArray, "~/observed_lanes", 1) if self.yield_enabled else None

        # Only the yield arm needs the world model / plan / map / flow.
        # Subscribing anyway in the baseline arm would put this node's
        # callback load on both arms unequally in the OTHER direction (it
        # would be equal, but pointless); more importantly, not subscribing
        # makes it unambiguous in a `ros2 topic info` that the baseline arm
        # cannot be reading Panoptex perception.
        if self.yield_enabled:
            self.create_subscription(Detection3DArray,
                                     self.world_objects_topic,
                                     self._objects_cb, 10)
            self.create_subscription(Path, self.plan_topic, self._plan_cb, 1)
            self.create_subscription(
                OccupancyGrid, self.map_topic, self._map_cb,
                QoSProfile(depth=1,
                           history=HistoryPolicy.KEEP_LAST,
                           reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
            if self.lane_band_enabled:
                # Same QoS as `map`: the prior is published once per update
                # cycle but latched, so a late subscriber still gets it.
                self.create_subscription(
                    OccupancyGrid, self.lane_band_topic, self._prior_cb,
                    QoSProfile(depth=1,
                               history=HistoryPolicy.KEEP_LAST,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL))
            if self.crossing_policy_enabled:
                # Same latched QoS as `map` and the prior: the coverage
                # node publishes at 2 Hz but transient-local, so a
                # supervisor started after the perception stack still gets
                # the current exposure map on its first tick instead of
                # spending its first crossings on the blind branch.
                self.create_subscription(
                    OccupancyGrid, self.coverage_topic, self._coverage_cb,
                    QoSProfile(depth=1,
                               history=HistoryPolicy.KEEP_LAST,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL))
                self.create_subscription(
                    Image, self.headway_topic, self._headway_cb,
                    QoSProfile(depth=1,
                               history=HistoryPolicy.KEEP_LAST,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL))
            if self.flow_enabled:
                if sample_flow_grid is None:
                    self.get_logger().warning(
                        "flow_enabled but risk_perception."
                        "predictive_risk_costmap_node could not be imported "
                        "-- corridor headings will use raw velocity only")
                else:
                    self.create_subscription(Image, self.flow_topic,
                                             self._flow_cb, 1)

        # Relative action name: namespaces cleanly under namespace:=robotN,
        # same reasoning as waypoint_runner's 'follow_waypoints'.
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 0.2
        self.timer = self.create_timer(period, self._tick)
        self._start_timer = self.create_timer(
            max(0.05, self.start_delay_sec), self._start_once)

        self.get_logger().info(
            f"mission_supervisor up: {len(self.waypoints)} waypoints from "
            f"{self.waypoints_file or '<inline>'}, loop={self.loop} "
            f"laps={self.laps} start_index={self.start_index} "
            f"yield_enabled={self.yield_enabled} -> {self.state_topic}")

    # ------------------------------------------------------------- loading

    def _load_waypoints(self) -> List[Dict]:
        """Same YAML schema as yahboomcar_nav's waypoints/*.yaml (top-level
        `frame_id` plus a `waypoints:` list of {name, x, y, yaw}, yaw in
        RADIANS) so the two packages' waypoint files are interchangeable."""
        path = self.waypoints_file
        if not path or not os.path.isfile(path):
            self.get_logger().error(
                f"waypoints_file '{path}' does not exist -- pass an absolute "
                "path to a yahboomcar_nav-format waypoint yaml")
            raise SystemExit(1)
        with open(path, "r") as handle:
            data = yaml.safe_load(handle) or {}

        file_frame = str(data.get("frame_id", self.default_frame_id))
        entries = data.get("waypoints") or []
        if not entries:
            self.get_logger().error(f"no waypoints in {path}")
            raise SystemExit(1)

        out: List[Dict] = []
        for i, entry in enumerate(entries):
            try:
                out.append({
                    "name": str(entry.get("name", f"wp_{i:03d}")),
                    "x": float(entry["x"]),
                    "y": float(entry["y"]),
                    "yaw": float(entry.get("yaw", 0.0)),
                    "frame_id": str(entry.get("frame_id", file_frame)),
                })
            except (KeyError, TypeError, ValueError) as exc:
                self.get_logger().error(
                    f"waypoint entry {i} in {path} is malformed: {exc}")
                raise SystemExit(1)
        return out

    # ----------------------------------------------------------- callbacks

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _objects_cb(self, msg: Detection3DArray) -> None:
        self._detections = msg

    def _plan_cb(self, msg: Path) -> None:
        """nav2's global plan for the CURRENT goal. Note this is the plan to
        whatever goal is outstanding -- including a hold or refuge goal --
        which is exactly what the re-evaluation in _eval_yielding wants."""
        if not msg.poses:
            self._plan_xy = None
            return
        self._plan_xy = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=float)
        self._plan_stamp = self._now()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        # True = free floor. Unknown (-1) is not free -- see map_free_max.
        self._map_free = (grid >= 0) & (grid <= self.map_free_max)
        self._map_info = {
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
        }

    def _prior_cb(self, msg: OccupancyGrid) -> None:
        """Rebuild the static lane band from /risk_perception/spatial_prior.

        Done here rather than per tick because the dilation is the expensive
        part and the prior changes on the order of seconds, not on the order
        of the 5 Hz control loop. The band is dilated by the CORRIDOR half
        width so a static lane covers the same footprint a live corridor
        would (corridor.make_lane_band explains the rest).
        """
        values = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        info = {"resolution": float(msg.info.resolution),
                "origin_x": float(msg.info.origin.position.x),
                "origin_y": float(msg.info.origin.position.y)}
        self._lane_band = make_lane_band(
            values, info,
            self.params["lane_band_min_value"],
            self.params["danger_half_width_m"])
        # The raw (undilated, unthresholded) S, for corridor.crossing_choice
        # -- "which of these crossings is the quietest place to cross".
        self._prior_grid = dict(info, values=values)
        self.get_logger().info(
            f"lane band rebuilt from {self.lane_band_topic}: "
            f"{int(self._lane_band['mask'].sum())} of "
            f"{values.size} cells ("
            f"value >= {self.params['lane_band_min_value']:.0f}, dilated "
            f"{self.params['danger_half_width_m']:.2f} m)",
            throttle_duration_sec=30.0)

    def _band(self) -> Optional[Dict]:
        """The lane band to plan refuges/holds against, or None when it is
        switched off or has not arrived. The missing case is logged ONCE and
        then behaves exactly as before the band existed -- a supervisor that
        refused to yield without a prior would be strictly worse than one
        that yields imperfectly."""
        if not self.lane_band_enabled:
            return None
        if self._lane_band is None and not self._warned_no_band:
            self._warned_no_band = True
            self.get_logger().warning(
                f"lane_band_enabled but nothing on {self.lane_band_topic} "
                "yet -- refuges and hold points fall back to live corridors "
                "only (they may land in a lane whose user is not currently "
                "tracked)")
        return self._lane_band

    def _flow_cb(self, msg: Image) -> None:
        """32FC3 (s, fx, fy) at the publisher's own grid resolution. Decoded
        with numpy rather than cv_bridge so this package needs no extra
        build dependency; the layout is the trivial one (row-major, 3
        float32 per pixel) and the encoding is asserted rather than
        guessed."""
        if msg.encoding != "32FC3":
            self.get_logger().warning(
                f"{self.flow_topic}: encoding {msg.encoding} != 32FC3, "
                "ignoring", throttle_duration_sec=10.0)
            return
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            self._flow_array = arr.reshape(msg.height, msg.width, 3)
        except ValueError as exc:
            self.get_logger().warning(
                f"{self.flow_topic}: cannot reshape to "
                f"({msg.height},{msg.width},3): {exc}",
                throttle_duration_sec=10.0)

    # ---------------------------------------------- coverage + headway (WP4)

    def _coverage_cb(self, msg: OccupancyGrid) -> None:
        """The analytic exposure map (risk_perception/coverage_mask_node):
        100 where some sensor covers the cell, 0 elsewhere. Kept raw and
        sampled by world coordinate, so it need not share this node's or
        the prior's grid geometry."""
        values = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        first = self._coverage is None
        self._coverage = {
            "values": values,
            "resolution": float(msg.info.resolution),
            "origin_x": float(msg.info.origin.position.x),
            "origin_y": float(msg.info.origin.position.y),
        }
        if first:
            covered = float((values >= self.params["coverage_cell_min"]).mean())
            self.get_logger().info(
                f"coverage map up on {self.coverage_topic}: "
                f"{100.0 * covered:.1f} % of {values.size} cells covered -- "
                "crossings whose approach zone is below "
                f"{100.0 * self.params['coverage_min']:.0f} % take the "
                "blind-crossing rule")

    def _headway_cb(self, msg: Image) -> None:
        """32FC3 (headway_mean, headway_count, last_pass_time) on the same
        geometry contract as the flow image -- see _flow_cb; sensor_msgs/
        Image carries no origin/resolution, so flow_resolution/flow_origin_*
        describe both."""
        if msg.encoding != "32FC3":
            self.get_logger().warning(
                f"{self.headway_topic}: encoding {msg.encoding} != 32FC3, "
                "ignoring", throttle_duration_sec=10.0)
            return
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            self._headway = arr.reshape(msg.height, msg.width, 3)
        except ValueError as exc:
            self.get_logger().warning(
                f"{self.headway_topic}: cannot reshape to "
                f"({msg.height},{msg.width},3): {exc}",
                throttle_duration_sec=10.0)

    def _coverage_grid(self) -> Optional[Dict]:
        """The exposure map to decide against, or None -- which makes
        crossing_policy fall straight back to plain gap acceptance. Warned
        ONCE: a supervisor that refused to cross without a coverage map
        would be strictly worse than the one that shipped before WP4."""
        if not self.crossing_policy_enabled:
            return None
        if self._coverage is None and not self._warned_no_coverage:
            self._warned_no_coverage = True
            self.get_logger().warning(
                f"crossing_policy_enabled but nothing on "
                f"{self.coverage_topic} yet -- crossings fall back to plain "
                "gap acceptance (no coverage gate, no blind-crossing rule). "
                "Is coverage_mask_node running?")
        return self._coverage

    def _sample_headway(self, x: float, y: float) -> Tuple[float, float, float]:
        """(headway_mean, headway_count, last_pass_time) at a map point --
        nearest cell, zeros/NO_PASS off the grid. Same nearest-cell rule as
        risk_perception's sample_flow_grid, open-coded so this node needs no
        import from a package it only optionally has."""
        if self._headway is None or self.flow_resolution <= 0.0:
            return 0.0, 0.0, NO_PASS
        rows, cols = self._headway.shape[0], self._headway.shape[1]
        col = int((x - self.flow_origin_x) / self.flow_resolution)
        row = int((y - self.flow_origin_y) / self.flow_resolution)
        if 0 <= row < rows and 0 <= col < cols:
            mean, count, last = self._headway[row, col]
            return float(mean), float(count), float(last)
        return 0.0, 0.0, NO_PASS

    def _crossing_key(self, xy) -> Tuple[int, int]:
        """Quantise a crossing point to a cell so the same physical crossing
        keeps its pass/wait bookkeeping across ticks, even though nav2
        republishes the plan (and therefore the exact entry point) every
        cycle. pass_radius_m is both the cell size and the pass-detection
        radius, which is the point: two crossings the robot could confuse
        are the same crossing as far as this memory is concerned."""
        cell = max(1e-3, float(self.params["pass_radius_m"]))
        return (int(math.floor(float(xy[0]) / cell)),
                int(math.floor(float(xy[1]) / cell)))

    def _note_passes(self, key: Tuple[int, int], xy, tracks: List[Dict],
                     now: float) -> None:
        """Record a LIVE pass: a corridor user whose centre is within
        pass_radius_m of the crossing point right now. This is the number
        the blind rule really wants -- the persisted last_pass_time channel
        is stamped in the clock of whichever session learned it (every sim
        session restarts near zero), so it cannot be differenced against
        this run's `now`. See _crossing_last_pass."""
        radius = float(self.params["pass_radius_m"])
        for tr in tracks:
            if math.hypot(float(tr["x"]) - float(xy[0]),
                          float(tr["y"]) - float(xy[1])) <= radius:
                self._last_observed_pass[key] = now
                break
        # Bounded memory: a 700 s run makes a handful of crossings per lap,
        # but a wandering plan could mint keys indefinitely.
        if len(self._last_observed_pass) > 64:
            oldest = min(self._last_observed_pass,
                         key=self._last_observed_pass.get)
            self._last_observed_pass.pop(oldest, None)
            self._crossing_wait_since.pop(oldest, None)

    def _crossing_last_pass(self, key: Tuple[int, int]) -> float:
        """When a user was last known to pass this crossing.

        Live observation if we have one; otherwise the moment this node
        started watching, which is an honest LOWER bound on the elapsed gap
        ("nothing has come past in the however-long we have been here").
        The persisted grid's own last_pass_time is deliberately NOT used for
        this difference -- it is a timestamp from the warm-up session's
        clock, and under sim time both sessions start near zero, so
        `now - loaded_last_pass` is a number with no meaning. The persisted
        channel is still what supplies headway_mean/headway_count, which
        ARE session-independent, and it is still published for analysis.
        """
        live = self._last_observed_pass.get(key)
        if live is not None:
            return float(live)
        if self._watch_since is None:
            return NO_PASS
        return float(self._watch_since)

    # -------------------------------------------------------- robot pose

    def _update_robot_pose(self) -> None:
        """Latest map->base_footprint. The previous pose is KEPT on a lookup
        failure (cached, not cleared): TF gaps of a few hundred ms are
        normal under sim time, and a corridor decision made against a
        200 ms-old pose is far better than no decision at all."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time(),
                timeout=Duration(seconds=self.tf_timeout_sec))
        except TransformException as exc:
            if not self._warned_tf:
                self.get_logger().warning(
                    f"no {self.map_frame} -> {self.base_frame} tf ({exc}); "
                    "yield logic idles until it appears",
                    throttle_duration_sec=5.0)
                self._warned_tf = True
            return
        self._warned_tf = False
        self._robot_xy = (float(tf.transform.translation.x),
                          float(tf.transform.translation.y))

    # ------------------------------------------------------------- tracks

    def _build_tracks(self) -> List[Dict]:
        """world_objects -> the plain dicts corridor.py works on. Same
        unpacking as risk_speed_governor._build_tracks (parse_class_id for
        the packed velocity/pmot fields, label_category for the group),
        plus `size` for logging."""
        tracks: List[Dict] = []
        if self._detections is None:
            return tracks
        now = self._now()
        for det in self._detections.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            label, kv = parse_class_id(str(hyp.class_id))
            stamp = det.header.stamp
            t = stamp.sec + stamp.nanosec * 1e-9
            tracks.append({
                "id": str(det.id),
                "label": label,
                "category": label_category(label),
                "score": float(hyp.score),
                "age_sec": 0.0 if t <= 0.0 else max(0.0, now - t),
                "x": float(det.bbox.center.position.x),
                "y": float(det.bbox.center.position.y),
                "vx": kv.get("vx", 0.0),
                "vy": kv.get("vy", 0.0),
                "pmot": kv.get("pmot", 0.0),
                "size": max(float(det.bbox.size.x), float(det.bbox.size.y)),
            })
        return tracks

    def _sample_flow(self, x: float, y: float):
        """(s, fx, fy) at a map point from the ONE subscribed flow image.

        Deliberately category-blind: flow_topic is the merged robot_group
        channel, so a `wheeled` Carter and a `robot` AMR snap their corridor
        headings -- and size their approach zones, via corridor.lane_speed --
        against the same learned lane. The tracker's label for a vehicle
        churns between the two categories within a single track's lifetime
        (see the object_tracker association_key gotcha), which is exactly
        what a per-category lookup would break on.
        """
        if self._flow_array is None or sample_flow_grid is None:
            return None
        rows, cols = self._flow_array.shape[0], self._flow_array.shape[1]
        return sample_flow_grid(self._flow_array, x, y,
                                self.flow_resolution,
                                self.flow_origin_x, self.flow_origin_y,
                                rows, cols)

    def _build_corridors(self, now: float) -> List[Corridor]:
        if self._robot_xy is None:
            return []
        users = select_corridor_users(self._build_tracks(), self._robot_xy,
                                      self.params)
        # Kept for the crossing policy: zone_users() must be asked about the
        # same filtered set (no self-reflection, no parked furniture), not
        # about the raw world model.
        self._users = users
        corridors = []
        for tr in users:
            c = make_corridor(tr, self.params,
                              self._sample_flow(tr["x"], tr["y"]))
            corridors.append(c)
            self._last_seen[str(tr.get("id", "?"))] = now
            # Every corridor user deposits its observed lane. Deliberately
            # the CORRIDOR's heading, not the raw velocity: where the
            # learned flow agreed it has already smoothed out the tracker's
            # 10-20 degrees of frame-to-frame jitter, and a line fitted from
            # jitter is a line nothing keeps out of.
            self._lanes.observe(c.user_id, (c.x, c.y), (c.ux, c.uy), now)
        self._lanes.expire(now)
        return corridors

    # ------------------------------------------------------- goal handling

    def _pose_msg(self, x: float, y: float, yaw: float,
                  frame_id: Optional[str] = None) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id or self.default_frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        _, _, qz, qw = yaw_to_quaternion(float(yaw))
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _send_goal(self, x: float, y: float, yaw: float, kind: str,
                   frame_id: Optional[str] = None) -> None:
        """Put a goal on the wire NOW. Callers that are replacing a goal
        which is still active must go through _transition_goal instead."""
        self._goal_seq += 1
        record = _Goal(seq=self._goal_seq, kind=kind,
                       xy=[float(x), float(y)], yaw=float(yaw),
                       wp_index=self._wp_index, lap=self._lap,
                       sent_t=self._now(),
                       after_self_cancel=self._cancels_in_flight > 0)
        self._active = record
        self._goal_xy = list(record.xy)

        self.get_logger().info(
            f"goal #{record.seq} ({kind}) sent to ({record.xy[0]:.2f}, "
            f"{record.xy[1]:.2f})"
            + (" while one of our cancels is still unanswered"
               if record.after_self_cancel else ""))

        goal = NavigateToPose.Goal()
        goal.pose = self._pose_msg(x, y, yaw, frame_id)
        future = self._client.send_goal_async(
            goal,
            feedback_callback=lambda _fb, g=record: g.note_progress())
        future.add_done_callback(
            lambda fut, g=record: self._on_goal_response(fut, g))

    def _transition_goal(self, x: float, y: float, yaw: float, kind: str,
                         frame_id: Optional[str] = None) -> None:
        """CANCEL THEN SEND -- the only way this node ever replaces a goal.

        nav2_util::SimpleActionServer runs one goal at a time and terminates
        the one it is running when a new one is accepted. A cancel that is
        still in flight when the replacement arrives can take the replacement
        down with it: in mppi_panoptex_3 a refuge goal was sent 7 ms after
        the waypoint goal it replaced was cancelled and came back status 6
        immediately, and because the old code only re-sent WAYPOINT goals the
        supervisor logged "staying put until the corridor releases" and stood
        in the lane while carter1 pushed the X3 five metres north.

        So: if a goal is active (or one of our cancels is still unanswered),
        the new goal is not sent. It is queued, the active goal is cancelled,
        and the queue is released by that goal's TERMINAL RESULT -- CANCELED,
        ABORTED or SUCCEEDED, whichever nav2 ends up reporting. Not by the
        CancelGoal service reply, which only says the request was accepted:
        mppi_panoptex_4 released on that reply and had refuge #25 killed
        21 ms later by the very cancel it thought it had waited out (see this
        module's docstring). `_tick`'s watchdog sends the queued goal anyway
        after cancel_timeout_s if no result ever comes.

        `goal_xy` is updated immediately regardless: the recorded state
        stream should say where the robot is being sent, not where the goal
        it is abandoning went.
        """
        self._pending_goal = {"x": float(x), "y": float(y), "yaw": float(yaw),
                              "kind": str(kind), "frame_id": frame_id,
                              "queued_t": self._now()}
        self._goal_xy = [float(x), float(y)]
        if self._active is None and self._cancels_in_flight == 0:
            # Nothing of ours is running, so there is no result to wait for;
            # clear any stale wait so the queue cannot wedge behind a record
            # whose result will never arrive.
            self._cancel_wait = None
            self._flush_pending_goal()
            return
        previous = self._active
        self._cancel_goal()
        self.get_logger().info(
            f"goal transition -> {kind} at ({float(x):.2f}, {float(y):.2f}): "
            "queued behind the terminal result of "
            + (f"goal #{previous.seq} ({previous.kind})"
               if previous is not None else "an earlier goal")
            + f" (at most {self.cancel_timeout_s:.1f} s)")
        # It may already have finished synchronously.
        self._flush_pending_goal()

    def _pending_blocked(self) -> bool:
        """Is the queued goal still waiting for nav2's action server to be
        genuinely idle? It is, while one of our goals is active, and while
        the goal the queued one replaces has not reported a terminal result.
        With no such record to wait on (nothing was active when the goal was
        queued), fall back to the cancel-response accounting."""
        if self._active is not None:
            return True
        waiting = self._cancel_wait
        if waiting is not None:
            return not waiting.done
        return self._cancels_in_flight > 0

    def _flush_pending_goal(self, force: bool = False) -> None:
        """Send the queued goal if the server is idle again (or if the
        watchdog says to send it regardless)."""
        pending = self._pending_goal
        if pending is None:
            return
        if not force and self._pending_blocked():
            return
        self._pending_goal = None
        self._cancel_wait = None
        self._send_goal(pending["x"], pending["y"], pending["yaw"],
                        pending["kind"], pending["frame_id"])

    def _release_queued_goal(self, record: _Goal,
                             status: Optional[int]) -> None:
        """A goal of ours reached a terminal result: release anything queued
        behind it.

        The queue normally waits for exactly this record. If the result comes
        from a DIFFERENT handle than the one we queued behind, release it
        anyway -- the expected result may never arrive (a rejected goal, a
        lifecycle transition), and a wedged queue is a refuge that is never
        sent, which is the failure this whole mechanism exists to prevent.
        """
        if self._pending_goal is None or self._active is not None:
            return
        expected = self._cancel_wait
        if expected is None:
            return
        which = ("as expected" if record is expected else
                 f"UNEXPECTED (we queued behind goal #{expected.seq} "
                 f"({expected.kind}), which has not reported) -- releasing "
                 "anyway rather than wedging the queue")
        pending = self._pending_goal
        self.get_logger().info(
            f"queued {pending['kind']} goal to ({pending['x']:.2f}, "
            f"{pending['y']:.2f}) released by the terminal result of goal "
            f"#{record.seq} ({record.kind}, status "
            f"{'rejected' if status is None else status}) -- {which}")
        self._cancel_wait = None
        self._flush_pending_goal(force=True)

    def _hold_in_place(self) -> None:
        """Stop where we are: cancel whatever is outstanding and report the
        robot's own position as the goal, because that is now honestly where
        it is being sent. Publishing the untouched waypoint as `goal_xy`
        while parked would make the recorded state stream claim the robot
        was still driving to it."""
        # Anything queued was queued to replace a goal we are now abandoning
        # altogether; sending it later would drive off after the yield ended.
        self._pending_goal = None
        self._cancel_goal()
        if self._robot_xy is not None:
            self._goal_xy = [float(self._robot_xy[0]),
                             float(self._robot_xy[1])]

    def _cancel_goal(self) -> None:
        """Cancel whatever is outstanding and detach it from the mission.

        The record is marked as ours-to-cancel and dropped from `_active`,
        so its late response/result callbacks are recognised by identity and
        ignored (see _on_result) rather than being credited to whatever goal
        is outstanding by the time they land. Safe to call with nothing
        outstanding.
        """
        record = self._active
        self._active = None
        if record is None:
            # Nothing of ours was running -- and in particular do NOT clear
            # an existing wait: this is the second transition inside one
            # round trip (_active already None because the FIRST transition
            # cancelled it), and the goal we are still waiting out is that
            # first transition's victim.
            return
        # Whatever is queued now waits for THIS record's terminal result.
        self._cancel_wait = record
        record.cancelled_by_us = True
        self._last_cancel_t = self._now()
        if record.handle is None:
            # Not accepted yet; _on_goal_response cancels it when the server
            # answers (it will see the record is no longer active). Counted
            # from here anyway so a queued replacement waits for that round
            # trip instead of racing the acceptance.
            record.cancel_deferred = True
            self._cancels_in_flight += 1
            return
        self._cancels_in_flight += 1
        future = record.handle.cancel_goal_async()
        future.add_done_callback(self._on_cancel_response)

    def _on_cancel_response(self, _future) -> None:
        """The CancelGoal SERVICE reply: nav2 accepted the cancel request and
        nothing more. It does NOT mean the goal is done -- the
        SimpleActionServer halts the BT and terminates the handle later, in
        its own work loop (mppi_panoptex_4). So this never releases a goal
        queued behind a record we are still waiting on; _pending_blocked
        keeps it queued until _on_result fires. The flush is still attempted
        for the case where nothing was active to wait for."""
        self._cancels_in_flight = max(0, self._cancels_in_flight - 1)
        self._flush_pending_goal()

    def _on_goal_response(self, future, record: _Goal) -> None:
        handle = future.result()
        if record is not self._active:
            # Cancelled or superseded before the server answered -- cancel it
            # so nav2 does not keep driving to a goal nobody is tracking, and
            # never let it reach the mission state machine.
            if handle is not None and handle.accepted:
                if not record.cancel_deferred:
                    self._cancels_in_flight += 1
                record.handle = handle
                # Subscribe to its result even though it is no longer ours:
                # a goal queued behind this one is waiting for exactly that
                # terminal result, not for the cancel response below.
                handle.get_result_async().add_done_callback(
                    lambda fut, g=record: self._on_result(fut, g))
                self._last_cancel_t = self._now()
                cancel = handle.cancel_goal_async()
                cancel.add_done_callback(self._on_cancel_response)
            else:
                # Rejected outright: no result will ever come for it, so it
                # must not hold a queued goal back. The count taken in
                # _cancel_goal still has to come back too.
                record.done = True
                self._release_queued_goal(record, None)
                if record.cancel_deferred:
                    self._on_cancel_response(None)
            return
        if handle is None or not handle.accepted:
            self._active = None
            record.done = True
            if (record.kind == GOAL_WAYPOINT
                    and self._reject_retries < self.max_reject_retries):
                self._reject_retries += 1
                self.get_logger().warning(
                    f"navigate_to_pose rejected the waypoint goal (nav2 not active "
                    f"yet?) -- retrying in {self.reject_retry_sec:.1f} s "
                    f"({self._reject_retries}/{self.max_reject_retries})")
                self._schedule_reject_retry()
                return
            self.get_logger().error(
                f"navigate_to_pose rejected the {record.kind} goal")
            if record.kind == GOAL_WAYPOINT:
                self._waypoint_failed(record, "was rejected by nav2")
            return
        self._reject_retries = 0

        record.handle = handle
        if record.kind == GOAL_WAYPOINT and self._state == RESUMING:
            self._set_state(NAVIGATING)
        handle.get_result_async().add_done_callback(
            lambda fut, g=record: self._on_result(fut, g))

    def _on_result(self, future, record: _Goal) -> None:
        if record.done:
            return
        record.done = True
        status = future.result().status

        if record is not self._active:
            # A goal we cancelled or superseded, reporting in late. This is
            # the mppi_panoptex_2 wp_001 case: a refuge goal terminated by
            # our own cancel (nav2 reports that as ABORTED, not CANCELED,
            # when a new goal preempts it) whose result landed after the
            # resumed waypoint goal had already been sent. It says nothing
            # about the waypoint and must never count as a missed one.
            self.get_logger().info(
                f"late result for the superseded {record.kind} goal "
                f"#{record.seq} (status {status}"
                f"{', we cancelled it' if record.cancelled_by_us else ''}) "
                "-- ignored", throttle_duration_sec=5.0)
            # It says nothing about the mission, but it DOES say nav2's
            # action server has finished with it -- which is what a goal
            # queued behind it has been waiting for.
            self._release_queued_goal(record, status)
            return
        self._active = None
        self._release_queued_goal(record, status)

        if record.kind != GOAL_WAYPOINT:
            # Arrived at (or failed to reach) a hold/refuge point. Arrival
            # parks the robot and the release test decides what next; a
            # failure is handled like a waypoint's, because a refuge that
            # never ran is the robot still standing in the lane.
            if status != GoalStatus.STATUS_SUCCEEDED:
                self._yield_goal_failed(record, status)
            return

        name = self.waypoints[record.wp_index]["name"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f"waypoint {record.wp_index} ({name}) reached")
            self._advance()
        elif status == GoalStatus.STATUS_CANCELED:
            # Not one of ours (ours are dropped above, by identity) --
            # somebody else cancelled, e.g. a lifecycle transition.
            self._waypoint_failed(record, "was cancelled externally", status)
        else:
            self._waypoint_failed(record, f"aborted (status {status})",
                                  status)

    def _near_a_cancel_of_ours(self, record: _Goal) -> bool:
        """Was this goal sent in the wake of one of OUR cancels?

        `after_self_cancel` (a cancel still unanswered at send time) is the
        strict reading and it is not enough. nav2 acts on a cancel long after
        the CancelGoal service has replied, so the goal sent FROM that reply
        -- mppi_panoptex_4's refuge #25 -- had our counter back at zero while
        nav2 was still halting the BT of the goal being cancelled. Asking
        instead "did we request a cancel within collateral_window_s of
        sending this?" covers both, and still never matches in the baseline
        arm, which never cancels anything at all.
        """
        if record.after_self_cancel:
            return True
        return record.sent_t - self._last_cancel_t <= self.collateral_window_s

    def _is_collateral(self, record: _Goal,
                       status: Optional[int] = None) -> bool:
        """Did this goal die of OUR cancel of a previous goal rather than of
        anything to do with the goal itself?

        nav2_util::SimpleActionServer terminates the goal it is running when
        a new one preempts it, and a cancel it has accepted but not yet
        processed can take a freshly accepted replacement down with it -- in
        mppi_panoptex_2 the resumed wp_001 goal came back ABORTED 6 ms after
        being sent and bt_navigator never even logged "Begin navigating" for
        it; in mppi_panoptex_3 the same thing happened to a freshly sent
        REFUGE goal, 7 ms after `navigating -> refuge`; in mppi_panoptex_4 to
        refuge #25, 21 ms after it was sent from the cancel response of the
        hold goal it replaced. Three signals have to agree: the goal went out
        in the wake of a cancel of ours (see _near_a_cancel_of_ours -- NOT
        our unanswered-cancel count, which lags nav2), nav2 never published a
        single feedback message for it (so it never ran), and it died inside
        collateral_window_s with ABORTED/CANCELED.

        Applies to every goal kind since 2026-09-09, with separate budgets:
        a waypoint may be re-sent MAX_COLLATERAL_RESENDS times before it is
        counted missed, a hold/refuge goal once before it degenerates to
        holding in place. _transition_goal should now keep this from ever
        firing; it stays as the second line of defence.
        """
        if record.progressed:
            return False
        if status is not None and status not in (GoalStatus.STATUS_ABORTED,
                                                 GoalStatus.STATUS_CANCELED):
            return False
        if self._now() - record.sent_t > self.collateral_window_s:
            return False
        if not self._near_a_cancel_of_ours(record):
            return False
        if record.kind == GOAL_WAYPOINT:
            return self._resends < MAX_COLLATERAL_RESENDS
        return self._yield_resends < MAX_YIELD_RESENDS

    def _yield_goal_failed(self, record: _Goal, status: int) -> None:
        """A hold/refuge goal ended without reaching its point.

        Until 2026-09-09 this branch only logged "staying put until the
        corridor releases" -- which, when the goal had been killed as
        collateral of our own cancel milliseconds after being sent, meant
        standing in exactly the lane the refuge existed to leave
        (mppi_panoptex_3: `refuge goal ended with status 6` 7 ms after
        `state: navigating -> refuge`, then carter1 pushed the X3 five metres
        north). So the never-ran test now applies here too: re-send once,
        and only then fall back to stopping where we are -- with `goal_xy`
        saying so, and at WARN, because that fallback is a robot parked
        wherever it happened to be when the yield started.
        """
        if self._is_collateral(record, status):
            self._yield_resends += 1
            self.get_logger().warning(
                f"{record.kind} goal #{record.seq} ended with status "
                f"{status} {self._now() - record.sent_t:.2f} s after being "
                "sent and without nav2 ever running it -- collateral damage "
                "of our own cancel, NOT a refuge we could not reach; "
                f"re-sending it (attempt {self._yield_resends} of "
                f"{MAX_YIELD_RESENDS})")
            self._send_goal(record.xy[0], record.xy[1], record.yaw,
                            record.kind)
            return
        self.get_logger().warning(
            f"{record.kind} goal #{record.seq} to ({record.xy[0]:.2f}, "
            f"{record.xy[1]:.2f}) ended with status {status} -- holding in "
            "place until the corridor releases")
        self._hold_in_place()

    def _waypoint_failed(self, record: _Goal, what: str,
                         status: Optional[int] = None) -> None:
        """A waypoint goal ended without reaching its pose: re-send it if it
        was collateral damage of our own cancel, otherwise count it missed
        and move on (the pre-existing behaviour, unchanged for the baseline
        arm, which never cancels anything)."""
        name = self.waypoints[record.wp_index]["name"]
        if self._is_collateral(record, status):
            self._resends += 1
            self.get_logger().warning(
                f"waypoint {record.wp_index} ({name}) {what} "
                f"{self._now() - record.sent_t:.2f} s after being sent and "
                "without nav2 ever running it -- collateral damage of our own "
                f"cancel, NOT a missed waypoint; re-sending it (attempt "
                f"{self._resends} of {MAX_COLLATERAL_RESENDS})")
            self._send_current_waypoint()
            return
        self.get_logger().warning(
            f"waypoint {record.wp_index} ({name}) {what} -- counted as "
            "missed, continuing")
        self._missed += 1
        self._advance()

    def _advance(self) -> None:
        """Waypoint finished (any outcome): step the list, count laps, and
        pause before the next goal."""
        # The collateral-resend budget is per waypoint, not per mission.
        self._resends = 0
        self._wp_index += 1
        if self._wp_index >= len(self.waypoints):
            self._lap += 1
            if not self.loop or (self.laps > 0 and self._lap >= self.laps):
                self._wp_index = len(self.waypoints) - 1
                self.get_logger().info(
                    f"mission complete: {self._lap} lap(s), "
                    f"{self._yield_count} yield(s), {self._missed} missed "
                    "waypoint(s)")
                self._set_state(COMPLETE)
                return
            self._wp_index = 0
        # No goal is outstanding between waypoints, so _publish_state falls
        # back to the waypoint we are about to drive to rather than
        # advertising the one that just finished.
        self._goal_xy = None
        self._pause_until = self._now() + self.waypoint_pause_sec
        self._set_state(PAUSED)

    def _schedule_reject_retry(self) -> None:
        """One-shot timer: re-send the current waypoint after reject_retry_sec."""
        if self._reject_timer is not None:
            self._reject_timer.cancel()

        def _fire() -> None:
            if self._reject_timer is not None:
                self._reject_timer.cancel()
                self._reject_timer = None
            if self._active is None:
                self._send_current_waypoint()

        self._reject_timer = self.create_timer(self.reject_retry_sec, _fire)

    def _send_current_waypoint(self) -> None:
        """Send the current waypoint straight away. Only called where no
        goal of ours is active: startup, after a waypoint finished, and the
        collateral re-send. A yield RESUMING to its waypoint has one to
        replace and goes through _resume_waypoint instead."""
        wp = self.waypoints[self._wp_index]
        self._send_goal(wp["x"], wp["y"], wp["yaw"], GOAL_WAYPOINT,
                        wp["frame_id"])

    def _resume_waypoint(self) -> None:
        """Go back to the interrupted waypoint, cancelling the hold/refuge
        goal first and waiting for that cancel to be answered -- see
        _transition_goal."""
        wp = self.waypoints[self._wp_index]
        self._transition_goal(wp["x"], wp["y"], wp["yaw"], GOAL_WAYPOINT,
                              wp["frame_id"])

    # ------------------------------------------------------------- startup

    def _start_once(self) -> None:
        self._start_timer.cancel()
        self.get_logger().info(
            f"waiting for the navigate_to_pose action server "
            f"({self.server_timeout_sec:.0f} s)...")
        if not self._client.wait_for_server(
                timeout_sec=self.server_timeout_sec):
            self.get_logger().error(
                "timed out waiting for the 'navigate_to_pose' action server. "
                "Is nav2 up and is bt_navigator in lifecycle_manager_"
                "navigation's node_names?")
            raise _StopRunner()
        self._started = True
        self._send_current_waypoint()
        self._set_state(NAVIGATING)

    # ---------------------------------------------------------------- tick

    def _tick(self) -> None:
        now = self._now()
        if self._watch_since is None:
            # "Nothing has passed since we started watching" needs a start.
            self._watch_since = now
        self._update_robot_pose()

        # WATCHDOG for _transition_goal's queue. nav2 finishing off every
        # cancelled goal is not something to bet a yield on: if the result
        # never comes, send the queued goal anyway and forget the cancel,
        # because a supervisor sitting on an unsent refuge is the failure
        # this whole mechanism exists to prevent.
        pending = self._pending_goal
        if pending is not None and \
                now - float(pending["queued_t"]) >= self.cancel_timeout_s:
            waiting = self._cancel_wait
            self.get_logger().warning(
                "no terminal result for "
                + (f"goal #{waiting.seq} ({waiting.kind})"
                   if waiting is not None else "the cancelled goal")
                + f" after {self.cancel_timeout_s:.1f} s")
            self.get_logger().info(
                f"queued {pending['kind']} goal to ({pending['x']:.2f}, "
                f"{pending['y']:.2f}) released by the cancel_timeout_s "
                "watchdog")
            self._cancels_in_flight = 0
            self._flush_pending_goal(force=True)

        if self._state == PAUSED and self._started \
                and self._pending_goal is None \
                and self._pause_until is not None and now >= self._pause_until:
            self._pause_until = None
            # Send first, then transition: _set_state publishes immediately,
            # and the transition message should already carry the new goal.
            # (Safe here, unlike in _release, because _on_goal_response only
            # special-cases the RESUMING state.)
            self._send_current_waypoint()
            self._set_state(NAVIGATING)

        if self.yield_enabled and self._started and self._state != COMPLETE:
            self._yield_tick(now)

        self._publish_state(now)

    # --------------------------------------------------------- yield logic

    def _yield_tick(self, now: float) -> None:
        if self._robot_xy is None:
            return
        corridors = self._build_corridors(now)
        # Recomputed every tick, not only while NAVIGATING, so the published
        # flag and the hold escalation agree about the waypoint we are
        # actually heading for. See _waypoint_gate_candidate.
        wp = self.waypoints[self._wp_index]
        self._waypoint_in_lane = self._lanes.in_lane(
            (float(wp["x"]), float(wp["y"])))
        self._publish_lanes(now)
        if self._state == NAVIGATING:
            self._eval_navigating(corridors, now)
        elif self._state in (HOLDING, REFUGE):
            self._eval_yielding(corridors, now)

    def _eval_navigating(self, corridors: List[Corridor], now: float) -> None:
        """Decide whether this tick is a yield tick, apply the
        confirm_ticks hysteresis, and act once it is confirmed.

        Refuge outranks hold: being IN a lane is strictly more urgent than
        being about to enter one, and a robot that is already inside a
        corridor has no gap to accept in the first place.
        """
        candidate = self._refuge_candidate(corridors)
        if candidate is None:
            candidate = self._waypoint_gate_candidate(corridors)
        if candidate is None:
            candidate = self._hold_candidate(corridors, now)

        if candidate is None:
            self._pending_kind = self._pending_user = self._pending_info = None
            self._pending_ticks = 0
            return

        if (candidate["kind"] == self._pending_kind
                and candidate["user"] == self._pending_user):
            self._pending_ticks += 1
        else:
            self._pending_kind = candidate["kind"]
            self._pending_user = candidate["user"]
            self._pending_ticks = 1
        self._pending_info = candidate

        # Hysteresis: a single noisy frame (a velocity spike, one flickering
        # detection) must not stop the mission. confirm_ticks consecutive
        # agreeing evaluations at publish_rate_hz -- 0.6 s at the shipped
        # 3 ticks / 5 Hz -- is short enough to still act in time given the
        # t_margin_sec slack baked into decide_crossing.
        if self._pending_ticks < self._required_ticks(now):
            return

        if candidate["kind"] == GOAL_REFUGE:
            self._enter_refuge(candidate, corridors, now)
        else:
            self._enter_hold(candidate, now, corridors)

    def _lane_clearance(self) -> float:
        """The hard minimum distance a refuge, a hold-in-place or the
        waypoint gate keeps from a remembered lane line.

        Derived from the geometry unless refuge_lane_clearance_m overrides
        it -- see corridor.refuge_lane_clearance. Every call site here asks
        THIS, never the raw parameter (which is 0.0 by default and means
        "derive"): mppi_panoptex_4's flat 1.0 m was wider than the 0.97 m
        aisle it had to fit in, so the only legal refuge was on the far
        side of the lane the robot was fleeing.
        """
        return refuge_lane_clearance(self.params)

    def _danger(self, c: Corridor) -> Corridor:
        """The containment view of one corridor -- see
        corridor.danger_corridor and corridor_contains' docstring for why
        containment and gap acceptance use different widths."""
        return danger_corridor(c, self.params)

    def _required_ticks(self, now: float) -> int:
        """How many agreeing evaluations a yield needs right now.

        Normally confirm_ticks. For post_loss_cooldown_sec after a yield was
        released because its track was LOST, just one: a loss-release is
        usually the tracker re-identifying the same physical Carter under a
        new id, and the robot is by definition sitting next to a lane that
        was busy a moment ago. Paying the full confirmation delay again
        there is how it ends up back in the lane before the "new" user is
        confirmed."""
        if now - self._last_loss_release_t < self.params[
                "post_loss_cooldown_sec"]:
            return 1
        return self.confirm_ticks

    def _refuge_candidate(self, corridors: List[Corridor]) -> Optional[Dict]:
        best = None
        for c in corridors:
            if not corridor_contains(self._danger(c), self._robot_xy):
                continue
            t_a = tta(c, self._robot_xy)
            if not (0.0 < t_a < self.params["t_yield_sec"]):
                continue
            if best is None or t_a < best["tta"]:
                best = {"kind": GOAL_REFUGE, "user": c.user_id, "tta": t_a,
                        "t_clear": None, "corridor": c,
                        "xy": [float(self._robot_xy[0]),
                               float(self._robot_xy[1])]}
        return best

    def _waypoint_gate_candidate(self,
                                 corridors: List[Corridor]) -> Optional[Dict]:
        """The approach gate for a waypoint that is ITSELF in a lane.

        Waypoint C of the sim triangle is (0.65, 0.5); carter1 patrols the
        line x = 1.27, and a wall at x ~ 0.3 means C and the spawn both sit
        0.6 m west of it. Reaching C is therefore not "arriving somewhere",
        it is entering the lane -- and the plan-crossing test in
        _hold_candidate does not see it, because the plan ENDS there instead
        of passing through, so there is no exit index and no crossing.

        So: whenever the current waypoint's lane-line clearance is below
        refuge_lane_clearance(), it may only be approached while no tracked
        user will sweep it within t_yield_sec. This is a gate on the
        approach, nothing more -- once the robot is there, the executor's
        own waypoint_pause_sec moves it on, and the lane-line clause of
        corridor.hold_in_place_is_unsafe keeps a yield from parking on it.
        """
        if not self._waypoint_in_lane:
            return None
        wp = self.waypoints[self._wp_index]
        target = (float(wp["x"]), float(wp["y"]))

        best = None
        for c in corridors:
            if not corridor_contains(self._danger(c), target):
                continue
            t_a = tta(c, target)
            if not (0.0 < t_a < self.params["t_yield_sec"]):
                continue
            if best is None or t_a < best["tta"]:
                best = {"kind": GOAL_HOLD, "user": c.user_id, "tta": t_a,
                        "t_clear": None, "corridor": c, "crossing": None,
                        "policy": {"mode": "waypoint_in_lane"},
                        "xy": [target[0], target[1]]}
        return best

    def _hold_candidate(self, corridors: List[Corridor],
                        now: float) -> Optional[Dict]:
        """Every crossing the plan makes of a live corridor, each put
        through the WP4 policy (coverage -> users -> timing or statistics;
        see corridor.crossing_policy), and the soonest hold among them.

        The policy can only ever ADD a hold to what decide_crossing already
        said, so this stays a superset of the pre-WP4 behaviour, and with no
        coverage map it is exactly the pre-WP4 behaviour.
        """
        if self._plan_xy is None or self._plan_xy.shape[0] < 2:
            self._crossing = None
            return None

        infos = []
        for c in corridors:
            info = crossing_decision(c, self._plan_xy, self.params)
            if info is None:
                continue
            info["corridor"] = c
            info["policy"] = self._crossing_decision(c, info, now)
            infos.append(info)
        if not infos:
            self._crossing = None
            return None

        best_info = None
        soonest = None
        for info in infos:
            if soonest is None or info["tta"] < soonest["tta"]:
                soonest = info
            if info["policy"]["decision"] != "hold":
                continue
            if best_info is None or info["tta"] < best_info["tta"]:
                best_info = info

        # "Choose where to cross" (paper): advisory only -- nothing here can
        # re-route nav2's plan, so the preference is reported, not obeyed.
        preferred = crossing_choice(infos, self._prior_grid)
        reported = best_info if best_info is not None else soonest
        self._crossing = self._crossing_report(reported, preferred)

        if best_info is None:
            return None
        c = best_info["corridor"]
        return {"kind": GOAL_HOLD, "user": c.user_id,
                "tta": best_info["tta"], "t_clear": best_info["t_clear"],
                "corridor": c, "crossing": best_info,
                "policy": best_info["policy"],
                "xy": [float(best_info["xy_entry"][0]),
                       float(best_info["xy_entry"][1])]}

    def _crossing_decision(self, c: Corridor, info: Dict,
                           now: float) -> Dict:
        """One crossing through the WP4 policy, with this node's live
        bookkeeping (observed passes, blind-wait clock) supplying the parts
        corridor.py cannot know.

        The blind-wait clock is started the first tick a crossing is held
        BLIND and cleared the moment that crossing is cleared to go, so
        blind_wait_max_s measures "how long we have been standing here
        waiting for this particular crossing", not the age of the node.
        """
        key = self._crossing_key(info["xy_entry"])
        self._note_passes(key, info["xy_entry"], self._users, now)

        coverage = self._coverage_grid()
        if coverage is None:
            return crossing_policy(info, c, self.params)

        mean, count, _persisted_last = self._sample_headway(
            float(info["xy_entry"][0]), float(info["xy_entry"][1]))
        policy = crossing_policy(
            info, c, self.params,
            coverage_grid=coverage,
            tracks=self._users,
            flow_sample=self._sample_flow(float(info["xy_entry"][0]),
                                          float(info["xy_entry"][1])),
            now=now,
            last_pass_time=self._crossing_last_pass(key),
            headway_mean=mean, headway_count=count,
            t_arrived=self._crossing_wait_since.get(key, -1.0))

        if policy["decision"] == "hold" and policy["mode"] == "blind":
            if key not in self._crossing_wait_since:
                self._crossing_wait_since[key] = now
                self.get_logger().info(
                    f"BLIND crossing at ({info['xy_entry'][0]:.2f}, "
                    f"{info['xy_entry'][1]:.2f}): only "
                    f"{100.0 * policy['zone_cov']:.0f} % of the "
                    f"{policy['v_lane']:.2f} m/s approach zone is covered "
                    f"(need {100.0 * self.params['coverage_min']:.0f} %); "
                    f"headway mean {mean:.1f} s from {int(count)} samples "
                    f"-- waiting (at most "
                    f"{self.params['blind_wait_max_s']:.0f} s)")
                policy["wait_s"] = 0.0
        elif policy["decision"] == "go":
            self._crossing_wait_since.pop(key, None)
        return policy

    def _crossing_report(self, info: Optional[Dict],
                         preferred: Optional[Dict]) -> Optional[Dict]:
        """The `crossing` block of /mission/state -- a SNAPSHOT taken the
        last time crossings were evaluated, i.e. while NAVIGATING. It is
        deliberately not refreshed during the hold it caused: what a run's
        analysis needs is the evidence the decision was made on, and the
        release is decided by the corridor geometry (user_passed /
        loss_release_guard), not by re-running this.

        `pref_xy` is crossing_choice's lowest-S crossing on the current plan
        -- telemetry for "would a route layer have picked somewhere else to
        cross", which nothing here can act on yet.
        """
        if info is None:
            return None
        policy = info["policy"]
        cov = policy["zone_cov"]
        pref = None
        if preferred is not None:
            pref = [round(float(preferred["xy_entry"][0]), 3),
                    round(float(preferred["xy_entry"][1]), 3)]
        return {
            "zone_cov": None if cov is None else round(float(cov), 3),
            "zone_users": list(policy["zone_users"]),
            "mode": policy["mode"],
            "wait_s": round(float(policy["wait_s"]), 2),
            "xy": [round(float(info["xy_entry"][0]), 3),
                   round(float(info["xy_entry"][1]), 3)],
            "v_lane": round(float(policy["v_lane"]), 3),
            "pref_xy": pref,
        }

    # ------------------------------------------------------------ entering

    def _yield_reason(self, candidate: Dict) -> Dict:
        return {"user": candidate["user"], "kind": candidate["kind"],
                "tta": round(float(candidate["tta"]), 2),
                "t_clear": (None if candidate["t_clear"] is None
                            else round(float(candidate["t_clear"]), 2)),
                "xy": [round(candidate["xy"][0], 3),
                       round(candidate["xy"][1], 3)]}

    def _enter_hold(self, candidate: Dict, now: float,
                    corridors: List[Corridor]) -> None:
        # `crossing` is None for the waypoint gate: there is no plan crossing
        # to stand back from, so the hold degenerates to "stop here" -- and
        # then escalates below if stopping here is itself in the way.
        info = candidate.get("crossing")
        target = None if info is None else hold_point(
            self._plan_xy, info["idx_entry"],
            self.params["hold_back_m"], self._band())

        # ESCALATION. A hold whose stand-off point turns out to be behind
        # the robot (or within hold_cancel_radius_m of it) degenerates into
        # "stand still" -- which is a yield only if standing still is out of
        # the way. In avoid_panoptex_2 it repeatedly was not: 8 of 21 yields
        # logged "the stand-off is already behind us", and the run's closest
        # approach (0.15 m ground truth) was the robot obediently standing
        # 0.65 m off lane 1's centre line while a Carter came down it. When
        # holding in place would leave us inside the danger corridor or the
        # lane band, step aside instead of standing there.
        stand_still = target is None or math.hypot(
            target[1][0] - self._robot_xy[0],
            target[1][1] - self._robot_xy[1]) <= self.hold_cancel_radius_m
        if stand_still and hold_in_place_is_unsafe(
                self._danger(candidate["corridor"]), self._robot_xy,
                self._band(),
                lanes=self._lanes.effective_lanes(),
                lane_clearance_m=self._lane_clearance(),
                lane_extension_m=self.params["lane_extension_m"]):
            self.get_logger().info(
                f"hold-in-place would leave us inside the danger zone "
                f"(user {candidate['user']}, tta={candidate['tta']:.1f} s, "
                f"danger half width "
                f"{self.params['danger_half_width_m']:.2f} m) "
                "-- stepping aside")
            self._enter_refuge(dict(candidate, kind=GOAL_REFUGE,
                                    xy=[float(self._robot_xy[0]),
                                        float(self._robot_xy[1])]),
                               corridors, now)
            return

        self._yield_count += 1
        self._yield = self._yield_reason(candidate)
        self._yield_ref_xy = np.asarray(
            candidate["xy"] if info is None else info["xy_entry"], dtype=float)
        self._yield_started_t = now
        self._pending_ticks = 0
        self._loss_clear_ticks = 0
        self._yield_resends = 0

        if target is None:
            self._hold_in_place()
            where = "in place (the stand-off is already behind us)"
        else:
            idx, xy = target
            dist = math.hypot(xy[0] - self._robot_xy[0],
                              xy[1] - self._robot_xy[1])
            if dist <= self.hold_cancel_radius_m:
                self._hold_in_place()
                where = f"in place ({dist:.2f} m from the stand-off point)"
            else:
                yaw = self._plan_heading(idx)
                self._transition_goal(xy[0], xy[1], yaw, GOAL_HOLD)
                where = f"at ({xy[0]:.2f}, {xy[1]:.2f}), {dist:.2f} m ahead"

        policy = candidate.get("policy") or {}
        mode = policy.get("mode", "gap")
        cov = policy.get("zone_cov")
        if mode == "waypoint_in_lane":
            why = (f"waypoint {self.waypoints[self._wp_index]['name']} is "
                   f"itself within {self._lane_clearance():.2f}"
                   f" m of a remembered lane line and user "
                   f"{candidate['user']} sweeps it in "
                   f"tta={candidate['tta']:.1f} s")
        else:
            why = (f"gap: user {candidate['user']} reaches the crossing in "
                   f"tta={candidate['tta']:.1f} s, we need "
                   f"t_clear={candidate['t_clear']:.1f} s "
                   f"(+{self.params['t_margin_sec']:.1f} s margin)")
        if mode == "blind":
            why = ("blind: the approach zone is only "
                   f"{0.0 if cov is None else 100.0 * cov:.0f} % covered and "
                   "the learned headway has not elapsed yet")
        self._set_state(HOLDING)
        self.get_logger().info(
            f"YIELD/hold #{self._yield_count} [{mode}] -- {why} -- "
            f"holding {where}")

    def _enter_refuge(self, candidate: Dict, corridors: List[Corridor],
                      now: float) -> None:
        self._yield_count += 1
        self._yield = self._yield_reason(candidate)
        self._yield_ref_xy = np.asarray(self._robot_xy, dtype=float)
        self._yield_started_t = now
        self._pending_ticks = 0
        self._loss_clear_ticks = 0
        self._yield_resends = 0

        # The search runs BEFORE anything is cancelled: _transition_goal
        # needs the active goal still there to hang the new one off, and
        # cancelling first only to find no refuge would leave the robot
        # stopped for the few milliseconds of the search either way.
        xy = self._find_refuge(corridors, candidate["corridor"])
        if xy is None:
            # Falls back to exactly the old behaviour -- stop where we are --
            # but at WARN, because if we got here from the escalation above
            # then "where we are" is known to be inside the danger zone.
            self._hold_in_place()
            self.get_logger().warning(
                f"YIELD/refuge #{self._yield_count}: user "
                f"{candidate['user']} enters our lane in "
                f"tta={candidate['tta']:.1f} s but NO refuge is reachable "
                f"(radius {self.params['refuge_radius_m']:.1f} m, clearance "
                f"{self.params['refuge_clearance_m']:.2f} m), not even a "
                "wall hug on our own side -- cancelling the goal and "
                "stopping where we are")
        else:
            yaw = math.atan2(xy[1] - self._robot_xy[1],
                             xy[0] - self._robot_xy[0])
            self._transition_goal(xy[0], xy[1], yaw, GOAL_REFUGE)
            side = "same" if (corridors and abs(lateral(
                candidate["corridor"], self._robot_xy)) > 1e-9
                and lateral(candidate["corridor"], xy)
                * lateral(candidate["corridor"], self._robot_xy) > 0) \
                else "other"
            self.get_logger().info(
                f"YIELD/refuge #{self._yield_count}: user "
                f"{candidate['user']} enters our lane in "
                f"tta={candidate['tta']:.1f} s (< t_yield "
                f"{self.params['t_yield_sec']:.1f} s) -- stepping aside to "
                f"({xy[0]:.2f}, {xy[1]:.2f}), "
                f"{math.hypot(xy[0] - self._robot_xy[0], xy[1] - self._robot_xy[1]):.2f} m, "
                f"{side} side of the lane")
        self._set_state(REFUGE)

    def _find_refuge(self, corridors: List[Corridor],
                     primary: Optional[Corridor]) -> Optional[np.ndarray]:
        """The refuge search, in up to two stages.

        Stage 1 is the shipped `refuge_radius_m` disc. Since 2026-09-09 a
        candidate must ALSO be `_lane_clearance()` clear of every remembered
        lane line, which is a hard filter with no least-bad fallback -- in
        the sim aisle that empties the 2.5 m disc outright (the wall at
        x ~ 0.3 leaves nothing legal west of carter1's line), so stage 2
        widens once to `refuge_radius_max_m` before the caller falls back to
        holding in place. Which stage answered is logged: a run that keeps
        needing stage 2 is a run whose refuge radius is wrong, and that has
        to be visible in the log rather than inferred.

        Both stages now carry the CROSSING GUARDS (corridor.lane_guards):
        no candidate on the far side of an approaching user's lane, unless
        the crossing itself fits inside that user's tta with
        refuge_cross_margin_s to spare. mppi_panoptex_4 crossed 1.9 m of
        danger band -- >= 7 s at the pessimistic crossing speed -- against a
        Carter 6.5 s out, because the far side was the only place that
        cleared the old flat 1.0 m lane clearance.

        Stage 3, when both discs come up empty, is the WALL HUG: the cell on
        OUR side that stands furthest off the traffic, below the clearance
        floor and possibly still inside the corridor, but never across it.
        Only if even that fails does the caller hold in place.
        """
        if self._map_free is None or self._map_info is None:
            self.get_logger().warning(
                f"no map on {self.map_topic} yet -- cannot search for a "
                "refuge", throttle_duration_sec=10.0)
            return None
        # primary first: find_refuge's side/perpendicular preference is
        # taken from corridors[0], and the lane we are fleeing is the one
        # whose geometry should decide which way "aside" is. primary is None
        # only when the lane MEMORY (not a live user) invalidated the last
        # target, in which case there is no side to prefer.
        ordered = [] if primary is None else [self._danger(primary)]
        ordered += [self._danger(c) for c in corridors if c is not primary]
        band = self._band()
        # EFFECTIVE lanes only (corridor.lane_is_effective): a 0.8 m stub
        # left by a track rounding a corner is not a lane, and a hall's worth
        # of them is what emptied the 2.5 m disc in mppi_panoptex_3 (32
        # remembered lanes, every refuge found at the 4.0 m stage).
        lanes = self._lanes.effective_lanes()
        # The lane lines we must not CROSS to get there: every live user's
        # own centre line, plus every remembered lane one of them is on.
        guards = lane_guards(ordered, self._robot_xy, lanes,
                             float(self.params["lane_extension_m"]))

        stages = [("refuge_radius_m", float(self.params["refuge_radius_m"]))]
        wide = float(self.params["refuge_radius_max_m"])
        if wide > stages[0][1]:
            stages.append(("refuge_radius_max_m", wide))

        xy = None
        for stage, radius in stages:
            xy = find_refuge(
                self._map_free, self._map_info, self._robot_xy,
                ordered,
                radius=radius,
                clearance=self.params["refuge_clearance_m"],
                prefer_side=primary is not None,
                lane_band=band,
                line_clearance_m=self.params["danger_half_width_m"],
                lanes=lanes,
                lane_clearance_m=self._lane_clearance(),
                lane_extension_m=self.params["lane_extension_m"],
                guards=guards,
                params=self.params)
            if xy is None:
                continue
            gap = float(self._lanes.effective_clearance(xy))
            msg = (f"refuge ({xy[0]:.2f}, {xy[1]:.2f}) found at {stage}="
                   f"{radius:.1f} m ({len(lanes)} of "
                   f"{len(self._lanes.lanes)} remembered lane(s) count, "
                   f"clearance "
                   f"{'inf' if math.isinf(gap) else f'{gap:.2f} m'})")
            # rclpy pins one severity per call site (file:line) -- calling
            # info() and warning() through the same line raised
            # "Logger severity cannot be changed between calls" and killed
            # the node mid-run (mppi_panoptex_3). Two explicit call sites.
            if stage == "refuge_radius_m":
                self.get_logger().info(msg)
            else:
                self.get_logger().warning(msg)
            break

        if xy is None:
            # WALL HUG (mppi_panoptex_4): nowhere on this side clears the
            # lane, and the far side is a head-on encounter with extra
            # steps. Stand as far off the traffic as the wall allows.
            xy = find_wall_hug(
                self._map_free, self._map_info, self._robot_xy,
                guards=guards,
                radius=float(self.params["refuge_radius_m"]),
                clearance=float(self.params["refuge_clearance_m"]),
                lanes=lanes, params=self.params,
                lane_extension_m=float(self.params["lane_extension_m"]))
            if xy is not None:
                gap = float(self._lanes.effective_clearance(xy))
                # Its own call site, at its own severity -- rclpy pins one
                # severity per line (see the two branches above).
                self.get_logger().warning(
                    f"WALL-HUG ({xy[0]:.2f}, {xy[1]:.2f}): no cell within "
                    f"{stages[-1][1]:.1f} m keeps "
                    f"{self._lane_clearance():.2f} m off every lane line "
                    "without crossing one in front of its user -- taking "
                    "the furthest-from-traffic cell on our own side "
                    f"(lane clearance "
                    f"{'inf' if math.isinf(gap) else f'{gap:.2f} m'}"
                    ") and holding there")
            return xy

        if xy is not None and in_lane_band(band, xy):
            # find_refuge's least-bad fallback fired: every candidate that
            # cleared the live corridors was still inside the static lane
            # band. Worth stepping to, but the run analysis needs to be able
            # to count these, hence WARN rather than INFO.
            self.get_logger().warning(
                f"refuge ({xy[0]:.2f}, {xy[1]:.2f}) is itself inside the "
                "lane band -- no fully clear cell within "
                f"{self.params['refuge_radius_m']:.1f} m; taking the least "
                "bad")
        return xy

    def _plan_heading(self, idx: int) -> float:
        """Heading of the plan at index idx -- the direction the robot is
        travelling there, so a hold point does not spin the robot round on
        arrival only to spin back when it resumes."""
        plan = self._plan_xy
        if plan is None or plan.shape[0] < 2:
            return self.waypoints[self._wp_index]["yaw"]
        j = min(max(idx, 0), plan.shape[0] - 2)
        return math.atan2(plan[j + 1][1] - plan[j][1],
                          plan[j + 1][0] - plan[j][0])

    # ------------------------------------------------------------ releasing

    def _eval_yielding(self, corridors: List[Corridor], now: float) -> None:
        """While HOLDING/REFUGE: watch for the release, and keep the
        hold/refuge point itself legal in the meantime."""
        assert self._yield is not None
        user = self._yield["user"]
        current = next((c for c in corridors if c.user_id == user), None)

        if current is None:
            # _last_seen is stamped only for tracks that are still CORRIDOR
            # USERS (see _build_corridors), so a user that parks in the lane
            # also ages out here. That is intended: a stationary agent owns
            # no corridor at all, and a stationary obstacle is Nav2's
            # costmap's problem, not this layer's.
            last = self._last_seen.get(user, self._yield_started_t)
            if now - last > self.params["lost_timeout_sec"]:
                self._release_on_loss(corridors, now, user, now - last)
            return
        self._loss_clear_ticks = 0

        # Hysteresis floor: never release inside min_hold_sec of committing.
        # Without it a user whose velocity estimate flickers can pass the
        # release test on the very tick after the hold started, producing a
        # stop-go-stop stutter directly in front of it.
        if now - self._yield_started_t < self.params["min_hold_sec"]:
            return

        if self._yield_ref_xy is not None and \
                user_passed(current, self._yield_ref_xy, self.params):
            behind = -float(along(current, self._yield_ref_xy))
            self._release(f"user {user} passed (the point we yielded for is "
                          f"{behind:.2f} m behind it, > half_width+"
                          f"{self.params['release_margin_m']:.2f} m)",
                          corridors, now)
            return

        # The user is still coming: is where we are waiting still a good
        # place? A refuge computed 3 s ago can be inside a corridor now (the
        # user turned, or a second agent appeared).
        self._reevaluate_target(corridors, now)

    def _release_on_loss(self, corridors: List[Corridor], now: float,
                         user: str, gone_for: float) -> None:
        """The track we were yielding for has vanished. Resume ONLY if the
        geometry says the point we yielded for is actually clear.

        A vanished track is not evidence of a clear lane: in
        avoid_panoptex_1 (2026-09-09) most releases were "track lost for
        3.2 s" and the robot drove straight back into the lane, because the
        SAME physical Carter had simply been re-identified (one vehicle,
        six ids: 27, 8, 28, 19, 60, 14). So the guard ignores identity
        entirely and asks the only question that matters: is ANY tracked
        user still sweeping the reference point within t_yield? If one is,
        the yield is re-bound to it and continues; if not, the all-clear
        must hold for confirm_ticks consecutive ticks -- one empty tick of
        the world model is exactly the kind of dropout that caused the
        problem -- before the mission resumes.
        """
        ref = self._yield_ref_xy if self._yield_ref_xy is not None \
            else np.asarray(self._robot_xy, dtype=float)
        blocker = loss_release_guard(corridors, ref, self.params)

        if blocker is not None:
            self._loss_clear_ticks = 0
            if self._yield is not None and blocker.user_id != user:
                self.get_logger().info(
                    f"track {user} lost for {gone_for:.1f} s, but user "
                    f"{blocker.user_id} is still sweeping the point we "
                    f"yielded for in tta={tta(blocker, ref):.1f} s -- "
                    "re-binding the yield to it instead of resuming")
                self._yield["user"] = blocker.user_id
                self._yield["tta"] = round(float(tta(blocker, ref)), 2)
                self._publish_state(now)
            self._reevaluate_target(corridors, now)
            return

        self._loss_clear_ticks += 1
        if self._loss_clear_ticks < self.confirm_ticks:
            return
        self._loss_clear_ticks = 0
        self._last_loss_release_t = now
        self._release(
            f"track {user} lost for {gone_for:.1f} s and no tracked user is "
            f"approaching the point we yielded for "
            f"({self.confirm_ticks} clear ticks; the next yield needs only "
            f"one tick for {self.params['post_loss_cooldown_sec']:.1f} s)",
            corridors, now)

    def _reevaluate_target(self, corridors: List[Corridor],
                           now: float) -> None:
        """Is the point we committed to still worth standing on?

        HYSTERESIS (2026-09-09, mppi_panoptex_2). This used to fire whenever
        any danger corridor merely contained the target with tta inside
        t_yield_sec -- which a corridor window does every time it slides
        along a lane the target happens to sit near, nine times in one
        contact episode, cancelling the goal (status 6) each time and
        leaving the robot in the lane it was trying to leave. Only two
        things may now disturb a committed target, both of them in
        corridor.refuge_recompute_reason: a user whose closest approach to
        it is inside refuge_recompute_tta_s, or its lane-line clearance
        having dropped below refuge_lane_clearance() - LANE_CLEARANCE_SLACK_M.
        And never more than once per refuge_recompute_min_s.
        """
        target = self._goal_xy if self._goal_xy is not None else \
            list(self._robot_xy)
        reason = refuge_recompute_reason(
            target, [self._danger(c) for c in corridors],
            self._lanes.effective_lanes(), self.params)
        if reason is None:
            return

        since = now - self._last_refuge_recompute_t
        if since < self.params["refuge_recompute_min_s"]:
            self.get_logger().info(
                f"{self._state} point ({target[0]:.2f}, {target[1]:.2f}) is "
                f"contested ({reason['why']}) but the last recompute was "
                f"{since:.1f} s ago (< refuge_recompute_min_s "
                f"{self.params['refuge_recompute_min_s']:.1f} s) -- staying "
                "put", throttle_duration_sec=5.0)
            return
        self._last_refuge_recompute_t = now

        blocker = reason["corridor"]
        if reason["why"] == "corridor":
            why = (f"user {blocker.user_id}'s danger corridor covers it and "
                   f"its closest approach is {reason['tta']:.1f} s away "
                   f"(< refuge_recompute_tta_s "
                   f"{self.params['refuge_recompute_tta_s']:.1f} s)")
        else:
            why = (f"it is only {reason['clearance']:.2f} m from a "
                   f"remembered lane line (need "
                   f"{self._lane_clearance():.2f} - "
                   f"{LANE_CLEARANCE_SLACK_M:.2f} m)")
        self.get_logger().info(
            f"{self._state} point ({target[0]:.2f}, {target[1]:.2f}): {why} "
            "-- recomputing")
        xy = self._find_refuge(corridors, blocker)
        if xy is None:
            self._hold_in_place()
            self.get_logger().warning(
                "no alternative refuge -- stopping where we are")
            self._set_state(REFUGE)
            return
        yaw = math.atan2(xy[1] - self._robot_xy[1], xy[0] - self._robot_xy[0])
        self._yield_resends = 0
        self._transition_goal(xy[0], xy[1], yaw, GOAL_REFUGE)
        self._yield_ref_xy = np.asarray(self._robot_xy, dtype=float)
        if self._yield is not None:
            # The reason dict now describes a refuge move, whatever it
            # started as -- the recorder should not see kind="hold" against a
            # robot that is driving off the lane.
            self._yield["kind"] = GOAL_REFUGE
            if blocker is not None:
                self._yield["user"] = blocker.user_id
        self._set_state(REFUGE)

    def _release(self, reason: str, corridors: List[Corridor],
                 now: float) -> None:
        """Go back to the interrupted waypoint -- unless the robot is (still
        or again) standing in an approaching user's lane, in which case
        resuming would drive straight back into it. That is where `never
        re-enter an approaching user's corridor` is enforced; the other half
        is that _eval_navigating re-runs the moment we are NAVIGATING again,
        so a lane that becomes a conflict during the resume is caught on the
        next tick like any other."""
        again = self._refuge_candidate(corridors)
        if again is not None:
            self.get_logger().info(
                f"release condition met ({reason}) but we are still inside "
                f"user {again['user']}'s corridor (tta={again['tta']:.1f} s) "
                "-- taking refuge instead of resuming")
            self._enter_refuge(again, corridors, now)
            return
        self.get_logger().info(
            f"release after {self._now() - self._yield_started_t:.1f} s: "
            f"{reason} -- resuming waypoint {self._wp_index} "
            f"({self.waypoints[self._wp_index]['name']})")
        self._pending_kind = self._pending_user = self._pending_info = None
        self._pending_ticks = 0
        self._yield_ref_xy = None
        # The blind-wait clock measures one wait at one crossing; a release
        # ends that wait, whatever caused it. The observed-pass memory is
        # NOT cleared -- that is the history the next blind decision needs.
        self._crossing_wait_since.clear()
        self._yield_resends = 0
        self._set_state(RESUMING)
        self._resume_waypoint()

    # ------------------------------------------------------------ plumbing

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        previous = self._state
        self._state = state
        self.get_logger().info(f"state: {previous} -> {state}")
        self._publish_state(self._now())

    def _publish_state(self, now: float) -> None:
        wp = self.waypoints[self._wp_index]
        goal_xy = self._goal_xy if self._goal_xy is not None else \
            [wp["x"], wp["y"]]
        payload = {
            "t": round(now, 3),
            "waypoint": self._wp_index,
            "name": wp["name"],
            "lap": self._lap,
            "state": self._state,
            "yield_count": self._yield_count,
            "yield": self._yield,
            "goal_xy": [round(float(goal_xy[0]), 3),
                        round(float(goal_xy[1]), 3)],
        }
        if self.yield_enabled:
            # WP4 telemetry, added ONLY in the panoptex arm: the baseline
            # arm's state stream must stay byte-identical to the pre-WP4
            # one, since it is the experiment's control. Same rule for the
            # two lane-memory keys below.
            payload["crossing"] = self._crossing
            payload["lanes"] = len(self._lanes.lanes)
            # How many of those actually keep a refuge out
            # (corridor.lane_is_effective). `lanes` alone was misleading:
            # mppi_panoptex_3 reported 32 and behaved as if every one of them
            # were a patrol line. A run whose two numbers diverge is a run
            # whose lane memory is fragmenting.
            payload["lanes_effective"] = len(self._lanes.effective_lanes())
            payload["waypoint_in_lane"] = bool(self._waypoint_in_lane)
        msg = String()
        msg.data = json.dumps(payload)
        self.state_pub.publish(msg)

    def _publish_lanes(self, now: float) -> None:
        """The remembered lane lines as one LINE_LIST marker for RViz.

        1 Hz, not the tick rate: this is a debug view of state that changes
        on the order of a lap, and the state topic already carries the count
        at full rate. One marker with every segment in it (rather than one
        marker per lane) means a lane that expires simply stops being drawn,
        with no per-id DELETE bookkeeping to get wrong.
        """
        if self.lanes_pub is None or now - self._lanes_pub_t < 1.0:
            return
        self._lanes_pub_t = now
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "observed_lanes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.55, 0.0
        marker.color.a = 0.85
        for lane in self._lanes.lanes:
            for x, y in lane.endpoints():
                marker.points.append(
                    Point(x=float(x), y=float(y), z=0.05))
        array = MarkerArray()
        array.markers.append(marker)
        self.lanes_pub.publish(array)

    def cancel(self) -> None:
        if self._active is not None and self._active.handle is not None:
            self._active.handle.cancel_goal_async()


def main(args=None):
    rclpy.init(args=args)
    node = MissionSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cancel()
    except _StopRunner:
        pass
    finally:
        # Only ever reached from main()'s own stack, never from a callback --
        # see _StopRunner's docstring for why that distinction matters here.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
