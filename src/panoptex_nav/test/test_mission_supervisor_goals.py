"""
Goal-bookkeeping tests for panoptex_nav.mission_supervisor -- the half of
the node that decides which NavigateToPose result belongs to which goal, and
therefore which of them may count as a MISSED waypoint.

Unlike test_corridor.py / test_risk_speed_governor.py (pure data in / data
out), this logic lives on the Node itself, so the node IS built here --
rclpy.init() plus a MissionSupervisor with parameter_overrides, no launch
file, no spin, no nav2. The action client is replaced by _FakeActionClient,
which hands out goal handles and result futures the test fires by hand; that
is what makes the ORDER of a real run's race reproducible:

    waypoint goal sent -> refuge goal sent (waypoint cancelled)
    -> release: refuge cancelled, waypoint re-sent
    -> the refuge goal's result finally lands

Regression, mppi_panoptex_2 (2026-09-09, x3_nav.log ~line 4802): that last
result was credited to the waypoint goal that had replaced it and logged as
"waypoint 0 (wp_001) aborted (status 6) -- counted as missed, continuing",
so a lap was lost and the robot drove back into the lane it had just refuged
from. Note status 6 is ABORTED, not CANCELED (action_msgs/msg/GoalStatus:
SUCCEEDED=4, CANCELED=5, ABORTED=6) -- nav2's SimpleActionServer reports a
goal that a new goal preempted as ABORTED even when a cancel was what
prompted the replacement, which is why "ignore CANCELED" alone would not
have caught it.

Run via `colcon test --packages-select panoptex_nav` (wired into
CMakeLists.txt through ament_cmake_pytest) or directly with
`python3 -m pytest test/test_mission_supervisor_goals.py` from this
package's root, with the workspace built and sourced.
"""

import os
import tempfile

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.parameter import Parameter

from panoptex_nav.mission_supervisor import (
    GOAL_HOLD,
    GOAL_REFUGE,
    GOAL_WAYPOINT,
    MAX_COLLATERAL_RESENDS,
    MAX_YIELD_RESENDS,
    NAVIGATING,
    REFUGE,
    RESUMING,
    MissionSupervisor,
)

WAYPOINTS_YAML = """\
frame_id: map
waypoints:
  - {name: wp_001, x: 3.4, y: 2.5, yaw: 0.0}
  - {name: wp_002, x: 3.4, y: 0.5, yaw: 0.0}
  - {name: wp_003, x: 0.65, y: 0.5, yaw: 0.0}
"""


# --------------------------------------------------------------- fake nav2

class _FakeFuture:
    """rclpy's Future reduced to what the supervisor uses: a result that can
    be set before or after the done callback is attached."""

    def __init__(self):
        self._result = None
        self._callbacks = []

    def add_done_callback(self, cb):
        self._callbacks.append(cb)
        if self._result is not None:
            cb(self)

    def result(self):
        return self._result

    def set_result(self, result):
        self._result = result
        for cb in list(self._callbacks):
            cb(self)


class _FakeResult:
    def __init__(self, status):
        self.status = status


class _FakeGoalHandle:
    def __init__(self, sent, accepted=True):
        self.sent = sent                 # the _SentGoal that owns it
        self.accepted = accepted
        self.result_future = _FakeFuture()
        self.cancel_future = _FakeFuture()
        self.cancel_requested = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_requested = True
        return self.cancel_future


class _SentGoal:
    def __init__(self, pose, feedback_callback):
        self.pose = pose
        self.feedback_callback = feedback_callback
        self.response_future = _FakeFuture()
        self.handle = None

    # -- what a test drives ------------------------------------------------
    def accept(self):
        self.handle = _FakeGoalHandle(self)
        self.response_future.set_result(self.handle)
        return self.handle

    def reject(self):
        self.handle = _FakeGoalHandle(self, accepted=False)
        self.response_future.set_result(self.handle)

    def feedback(self):
        """nav2 publishing feedback == the goal actually started running."""
        if self.feedback_callback is not None:
            self.feedback_callback(object())

    def finish(self, status):
        self.handle.result_future.set_result(_FakeResult(status))

    def ack_cancel(self):
        self.handle.cancel_future.set_result(object())

    @property
    def xy(self):
        return (round(self.pose.pose.position.x, 3),
                round(self.pose.pose.position.y, 3))


class _FakeActionClient:
    def __init__(self):
        self.goals = []

    def send_goal_async(self, goal, feedback_callback=None):
        sent = _SentGoal(goal.pose, feedback_callback)
        self.goals.append(sent)
        return sent.response_future

    def wait_for_server(self, timeout_sec=None):
        return True


# ------------------------------------------------------------------ rig

@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def waypoints_file():
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(WAYPOINTS_YAML)
    handle.close()
    yield handle.name
    os.unlink(handle.name)


def make_node(waypoints_file, yield_enabled=True, overrides=()):
    node = MissionSupervisor(parameter_overrides=[
        Parameter("waypoints_file", value=waypoints_file),
        Parameter("yield_enabled", value=yield_enabled),
        Parameter("use_sim_time", value=False),
    ] + list(overrides))
    node._client = _FakeActionClient()
    node._started = True
    # Timers would only fire under spin(), which no test does, but a node
    # that is never spun still holds them; drop them so nothing can tick.
    node.timer.cancel()
    node._start_timer.cancel()
    return node


@pytest.fixture
def node(waypoints_file):
    n = make_node(waypoints_file)
    yield n
    n.destroy_node()


def send_waypoint(node, *, running=True):
    """Send the current waypoint, have nav2 accept it, and (by default) have
    nav2 report feedback on it -- i.e. the goal is genuinely under way."""
    node._send_current_waypoint()
    goal = node._client.goals[-1]
    goal.accept()
    if running:
        goal.feedback()
    return goal


# ------------------------------------------------- the mppi_panoptex_2 race

def test_late_refuge_result_does_not_skip_the_waypoint(node):
    """waypoint -> refuge (waypoint cancelled) -> resume -> the refuge goal's
    result lands LATE. The waypoint must be untouched by it."""
    wp_goal = send_waypoint(node)
    node._set_state(NAVIGATING)
    assert node._wp_index == 0

    # --- yield: cancel the waypoint, drive to a refuge
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    refuge_goal = node._client.goals[-1]
    refuge_goal.accept()
    refuge_goal.feedback()
    assert wp_goal.handle.cancel_requested

    # --- release: cancel the refuge and re-send the SAME waypoint, before
    #     either the cancel or the refuge result has come back (the race).
    node._cancel_goal()
    node._set_state(RESUMING)
    node._send_current_waypoint()
    resumed = node._client.goals[-1]
    resumed.accept()
    assert node._state == NAVIGATING
    assert resumed.xy == (3.4, 2.5)          # still wp_001

    # --- now the stale results arrive, in the order the run logged them.
    # nav2 reports a preempted goal as ABORTED(6) even though we cancelled it.
    refuge_goal.finish(GoalStatus.STATUS_ABORTED)
    wp_goal.finish(GoalStatus.STATUS_CANCELED)

    assert node._wp_index == 0, "the interrupted waypoint was skipped"
    assert node._missed == 0, "a cancelled hold/refuge goal counted as missed"
    assert node._lap == 0
    assert node._state == NAVIGATING
    assert node._active is not None and node._active.kind == "waypoint"

    # ... and the resumed goal still completes the waypoint normally.
    resumed.feedback()
    resumed.finish(GoalStatus.STATUS_SUCCEEDED)
    assert node._wp_index == 1
    assert node._missed == 0


def test_late_canceled_refuge_result_is_ignored(node):
    """Same replay, but nav2 reports the cancelled refuge goal as CANCELED
    (the status the cancel path produces when nothing preempts it)."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    refuge_goal = node._client.goals[-1]
    refuge_goal.accept()

    node._cancel_goal()
    node._set_state(RESUMING)
    node._send_current_waypoint()
    node._client.goals[-1].accept()

    refuge_goal.finish(GoalStatus.STATUS_CANCELED)

    assert node._wp_index == 0
    assert node._missed == 0
    assert node._state == NAVIGATING


def test_stale_result_cannot_flip_the_state_machine(node):
    """A result from an older handle must not advance, pause or complete
    anything -- even a SUCCEEDED one."""
    wp_goal = send_waypoint(node)
    node._set_state(NAVIGATING)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._client.goals[-1].accept()
    node._set_state(REFUGE)

    wp_goal.finish(GoalStatus.STATUS_SUCCEEDED)

    assert node._wp_index == 0
    assert node._missed == 0
    assert node._state == REFUGE
    assert node._active is not None and node._active.kind == GOAL_REFUGE


def test_resumed_waypoint_aborted_by_our_own_cancel_is_resent(node):
    """The other half of the same race, and what mppi_panoptex_2 actually
    hit: nav2 takes the freshly accepted waypoint goal down together with
    the refuge goal we cancelled ("Aborting handle." / "Goal canceled" back
    to back, no "Begin navigating" in between). The goal never ran, so it is
    re-sent, not counted as missed."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._client.goals[-1].accept()

    node._cancel_goal()                       # cancel ack still outstanding
    node._set_state(RESUMING)
    node._send_current_waypoint()
    resumed = node._client.goals[-1]
    resumed.accept()
    resumed.finish(GoalStatus.STATUS_ABORTED)  # no feedback: it never ran

    assert node._missed == 0
    assert node._wp_index == 0
    retry = node._client.goals[-1]
    assert retry is not resumed
    assert retry.xy == (3.4, 2.5)


def test_collateral_resends_are_bounded(node):
    """A waypoint that keeps dying that way is retried, then declared missed
    -- the mission can never sit in a resend loop."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._client.goals[-1].accept()
    node._cancel_goal()
    node._send_current_waypoint()

    for _ in range(MAX_COLLATERAL_RESENDS + 1):
        goal = node._client.goals[-1]
        goal.accept()
        goal.finish(GoalStatus.STATUS_ABORTED)

    assert node._missed == 1
    assert node._wp_index == 1


# ------------------------------------------------- genuine waypoint failures

def test_waypoint_abort_counts_as_missed(node):
    """No cancel of ours anywhere near it: an ABORTED waypoint is a missed
    waypoint, exactly as before."""
    goal = send_waypoint(node)
    goal.finish(GoalStatus.STATUS_ABORTED)

    assert node._missed == 1
    assert node._wp_index == 1


def test_running_waypoint_aborted_after_a_cancel_still_counts_as_missed(node):
    """Sent during our own cancel, but nav2 DID run it (feedback arrived) --
    so its abort is the waypoint's own failure, not collateral damage."""
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._client.goals[-1].accept()
    node._cancel_goal()
    goal = send_waypoint(node)                 # accepted AND fed back
    goal.finish(GoalStatus.STATUS_ABORTED)

    assert node._missed == 1
    assert node._wp_index == 1


def test_external_cancel_of_the_active_waypoint_counts_as_missed(node):
    goal = send_waypoint(node)
    goal.finish(GoalStatus.STATUS_CANCELED)

    assert node._missed == 1
    assert node._wp_index == 1


def test_rejected_waypoint_is_retried_then_missed(node):
    """nav2 rejects goals while bt_navigator is still activating (2026-09-10:
    all six laps were 'completed' in 4 s that way). A rejection now schedules
    a retry of the SAME waypoint; only after max_reject_retries rejections is
    it counted missed and the list advances."""
    node.max_reject_retries = 2
    node._send_current_waypoint()
    node._client.goals[-1].reject()
    assert node._missed == 0
    assert node._wp_index == 0
    assert node._reject_retries == 1
    assert node._reject_timer is not None

    # Fire the retry timer by hand (no executor in the test), reject again,
    # then a third rejection exhausts the budget.
    node._reject_timer.cancel(); node._reject_timer = None
    node._send_current_waypoint()
    node._client.goals[-1].reject()
    assert node._reject_retries == 2 and node._missed == 0
    node._reject_timer.cancel(); node._reject_timer = None
    node._send_current_waypoint()
    node._client.goals[-1].reject()
    assert node._missed == 1
    assert node._wp_index == 1


def test_waypoint_success_advances_and_laps(node):
    for expected in (1, 2):
        goal = send_waypoint(node)
        goal.finish(GoalStatus.STATUS_SUCCEEDED)
        assert node._wp_index == expected
    goal = send_waypoint(node)
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    assert node._wp_index == 0
    assert node._lap == 1
    assert node._missed == 0


# ------------------------------------------------------------- baseline arm

def test_baseline_arm_behaviour_is_unchanged(waypoints_file):
    """yield_enabled:=false never cancels anything, so no goal can ever be
    flagged collateral: every non-SUCCEEDED result is a missed waypoint and
    the list advances, which is the control arm's contract."""
    node = make_node(waypoints_file, yield_enabled=False)
    try:
        for status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
            goal = send_waypoint(node, running=False)
            goal.finish(status)
        assert node._missed == 2
        assert node._wp_index == 2
        assert node._cancels_in_flight == 0
        assert node._resends == 0

        goal = send_waypoint(node)
        goal.finish(GoalStatus.STATUS_SUCCEEDED)
        assert node._wp_index == 0
        assert node._lap == 1
        assert node._missed == 2
    finally:
        node.destroy_node()


# ------------------------------------- cancel then send (mppi_panoptex_3)

def test_a_refuge_is_queued_until_the_waypoint_reports_a_terminal_result(node):
    """THE mppi_panoptex_3 defect, and its mppi_panoptex_4 sequel. nav2's
    SimpleActionServer runs one goal at a time and can take a freshly
    accepted goal down together with the goal it replaced: run 3 logged
    `refuge goal ended with status 6` seven milliseconds after `state:
    navigating -> refuge`, right behind the superseded waypoint goal's late
    result.

    Waiting for the CancelGoal SERVICE reply is not enough -- that reply only
    says nav2 ACCEPTED the cancel, which it then processes asynchronously in
    its work loop. mppi_panoptex_4 released the queue on it and had refuge
    goal #25 aborted 21 ms later by the very cancel it thought it had waited
    out. The queue is released by the previous goal's TERMINAL RESULT."""
    wp_goal = send_waypoint(node)
    node._set_state(NAVIGATING)
    assert len(node._client.goals) == 1

    node._transition_goal(3.58, -0.57, 0.0, GOAL_REFUGE)

    # Cancelled, queued, and NOTHING new on the wire.
    assert wp_goal.handle.cancel_requested
    assert len(node._client.goals) == 1
    assert node._active is None
    assert node._pending_goal is not None
    assert node._cancels_in_flight == 1
    # ...but the recorded state stream already says where we are going.
    assert node._goal_xy == [3.58, -0.57]

    # The cancel service answers: accepted, not finished. STILL nothing on
    # the wire -- this is exactly the moment mppi_panoptex_4 sent goal #25.
    wp_goal.ack_cancel()
    assert len(node._client.goals) == 1
    assert node._pending_goal is not None
    assert node._cancels_in_flight == 0

    # nav2 finishes halting the cancelled goal. NOW the refuge may go.
    wp_goal.finish(GoalStatus.STATUS_CANCELED)

    assert len(node._client.goals) == 2
    refuge = node._client.goals[-1]
    assert refuge.xy == (3.58, -0.57)
    assert node._pending_goal is None
    refuge.accept()
    assert node._active is not None and node._active.kind == GOAL_REFUGE

    # Exactly once: a duplicate/late result cannot mint a second refuge.
    wp_goal.finish(GoalStatus.STATUS_ABORTED)
    assert len(node._client.goals) == 2
    assert node._missed == 0
    assert node._wp_index == 0


def test_a_hold_is_queued_the_same_way_and_only_the_last_one_is_sent(node):
    """Two transitions inside one cancel round trip (a refuge that is
    re-decided before nav2 answers): the queue holds ONE goal, and it is the
    one the policy last asked for -- not both, in sequence, into the same
    race."""
    wp_goal = send_waypoint(node)
    node._transition_goal(1.0, 2.0, 0.0, GOAL_HOLD)
    node._transition_goal(3.0, 4.0, 0.0, GOAL_REFUGE)

    assert len(node._client.goals) == 1
    assert node._pending_goal["kind"] == GOAL_REFUGE

    wp_goal.ack_cancel()
    assert len(node._client.goals) == 1        # the ack is not the result

    wp_goal.finish(GoalStatus.STATUS_CANCELED)

    assert len(node._client.goals) == 2
    assert node._client.goals[-1].xy == (3.0, 4.0)


def test_the_resumed_waypoint_is_queued_behind_the_refuge_cancel(node):
    """The release path, which is where mppi_panoptex_2 lost wp_001: the
    waypoint is not re-sent until the refuge goal's cancel comes back."""
    wp_goal = send_waypoint(node)
    node._transition_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    wp_goal.ack_cancel()
    wp_goal.finish(GoalStatus.STATUS_CANCELED)
    refuge = node._client.goals[-1]
    refuge.accept()
    node._set_state(REFUGE)

    node._set_state(RESUMING)
    node._resume_waypoint()
    assert len(node._client.goals) == 2
    assert refuge.handle.cancel_requested
    assert node._state == RESUMING

    refuge.ack_cancel()
    assert len(node._client.goals) == 2        # the ack is not the result
    refuge.finish(GoalStatus.STATUS_CANCELED)
    resumed = node._client.goals[-1]
    assert resumed.xy == (3.4, 2.5)
    resumed.accept()
    assert node._state == NAVIGATING
    assert node._active.kind == GOAL_WAYPOINT


def test_the_watchdog_sends_a_queued_goal_if_no_cancel_response_arrives(
        waypoints_file):
    """nav2 answering every cancel is not something to bet a yield on. A
    supervisor sitting on an unsent refuge is the failure this whole
    mechanism exists to prevent, so cancel_timeout_s bounds the wait."""
    node = make_node(waypoints_file,
                     overrides=[Parameter("cancel_timeout_s", value=0.0)])
    try:
        wp_goal = send_waypoint(node)
        node._transition_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
        assert len(node._client.goals) == 1      # queued, nothing acked

        # The cancel is answered but the goal never terminates -- the case
        # the terminal-result gate would otherwise sit on forever.
        wp_goal.ack_cancel()
        assert len(node._client.goals) == 1

        node._tick()                             # no result, ever

        assert len(node._client.goals) == 2
        assert node._client.goals[-1].xy == (3.58, -0.57)
        assert node._pending_goal is None
        # The unanswered cancel is forgotten too, or every later goal would
        # be flagged as sent-during-a-cancel for the rest of the run.
        assert node._cancels_in_flight == 0
    finally:
        node.destroy_node()


def test_a_queued_goal_is_dropped_when_the_yield_gives_up(node):
    """_hold_in_place is "stop here, we are not going anywhere": whatever was
    queued was queued to replace a goal we are now abandoning, and sending it
    when the cancel lands would drive off after the yield had ended."""
    wp_goal = send_waypoint(node)
    node._transition_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._hold_in_place()

    assert node._pending_goal is None
    wp_goal.ack_cancel()
    assert len(node._client.goals) == 1


# ------------------------------ collateral damage to a hold/refuge goal

def test_collateral_aborted_refuge_goal_is_resent_once(node):
    """The second line of defence, now covering hold/refuge goals. Before
    this the branch logged "staying put until the corridor releases" for a
    refuge goal that had been killed 7 ms after being sent -- i.e. the robot
    stood in the lane it was trying to leave while carter1 pushed it five
    metres north (mppi_panoptex_3)."""
    send_waypoint(node)
    node._cancel_goal()                        # cancel ack still outstanding
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    refuge = node._client.goals[-1]
    refuge.accept()
    refuge.finish(GoalStatus.STATUS_ABORTED)   # no feedback: it never ran

    retry = node._client.goals[-1]
    assert retry is not refuge
    assert retry.xy == (3.58, -0.57)
    assert node._active is not None and node._active.kind == GOAL_REFUGE
    assert node._yield_resends == MAX_YIELD_RESENDS
    assert node._missed == 0
    assert node._wp_index == 0


def test_a_refuge_killed_by_a_lagging_cancel_is_resent_then_parks(node):
    """mppi_panoptex_4 (x3_nav.log ~4184): refuge goal #25 was sent from the
    cancel RESPONSE of hold goal #24, so by our own accounting no cancel of
    ours was outstanding (`after_self_cancel` False) -- and nav2's work loop
    then applied that very cancel to #25, aborting it 21 ms later with no
    feedback and no "Begin navigating". The old rule needed an unanswered
    cancel and so classified it as a refuge we could not reach: "holding in
    place until the corridor releases", i.e. standing in the corridor. It is
    collateral damage, so it is re-sent once -- and only a SECOND failure
    parks the robot."""
    wp_goal = send_waypoint(node)
    node._transition_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    wp_goal.ack_cancel()
    wp_goal.finish(GoalStatus.STATUS_CANCELED)

    refuge = node._client.goals[-1]
    assert refuge.xy == (3.58, -0.57)
    refuge.accept()
    # The whole point: our cancel accounting is clean when this goes out.
    assert node._cancels_in_flight == 0
    assert not node._active.after_self_cancel

    refuge.finish(GoalStatus.STATUS_ABORTED)     # 30 ms later, no feedback

    retry = node._client.goals[-1]
    assert retry is not refuge
    assert retry.xy == (3.58, -0.57)
    assert node._yield_resends == MAX_YIELD_RESENDS
    assert node._active is not None and node._active.kind == GOAL_REFUGE

    retry.accept()
    retry.finish(GoalStatus.STATUS_ABORTED)

    assert node._client.goals[-1] is retry       # no third attempt
    assert node._active is None                  # parked
    assert node._missed == 0
    assert node._wp_index == 0


def test_a_refuge_that_keeps_failing_falls_back_to_holding_in_place(node):
    """Bounded, like the waypoint budget: one re-send, then stop where we
    are. There is nothing to miss -- standing still is a worse yield, not a
    lost waypoint -- so a second retry would buy nothing."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    node._client.goals[-1].accept()
    node._client.goals[-1].finish(GoalStatus.STATUS_ABORTED)

    retry = node._client.goals[-1]
    retry.accept()
    retry.finish(GoalStatus.STATUS_ABORTED)

    assert node._client.goals[-1] is retry     # no third attempt
    assert node._active is None                # parked
    assert node._missed == 0
    assert node._wp_index == 0


def test_a_refuge_that_genuinely_failed_is_not_resent(node):
    """nav2 ran it (feedback arrived) and could not reach the point: that is
    a refuge we cannot have, not collateral damage. Hold in place."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    refuge = node._client.goals[-1]
    refuge.accept()
    refuge.feedback()
    refuge.finish(GoalStatus.STATUS_ABORTED)

    assert node._client.goals[-1] is refuge
    assert node._active is None
    assert node._yield_resends == 0


def test_a_refuge_that_arrived_leaves_the_robot_parked(node):
    """The SUCCEEDED case is untouched: arrival parks the robot and the
    release test decides what happens next."""
    send_waypoint(node)
    node._cancel_goal()
    node._send_goal(3.58, -0.57, 0.0, GOAL_REFUGE)
    refuge = node._client.goals[-1]
    refuge.accept()
    refuge.feedback()
    refuge.finish(GoalStatus.STATUS_SUCCEEDED)

    assert node._client.goals[-1] is refuge
    assert node._active is None
    assert node._missed == 0
