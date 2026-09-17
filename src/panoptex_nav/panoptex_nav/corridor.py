#!/usr/bin/env python3
"""
corridor.py  --  WP-B: the geometry and gap-acceptance core of the
                 Panoptex "polite mission supervisor".

Everything in this module is a PURE function over numpy arrays and plain
dicts: no rclpy, no Node, no clock, no TF. `mission_supervisor.py` is the
only ROS-aware half; it gathers tracks / plan / map / TF each tick, calls
in here, and turns the answers into Nav2 goals. Same split (and the same
reason) as `risk_speed_governor.compute_cap`: the policy is data in /
data out, so it is unit-tested directly in `test/test_corridor.py` with no
ROS graph, no spin and no simulator.

WHAT A "CORRIDOR" IS
--------------------
The Panoptex world model (`/risk_perception/world_objects`) reports other
moving agents -- in the warehouse scenario, the Carter AMRs -- with a
map-frame position and velocity. A *corridor* is the swept lane that one
such agent is about to occupy: a rectangle starting `back_margin` metres
BEHIND the agent, running `|v| * t_corridor` metres ahead of it along its
heading, `half_width` metres either side of that centre line.

    lateral = +w   +-------------------------------------+
                   |            <-- corridor -->         |
    centre line    |  x====U=============================>|   heading
                   |     (user)                          |
    lateral = -w   +-------------------------------------+
                   ^      ^                              ^
        along = -back_margin  along = 0        along = |v| * t_corridor

`along` is measured FROM THE USER (not from the rectangle's back edge), so
it is directly comparable to time: `along / |v|` is the seconds until the
user reaches that point (`tta()`). That is the whole point of the
parameterisation -- the yield decision is a time comparison, not a distance
comparison.

The back margin exists because a lane the user has *just* left is still not
a place to drive into: the X3 is 0.25 m long and slow, and a Carter that
has passed by 0.3 m is still close enough that entering its lane invites a
re-encounter if it stops or reverses. It also gives the release test
(`user_passed`) something to be hysteretic about.

The heading is the agent's raw velocity, EXCEPT where the learned
Spatial-Flow prior (`/risk_perception/spatial_flow/<category>`, see
`spatial_prior_node.py`'s docstring) has enough evidence at the agent's own
cell and broadly agrees with the observed velocity -- then the flow heading
wins. Rationale: the tracker's instantaneous velocity is noisy and jitters
by 10-20 degrees frame to frame, which at a 10 s horizon swings the far end
of the corridor by metres; the flow field is an EMA over the whole run of
what actually travels through that cell, so where it agrees it is the
better long-horizon estimate of the same lane. Where it *disagrees* by more
than `flow_snap_deg` we must NOT snap: that is the case of an agent
genuinely departing from the learned lane (a Carter turning off, or a
second lane crossing the first), and forcing it back onto the historical
heading would point the corridor at empty floor while the agent drives
somewhere else. `flow_min_conf` guards the other failure: an unvisited cell
reports s=0 with f=(0,0), which is "no data", not "observed stationary"
(spatial_prior_node's docstring is explicit about this).

THE TWO YIELD DECISIONS
-----------------------
1. GAP ACCEPTANCE at a crossing (`plan_corridor_crossing` +
   `decide_crossing`). The robot's global plan crosses the corridor. Can it
   get across before the user arrives? Time needed:

       t_clear = (d_entry + (2*half_width + clear_margin)) / v_cross

   -- drive the remaining `d_entry` metres to the corridor edge, then the
   full width of the lane plus a margin, all at the *conservative* speed
   `v_cross` (0.20 m/s, not the X3's nominal 0.3-0.5: nav2 decelerates into
   and accelerates out of the crossing, and this number has to be a lower
   bound or the whole decision is optimistic in exactly the situation where
   optimism costs a collision). Time available: `tta`, the user's own
   along-distance to the crossing over its speed. Hold iff
   `0 < tta < t_clear + t_margin`. `tta <= 0` means the user is past the
   crossing already -- go.

2. REFUGE (`find_refuge`). The robot is ALREADY inside a corridor when its
   user starts approaching (typically: nav2 routed the X3 up the middle of
   a Carter lane). There is nothing to accept a gap for -- the robot has to
   leave the lane. `find_refuge` picks the nearest free, obstacle-clear
   cell outside every corridor, preferring the side the robot is already on
   (crossing the lane to reach the far side would be a worse encounter than
   the one being avoided) and directions perpendicular to the lane (the
   shortest way out of a long thin rectangle is across it).

LANE-LINE MEMORY (2026-09-09, mppi_panoptex_2)
----------------------------------------------
A corridor is an INSTANTANEOUS window: `back_margin` behind the track's
current position, `speed * t_corridor` ahead of it. That window is the right
object for "will this user sweep the point I am standing on", and it is the
WRONG object for "is this a place to stand". mppi_panoptex_2 made the
difference expensive: carter1 patrols the line x = 1.27 from y = -4 to y = 7,
so a cell at (1.27, -3) is on its permanent centre line, but with the Carter
at y = 4 heading north that cell is 7 m behind it -- outside the window, and
therefore an acceptable refuge. `find_refuge` took it (10 of the run's 25
refuge targets were within 0.6 m of the centre line, one of them 0.01 m), the
window swung back over it a tick or two later, and the refuge goal was
cancelled and recomputed -- nine times in one contact episode. The X3 spent
312 s of 701 s within 0.6 m of the lane line; a clean crossing needs ~7 s.

The learned lane band (`make_lane_band` on `/risk_perception/spatial_prior`)
did not catch it either: the prior read 2-15 at the contact cells against a
`lane_band_min_value` of 5, i.e. right at the noise floor of an EMA that has
seen a handful of passes.

So the session remembers, per corridor user, the LINE it has actually been
observed driving along -- `LaneMemory` / `ObservedLane`, a segment from where
we first saw that track to where we last saw it, merged with any collinear
segment within `lane_merge_dist_m` so one patrol accumulates into one lane
rather than one lane per track id (and ids churn: one Carter carried six in
one run). `lane_line_clearance()` measures distance to those segments
EXTENDED by `lane_extension_m` past both observed ends, because the patrol
goes further than the cameras watched it go.

LANE HYGIENE (2026-09-09, mppi_panoptex_3)
------------------------------------------
That memory fragmented. Two Carter patrol lines in the hall produced THIRTY
TWO remembered lanes, each extended 2 m past its own ends and each demanding
`refuge_lane_clearance_m` (1.0 m) of its own, which left almost nothing legal
inside the 2.5 m refuge disc and pushed refuges out to the 4.0 m stage --
3.5 m away, with a Carter closing. Three things caused it and three things
fix it:

  * the merge test rejected segments it should have accepted, because it
    demanded BOTH endpoints of the new segment be within
    `lane_merge_dist_m` of the existing line. A 2-4 m piece of a jittering
    track tilts about its centre, swinging an endpoint half a metre while
    the midpoint barely moves. `lanes_collinear()` now tests the MIDPOINT
    (with the angle test, widened to 25 deg, doing the work of keeping a
    genuinely different line out) plus an along-line gap of at most
    `lane_merge_gap_m` -- which is the one thing the old test was too
    permissive about, since longitudinal separation was ignored entirely.
  * an insert merged the new segment into the first lane it matched and
    stopped, so two lanes that a later observation had made collinear with
    each other stayed apart forever. `LaneMemory._compact()` now re-merges
    the whole set after every insert, transitively.
  * a fragment blocked refuges from its first tick. `lane_is_effective()`
    now requires `lane_min_length_m` of observed extent and
    `lane_min_points` sightings before a lane counts for the clearance
    rule; `effective_lanes()` / `effective_clearance()` are what the refuge
    search, the hold escalation and the waypoint gate ask. Short lanes are
    still remembered and still grow. Finally `lane_max` caps the memory at
    a warehouse-plausible 8, dropping the shortest/oldest.

That memory is used in three places, all of them "where may I stand", never
"may I cross": the refuge search rejects any cell inside
`refuge_lane_clearance()` of a remembered line, `hold_in_place_is_unsafe()`
counts standing on one as being in the way, and `refuge_recompute_reason()`
uses it as the ONLY lane-geometry reason to disturb a committed refuge. It is
deliberately not evidence for holding: a line somebody drove down an hour ago
is not a user approaching, and turning it into one would stop the robot
everywhere.

NEVER CROSS A LANE TO REACH A REFUGE (2026-09-09, mppi_panoptex_4)
------------------------------------------------------------------
The refuge search always *preferred* the robot's own side of the lane. It
was only a preference, ranked below the lane-line clearance, and in
mppi_panoptex_4 the geometry made it lose. carter1 patrols x = 1.27; a
shelf runs at x ~ 0.30; the aisle between them is 0.97 m wide, and
`refuge_lane_clearance_m` was a flat 1.0 m -- so NOTHING on the robot's own
side could ever be legal, and the search dutifully answered with the far
side:

    refuge (2.73, 1.93) found at refuge_radius_m=2.5 m (clearance 1.46 m)
    YIELD/refuge #6: user 2 enters our lane in tta=6.5 s -- stepping aside
      to (2.73, 1.93), ..., other side of the lane

Crossing 1.9 m of danger band at <= 0.26 m/s takes >= 7 s; the Carter was
6.5 s away. The robot was met in the middle of the lane, released standing
ON the centre line at (1.27, 1.85), and pushed to (1.31, 6.87). Runs 2-3
show the same mechanism.

Three changes, in this module and in mission_supervisor:

  1. `LaneGuard` / `guard_blocks` / `crossing_rejected`: a candidate on the
     far side of any guarded lane line is HARD-REJECTED unless
     `tta > crossing_time + refuge_cross_margin_s`, where crossing_time is
     `(|robot offset| + |candidate offset| + 2*half_width) / v_cross_mps`.
     That is the gap-acceptance test of the module's first half, applied to
     the one decision that was still choosing to cross a lane without
     making it. Guards come from every corridor user and from every
     remembered lane a user is currently on (`lane_guards`).
  2. `refuge_lane_clearance()`: the clearance floor is DERIVED from the
     geometry (user half width + robot radius + margin, floored at
     `refuge_lane_clearance_min_m`) instead of being a flat 1.0 m, so it
     cannot exceed the aisle it has to fit inside. 0.75 m by default;
     `refuge_lane_clearance_m` > 0 still overrides it outright.
  3. `find_wall_hug()`: when nothing passes, take the free, obstacle-clear
     cell on OUR side that stands furthest off the traffic and hold there.
     Below the clearance floor and possibly still inside the corridor --
     but on this side of it, which crossing never is.

Ranking changed with them: same side now outranks lane clearance, so half a
metre of extra clearance can no longer buy a lane crossing.

CROSSING A LANE YOU CANNOT SEE (WP4)
------------------------------------
Gap acceptance above answers "can I beat the user I can see?". Four runs
against Isaac (`avoid_panoptex_1`-`4`) showed that is the wrong question
half the time: the overhead cameras track a Carter within 0.7 m only 89 % of
the time in the south half and 33 % in the north, and a decision made on
"nobody is tracked" is not a decision that the lane is clear -- it is a
decision made with no information at all, and the robot took it every time.

So the policy now asks two questions before the timing one:

  1. WHERE would a user have to be for this crossing to matter?
     `approach_zone()` -- the strip of the lane upstream of the crossing
     point, `v_lane * (t_clear + t_margin)` long, i.e. exactly the reach of
     anything that could arrive while we are still in the lane. Anything
     further back cannot reach us in time; anything inside it can.

  2. Can we SEE that strip? `zone_coverage()` against
     `/risk_perception/coverage` (the analytic exposure map -- overhead
     camera footprints plus lidar line of sight; see
     risk_perception/coverage_mask_node.py). This is the paper's
     "visibility-normalized place statistics" future-work item: the same
     empty grid cell means two opposite things depending on whether a
     sensor was ever looking at it.

Then, in order:

  covered + zone empty        -> cross. Nothing that could reach the
                                 crossing in time exists, and we can see
                                 that this is true.
  covered + a user in it      -> the gap acceptance above, unchanged. This
                                 is the branch every previous run took.
  NOT covered ("blind")       -> `blind_crossing_ok()`. Fall back to the
                                 learned traffic STATISTIC instead of the
                                 current observation: cross once more than a
                                 typical headway has elapsed since the last
                                 pass we actually saw (needs
                                 `headway_min_count` samples of evidence),
                                 or, with no such statistic, after a bounded
                                 `blind_wait_max_s`. The visible PART of the
                                 zone must be empty either way.

INVARIANT, and the reason this is safe to ship: `crossing_policy()` can only
ever turn a "go" into a "hold", never the reverse. Whenever the old timing
test says hold, the answer is hold, whatever the coverage and headway say.
The new branches exist to catch the crossings the old test waved through
because it had nothing to look at, not to overrule it when it does.

`crossing_choice()` is the paper's other half of this ("choose WHERE to
cross"): given several crossings on a plan, prefer the one whose cells carry
the lowest learned occupancy S. It is a pure ranking function here --
nothing in this stack can re-route Nav2's plan today, so the supervisor uses
it as telemetry rather than as a control input; see its docstring.

None of this is a collision avoider -- Nav2's own costmap layers and the
`risk_speed_governor` in this package remain responsible for that. This is
a *politeness* layer: it keeps the X3 out of another agent's lane in the
first place, so the reactive layers rarely have to fire at all.
"""

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional -- see _clearance_field() and _dilate_disc()
    from scipy.ndimage import (binary_dilation as _binary_dilation,
                               distance_transform_edt as _distance_transform_edt)
except ImportError:  # pragma: no cover - exercised only where scipy is absent
    _binary_dilation = None
    _distance_transform_edt = None


# Categories that can own a corridor under corridor_users_any_mover: false
# (the pre-2026-09-11 behaviour -- see CORRIDOR_DEFAULTS and
# select_corridor_users). A person does NOT: people stop, turn on the spot
# and step around obstacles, so a 10 s swept rectangle along their current
# velocity is fiction. People are handled by the distance-only branch of
# risk_speed_governor (see its module docstring for the same argument made
# about CPA), not here.
CORRIDOR_CATEGORIES = ("robot", "wheeled")

# WP-A, 2026-09-11: categories EXCLUDED from owning a corridor under the
# default corridor_users_any_mover: true -- everything else (furniture,
# unknown, or a tracker-promoted "wheeled" table) may, once it clears
# pmot_min/min_user_speed_mps like any other candidate. Only "person" is
# listed, for the same reason CORRIDOR_CATEGORIES above excludes it: a
# person's swept-rectangle prediction is fiction, not a class veto on risk.
NON_CORRIDOR_CATEGORIES = ("person",)

# Every threshold in this module, in one place -- shared by
# mission_supervisor's declare_parameter calls and by
# config/mission_supervisor.yaml, so the three cannot drift apart. All
# floats (mission_supervisor declares the one integer knob, confirm_ticks,
# itself).
CORRIDOR_DEFAULTS: Dict[str, float] = {
    # --- track -> corridor-user gating -------------------------------
    # Tracks within this radius of the robot are the X3 ITSELF: the
    # overhead cameras see it as a "mobile robot" like any other and the
    # tracker has no notion of "self". Nothing else can legitimately be
    # this close and still be worth yielding to.
    "self_exclusion_radius_m": 0.7,
    "pmot_min": 0.5,
    "min_user_speed_mps": 0.2,
    # WP-A, 2026-09-11 (default 1.0 = true; 1.0/0.0, not true/false -- every
    # CORRIDOR_DEFAULTS entry is declared as a float, see this dict's own
    # top comment, and mission_supervisor.py's declare_parameter loop
    # infers a DOUBLE type from that Python float default; a bool literal
    # in the yaml override would mismatch it and fail at node startup, same
    # reasoning as risk_speed_governor's act_on_any_mover). Any category
    # except NON_CORRIDOR_CATEGORIES ("person") may own a corridor once it
    # clears pmot_min/min_user_speed_mps like any other candidate -- a
    # tracker-promoted "table" (category relabelled "wheeled" by
    # object_tracker_node.Track.update_promotion) is not the only mover
    # that gets to reserve a lane; a genuinely moving piece of furniture
    # the promotion logic hasn't caught yet should too. 0.0 restores the
    # pre-2026-09-11 behaviour: only CORRIDOR_CATEGORIES ("robot",
    # "wheeled") may own a corridor. See select_corridor_users().
    "corridor_users_any_mover": 1.0,

    # --- corridor shape ----------------------------------------------
    "t_corridor_sec": 10.0,
    "back_margin_m": 1.0,
    "corridor_half_width_m": 0.55,
    # CONTAINMENT half width -- "am I in the way?" -- as opposed to the
    # nominal corridor_half_width_m used for "can I cross in time?".
    # 0.55 + 0.35 m of track-position error. The tracker's map-frame
    # position for a Carter was 0.1-0.4 m off ground truth in
    # avoid_panoptex_2, so a robot the nominal corridor called "outside"
    # (lateral 0.65 m) was in fact 0.15 m from being run over. Widening the
    # corridor everywhere would instead make the robot refuse crossings it
    # can comfortably make, so the two questions get two widths: see
    # corridor_contains()'s half_width override and danger_corridor().
    "danger_half_width_m": 0.90,

    # --- flow snapping -----------------------------------------------
    "flow_min_conf": 0.3,
    "flow_snap_deg": 30.0,

    # --- gap acceptance ----------------------------------------------
    "clear_margin_m": 0.3,
    "v_cross_mps": 0.20,
    "t_margin_sec": 2.0,
    "hold_back_m": 0.7,

    # --- refuge -------------------------------------------------------
    "t_yield_sec": 8.0,
    # 2.5 m, not the original 1.5: with the LANE BAND excluded (see
    # make_lane_band) the nearest legal refuge from the middle lane of the
    # warehouse aisle is the east column at x > 2.8, which is ~2 m away.
    # A 1.5 m disc found nothing there and fell back to cells that were
    # merely outside the ACTIVE corridors -- i.e. inside the other lane
    # (avoid_panoptex_1, 2026-09-09: refuges at (1.73,-4.22), (1.23,-3.22),
    # (2.03,-3.07), (2.53,-3.07), all in the aisle, followed by six
    # "recomputing" thrashes as the second lane woke up).
    "refuge_radius_m": 2.5,
    "refuge_clearance_m": 0.45,
    # Widen the search disc ONCE to this before giving up and holding in
    # place. mppi_panoptex_2's refuges were all found comfortably inside
    # 2.5 m -- and 10 of 25 of them landed within 0.6 m of the Carter's
    # permanent centre line, because that was the nearest cell the
    # instantaneous-corridor test happened to allow. Requiring
    # refuge_lane_clearance_m below makes the 2.5 m disc come up empty in
    # the aisle, and "nowhere to go" must mean "nowhere within 4 m", not
    # "nowhere within 2.5".
    "refuge_radius_max_m": 4.0,

    # --- observed-lane memory (session-local) --------------------------
    # See the "LANE-LINE MEMORY" section of the module docstring. A tracked
    # user's corridor is an INSTANTANEOUS window; these remember the LINE it
    # has been driving along for the whole session.
    #
    # How long a remembered lane survives with nothing driving down it.
    # 600 s ~ a whole study run: a patrol lane does not stop being a lane
    # because its AMR is at the far end of the hall.
    "lane_memory_s": 600.0,
    # Two observed segments merge into one lane when ALL THREE hold, in the
    # existing lane's own frame: their (undirected) headings agree within
    # lane_merge_angle_deg, the new segment's MIDPOINT lies within
    # lane_merge_dist_m of the existing line, and the along-line gap between
    # the two extents is at most lane_merge_gap_m. Undirected on purpose: a
    # patrol drives the same line in both directions.
    #
    # 25 deg / 1.0 m, not the original 15 deg / 0.6 m tested at BOTH
    # endpoints (mppi_panoptex_3, 2026-09-09): the session memory fragmented
    # into 32 lanes over two Carter patrol lines, every fragment's 2 m-
    # extended segment demanding refuge_lane_clearance_m of its own, which
    # emptied the 2.5 m refuge disc outright and pushed every refuge out to
    # the 4.0 m stage. A 2-4 m piece of a jittering track tilts by up to
    # ~20 deg and its far endpoint swings by more than half a metre while
    # its midpoint barely moves, so the midpoint is the robust statistic and
    # the endpoint test was the thing doing the fragmenting.
    "lane_merge_angle_deg": 25.0,
    "lane_merge_dist_m": 1.0,
    # Along-line separation two collinear stretches may have and still be
    # one lane. Overlapping stretches score 0. Not unbounded (which is what
    # this used to be): two aisles end to end across a hall are two lanes,
    # and joining them would lay a fictional line over the floor between.
    "lane_merge_gap_m": 3.0,
    # A remembered lane is extended by this much beyond BOTH observed
    # endpoints when clearance is measured against it. We only ever see the
    # stretch the cameras cover; carter1 patrols x=1.27 from y=-4 to y=7 and
    # was tracked over rather less than that.
    "lane_extension_m": 2.0,
    # Hard minimum distance a refuge must keep from every remembered lane
    # line -- an EXPLICIT OVERRIDE, used exactly as given whenever it is
    # > 0. 0.0 (the shipped default) means "derive it from the geometry",
    # which is what refuge_lane_clearance() below does:
    # max(refuge_lane_clearance_min_m,
    #     refuge_user_half_width_m + refuge_robot_radius_m
    #     + refuge_lane_margin_m) = max(0.60, 0.30 + 0.15 + 0.30) = 0.75 m.
    #
    # It used to be a flat 1.0 m (danger_half_width_m + 0.1), and in
    # mppi_panoptex_4 that number is precisely what drove the robot ACROSS
    # carter1's lane. The aisle west of the line x = 1.27 is 0.97 m wide (a
    # shelf runs at x ~ 0.30), so NO cell on the robot's own side could ever
    # clear 1.0 m; the search went to the far side, at (2.73, 1.93), and
    # crossing 1.9 m of danger band at <= 0.26 m/s takes >= 7 s against a
    # Carter that was 6.5 s away. A clearance floor has to be what the
    # geometry actually needs -- the widest user's half width, plus our own
    # radius, plus air -- not a round number that makes the only reachable
    # refuge illegal.
    "refuge_lane_clearance_m": 0.0,
    # The three terms of the derived floor, and its own hard minimum.
    # 0.30 = a Carter's half width; 0.15 = the X3's robot_radius; 0.30 m of
    # air between the two bodies.
    "refuge_user_half_width_m": 0.30,
    "refuge_robot_radius_m": 0.15,
    "refuge_lane_margin_m": 0.30,
    "refuge_lane_clearance_min_m": 0.60,
    # NEVER CROSS A LANE TO REACH A REFUGE. A candidate on the far side of a
    # lane line from the robot is legal only when that lane's user is
    # further away, in time, than the crossing itself takes plus this
    # margin. See LaneGuard / guard_blocks.
    "refuge_cross_margin_s": 2.0,
    # EVIDENCE FLOOR for that hard filter. A lane keeps refuges out only
    # once it has been observed over at least lane_min_length_m and from at
    # least lane_min_points sightings. A 0.8 m stub deposited by a track
    # turning a corner, or by a 1 s glimpse of something crossing the hall,
    # is not a lane -- and in mppi_panoptex_3 a pile of exactly those was
    # what left no legal refuge within 2.5 m. Short lanes are still
    # remembered (they grow into real ones); they simply do not vote.
    "lane_min_length_m": 1.5,
    "lane_min_points": 5.0,
    # Hard cap on how many lanes are remembered at all. A warehouse hall has
    # a handful; 32 (the old cap) was never a real lane count, it was track
    # churn. The shortest/oldest are dropped first.
    "lane_max": 8.0,

    # --- refuge hysteresis --------------------------------------------
    # Once a refuge goal is COMMITTED, a corridor window sliding over it is
    # not a reason to recompute -- that is what produced nine
    # "recomputing" cycles in one episode of mppi_panoptex_2, each one
    # cancelling the goal (status 6) and leaving the robot in the lane it
    # was trying to leave. Recompute only for a user whose closest approach
    # to the committed point is inside this horizon...
    "refuge_recompute_tta_s": 6.0,
    # ...and never more often than this.
    "refuge_recompute_min_s": 2.0,

    # --- static lane band ---------------------------------------------
    # Spatial-prior occupancy (0-100) at or above which a cell counts as
    # "traffic happens here". Deliberately low: the prior is an EMA that
    # saturates only on heavily repeated traversals, and one pass of a
    # Carter is already enough evidence that a cell is somebody's lane.
    "lane_band_min_value": 5.0,

    # --- crossing policy: coverage + headway (WP4) --------------------
    # See the "CROSSING A LANE YOU CANNOT SEE" section of the module
    # docstring. All floats (headway_min_count included) so this table
    # stays uniform -- mission_supervisor declares them straight from here.
    #
    # Fraction of the approach zone that must be covered by SOME sensor
    # (analytic camera footprint or lidar line of sight) before the timing
    # test is worth anything. 0.8 rather than 1.0: the far upstream corner
    # of a zone is routinely clipped by a shelf, and demanding a perfect
    # view would put every crossing on the blind branch.
    "coverage_min": 0.8,
    # /risk_perception/coverage value at or above which a cell counts as
    # covered. That grid is 0 or 100 (any-sensor OR), so anything strictly
    # between is a resampling artefact; 50 is the midpoint.
    "coverage_cell_min": 50.0,
    # Speed used to size the approach zone, from the learned Spatial-Flow
    # field at the crossing cell. Floored because an unlearned or
    # bidirectionally-cancelled cell reports near zero, which would collapse
    # the zone to nothing and declare every crossing safe; capped because a
    # single noisy EMA cell must not stretch the zone across the whole hall.
    # The observed speed of a tracked user is taken as a further floor by
    # the caller -- see lane_speed().
    "v_lane_min_mps": 0.4,
    "v_lane_max_mps": 1.0,
    # Headway samples needed before the learned mean gap is trusted at all.
    "headway_min_count": 3.0,
    # Wait this many mean headways since the last observed pass before
    # crossing blind. 1.0 = "longer than the typical gap has elapsed".
    "headway_factor": 1.0,
    # ...and when there IS no trustworthy headway (a cold prior, or a lane
    # nothing has ever been seen using), wait at most this long and then go.
    # A supervisor that waits forever for a statistic it will never get is
    # a stopped robot, which is its own failure mode.
    "blind_wait_max_s": 15.0,
    # A corridor user's centre within this distance of the crossing point
    # counts as a PASS of that crossing (mission_supervisor's live
    # last_observed_pass, which beats any persisted timestamp).
    "pass_radius_m": 0.5,

    # --- release / hysteresis ----------------------------------------
    "release_margin_m": 0.3,
    "lost_timeout_sec": 3.0,
    "min_hold_sec": 1.0,
    # After a yield released because the track was LOST (rather than seen to
    # pass), a fresh yield is confirmed in a single tick for this long. Track
    # ids churn constantly (one Carter was ids 27, 8, 28, 19, 60, 14 inside
    # one run), so a loss-release is usually a re-identification, not a
    # departure, and the normal confirm_ticks delay would let the robot drive
    # back into the lane before the "new" user is confirmed.
    "post_loss_cooldown_sec": 2.0,
}

# A candidate refuge whose direction from the robot is within this angle of
# the corridor's normal counts as "perpendicular" for the preference sort
# (see find_refuge). Not a tuned policy threshold -- it is the natural
# half-way split between "across the lane" and "along the lane", so it is a
# module constant rather than another ROS parameter to get wrong.
_PERPENDICULAR_TOLERANCE_DEG = 45.0

# Bin width for find_refuge's "further from every remembered lane line is
# better" preference. Coarse on purpose: it must dominate the side/distance
# preferences between genuinely different standing places without turning
# every stray centimetre into a reason to drive further.
_LANE_RANK_BIN_M = 0.5

_EPS = 1e-9


@dataclass(frozen=True)
class Corridor:
    """One agent's swept lane. Built by make_corridor(); read by every
    other function here. Immutable on purpose -- a corridor is a snapshot
    of one tick's track state, never mutated in place; the supervisor
    rebuilds them all every tick from the live world model."""

    user_id: str
    x: float            # user position, map frame -- the origin of `along`
    y: float
    ux: float           # unit heading (velocity, or the snapped flow heading)
    uy: float
    speed: float        # |v| of the RAW velocity, even when the heading snapped
    half_width: float
    back_margin: float
    length_ahead: float  # = speed * t_corridor_sec
    heading_source: str  # "velocity" | "flow"  (for logging / debugging)
    # WP-A, 2026-09-11: the track's own category, carried through so
    # crossing_choice() can use it as a tie-breaker (see that function's
    # docstring) now that select_corridor_users() no longer restricts
    # corridor ownership to CORRIDOR_CATEGORIES by default. Defaulted so
    # every pre-2026-09-11 direct Corridor(...) construction (tests
    # included) is unaffected.
    category: str = "unknown"


# --------------------------------------------------------------- helpers

def _unit(vx: float, vy: float) -> Tuple[float, float, float]:
    """(ux, uy, norm); (1, 0, 0) for a zero vector so callers never divide
    by zero -- a zero-speed corridor has length_ahead 0 anyway and is
    filtered out upstream by min_user_speed_mps."""
    n = math.hypot(vx, vy)
    if n < _EPS:
        return 1.0, 0.0, 0.0
    return vx / n, vy / n, n


def _as_xy(xy) -> np.ndarray:
    return np.asarray(xy, dtype=float)


# ------------------------------------------------------- corridor building

def select_corridor_users(tracks: Sequence[Dict],
                          robot_xy: Tuple[float, float],
                          params: Dict) -> List[Dict]:
    """Filter the world model down to the tracks that may own a corridor.

    tracks: dicts as built by mission_supervisor._build_tracks() --
        {id, label, category, x, y, vx, vy, pmot, score, age_sec, size}.
    Drops, in order: the robot itself (within self_exclusion_radius_m --
    the tracker cannot tell the X3 apart from any other mobile robot),
    category (see below), and anything not convincingly moving (pmot below
    pmot_min, or a speed below min_user_speed_mps -- a parked Carter has no
    lane to reserve).

    Category gate (WP-A, 2026-09-11): under the default
    corridor_users_any_mover (true), ANY category except
    NON_CORRIDOR_CATEGORIES ("person") may own a corridor -- a mover is a
    mover, and a tracker-promoted "table" (relabelled category "wheeled")
    is not the only kind of furniture-labelled mover that deserves a lane.
    corridor_users_any_mover: false restores the pre-2026-09-11 behaviour:
    only CORRIDOR_CATEGORIES ("robot", "wheeled") may own one at all.
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    self_r = float(params["self_exclusion_radius_m"])
    pmot_min = float(params["pmot_min"])
    v_min = float(params["min_user_speed_mps"])
    any_mover = bool(params.get("corridor_users_any_mover", 1.0))

    users: List[Dict] = []
    for tr in tracks:
        x, y = float(tr["x"]), float(tr["y"])
        if math.hypot(x - rx, y - ry) <= self_r:
            continue
        category = tr.get("category", "unknown")
        if any_mover:
            if category in NON_CORRIDOR_CATEGORIES:
                continue
        elif category not in CORRIDOR_CATEGORIES:
            continue
        if float(tr.get("pmot", 0.0)) < pmot_min:
            continue
        if math.hypot(float(tr.get("vx", 0.0)), float(tr.get("vy", 0.0))) < v_min:
            continue
        users.append(tr)
    return users


def make_corridor(track: Dict, params: Dict,
                  flow_sample: Optional[Tuple[float, float, float]] = None
                  ) -> Corridor:
    """Build the swept lane for one moving agent.

    flow_sample: (s, fx, fy) sampled at the agent's own cell from its
        category's Spatial-Flow image (risk_perception's
        sample_flow_grid()), or None when no flow is subscribed. Snapping
        rule and its rationale: see the module docstring.
    """
    vx = float(track.get("vx", 0.0))
    vy = float(track.get("vy", 0.0))
    ux, uy, speed = _unit(vx, vy)
    source = "velocity"

    if flow_sample is not None and speed > _EPS:
        s, fx, fy = (float(flow_sample[0]), float(flow_sample[1]),
                     float(flow_sample[2]))
        gx, gy, gn = _unit(fx, fy)
        if s > float(params["flow_min_conf"]) and gn > _EPS:
            dot = max(-1.0, min(1.0, ux * gx + uy * gy))
            if math.degrees(math.acos(dot)) < float(params["flow_snap_deg"]):
                ux, uy, source = gx, gy, "flow"

    return Corridor(
        user_id=str(track.get("id", "?")),
        x=float(track["x"]), y=float(track["y"]),
        ux=ux, uy=uy, speed=speed,
        half_width=float(params["corridor_half_width_m"]),
        back_margin=float(params["back_margin_m"]),
        length_ahead=speed * float(params["t_corridor_sec"]),
        heading_source=source,
        category=str(track.get("category", "unknown")))


# ------------------------------------------------------ corridor geometry

def along(c: Corridor, xy):
    """Signed distance along the centre line, measured FROM THE USER
    (positive = ahead of the user, negative = behind it). Accepts a single
    (2,) point or any array whose last axis is 2, returning a float or an
    array of matching leading shape."""
    p = _as_xy(xy)
    out = (p[..., 0] - c.x) * c.ux + (p[..., 1] - c.y) * c.uy
    return float(out) if np.ndim(out) == 0 else out


def lateral(c: Corridor, xy):
    """Signed distance across the centre line: positive = to the LEFT of
    the user's heading. Same broadcasting rules as along()."""
    p = _as_xy(xy)
    out = -(p[..., 0] - c.x) * c.uy + (p[..., 1] - c.y) * c.ux
    return float(out) if np.ndim(out) == 0 else out


def corridor_contains(c: Corridor, xy, half_width: Optional[float] = None):
    """Is the point inside the rectangle? bool for a single point, a bool
    array for a stack of them.

    half_width overrides the corridor's own, which is how the two questions
    this module asks are kept apart:

      * GAP ACCEPTANCE ("will the plan cross this lane, and can I be out the
        far side in time?") uses the nominal corridor_half_width_m. Being
        wrong by 10 cm there costs a crossing the robot could have made; the
        crossing itself is transient and the timing margin absorbs it.

      * CONTAINMENT ("am I standing in this lane?") uses the wider
        danger_half_width_m -- see danger_corridor(). Being wrong there
        costs a collision, because the answer decides whether the robot
        stands still or steps aside, and the track position it is computed
        from carries a real error (0.1-0.4 m observed).
    """
    w = c.half_width if half_width is None else float(half_width)
    a = along(c, xy)
    l = lateral(c, xy)
    inside = ((a >= -c.back_margin) & (a <= c.length_ahead)
              & (np.abs(l) <= w))
    return bool(inside) if np.ndim(inside) == 0 else inside


def danger_corridor(c: Corridor, params: Dict) -> Corridor:
    """The same lane widened to danger_half_width_m -- the object to ask
    every containment question of. Never narrows a corridor that is already
    wider (a fast, wide user keeps its own width)."""
    w = max(float(params["danger_half_width_m"]), c.half_width)
    return c if w == c.half_width else replace(c, half_width=w)


def refuge_lane_clearance(params: Dict) -> float:
    """The hard minimum distance a refuge or a hold point keeps from a lane
    line, in metres.

    `refuge_lane_clearance_m` is an explicit OVERRIDE and wins whenever it
    is > 0. Otherwise the number is DERIVED from the geometry it is meant
    to encode -- half the widest user, plus our own radius, plus
    `refuge_lane_margin_m` of air between the two bodies -- and floored at
    `refuge_lane_clearance_min_m`. With the shipped defaults that is
    max(0.60, 0.30 + 0.15 + 0.30) = 0.75 m rather than the old flat 1.0 m.

    EVERY "where may I stand" call site asks this, never the raw parameter:
    find_refuge's hard filter, hold_in_place_is_unsafe, the waypoint gate
    (LaneMemory.in_lane) and refuge_recompute_reason -- a floor that
    different call sites disagree about is a robot that steps off a lane
    and is immediately told it is still on one. See the parameter's comment
    in CORRIDOR_DEFAULTS for why 1.0 m was worse than no filter at all in a
    0.97 m aisle.
    """
    override = float(params.get("refuge_lane_clearance_m", 0.0))
    if override > 0.0:
        return override
    derived = (float(params.get("refuge_user_half_width_m", 0.30))
               + float(params.get("refuge_robot_radius_m", 0.15))
               + float(params.get("refuge_lane_margin_m", 0.30)))
    return max(float(params.get("refuge_lane_clearance_min_m", 0.60)),
               derived)


def hold_in_place_is_unsafe(danger: Optional[Corridor], robot_xy,
                            lane_band: Optional[Dict] = None,
                            lanes: Sequence["ObservedLane"] = (),
                            lane_clearance_m: float = 0.0,
                            lane_extension_m: float = 0.0) -> bool:
    """Would standing still leave the robot in the way?

    True when the robot is inside the approaching user's DANGER corridor,
    inside the static lane band, or (2026-09-09) within `lane_clearance_m`
    of a remembered lane line. This is the escalation test: a hold that
    resolves to "stay exactly where you are" is only a yield if where you
    are is out of the traffic. In avoid_panoptex_2 it was not -- 8 of 21
    yields logged "holding in place (the stand-off is already behind us)"
    and one of those was the closest approach of the run (0.15 m), the robot
    parked 0.65 m off the lane centre line while a Carter came down it.

    The lane-line clause is also what keeps the robot from PARKING on a
    waypoint that sits in a lane (waypoint C of the sim triangle is 0.62 m
    off carter1's line): arriving there is a crossing, standing there is
    not a yield.
    """
    if danger is not None and corridor_contains(danger, robot_xy):
        return True
    if in_lane_band(lane_band, robot_xy):
        return True
    if lanes and lane_clearance_m > 0.0:
        return bool(lane_line_clearance(robot_xy, lanes, lane_extension_m)
                    < float(lane_clearance_m))
    return False


def tta(c: Corridor, xy) -> float:
    """Seconds until the user reaches `xy`'s along-coordinate.

    ONLY a positive return means "the user is approaching that point";
    <= 0 means it has already passed it (callers treat that as no
    conflict). A stationary user (speed below the numerical floor) returns
    +inf -- never approaching, never a reason to yield -- which is
    consistent with select_corridor_users() having filtered it out anyway.
    """
    if c.speed < _EPS:
        return float("inf")
    return float(along(c, xy)) / c.speed


def user_passed(c: Corridor, xy_ref, params: Dict) -> bool:
    """Release test: has the user cleared the point we are yielding for?

    True once the reference point (the crossing, or the robot's own
    position for a refuge) is more than `half_width + release_margin_m`
    BEHIND the user. Not simply `along < 0`: at the moment the user's
    centre passes the crossing, its body still occupies the crossing, and
    releasing there would send the X3 into its flank. The extra margin is
    also what keeps this from chattering against the hold test.
    """
    return float(along(c, xy_ref)) < -(c.half_width
                                       + float(params["release_margin_m"]))


# ------------------------------------------------------- gap acceptance

def path_length_to(plan_xy: np.ndarray, idx: int) -> float:
    """Arc length along the plan polyline from its first point to plan[idx].
    The plan nav2 publishes starts at the robot, so this is "how far the
    robot still has to drive to reach that point"."""
    plan = np.asarray(plan_xy, dtype=float).reshape(-1, 2)
    if idx <= 0 or plan.shape[0] < 2:
        return 0.0
    idx = min(idx, plan.shape[0] - 1)
    seg = np.linalg.norm(np.diff(plan[:idx + 1], axis=0), axis=1)
    return float(seg.sum())


def plan_corridor_crossing(plan_xy: np.ndarray, c: Corridor):
    """Where does the global plan enter and leave this corridor?

    Returns None when the plan never enters it, else
    (idx_entry, idx_exit, d_entry, xy_entry, xy_exit):
      idx_entry  first plan index inside the corridor,
      idx_exit   last index of that FIRST CONSECUTIVE run (a plan that
                 re-enters the same lane later -- an out-and-back leg --
                 gets its second crossing decided later, on a later tick,
                 against a freshly rebuilt corridor; deciding both at once
                 would use a 10 s-old prediction for the far one),
      d_entry    arc length from the plan's start (= the robot) to entry,
      xy_entry / xy_exit  the two points themselves.
    """
    plan = np.asarray(plan_xy, dtype=float).reshape(-1, 2)
    if plan.shape[0] == 0:
        return None
    inside = np.asarray(corridor_contains(c, plan)).reshape(-1)
    hits = np.flatnonzero(inside)
    if hits.size == 0:
        return None

    idx_entry = int(hits[0])
    idx_exit = idx_entry
    while idx_exit + 1 < plan.shape[0] and inside[idx_exit + 1]:
        idx_exit += 1

    d_entry = path_length_to(plan, idx_entry)
    return (idx_entry, idx_exit, d_entry,
            plan[idx_entry].copy(), plan[idx_exit].copy())


def t_clear(d_entry: float, c: Corridor, params: Dict) -> float:
    """Seconds the robot needs to get from where it is now to fully out the
    far side of the lane: (d_entry + lane width + margin) / v_cross. See
    the module docstring for why v_cross is deliberately pessimistic."""
    span = 2.0 * c.half_width + float(params["clear_margin_m"])
    v = float(params["v_cross_mps"])
    if v < _EPS:
        return float("inf")
    return (float(d_entry) + span) / v


def decide_crossing(tta_s: float, t_clear_s: float, params: Dict) -> str:
    """"go" | "hold" -- hold iff the user arrives inside the window the
    robot needs, plus t_margin_sec of slack. A non-positive tta (user
    already past) or an infinite one (user stationary) is always "go"."""
    if not math.isfinite(tta_s):
        return "go"
    return ("hold" if 0.0 < tta_s < t_clear_s + float(params["t_margin_sec"])
            else "go")


def crossing_decision(c: Corridor, plan_xy: np.ndarray,
                      params: Dict) -> Optional[Dict]:
    """Convenience wrapper the supervisor calls once per corridor per tick:
    crossing geometry + tta + t_clear + decide_crossing, as one dict, or
    None if the plan does not cross this corridor at all."""
    crossing = plan_corridor_crossing(plan_xy, c)
    if crossing is None:
        return None
    idx_entry, idx_exit, d_entry, xy_entry, xy_exit = crossing
    t_a = tta(c, xy_entry)
    t_c = t_clear(d_entry, c, params)
    return {
        "user": c.user_id,
        "category": c.category,
        "idx_entry": idx_entry,
        "idx_exit": idx_exit,
        "d_entry": d_entry,
        "xy_entry": xy_entry,
        "xy_exit": xy_exit,
        "tta": t_a,
        "t_clear": t_c,
        "decision": decide_crossing(t_a, t_c, params),
    }


def hold_point(plan_xy: np.ndarray, idx_entry: int, hold_back_m: float,
               lane_band: Optional[Dict] = None):
    """Where to wait: walk BACK along the plan from the corridor entry until
    `hold_back_m` of arc length has been covered AND the resulting point is
    outside the static lane band.

    The band condition is what stops the robot waiting for lane 1 while
    parked in lane 2 -- with two lanes 0.8 m apart, a 0.7 m stand-off from
    one of them lands squarely in the other. When the plan runs down the
    band for its whole remaining length there is no such point, and the
    answer is the same as for a stand-off behind the robot: hold in place.

    Returns (idx, xy) for that plan point, or None meaning "hold in place" --
    the walk-back ran off the start of the plan, i.e. the requested standoff
    is behind the robot's current position, so the robot is already at (or
    past) where it should stop and the supervisor cancels the goal instead
    of driving backwards to a stand-off point.
    """
    plan = np.asarray(plan_xy, dtype=float).reshape(-1, 2)
    if plan.shape[0] == 0 or idx_entry <= 0:
        return None
    idx_entry = min(int(idx_entry), plan.shape[0] - 1)
    walked = 0.0
    i = idx_entry
    while i > 0:
        walked += float(np.linalg.norm(plan[i] - plan[i - 1]))
        i -= 1
        if walked >= float(hold_back_m) - 1e-9 \
                and not in_lane_band(lane_band, plan[i]):
            return i, plan[i].copy()
    return None


# ------------------------------------------- crossing policy (WP4)
#
# "no pass has ever been recorded here" -- the same sentinel
# spatial_prior_node.NO_PASS writes into its last_pass_time channel. Kept as
# a literal here rather than imported so corridor.py stays importable with
# no risk_perception on the path (mission_supervisor already guards that
# import; this module must not need one at all).
NO_PASS = -1.0


def lane_speed(flow_sample: Optional[Sequence[float]], params: Dict,
               observed_speed: float = 0.0) -> float:
    """How fast traffic runs through the crossing cell, for sizing the
    approach zone.

    flow_sample: (s, fx, fy) from the crossing cell of
        `/risk_perception/spatial_flow/<c>` (risk_perception's
        sample_flow_grid), or None.

    |F| is the learned raw-velocity EMA, so it IS a speed in m/s -- but it
    is an average, and at a cell used in both directions the two headings
    partially cancel (spatial_prior_node's own "honest limitations"), which
    reads as a slower lane than the real one. Hence the clamp to
    [v_lane_min_mps, v_lane_max_mps] and, on top of it, the floor at the
    speed of the user actually being tracked: the zone must always be long
    enough to contain anything that can reach the crossing in time, or the
    "zone is empty -> cross" branch would be reasoning about a strip too
    short to hold the very vehicle it is deciding about. `observed_speed` is
    NOT capped -- a genuinely fast user lengthens the zone without limit.
    """
    v_min = float(params["v_lane_min_mps"])
    v_max = float(params["v_lane_max_mps"])
    learned = 0.0
    if flow_sample is not None:
        s = float(flow_sample[0])
        if s > 0.0:
            learned = math.hypot(float(flow_sample[1]), float(flow_sample[2]))
    return max(min(max(learned, v_min), v_max), float(observed_speed))


def approach_zone(point, flow_dir, v_lane: float, t_clear_s: float,
                  t_margin_s: float, half_width: float) -> np.ndarray:
    """The strip of lane a user would have to be in to matter at `point`.

    A rectangle starting AT the crossing point and running UPSTREAM (along
    -flow_dir) for `v_lane * (t_clear_s + t_margin_s)` metres,
    `half_width` either side of the lane's centre line through the point.
    That length is not a tuning choice: it is exactly the distance a user
    travelling at `v_lane` covers in the time we need to get across plus the
    margin, so "empty zone" and "nobody can reach the crossing before we are
    out of it" are the same statement.

    Returned as a (4, 2) polygon in walk order (near-left, near-right,
    far-right, far-left), the shape zone_coverage()/zone_users() consume.
    A zero-length zone (a stationary lane, or a degenerate t_clear) still
    returns a valid degenerate polygon rather than None; every consumer
    handles an empty cell set.
    """
    p = _as_xy(point).reshape(2)
    ux, uy, _ = _unit(float(flow_dir[0]), float(flow_dir[1]))
    nx, ny = -uy, ux
    length = max(0.0, float(v_lane)) * max(0.0, float(t_clear_s)
                                           + float(t_margin_s))
    w = float(half_width)
    back = p - np.array([ux, uy]) * length
    return np.array([
        [p[0] + nx * w, p[1] + ny * w],
        [p[0] - nx * w, p[1] - ny * w],
        [back[0] - nx * w, back[1] - ny * w],
        [back[0] + nx * w, back[1] + ny * w],
    ], dtype=float)


def polygon_contains(poly, xy):
    """Even-odd point-in-polygon, vectorised over `xy` (a single (2,) point
    or any array whose last axis is 2), returning a bool or a bool array.
    General rather than a convex half-plane test so the same helper works if
    a zone ever stops being a rectangle."""
    p = _as_xy(xy)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    px, py = p[..., 0], p[..., 1]
    inside = np.zeros(np.shape(px), dtype=bool)
    if poly.shape[0] < 3:
        return bool(inside) if np.ndim(inside) == 0 else inside
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x0, y0 = poly[i - 1]
        if abs(y1 - y0) < _EPS:
            continue
        straddles = (y1 > py) != (y0 > py)
        x_cross = x1 + (py - y1) * (x0 - x1) / (y0 - y1)
        inside ^= straddles & (px < x_cross)
    return bool(inside) if np.ndim(inside) == 0 else inside


def _grid_cells_in(poly, grid: Dict):
    """(values, mask) for the cells of `grid` whose centres fall inside
    `poly`. `grid` is {"values" (2D, row 0 at the origin), "resolution",
    "origin_x", "origin_y"} -- the same convention as make_lane_band's
    band dict and as OccupancyGrid.data. Returns (None, None) when the
    polygon misses the grid entirely."""
    values = np.asarray(grid["values"])
    if values.ndim != 2 or values.size == 0:
        return None, None
    res = float(grid["resolution"])
    if res < _EPS:
        return None, None
    ox, oy = float(grid["origin_x"]), float(grid["origin_y"])
    rows, cols = values.shape
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)

    c0 = max(0, int(math.floor((poly[:, 0].min() - ox) / res)))
    c1 = min(cols - 1, int(math.ceil((poly[:, 0].max() - ox) / res)))
    r0 = max(0, int(math.floor((poly[:, 1].min() - oy) / res)))
    r1 = min(rows - 1, int(math.ceil((poly[:, 1].max() - oy) / res)))
    if c1 < c0 or r1 < r0:
        return None, None

    cc = np.arange(c0, c1 + 1, dtype=float).reshape(1, -1)
    rr = np.arange(r0, r1 + 1, dtype=float).reshape(-1, 1)
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    wx = np.broadcast_to(ox + (cc + 0.5) * res, shape)
    wy = np.broadcast_to(oy + (rr + 0.5) * res, shape)
    inside = np.asarray(polygon_contains(poly, np.stack([wx, wy], axis=-1)))
    return values[r0:r1 + 1, c0:c1 + 1], inside


def grid_value_at(grid: Optional[Dict], xy, default: float = 0.0) -> float:
    """Nearest-cell lookup by WORLD coordinate, `default` off the grid or
    with no grid at all. Same geometry contract as _grid_cells_in."""
    if grid is None:
        return float(default)
    values = np.asarray(grid["values"])
    if values.ndim != 2 or values.size == 0:
        return float(default)
    res = float(grid["resolution"])
    if res < _EPS:
        return float(default)
    p = _as_xy(xy).reshape(2)
    col = int(math.floor((p[0] - float(grid["origin_x"])) / res))
    row = int(math.floor((p[1] - float(grid["origin_y"])) / res))
    rows, cols = values.shape
    if 0 <= row < rows and 0 <= col < cols:
        return float(values[row, col])
    return float(default)


def zone_coverage(zone, coverage_grid: Optional[Dict],
                  min_value: float = 50.0) -> float:
    """Fraction of the zone's cells that SOME sensor covers, in [0, 1].

    coverage_grid: {"values", "resolution", "origin_x", "origin_y"} built
        from /risk_perception/coverage (0 or 100).

    Zero when there is no coverage grid at all, when the zone falls off the
    grid, and when the zone is degenerate. Every one of those is "we cannot
    justify a claim to see this", which is the direction that makes the
    caller MORE careful -- an exposure map that fails open would be worse
    than not having one.
    """
    if coverage_grid is None:
        return 0.0
    values, inside = _grid_cells_in(zone, coverage_grid)
    if values is None or not inside.any():
        return 0.0
    return float((values[inside] >= float(min_value)).mean())


def zone_users(zone, tracks: Sequence[Dict]) -> List[Dict]:
    """The tracks whose centre lies inside the zone.

    Pass the CORRIDOR USERS (select_corridor_users' output), not the raw
    world model: the robot's own reflection and every parked pallet would
    otherwise count as somebody about to drive through the crossing.
    """
    out: List[Dict] = []
    for tr in tracks:
        if polygon_contains(zone, (float(tr["x"]), float(tr["y"]))):
            out.append(tr)
    return out


def _prefers_category(candidate: Dict, current: Dict) -> bool:
    """crossing_choice's tie-break ONLY: True when `candidate`'s category
    outranks `current`'s -- CORRIDOR_CATEGORIES (robot/wheeled) preferred
    over anything else. Never consulted unless the S-score above is
    already an exact tie with the incumbent (see crossing_choice) --
    category is not evidence about WHERE to cross, only a preference
    between two otherwise-equal options (WP-A, 2026-09-11: now that
    select_corridor_users() no longer restricts corridor ownership to
    CORRIDOR_CATEGORIES by default, a crossing's `category` key may be
    "furniture"/"unknown" too, and this keeps the tie-break's own
    preference unchanged rather than becoming meaningless)."""
    return (candidate.get("category") in CORRIDOR_CATEGORIES
            and current.get("category") not in CORRIDOR_CATEGORIES)


def crossing_choice(plan_crossings: Sequence[Dict],
                    s_grid: Optional[Dict] = None,
                    samples: int = 21) -> Optional[Dict]:
    """"Choose where to cross": of several crossings, the one whose cells
    carry the LOWEST learned occupancy S.

    plan_crossings: dicts as returned by crossing_decision() (`xy_entry` /
        `xy_exit` for the S-score, `category` for the tie-break below), in
        plan order.
    s_grid: {"values", "resolution", "origin_x", "origin_y"} from
        /risk_perception/spatial_prior (0-100), or None.

    Each crossing is scored by the mean S over `samples` points along its
    entry->exit segment. An outright-better score (beyond a small epsilon)
    always wins; an EXACT tie breaks by category (WP-A, 2026-09-11) --
    prefer CORRIDOR_CATEGORIES (robot/wheeled) over a furniture/unknown
    corridor user (see _prefers_category) -- and only then falls back to
    whichever crossing came first, i.e. the nearest one along the plan. A
    missing/unusable grid, or a single crossing, also falls back to the
    FIRST crossing outright. That fallback is why this is safe to call
    unconditionally.

    Note what this does NOT do: nothing in this stack can re-route Nav2's
    global plan, so the supervisor reports the preference rather than
    steering by it (see mission_supervisor._publish_state's `crossing`
    block). The ranking is the paper's; acting on it needs a route layer
    that does not exist yet.
    """
    crossings = list(plan_crossings)
    if not crossings:
        return None
    if s_grid is None or len(crossings) == 1:
        return crossings[0]
    best, best_score = crossings[0], float("inf")
    n = max(2, int(samples))
    for info in crossings:
        a = _as_xy(info["xy_entry"]).reshape(2)
        b = _as_xy(info["xy_exit"]).reshape(2)
        ts = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        pts = a.reshape(1, 2) + ts * (b - a).reshape(1, 2)
        vals = [grid_value_at(s_grid, p, default=0.0) for p in pts]
        score = float(np.mean(vals)) if vals else float("inf")
        if score < best_score - 1e-9:
            best, best_score = info, score
        elif abs(score - best_score) <= 1e-9 and _prefers_category(info, best):
            best = info
    return best


def blind_crossing_ok(now: float, last_pass_time: Optional[float],
                      headway_mean: float, headway_count: float,
                      visible_zone_empty: bool, t_arrived: float,
                      params: Dict) -> bool:
    """May we cross a lane whose approach zone we cannot see?

    now:            current clock, same base as last_pass_time/t_arrived.
    last_pass_time: when a user was last OBSERVED passing this crossing, or
                    NO_PASS/None for "never / unknown".
    headway_mean, headway_count: the learned statistic at the crossing cell
                    (risk_perception's /risk_perception/spatial_headway/<c>).
    visible_zone_empty: no tracked user inside the part of the zone that IS
                    covered. Necessary in every branch -- a statistic never
                    overrules something we can actually see.
    t_arrived:      when we started waiting here (the refuge/hold pose), or
                    < 0 for "not waiting yet".

    Two branches:
      * with `headway_min_count` samples of evidence AND a known last pass,
        wait one `headway_factor` x mean headway since that pass. This is
        the real rule: "longer than a typical gap has gone by, so the lane
        is between vehicles."
      * otherwise -- a cold prior, or a crossing nothing has ever been seen
        using -- there is no statistic to wait for, so wait a bounded
        `blind_wait_max_s` and then go. Waiting forever for evidence that
        will never arrive is a stopped robot, and the MPPI controller plus
        the collision monitor remain underneath either way.
    """
    if not visible_zone_empty:
        return False
    have_pass = (last_pass_time is not None
                 and float(last_pass_time) >= 0.0)
    if float(headway_count) >= float(params["headway_min_count"]) and have_pass:
        return (float(now) - float(last_pass_time)
                >= float(headway_mean) * float(params["headway_factor"]))
    if t_arrived is None or float(t_arrived) < 0.0:
        return False
    return float(now) - float(t_arrived) >= float(params["blind_wait_max_s"])


def crossing_policy(info: Dict, c: Corridor, params: Dict,
                    coverage_grid: Optional[Dict] = None,
                    tracks: Sequence[Dict] = (),
                    flow_sample: Optional[Sequence[float]] = None,
                    now: float = 0.0,
                    last_pass_time: Optional[float] = NO_PASS,
                    headway_mean: float = 0.0,
                    headway_count: float = 0.0,
                    t_arrived: float = -1.0) -> Dict:
    """The full crossing decision -- coverage, then users, then timing or
    statistics. See the module docstring's "CROSSING A LANE YOU CANNOT SEE".

    info: one crossing_decision() dict (xy_entry / tta / t_clear / decision).
    c:    the corridor that crossing belongs to; its heading is the lane
          direction the zone is built along (already flow-snapped by
          make_corridor, so the zone follows the learned lane rather than a
          jittering instantaneous velocity).

    Returns {"decision": "go"|"hold", "mode": "seen"|"gap"|"blind",
             "zone", "zone_cov", "zone_users", "v_lane", "wait_s"}.

    THE INVARIANT: the returned decision is "hold" whenever
    decide_crossing() alone would have held. The coverage/headway branches
    can only ADD holds -- specifically the ones the timing test waved
    through because nothing was tracked in a strip nobody could see. Modes:

      "gap"   the classic answer (also the answer when no coverage grid has
              arrived at all -- with no exposure map this degrades exactly
              to the pre-WP4 behaviour, which is what makes the coverage
              node optional rather than load-bearing),
      "seen"  covered, and nothing in the zone that could reach us in time,
      "blind" not covered; blind_crossing_ok() decided.
    """
    classic = decide_crossing(float(info["tta"]), float(info["t_clear"]),
                              params)
    wait_s = 0.0 if (t_arrived is None or float(t_arrived) < 0.0) \
        else max(0.0, float(now) - float(t_arrived))

    if coverage_grid is None:
        return {"decision": classic, "mode": "gap", "zone": None,
                "zone_cov": None, "zone_users": [], "v_lane": 0.0,
                "wait_s": wait_s}

    v_lane = lane_speed(flow_sample, params, observed_speed=c.speed)
    zone = approach_zone(info["xy_entry"], (c.ux, c.uy), v_lane,
                         float(info["t_clear"]),
                         float(params["t_margin_sec"]),
                         float(params["danger_half_width_m"]))
    cell_min = float(params["coverage_cell_min"])
    cov = zone_coverage(zone, coverage_grid, cell_min)
    users = zone_users(zone, tracks)

    if cov >= float(params["coverage_min"]):
        mode = "gap" if users else "seen"
        decision = classic if users else "go"
    else:
        mode = "blind"
        visible = [u for u in users
                   if grid_value_at(coverage_grid, (u["x"], u["y"]), 0.0)
                   >= cell_min]
        decision = "go" if blind_crossing_ok(
            now, last_pass_time, headway_mean, headway_count,
            not visible, t_arrived, params) else "hold"

    if classic == "hold":
        decision = "hold"          # the invariant, enforced in one place
    return {"decision": decision, "mode": mode, "zone": zone,
            "zone_cov": cov, "zone_users": [str(u.get("id", "?"))
                                            for u in users],
            "v_lane": v_lane, "wait_s": wait_s}


def loss_release_guard(corridors: Sequence[Corridor], xy_ref,
                       params: Dict) -> Optional[Corridor]:
    """Is it safe to stop yielding for `xy_ref` when the track we were
    yielding for has simply DISAPPEARED?

    Returns the corridor of the soonest-arriving user that still sweeps
    `xy_ref` within t_yield_sec, or None if nobody does. IDENTITY IS NOT
    CONSIDERED: a lost track is far more often a re-identification than a
    departure (one Carter carried six different ids inside one 700 s run),
    so "the track we were watching is gone" is worthless as evidence that
    the lane is clear -- only the geometry of whoever is currently tracked
    is. Both conditions matter: the point must be INSIDE the corridor (a
    user on a parallel lane has a perfectly positive tta to the same
    along-coordinate and would otherwise block the release forever) and the
    user must still be approaching it.
    """
    best = None
    best_tta = float("inf")
    for c in corridors:
        if not corridor_contains(c, xy_ref):
            continue
        t_a = tta(c, xy_ref)
        if 0.0 < t_a < float(params["t_yield_sec"]) and t_a < best_tta:
            best, best_tta = c, t_a
    return best


# ------------------------------------------------------------ lane band

def _shift_or(dst: np.ndarray, src: np.ndarray, dr: int, dc: int) -> None:
    """dst |= src shifted by (dr, dc), clipped at the array edges."""
    rows, cols = src.shape
    r0d, r1d = max(0, dr), rows + min(0, dr)
    c0d, c1d = max(0, dc), cols + min(0, dc)
    if r0d >= r1d or c0d >= c1d:
        return
    dst[r0d:r1d, c0d:c1d] |= src[r0d - dr:r1d - dr, c0d - dc:c1d - dc]


def _dilate_disc(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation by a disc of `radius_cells`. scipy where available,
    otherwise an OR of the shifted copies -- identical result, just slower,
    and this runs once per spatial-prior message, not per tick."""
    if radius_cells <= 0:
        return mask.copy()
    r = int(radius_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    if _binary_dilation is not None:
        return _binary_dilation(mask, structure=disc)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if disc[dr + r, dc + r]:
                _shift_or(out, mask, dr, dc)
    return out


def make_lane_band(values: np.ndarray, info: Dict, min_value: float,
                   dilate_m: float) -> Dict:
    """The STATIC traffic lanes, from the Spatial-Prior occupancy grid.

    An `active corridor` only exists while a user is being tracked driving
    down it. That is not enough to pick a refuge by: in the warehouse aisle
    the two AMR lanes are 0.8 m apart, and stepping out of the one with a
    live track into the one whose Carter happens to be round the corner is
    not stepping aside at all -- it is queueing for the next encounter.
    (Measured: avoid_panoptex_1 chose four refuges inside the aisle and then
    thrashed through six recomputes as the second lane woke up.)

    `/risk_perception/spatial_prior` already knows where traffic goes: it is
    the element-wise max over every category's learned occupancy channel S,
    accumulated across the whole run (see spatial_prior_node.py's
    docstring). Thresholding it at `min_value` and dilating by `dilate_m`
    (= the corridor half width, so the band covers a body width either side
    of the learned centre of travel, exactly like a live corridor does)
    gives a static keep-out that holds whether or not anyone is currently
    driving down it.

    values: the OccupancyGrid data as a 2D array (rows, cols), row 0 at the
        origin -- the same convention as find_refuge's map_grid.
    info:   {"resolution", "origin_x", "origin_y"} of THAT grid; it need not
            match the nav map's, because in_lane_band() looks candidates up
            by world coordinate rather than by index.

    Returns the band as a dict {"mask", "resolution", "origin_x",
    "origin_y"}; feed it to find_refuge()/hold_point()/in_lane_band().
    Note that unknown cells (-1) are below any sensible min_value and so are
    never band -- "never observed" is not "known to be a lane".
    """
    vals = np.asarray(values)
    res = float(info["resolution"])
    seed = vals >= float(min_value)
    radius_cells = int(math.ceil(float(dilate_m) / res)) if res > _EPS else 0
    return {
        "mask": _dilate_disc(seed, radius_cells),
        "resolution": res,
        "origin_x": float(info["origin_x"]),
        "origin_y": float(info["origin_y"]),
    }


def in_lane_band(band: Optional[Dict], xy):
    """Is the point inside the static lane band? Nearest-cell lookup by
    world coordinate (so the band may be on its own grid geometry); outside
    the band's extent, and for band=None, the answer is False -- "no
    evidence" must never block a refuge."""
    p = _as_xy(xy)
    if band is None:
        out = np.zeros(p.shape[:-1], dtype=bool)
        return bool(out) if out.ndim == 0 else out
    mask = band["mask"]
    rows, cols = mask.shape
    res = float(band["resolution"])
    col = np.floor((p[..., 0] - band["origin_x"]) / res).astype(int)
    row = np.floor((p[..., 1] - band["origin_y"]) / res).astype(int)
    ok = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    out = ok & mask[np.clip(row, 0, rows - 1), np.clip(col, 0, cols - 1)]
    return bool(out) if np.ndim(out) == 0 else out


# ------------------------------------------------- observed-lane memory

# A track's fit is restarted rather than extended once its id has been
# silent for this multiple of lost_timeout_sec: ids are recycled, and
# joining a vanished Carter's start point to a new one's position would mint
# a lane across floor nothing ever drove on.
_LANE_FIT_RESTART_FACTOR = 2.0

# Below this the observed start->latest displacement is noise, not a
# direction, so the segment takes the corridor's own (flow-snapped) heading.
_LANE_FIT_MIN_LEN_M = 0.25

# The running fit is count-weighted, and the weight saturates here: after
# ~50 merges one more observation must not still be able to swing the line.
_LANE_BLEND_MAX_COUNT = 50

# Fallbacks for the lane-hygiene parameters, used only when a caller hands
# LaneMemory a params dict that predates them (every in-tree caller builds it
# from CORRIDOR_DEFAULTS, so these are belt and braces).
_LANE_MEMORY_MAX = 8
_LANE_MERGE_GAP_M = 3.0
_LANE_MIN_LENGTH_M = 1.5
_LANE_MIN_POINTS = 5

# A committed refuge is disturbed by lane geometry only once its clearance
# has fallen this far BELOW refuge_lane_clearance() -- plain hysteresis, so
# a lane whose fitted line creeps by a centimetre cannot cancel a goal.
LANE_CLEARANCE_SLACK_M = 0.2


@dataclass
class ObservedLane:
    """One line somebody has actually been observed driving along.

    (x0, y0) -> (x1, y1) are the extremes of everything merged into it and
    (ux, uy) is the unit direction from the first to the second, so the
    segment is fully described by the endpoints alone; the direction is
    carried because merging needs it before the endpoints are recomputed.

    Mutable, unlike Corridor: a Corridor is one tick's snapshot and is
    rebuilt from scratch every tick, whereas a lane is session state that
    accumulates. LaneMemory replaces list entries wholesale rather than
    mutating in place, but the dataclass is left unfrozen so a caller
    (a test, the marker publisher) can build one directly.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    ux: float
    uy: float
    last_seen: float
    count: int = 1
    users: Tuple[str, ...] = ()

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.x0, self.y0), (self.x1, self.y1)


def make_observed_lane(p0, p1, heading, now: float,
                       user: str = "?") -> ObservedLane:
    """One segment, from `p0` to `p1`, oriented by the endpoints where they
    are far enough apart to mean anything and by `heading` (the corridor's
    flow-snapped unit heading) where they are not -- a track seen once has a
    heading but no extent, and still deposits a lane."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    ux, uy, n = _unit(x1 - x0, y1 - y0)
    if n < _LANE_FIT_MIN_LEN_M:
        ux, uy, _ = _unit(float(heading[0]), float(heading[1]))
    return ObservedLane(x0=x0, y0=y0, x1=x1, y1=y1, ux=ux, uy=uy,
                        last_seen=float(now), count=1, users=(str(user),))


def lane_along_gap(a: ObservedLane, b: ObservedLane) -> float:
    """Along-line separation of `b`'s extent from `a`'s, in `a`'s frame.

    0.0 whenever the two projections overlap at all; otherwise the metres of
    line between them. Purely longitudinal -- the lateral half of "same
    line?" is the midpoint offset in lanes_collinear.
    """
    ts = [(px - a.x0) * a.ux + (py - a.y0) * a.uy
          for px, py in ((b.x0, b.y0), (b.x1, b.y1))]
    t0, t1 = min(ts), max(ts)
    return float(max(0.0, t0 - a.length, -t1))


def lanes_collinear(a: ObservedLane, b: ObservedLane, params: Dict) -> bool:
    """Do these two segments describe the same line?

    Three tests, ALL of them taken in `a`'s own frame:

      * UNDIRECTED angle (|cos|, so a patrol's two directions of travel are
        the same lane -- the whole reason this is a line and not a corridor)
        within lane_merge_angle_deg;
      * `b`'s MIDPOINT within lane_merge_dist_m of `a`'s infinite line;
      * the along-line gap between the two extents at most
        lane_merge_gap_m (0 when they overlap).

    The midpoint, not both endpoints (mppi_panoptex_3, 2026-09-09). A 2-4 m
    piece of a jittering track tilts by up to ~20 deg about its centre, which
    swings an endpoint by more than half a metre while the midpoint barely
    moves; testing endpoints therefore refused merges the geometry plainly
    wanted, and the memory fragmented into 32 lanes over two patrol lines --
    each fragment then demanding refuge_lane_clearance_m of its own, which
    left no legal refuge within 2.5 m of anywhere in the aisle. The angle
    test is what keeps a genuinely different line out; the midpoint test is
    what keeps the same line in.

    The gap test is new for the opposite reason: longitudinal separation used
    to be ignored entirely, so two collinear stretches at opposite ends of
    the hall merged into one lane whose fitted line covered the floor
    between them, which nothing had ever driven on.
    """
    cos_min = math.cos(math.radians(float(params["lane_merge_angle_deg"])))
    if abs(a.ux * b.ux + a.uy * b.uy) < cos_min:
        return False
    mx, my = (b.x0 + b.x1) * 0.5, (b.y0 + b.y1) * 0.5
    if abs(-(mx - a.x0) * a.uy + (my - a.y0) * a.ux) > \
            float(params["lane_merge_dist_m"]):
        return False
    gap_max = float(params.get("lane_merge_gap_m", _LANE_MERGE_GAP_M))
    return lane_along_gap(a, b) <= gap_max


def lanes_mergeable(a: ObservedLane, b: ObservedLane, params: Dict) -> bool:
    """lanes_collinear() made symmetric -- the compaction pass has no
    "existing" and "new" segment to privilege, and a long lane's frame and a
    short one's disagree about the midpoint offset whenever the short one is
    the tilted fragment."""
    return (lanes_collinear(a, b, params)
            or lanes_collinear(b, a, params))


def lane_is_effective(lane: ObservedLane, params: Dict) -> bool:
    """Has this lane earned the right to keep a refuge out?

    Only lanes observed over at least lane_min_length_m AND from at least
    lane_min_points sightings count for the refuge_lane_clearance() rule.
    Everything shorter is still remembered -- it grows into a real lane if
    traffic keeps using it -- but a stub left by a track rounding a corner,
    or by a one-second glimpse, must not block the only refuge in the aisle
    (mppi_panoptex_3: 32 remembered lanes, refuges at 3.5 m).
    """
    min_len = float(params.get("lane_min_length_m", _LANE_MIN_LENGTH_M))
    min_pts = int(float(params.get("lane_min_points", _LANE_MIN_POINTS)))
    return lane.length >= min_len and int(lane.count) >= min_pts


def merge_lanes(a: ObservedLane, b: ObservedLane, now: float) -> ObservedLane:
    """`a` updated to also cover `b`: a count-weighted running fit.

    Both sides are weighted by their own observation count (saturating at
    _LANE_BLEND_MAX_COUNT), then all four endpoints are projected onto the
    blended line and the extremes kept -- so the lane only ever grows in
    extent while its line itself settles down as evidence accumulates. A
    fresh sighting carries count 1, which is exactly the old
    "a.count against 1" behaviour; the weights matter only for the
    compaction pass, where two established lanes meet and the better-observed
    one must win.
    """
    w = float(min(int(a.count), _LANE_BLEND_MAX_COUNT))
    wb = float(min(max(int(b.count), 1), _LANE_BLEND_MAX_COUNT))
    sign = 1.0 if (a.ux * b.ux + a.uy * b.uy) >= 0.0 else -1.0
    ux, uy, n = _unit(w * a.ux + sign * wb * b.ux,
                      w * a.uy + sign * wb * b.uy)
    if n < _EPS:
        ux, uy = a.ux, a.uy

    a_mid = ((a.x0 + a.x1) * 0.5, (a.y0 + a.y1) * 0.5)
    b_mid = ((b.x0 + b.x1) * 0.5, (b.y0 + b.y1) * 0.5)
    ox = (w * a_mid[0] + wb * b_mid[0]) / (w + wb)
    oy = (w * a_mid[1] + wb * b_mid[1]) / (w + wb)

    ts = [(px - ox) * ux + (py - oy) * uy
          for px, py in ((a.x0, a.y0), (a.x1, a.y1),
                         (b.x0, b.y0), (b.x1, b.y1))]
    t0, t1 = min(ts), max(ts)
    users = tuple(dict.fromkeys(a.users + b.users))[:8]
    return ObservedLane(x0=ox + t0 * ux, y0=oy + t0 * uy,
                        x1=ox + t1 * ux, y1=oy + t1 * uy,
                        ux=ux, uy=uy, last_seen=float(now),
                        count=int(a.count) + max(int(b.count), 1),
                        users=users)


def lane_line_clearance(xy, lanes: Sequence[ObservedLane],
                        extension_m: float = 0.0):
    """Perpendicular distance from `xy` to the NEAREST remembered lane.

    Each segment is extended by `extension_m` beyond both observed
    endpoints before the distance is taken -- we only ever watched part of
    the patrol (carter1 runs y = -4 to y = 7; the cameras see rather less),
    and the unwatched continuation is exactly as much of a lane as the
    watched part.

    Same broadcasting rules as along()/lateral(): a (2,) point gives a
    float, any array whose last axis is 2 gives an array of the matching
    leading shape. With no lanes remembered the answer is +inf -- "no
    evidence" must never block a refuge, the same rule in_lane_band()
    follows for a missing band.
    """
    p = _as_xy(xy)
    ext = max(0.0, float(extension_m))
    best = np.full(p.shape[:-1], np.inf, dtype=float)
    for ln in lanes:
        length = ln.length
        t = (p[..., 0] - ln.x0) * ln.ux + (p[..., 1] - ln.y0) * ln.uy
        t = np.clip(t, -ext, length + ext)
        best = np.minimum(best, np.hypot(p[..., 0] - (ln.x0 + t * ln.ux),
                                         p[..., 1] - (ln.y0 + t * ln.uy)))
    return float(best) if np.ndim(best) == 0 else best


class LaneMemory:
    """Session-local memory of where corridor users have actually driven.

    Pure state + geometry -- no clock of its own, no ROS: the caller passes
    `now` in and holds the instance (mission_supervisor does, for the life
    of the node). `params` is kept BY REFERENCE so a live parameter change
    reaches it the same way it reaches every other function here.

    Two levels, and the split matters:

      * per track id, a running fit (first position seen -> latest position
        seen, plus the corridor's flow-snapped heading). Track ids churn --
        one Carter carried six inside one run -- and a fit is restarted
        rather than extended when its id has been silent long enough that
        it is probably a different vehicle.
      * across ids, the merged `lanes`: every fit that is collinear with an
        existing lane and close to it extends that lane instead of adding
        another. This is what turns six ids' worth of fragments into one
        line at x = 1.27.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.lanes: List[ObservedLane] = []
        # user id -> {"x0","y0","x1","y1","hx","hy","t"}
        self._fits: Dict[str, Dict[str, float]] = {}

    # -- observation ----------------------------------------------------

    def observe(self, user_id, xy, heading, now: float) -> None:
        """Deposit one tick's sighting of one corridor user.

        `heading` is the corridor's (ux, uy) -- flow-snapped where the
        learned flow agreed, raw velocity otherwise; see make_corridor.
        Callers pass ONLY corridor users (category, pmot_min,
        min_user_speed_mps already applied by select_corridor_users): a
        parked forklift has no lane, and neither does a person.
        """
        key = str(user_id)
        x, y = float(xy[0]), float(xy[1])
        restart_after = (_LANE_FIT_RESTART_FACTOR
                         * float(self.params["lost_timeout_sec"]))
        fit = self._fits.get(key)
        if fit is None or float(now) - fit["t"] > restart_after:
            fit = {"x0": x, "y0": y, "x1": x, "y1": y,
                   "hx": 1.0, "hy": 0.0, "t": float(now)}
            self._fits[key] = fit
        fit["x1"], fit["y1"] = x, y
        fit["hx"], fit["hy"], _ = _unit(float(heading[0]), float(heading[1]))
        fit["t"] = float(now)

        self._merge(make_observed_lane((fit["x0"], fit["y0"]), (x, y),
                                       (fit["hx"], fit["hy"]), now, key),
                    now)

    def _merge(self, seg: ObservedLane, now: float) -> None:
        """Fold one sighting's segment in, then re-compact and cap.

        The insert alone is not enough (mppi_panoptex_3): it merges the new
        segment into the FIRST lane it is collinear with, which leaves two
        lanes that were never collinear with each other still separate even
        after a third observation has made them so. The compaction pass
        below closes that transitively, so a line that arrives in pieces
        ends up as one lane however the pieces were ordered.
        """
        for i, lane in enumerate(self.lanes):
            if lanes_collinear(lane, seg, self.params):
                self.lanes[i] = merge_lanes(lane, seg, now)
                break
        else:
            self.lanes.append(seg)
        self._compact(now)
        self._cap()

    def _compact(self, now: float) -> None:
        """Merge every pair of remembered lanes that now satisfies the merge
        test, repeatedly, until none does.

        Transitive by construction: merging A with B produces a lane whose
        blended line may in turn be mergeable with C, so the scan restarts
        after every merge. Bounded by lane_max (and by the fact that each
        pass strictly shortens the list), so the loop cannot run away.
        """
        merged = True
        while merged and len(self.lanes) > 1:
            merged = False
            for i in range(len(self.lanes)):
                for j in range(i + 1, len(self.lanes)):
                    a, b = self.lanes[i], self.lanes[j]
                    if not lanes_mergeable(a, b, self.params):
                        continue
                    # Keep the better-observed lane as the frame the merge
                    # is taken in, and never move `last_seen` forward: a
                    # compaction is bookkeeping, not a sighting.
                    if int(b.count) > int(a.count):
                        a, b = b, a
                    self.lanes[i] = merge_lanes(
                        a, b, max(a.last_seen, b.last_seen))
                    del self.lanes[j]
                    merged = True
                    break
                if merged:
                    break

    def _cap(self) -> None:
        """Keep at most lane_max lanes, dropping the shortest/oldest first.

        Ranked by (does it count for the clearance rule, how long, how
        recently seen): a 0.4 m stub from a track rounding a corner is what
        should go when the memory is full, never the patrol line.
        """
        limit = max(1, int(float(self.params.get("lane_max",
                                                 _LANE_MEMORY_MAX))))
        if len(self.lanes) <= limit:
            return
        self.lanes.sort(
            key=lambda ln: (1 if lane_is_effective(ln, self.params) else 0,
                            ln.length, ln.last_seen),
            reverse=True)
        del self.lanes[limit:]

    def expire(self, now: float) -> int:
        """Drop lanes (and dead per-track fits) older than lane_memory_s.
        Returns how many lanes are left."""
        ttl = float(self.params["lane_memory_s"])
        self.lanes = [ln for ln in self.lanes
                      if float(now) - ln.last_seen <= ttl]
        self._fits = {k: f for k, f in self._fits.items()
                      if float(now) - f["t"] <= ttl}
        return len(self.lanes)

    # -- queries --------------------------------------------------------

    def clearance(self, xy):
        """lane_line_clearance() against EVERY remembered lane, with the
        configured lane_extension_m already applied. Raw geometry, for
        telemetry and for callers that want the memory as it stands; the
        refuge_lane_clearance() rule uses effective_clearance() below."""
        return lane_line_clearance(xy, self.lanes,
                                   float(self.params["lane_extension_m"]))

    def effective_lanes(self) -> List[ObservedLane]:
        """The remembered lanes that carry enough evidence to keep a refuge
        out -- see lane_is_effective(). This is the list every "where may I
        stand" call site should be handed."""
        return [ln for ln in self.lanes
                if lane_is_effective(ln, self.params)]

    def effective_clearance(self, xy):
        """clearance() over effective_lanes() only. Fragments do not vote,
        so a hall full of 0.8 m stubs cannot empty the refuge disc."""
        return lane_line_clearance(xy, self.effective_lanes(),
                                   float(self.params["lane_extension_m"]))

    def in_lane(self, xy) -> bool:
        """Is this point too close to a remembered lane to stand on?
        The single question the refuge search, the hold-in-place escalation
        and the waypoint gate all ask -- and it is asked of the EFFECTIVE
        lanes only."""
        return bool(self.effective_clearance(xy)
                    < refuge_lane_clearance(self.params))


def refuge_recompute_reason(target_xy, danger_corridors: Sequence[Corridor],
                            lanes: Sequence[ObservedLane],
                            params: Dict) -> Optional[Dict]:
    """Why a COMMITTED hold/refuge point is no longer good enough -- or None
    while it still is.

    This is the hysteresis half of the lane-line fix. The old test ("any
    danger corridor contains the target and its user arrives within
    t_yield_sec") re-fired every time a corridor window slid over a point
    the robot was already parked on, and each firing cancelled the goal:
    nine recompute cycles inside one mppi_panoptex_2 contact episode, status
    6 every time, the robot still in the lane at the end of them. Committing
    to a refuge is worth something in itself, so only two things may
    disturb one:

      * a user whose CLOSEST APPROACH to the committed point is inside
        refuge_recompute_tta_s -- a genuinely imminent sweep, not a window
        edge. For a constant-velocity corridor the closest approach happens
        exactly when the user's along-coordinate reaches the point's, so
        this is tta(); a NEGATIVE tta means the approach is in the past and
        the user is receding, which is not a reason for anything.
      * the point's own lane-line clearance having dropped more than
        LANE_CLEARANCE_SLACK_M below refuge_lane_clearance(), i.e. a newly
        remembered lane has been laid over where we are standing.

    `danger_corridors` must already be widened (danger_corridor()): this is
    a containment question. Returns {"why", "corridor", "tta", "clearance"}.
    """
    horizon = float(params["refuge_recompute_tta_s"])
    for c in danger_corridors:
        if not corridor_contains(c, target_xy):
            continue
        t_a = tta(c, target_xy)
        if 0.0 < t_a < horizon:
            return {"why": "corridor", "corridor": c, "tta": float(t_a),
                    "clearance": None}

    need = refuge_lane_clearance(params) - LANE_CLEARANCE_SLACK_M
    gap = float(lane_line_clearance(target_xy, lanes,
                                    float(params["lane_extension_m"])))
    if gap < need:
        return {"why": "lane", "corridor": None, "tta": None,
                "clearance": gap}
    return None


# --------------------------------------------------- lane-crossing guards
#
# NEVER CROSS A LANE TO REACH A REFUGE (mppi_panoptex_4, 2026-09-09). A
# refuge is somewhere to be while somebody drives past. Reaching one on the
# FAR side of that somebody's lane means driving the whole lane -- both
# perpendicular offsets plus twice the danger half width -- at the same
# pessimistic v_cross_mps the gap-acceptance rule uses, and nothing checked
# that clock: the run left (0.82, 0.57) for a refuge at (2.73, 1.93) with
# carter1 6.5 s out, needed >= 7 s of crossing, and was met in the middle of
# the lane ("speed cap 60% ... t_cpa_s 0.24, d_cpa_m 0.39", then released
# standing ON the centre line at (1.27, 1.85) and pushed to (1.31, 6.87)).
#
# So the far side of a lane is now a HARD rejection unless the timing says
# otherwise -- the same test gap acceptance makes at a crossing, applied to
# the refuge search, which is the one place that was still choosing to
# cross a lane without asking.


@dataclass(frozen=True)
class LaneGuard:
    """One lane line that must not be crossed to reach a refuge.

    x, y, ux, uy: a point on the line and its unit direction -- a corridor
        user's own centre line, or a remembered lane somebody is currently
        driving down.
    half_width: that lane's half width, the band to be cleared on top of
        the two perpendicular offsets.
    tta_s: seconds until the guard's user reaches the ROBOT's along
        coordinate -- the same number the yield decision itself is made on.
        <= 0 (already past) or +inf (nobody on it) means this guard never
        blocks anything, exactly as decide_crossing treats them.
    """

    x: float
    y: float
    ux: float
    uy: float
    half_width: float
    tta_s: float
    source: str = "user"     # "user" | "lane" -- logging / debugging only
    user_id: str = "?"


def guard_offset(g: LaneGuard, xy):
    """Signed perpendicular offset from the guard's line: lateral() for
    something that is not a Corridor. Same broadcasting rules as along()."""
    p = _as_xy(xy)
    out = -(p[..., 0] - g.x) * g.uy + (p[..., 1] - g.y) * g.ux
    return float(out) if np.ndim(out) == 0 else out


def lane_guards(corridors: Sequence[Corridor], robot_xy,
                lanes: Sequence["ObservedLane"] = (),
                lane_extension_m: float = 0.0) -> List[LaneGuard]:
    """Every lane line the robot must not cross to reach a refuge.

    One guard per corridor user (its own centre line), plus one per
    REMEMBERED lane that a corridor user is currently driving down -- "on
    it" meaning the user's own position is within its corridor half width
    of that line, extended by lane_extension_m like every other lane query
    here. The second kind matters because a remembered lane is longer and
    straighter than any instantaneous window: the line at x = 1.27 is the
    thing not to cross, whatever this tick's rectangle happens to cover.

    A remembered lane with NOBODY on it is deliberately not a guard. It
    still keeps refuges off itself (refuge_lane_clearance), but there is no
    one on it to be hit by, and promoting every remembered line to a wall
    would strand the robot in the first aisle with lanes on both sides.

    Pass the corridors already widened by danger_corridor(): the width used
    here is the band the robot has to drive across.
    """
    guards: List[LaneGuard] = []
    for c in corridors:
        guards.append(LaneGuard(x=c.x, y=c.y, ux=c.ux, uy=c.uy,
                                half_width=c.half_width,
                                tta_s=tta(c, robot_xy),
                                source="user", user_id=c.user_id))
    for ln in lanes:
        on_it = [c for c in corridors
                 if float(lane_line_clearance((c.x, c.y), [ln],
                                              lane_extension_m))
                 <= c.half_width]
        if not on_it:
            continue
        ttas = [tta(c, robot_xy) for c in on_it]
        # The soonest user still APPROACHING owns the guard; if every one of
        # them is past, the smallest (most negative) tta is carried and the
        # guard blocks nothing, which is the same answer either way.
        i = min(range(len(on_it)), key=lambda k: (ttas[k] <= 0.0, ttas[k]))
        guards.append(LaneGuard(
            x=ln.x0, y=ln.y0, ux=ln.ux, uy=ln.uy,
            half_width=max(c.half_width for c in on_it),
            tta_s=ttas[i], source="lane", user_id=on_it[i].user_id))
    return guards


def guard_crossing_time(g: LaneGuard, robot_xy, xy, params: Dict):
    """Seconds to get from the robot to `xy` across the guard's lane:

        (|robot offset| + |candidate offset| + 2*half_width) / v_cross_mps

    -- both perpendicular offsets plus the full width of the lane, at the
    deliberately pessimistic v_cross_mps (0.20 m/s) that t_clear() already
    uses for the same reason: this number has to be a LOWER bound on our
    own speed or the decision is optimistic exactly where optimism costs a
    collision. Same broadcasting rules as along()."""
    v = float(params["v_cross_mps"])
    span = (abs(float(guard_offset(g, robot_xy)))
            + np.abs(np.asarray(guard_offset(g, xy), dtype=float))
            + 2.0 * float(g.half_width))
    out = np.full(span.shape, np.inf) if v < _EPS else span / v
    return float(out) if np.ndim(out) == 0 else out


def guard_blocks(g: LaneGuard, robot_xy, xy, params: Dict):
    """Would reaching `xy` mean crossing this lane in front of its user?
    True = REJECT the candidate. Bool for one point, a bool array for a
    stack of them.

    Three things have to hold at once: the candidate is on the far side of
    the line from the robot (the perpendicular offsets have opposite
    signs), the user is actually approaching (tta > 0 -- a receding one is
    not a reason for anything, the same convention as decide_crossing and
    refuge_recompute_reason), and it arrives within the crossing time plus
    refuge_cross_margin_s.

    A robot standing ON the line has no far side and blocks nothing: every
    way out of a lane you are already in the middle of is a crossing, and
    the clearance ranking is what chooses between them.
    """
    lat_c = np.asarray(guard_offset(g, xy), dtype=float)
    lat_r = float(guard_offset(g, robot_xy))
    t_a = float(g.tta_s)
    if abs(lat_r) <= _EPS or not math.isfinite(t_a) or t_a <= 0.0:
        out = np.zeros(lat_c.shape, dtype=bool)
    else:
        need = (np.asarray(guard_crossing_time(g, robot_xy, xy, params),
                           dtype=float)
                + float(params["refuge_cross_margin_s"]))
        out = (lat_c * lat_r < 0.0) & (t_a <= need)
    return bool(out) if np.ndim(out) == 0 else out


def crossing_rejected(guards: Sequence[LaneGuard], robot_xy, xy,
                      params: Dict):
    """guard_blocks() OR-ed over every guard -- the hard filter that
    find_refuge and find_wall_hug both apply to their candidate discs."""
    p = _as_xy(xy)
    out = np.zeros(p.shape[:-1], dtype=bool)
    for g in guards:
        out = out | np.asarray(guard_blocks(g, robot_xy, p, params))
    return bool(out) if np.ndim(out) == 0 else out


# --------------------------------------------------------------- refuge

def _clearance_field(free: np.ndarray, res: float) -> np.ndarray:
    """Metres from each free cell to the nearest NON-free cell.

    `free` is expected to be a small window already padded by at least
    `clearance / res` cells around the cells that will actually be queried,
    and is additionally ringed here with one occupied cell so that "off the
    edge of the window" counts as blocked rather than as open floor. With
    that padding the returned distances are exact for every queried cell up
    to the clearance being tested, which is all the caller compares.

    scipy's EDT does this in one C pass; without scipy we fall back to a
    dilation-style brute force over the (small) neighbourhood -- correct,
    just slower, so refuge search still works on a bare install.
    """
    padded = np.zeros((free.shape[0] + 2, free.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = free

    if _distance_transform_edt is not None:
        dist = _distance_transform_edt(padded) * float(res)
        return dist[1:-1, 1:-1]

    # Brute force: for every free cell, the distance to the nearest blocked
    # cell within a square neighbourhood. Only distances that matter are the
    # short ones (we compare against `clearance`), so cap the search radius
    # generously and report anything beyond it as "far enough".
    rows, cols = padded.shape
    max_cells = max(1, int(math.ceil(max(free.shape) / 2.0)))
    out = np.full(padded.shape, float(max_cells) * float(res))
    blocked_r, blocked_c = np.nonzero(~padded)
    if blocked_r.size:
        rr = np.arange(rows).reshape(-1, 1)
        cc = np.arange(cols).reshape(1, -1)
        best = np.full(padded.shape, np.inf)
        for br, bc in zip(blocked_r, blocked_c):
            d2 = (rr - br) ** 2 + (cc - bc) ** 2
            np.minimum(best, d2, out=best)
        out = np.sqrt(best) * float(res)
    out[~padded] = 0.0
    return out[1:-1, 1:-1]


def _candidate_window(map_grid: np.ndarray, map_info: Dict,
                      robot_xy: Tuple[float, float],
                      radius: float, clearance: float):
    """The search disc around the robot, as grids: (mask, wx, wy, dist, pts).

    `mask` is free floor at least `clearance` metres from anything occupied
    or unknown and within `radius` of the robot; `wx`/`wy` are the cell
    centres, `pts` their (rows, cols, 2) stack and `dist` their distance
    from the robot. None when the map is unusable or the disc misses it
    entirely. Shared by find_refuge and find_wall_hug so the two can never
    disagree about what a candidate cell even is.
    """
    grid = np.asarray(map_grid, dtype=bool)
    if grid.ndim != 2 or grid.size == 0:
        return None
    rows, cols = grid.shape
    res = float(map_info["resolution"])
    ox = float(map_info["origin_x"])
    oy = float(map_info["origin_y"])
    if res < _EPS:
        return None
    rx, ry = float(robot_xy[0]), float(robot_xy[1])

    # Candidate window = the search disc; the clearance field needs that
    # window padded by the clearance radius so its distances are exact.
    def _span(lo_world, hi_world, origin, limit):
        lo = int(math.floor((lo_world - origin) / res))
        hi = int(math.ceil((hi_world - origin) / res))
        return max(0, lo), min(limit - 1, hi)

    r0, r1 = _span(ry - radius, ry + radius, oy, rows)
    c0, c1 = _span(rx - radius, rx + radius, ox, cols)
    if r1 < r0 or c1 < c0:
        return None

    pad = int(math.ceil(clearance / res)) + 1
    wr0, wr1 = max(0, r0 - pad), min(rows - 1, r1 + pad)
    wc0, wc1 = max(0, c0 - pad), min(cols - 1, c1 + pad)

    window = grid[wr0:wr1 + 1, wc0:wc1 + 1]
    clear_m = _clearance_field(window, res)
    # Slice the candidate disc back out of the padded window.
    sub = (slice(r0 - wr0, r1 - wr0 + 1), slice(c0 - wc0, c1 - wc0 + 1))
    free = window[sub]
    clear_m = clear_m[sub]

    shape = (r1 - r0 + 1, c1 - c0 + 1)
    rr = np.arange(r0, r1 + 1, dtype=float).reshape(-1, 1)
    cc = np.arange(c0, c1 + 1, dtype=float).reshape(1, -1)
    wx = np.ascontiguousarray(np.broadcast_to(ox + (cc + 0.5) * res, shape))
    wy = np.ascontiguousarray(np.broadcast_to(oy + (rr + 0.5) * res, shape))
    pts = np.stack([wx, wy], axis=-1)
    dist = np.hypot(wx - rx, wy - ry)
    mask = free & (dist <= radius) & (clear_m >= clearance)
    return mask, wx, wy, dist, pts


def find_refuge(map_grid: np.ndarray, map_info: Dict,
                robot_xy: Tuple[float, float],
                corridors: Sequence[Corridor],
                radius: float = 2.5,
                clearance: float = 0.45,
                prefer_side: bool = True,
                lane_band: Optional[Dict] = None,
                line_clearance_m: float = 0.0,
                lanes: Sequence[ObservedLane] = (),
                lane_clearance_m: float = 0.0,
                lane_extension_m: float = 0.0,
                guards: Sequence[LaneGuard] = (),
                params: Optional[Dict] = None) -> Optional[np.ndarray]:
    """Nearest spot to step aside to, or None if there is nowhere to go.

    map_grid: 2D bool array, True = FREE. Row 0 is the origin row, i.e.
        grid[row, col] with col = (wx - origin_x)/res, row = (wy - origin_y)/res
        -- the OccupancyGrid / spatial_flow convention used everywhere else
        in Panoptex (see risk_perception.sample_flow_grid).
    map_info: {"resolution", "origin_x", "origin_y"}.
    corridors: every corridor currently known -- a refuge must be outside
        ALL of them, not merely outside the one being fled: stepping out of
        one Carter's lane into another's is not a refuge. Pass these already
        widened by danger_corridor(): this is a containment question, and
        the tracked position of the user carries 0.1-0.4 m of error.
    line_clearance_m: prefer candidates at least this far (laterally) from
        corridors[0]'s centre LINE, extended beyond the rectangle's ends.
        A cell 3 m behind the user is outside its corridor but still on its
        line, and a user that stops short, reverses or is simply tracked
        0.3 m off its true position turns that cell back into the lane.
        Ranks after the lane band and before the side preference; 0 = off.
    lanes / lane_clearance_m / lane_extension_m: the session's remembered
        lane lines (LaneMemory.lanes) and the HARD minimum distance a
        candidate must keep from every one of them, each extended by
        lane_extension_m past its observed ends. Unlike `lane_band` this is
        a rejection with no least-bad fallback -- a cell on a patrol's
        centre line is not a refuge however convenient it is, and taking it
        anyway is what mppi_panoptex_2 did 10 times out of 25. When it
        empties the disc the CALLER widens the search (see
        refuge_radius_max_m) and then holds in place, which is the honest
        answer. 0 (the default) or an empty `lanes` disables it, so every
        pre-2026-09-09 call site behaves exactly as before.
    radius / clearance: search disc around the robot, and the minimum
        distance the refuge must keep from any non-free cell (unknown cells
        count as non-free -- see mission_supervisor's map_free_max).
    lane_band: the STATIC lane keep-out from make_lane_band(), or None.
        Candidates inside it are rejected outright as long as ANY candidate
        outside it survives the other filters -- a lane with no live track
        in it right now is still a lane, and this is the fix for
        avoid_panoptex_1's aisle refuges. When the band swallows every legal
        cell (a robot that is already deep inside the traffic pattern) the
        in-band candidates are used rather than returning None, but they
        sort last, so "least bad" still beats "stop dead in the aisle".
    guards / params: the lane lines that must not be CROSSED to reach the
        refuge (lane_guards()), and the parameter dict guard_blocks() reads
        v_cross_mps and refuge_cross_margin_s from (CORRIDOR_DEFAULTS when
        None). A HARD rejection like the lane lines above, and for the same
        reason: the far side looks cheapest exactly when our own side is
        too narrow to stand in, which is precisely when crossing is worst.
        Empty (the default) restores the pre-2026-09-09 behaviour.
    prefer_side: prefer candidates on the side of corridors[0]'s centre line
        that the robot is ALREADY on -- crossing the lane to reach the far
        side is a worse encounter than the one being avoided. Ranking is
        (outside the band, own side, Euclidean distance), with a
        perpendicular-escape preference only as a final tie-break between
        cells at identical distance. Pass False for "just take the nearest
        legal cell".

    Returns np.array([x, y]) (a cell centre) or None. None is a real answer,
    not an error: the supervisor's response is to cancel its goal and stop
    where it is, which is the correct behaviour in a corridor too narrow to
    step out of.
    """
    win = _candidate_window(map_grid, map_info, robot_xy, radius, clearance)
    if win is None:
        return None
    mask, wx, wy, dist, pts = win
    shape = mask.shape
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    params = CORRIDOR_DEFAULTS if params is None else params

    for c in corridors:
        if not mask.any():
            break
        mask &= ~np.asarray(corridor_contains(c, pts))

    # Remembered lane lines: a HARD filter, applied alongside the live
    # corridors and before any ranking. See the argument's docstring above
    # for why this one has no least-bad fallback.
    lane_gap = np.full(shape, np.inf)
    if lanes and lane_clearance_m > 0.0 and mask.any():
        lane_gap = np.asarray(lane_line_clearance(pts, lanes,
                                                  lane_extension_m),
                              dtype=float)
        mask &= lane_gap >= float(lane_clearance_m)

    # NEVER CROSS A LANE TO REACH A REFUGE (mppi_panoptex_4). A hard filter
    # like the remembered lines above, applied before any ranking: a cell
    # whose only route from here is across an approaching user's lane is
    # not a refuge, it is a head-on encounter with extra steps. When it
    # empties the disc the caller widens the search once and then hugs the
    # wall on this side (find_wall_hug) rather than crossing.
    if guards and mask.any():
        mask &= ~np.asarray(crossing_rejected(guards, (rx, ry), pts, params))

    if not mask.any():
        return None

    # Static lane band: primary rejection AND primary sort key. Rejecting
    # only when something outside the band survives keeps this a preference
    # of last resort rather than a way to answer None (and stop in the
    # aisle) where a merely-imperfect spot exists.
    band_pen = np.asarray(in_lane_band(lane_band, pts)).astype(int)
    if lane_band is not None and np.any(mask & (band_pen == 0)):
        mask &= (band_pen == 0)

    # Strong preference for standing FURTHER from every remembered lane,
    # quantised into _LANE_RANK_BIN_M bins so it outranks the side and
    # distance preferences without letting one extra centimetre of clearance
    # buy an arbitrary detour. Capped one metre past the hard minimum: past
    # that, "further from the lane" stops being worth more driving.
    lane_pen = np.zeros(mask.shape, dtype=int)
    if lanes and lane_clearance_m > 0.0:
        capped = np.minimum(lane_gap, float(lane_clearance_m) + 1.0)
        lane_pen = -np.floor(capped / _LANE_RANK_BIN_M).astype(int)

    line_pen = np.zeros(mask.shape, dtype=int)
    if line_clearance_m > 0.0 and corridors:
        line_pen = (np.abs(np.asarray(lateral(corridors[0], pts)))
                    < float(line_clearance_m)).astype(int)

    side_pen = np.zeros(mask.shape, dtype=int)
    perp_pen = np.zeros(mask.shape, dtype=int)
    if prefer_side and corridors:
        ref = corridors[0]
        lat_robot = float(lateral(ref, (rx, ry)))
        lat_cand = np.asarray(lateral(ref, pts))
        if abs(lat_robot) > _EPS:
            # 1 = the candidate is on the far side of the lane centre line
            # from the robot, i.e. reaching it means crossing the lane.
            side_pen = (lat_cand * lat_robot < 0.0).astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            dirx = np.where(dist > _EPS, (wx - rx) / np.maximum(dist, _EPS), 0.0)
            diry = np.where(dist > _EPS, (wy - ry) / np.maximum(dist, _EPS), 0.0)
        # |cos| between the escape direction and the lane normal (-uy, ux).
        perp = np.abs(dirx * (-ref.uy) + diry * ref.ux)
        perp_pen = (perp < math.cos(math.radians(_PERPENDICULAR_TOLERANCE_DEG))
                    ).astype(int)

    # np.lexsort sorts by the LAST key first: band, then SIDE, then binned
    # clearance from the remembered lane lines, then clearance from the live
    # user's line, then distance, with perpendicularity breaking exact
    # distance ties only.
    #
    # Side outranks clearance since mppi_panoptex_4: it used to sort below
    # it, so a far-side cell with 1.5 m of lane clearance beat a same-side
    # cell with 1.0 m, and the robot crossed the lane to buy half a metre.
    # Staying on this side is worth more than any amount of clearance on the
    # other one -- the far side is now hard-rejected outright while the user
    # is close (guards, above), and merely last-resort when it is not.
    flat = np.flatnonzero(mask.reshape(-1))
    order = np.lexsort((perp_pen.reshape(-1)[flat],
                        dist.reshape(-1)[flat],
                        line_pen.reshape(-1)[flat],
                        lane_pen.reshape(-1)[flat],
                        side_pen.reshape(-1)[flat],
                        band_pen.reshape(-1)[flat]))
    best = flat[order[0]]
    return np.array([wx.reshape(-1)[best], wy.reshape(-1)[best]], dtype=float)


def find_wall_hug(map_grid: np.ndarray, map_info: Dict,
                  robot_xy: Tuple[float, float],
                  guards: Sequence[LaneGuard] = (),
                  radius: float = 2.5,
                  clearance: float = 0.45,
                  lanes: Sequence[ObservedLane] = (),
                  lane_extension_m: float = 0.0,
                  params: Optional[Dict] = None) -> Optional[np.ndarray]:
    """The least-bad standing place on the robot's OWN side of the traffic.

    The fallback for "find_refuge found nothing". A 0.97 m strip between a
    shelf at x ~ 0.30 and a patrol line at x = 1.27 contains no cell that
    clears refuge_lane_clearance() and no cell outside the danger corridor
    either, so the two answers the search used to have were both bad: cross
    the lane (>= 7 s of exposure against a Carter 6.5 s out --
    mppi_panoptex_4) or stand still in the middle of it.

    This is the third answer: get as far from the traffic as the wall
    allows, and wait there. It relaxes exactly the two things the strip
    cannot satisfy -- the lane clearance floor and corridor containment --
    and NOTHING else. The cell is still free floor, still `clearance`
    metres from anything occupied or unknown, still within `radius`, and
    still on our side of every guard.

    Ranked by distance to the NEAREST lane (every guard's line and every
    remembered lane), largest first, with the nearest such cell winning
    ties: the point is to be as far off the traffic as this side goes, for
    as little driving as possible. Returns np.array([x, y]) or None -- None
    still means "hold in place", which by then really is the last resort.
    """
    params = CORRIDOR_DEFAULTS if params is None else params
    win = _candidate_window(map_grid, map_info, robot_xy, radius, clearance)
    if win is None:
        return None
    mask, wx, wy, dist, pts = win
    if guards and mask.any():
        mask &= ~np.asarray(crossing_rejected(guards, robot_xy, pts, params))
    if not mask.any():
        return None

    gap = np.full(mask.shape, np.inf)
    for g in guards:
        gap = np.minimum(gap, np.abs(np.asarray(guard_offset(g, pts),
                                                dtype=float)))
    if lanes:
        gap = np.minimum(gap, np.asarray(
            lane_line_clearance(pts, lanes, lane_extension_m), dtype=float))

    flat = np.flatnonzero(mask.reshape(-1))
    # Primary key last, as everywhere else here: furthest from the traffic,
    # nearest to the robot among equals. With no lane to be far from at all
    # (-inf everywhere) that degenerates to "the nearest legal cell".
    order = np.lexsort((dist.reshape(-1)[flat], -gap.reshape(-1)[flat]))
    best = flat[order[0]]
    return np.array([wx.reshape(-1)[best], wy.reshape(-1)[best]], dtype=float)
