#!/usr/bin/env python3
"""
spatial_prior_node.py  --  the Spatial-Flow prior: persistent, PLACE-indexed
                            occupancy (S) and heading (F), per label category

object_tracker_node remembers objects (identity, velocity, movability) but
forgets everything the moment a track dies (bounded further by its
graveyard). This node remembers *places*: for each label category
(person / robot / wheeled / furniture / unknown -- see
risk_visualization.label_category), a map-aligned grid pair

    S[i,j] in [0,1]     occupancy channel  -- P(something of this category
                                               moves here)
    F[i,j] = (fx, fy)   heading channel    -- EMA of the RAW (unnormalized)
                                               velocity observed here, so it
                                               substitutes directly into a
                                               constant-velocity rollout
                                               without a separately
                                               estimated speed

accumulated across a run and saved to disk between runs. Per-category
grids are allocated lazily (CategoryFlowGrid, below), the first time a
track of that category clears both gates -- most scenes only ever produce
one or two live categories (typically "person" and "wheeled"), not all of
them.

A category scope plus two Behavioral-prior gates decide whether a track
deposits into its category's grid this tick:

    scope  (category):          label_category(label) in flow_categories
    gate 1 (class-level, cheap): p_movable >= p_movable_min
    gate 2 (instant-level):      p_motion  >= p_motion_min  and
                                  confidence >= min_confidence

The scope is a hard filter (from user-a/sandbox): only categories we
publish a flow topic for (person / robot / wheeled) may deposit into S/F --
furniture and unknown are static by nature and were only ever adding noise
to the aggregate S grid. It applies to S/F ONLY: the WP-C activity channel
A below is class-agnostic by definition and ignores both the scope and
gate 1. Gate 1 is a pure performance pre-filter, NOT a correctness gate -- it lets
furniture-class tracks skip the (small but nonzero) per-cell deposit math
entirely, before gate 2 is even evaluated. Correctness rests entirely on
gate 2: p_motion decays fast enough (see object_tracker_node's
p_motion_decay) that a track whose p_movable is transiently elevated stops
depositing within about a second of actually stopping, regardless of
gate 1's exact threshold.

Deposit rule, per update of length dt, for every confirmed track clearing
both gates, into ITS category's grid:

    k        = learn_rate * dt * p_motion * confidence
    S_cell  += k * (1 - S_cell)              for cells within the track footprint
    F_cell  += k * (v_obs - F_cell)          same cells, same k

S's update is an EMA toward 1 (repeated traversals saturate rather than
diverge; one lucky frame cannot brand a cell). F's update is an EMA toward
whatever velocity is currently being observed there. Everywhere else, both
channels of every category decay with `forget_half_life_s` (hours, not
seconds) -- these are lifelong statistics, not another copy of the
instantaneous costmap.

On shutdown (and every `autosave_period_sec`) every category's S and F are
written to `persist_path` together with the shared geometry; on start they
are loaded back if the geometry still matches. Geometry mismatch -> warn
and start empty rather than silently smearing yesterday's doorway onto
today's map origin. A persisted file from before this category/F-channel
schema existed is detected and treated the same way -- start empty under
the new schema rather than guessing which category old undifferentiated
data belonged to.

Save-on-shutdown fires on SIGINT under `ros2 launch` and when the installed
executable is signalled directly. It does NOT fire under `ros2 run` + `kill`:
that wrapper does not pass the signal down, the process just dies, and the
session's learning is lost back to the last autosave. Verified, not assumed --
so leave `autosave_period_sec` on, and prefer Ctrl-C in the launch terminal.

FROZEN (the study-run contract, README section 8 / CLAUDE.md's "the spatial
prior must be frozen during a study run" gotcha)
--------------------------------------------------------------------------
`learn_rate <= 0` -- what every study run already pins it to -- or the
explicit `frozen` param (default False, forces the same thing regardless of
`learn_rate`'s value) means this node must be inert with respect to its own
persisted state: no decay, no deposits, no headway samples, no save, ever,
on any path. Deposits and headway samples were already gated (see
`deposit_into_grid`'s `k <= 0` check and `record_pass_into_grid`'s
`learn_rate <= 0` check); what was NOT gated, and is the bug this closes,
is `forget_half_life_s` decay (applied unconditionally every `_update`,
independent of `learn_rate`) and both save paths (the `autosave_period_sec`
timer and `destroy_node`'s save-on-exit) -- a frozen study run's warm-up
prior was still fading every tick and then being persisted right back to
disk faded, silently defeating the whole point of freezing it. See
`is_frozen`/`decay_factor`/`save_prior` below.

Research logging (from user-a/sandbox; off unless `debug_log_dir` or
`debug_log_path` is set): one CSV, mixed rows keyed by a `kind` column --
  kind=deposit : one per confirmed track evaluated per update tick, with
                 the category scope and BOTH Behavioral gates spelled out
                 (p_movable vs p_movable_min, p_motion vs p_motion_min,
                 confidence vs min_confidence), the resulting deposit
                 radius / gain k, the raw (vx, vy) that would seed F, and
                 whether it deposited. This is the "why is nothing being
                 learned" log.
  kind=grid    : one per category per publish -- S max/mean/nonzero-cell
                 count and the F-heading speed max/mean, i.e. how much the
                 Spatial-Flow prior has actually accumulated so far.
Feed it to tools/prior_report.py.

Published on two kinds of topic:

  /risk_perception/spatial_prior (OccupancyGrid, 0-100): the element-wise
      MAX of every category's S channel, for backward compatibility with
      predictive_risk_costmap_node's floor use (Nav2 does not care WHICH
      category made a place risky). Behaviorally identical to before
      whenever only one category is currently active, which is the common
      case.

  /risk_perception/spatial_headway/<category> (sensor_msgs/Image, 32FC3,
      one topic per category named in `flow_categories`, same geometry
      contract as spatial_flow below): channels (headway_mean,
      headway_count, last_pass_time). See "HEADWAY" below.

  /risk_perception/spatial_flow/<category> (sensor_msgs/Image, 32FC3, one
      topic per category named in `flow_categories`): channels (s, fx, fy)
      for THAT category only, at this node's own grid resolution. Bundling
      s alongside (fx, fy) is deliberate -- predictive_risk_costmap_node's
      rollout blend needs to know how much evidence backs a cell's F before
      trusting it, and the aggregate S above is the wrong number for that
      (a cell can read busy in aggregate from "person" traffic while a
      DIFFERENT category's own s there is still 0, meaning that category's
      f=(0,0) is genuinely "no data," not "observed stationary"). Published
      even for a category with zero deposits so far (an all-zero image),
      so a subscriber sees a stable topic from startup rather than one that
      appears only once evidence exists.

  /risk_perception/spatial_flow/<group>_group and
  /risk_perception/spatial_headway/<group>_group (same 32FC3 contract,
      same transient_local QoS for the headway one): a MERGED view across
      the categories named in one `flow_groups` entry ("cat1,cat2,..."),
      published under the FIRST named category's name suffixed `_group`.
      Motivation: the tracker's `association_groups` can fuse two label
      categories into one identity group (e.g. `["robot,wheeled"]`, because
      a mislabeled Carter flips between "robot" and "wheeled" frame to
      frame -- see CLAUDE.md/README), so evidence about the SAME physical
      traffic ends up split across two per-category grids depending on
      which label happened to win that instant. A consumer that cares about
      "AMR/cart traffic" as one thing, not two half-populated ones, reads
      the group channel instead. Deposits themselves stay strictly
      per-category (unchanged) -- this is a read-time merge only, computed
      fresh at publish time, never persisted. Per cell: S is the elementwise
      MAX across members (same convention as the aggregate topic above); F
      and the three headway fields are NOT blended -- they are taken
      together from whichever single member "wins" that cell (the one with
      the larger S for flow, the one with the larger headway_count for
      headway), so the published (s, fx, fy) / (mean, count, last_pass)
      tuple always describes one real member's observation, never an
      average of two categories' unrelated headings. See
      `merge_group_flow`/`merge_group_headway` below.

HEADWAY (WP4)
-------------
S says a cell is busy; it does not say how OFTEN. A robot that has to cross
a lane it cannot currently see needs the second number: the typical gap
between successive users of that cell, so "nothing has come past for longer
than the usual gap" becomes evidence the lane is clear (mission_supervisor's
blind-crossing rule -- see corridor.blind_crossing_ok). Three more channels,
per category, per cell, learned under exactly the same gates and the same
learn_rate > 0 condition as S and F:

    last_pass_time   float32, -1 = no pass ever recorded here
    headway_mean     float32, EMA (alpha `headway_alpha`) of the observed gap
    headway_count    int32, how many gap samples the mean is made of

plus a bookkeeping grid, `last_pass_id` (int32, -1 = none), holding the id of
the last track that deposited at that cell. A "pass" is a track that clears
both Behavioral gates at a cell whose `last_pass_id` is a DIFFERENT track;
the sample is `now - last_pass_time`, i.e. the gap since the previous
vehicle was last there. Two guards matter:

  * the id test is what makes this a headway rather than a frame counter --
    one track sitting on a cell for 30 s refreshes `last_pass_time` and
    contributes no sample;
  * `headway_min_s` (2 s) drops samples shorter than that. Track ids churn
    constantly in this system (one Carter carried six ids inside one 700 s
    run -- see object_tracker_node's association_key note), so without the
    floor a single vehicle whose id flips mid-crossing would deposit a
    fraction-of-a-second "headway" and drag the learned mean toward zero,
    which is the dangerous direction: it would tell the supervisor the lane
    is far busier than it is and never let it cross.

`headway_count` is therefore also the confidence of the statistic, and the
supervisor refuses to act on a headway backed by fewer than a handful of
samples (falling back to a bounded wait instead).

Persisted in the same `.npz`, under the same geometry guard, with the same
per-category key scheme. MISSING KEYS ARE NOT AN ERROR: a file written
before these channels existed loads its S/F normally and starts the headway
channels empty, so every warm-up file on disk keeps working.

ACTIVITY (WP-C)
----------------
S/F/headway (above) are a LIFELONG statistic (`forget_half_life_s`, hours,
frozen for every study run -- see "FROZEN" above): they answer "where are
the lanes" and must survive a quiet Tuesday afternoon without forgetting
the Monday-morning traffic, which is exactly why `lane_layer`,
`mission_supervisor`'s static lane band, and the headway statistic all
need them un-decayed for the length of a run. That same property makes S
useless for a different, narrower question a planner also needs answered
every few seconds: "is something actually moving near me RIGHT NOW" --
Thomas et al. 2021's "latent risk at busy doorways." A single class-
agnostic channel, A[i,j] in [0,1], answers that:

    A is ONE shared grid, not one per category -- a mover is a mover (the
    2026-09-11 class-agnostic decision; see the plan). It ignores BOTH
    Behavioral gates S uses: `p_movable_min` (a chair being carried still
    counts) and category (label never gates or weights this channel).
    The only gates are `activity_pmot_min` (an instantaneous-motion floor,
    independent of and usually looser than `p_motion_min`) and the same
    `min_confidence` every other deposit here already requires.

    Deposit, every confirmed track clearing both gates, same footprint as
    S (bbox-based radius, clamped to [min_deposit_radius_m,
    max_deposit_radius_m]):

        k       = activity_learn_rate * dt * p_motion * min(1, confidence)
        A_cell += k * (1 - A_cell)         -- same EMA-toward-1 maths as
                                               S's own update; see
                                               ema_deposit_clipped(),
                                               factored out of
                                               deposit_into_grid so both
                                               channels share one formula.

    Decay, EVERY `_update` tick, unconditionally:

        A *= 0.5 ** (dt / activity_half_life_s)     -- default 300 s (5 min)

    and this is the whole point of the channel: decay and deposit are
    NEITHER gated by `is_frozen`/`learn_rate`/the explicit `frozen` param.
    A is run-time state, not a learned prior -- a study run freezes S so
    the warm-up lanes survive the run, but must NOT freeze A, or the one
    channel meant to say "a Carter is near the crossing right now" would
    go stale for the run's entire duration. Consequently A is also NEVER
    written to `persist_path` / the `.npz` (see GRID_ARRAYS, save_prior,
    load_grid_arrays -- none of them mention "activity"): a 5-minute
    memory has no business surviving a process restart, and doing so would
    also break the geometry-guard/schema-version reasoning those functions
    rely on for S/F/headway.

    Published as its own `nav_msgs/OccupancyGrid` (0-100, same
    header/info construction as the aggregate `/risk_perception/
    spatial_prior` topic above -- see `occupancy_grid_data()` /
    `_publish_activity()`) on `activity_topic`
    (`/risk_perception/activity_prior`) at `activity_publish_rate`. Master
    switch `activity_enabled` (default true) creates or omits the
    publisher/timer/deposit/decay entirely; when false this channel does
    not exist, at zero cost, rather than existing and reading zero.
    `activity_viz_topic` (default "", i.e. unset) optionally mirrors the
    same message onto a second, distinctly-named topic -- purely a
    convenience for giving A its own layer name in RViz alongside other
    OccupancyGrid topics; the primary `activity_topic` is already an
    ordinary OccupancyGrid and renders with Nav2's usual costmap colour
    scheme without it.

    Self-exclusion: inherited for free. A is deposited from the exact same
    `self.latest.detections` (`/risk_perception/world_objects`) S reads,
    which is already the tracker's output AFTER whatever self/robot
    filtering the upstream detector chain applies (see
    `global_cam_projector_node._resolve_robot_xy` / CLAUDE.md's
    self-exclusion gotcha) -- there is no second, A-specific filter to
    keep in sync with S's, because there never was a separate one IN this
    node to begin with.

    Consumer contract (predictive_risk_costmap_node, a follow-up, NOT
    implemented here): compose by MAX, same convention as every other
    prior in this codebase (`_splat_into`, `srm.py`, `nav2_risk_layer`),
    seeding the lowest SRM level only -- bound
    `activity_weight <= srm_levels[0] + 0.05` (0.20 + 0.05 = 0.25 at this
    repo's current `srm_levels: [0.2, 0.5, 0.8]`). This is a ceiling, not
    a target: A is a coarse "something is nearby recently" signal with no
    class/consequence weighting behind it at all, so it must never be able
    to outrank a real, class-priced SRM level on its own -- see this
    node's report/handover note for the exact topic/message/param
    contract a subscriber needs.

Honest limitations, for the paper:
  * this is an OBSERVED-motion histogram, not an occupancy prior -- cells the
    cameras never cover stay at 0 and are indistinguishable from cells that
    are genuinely never crossed. There is no visibility/exposure normalisation.
  * it is map-frame and therefore only as good as localisation; a map origin
    change invalidates the saved grid (which is why the geometry check exists).
  * F is a per-cell average of whatever velocities were observed there; at a
    location genuinely used by traffic going both directions, opposing
    velocities partially cancel in the EMA rather than being represented as
    two distinct headings.
  * `last_pass_time` is stamped with THIS session's clock. Under sim time
    every session restarts near zero, so a persisted `last_pass_time` is
    only meaningful within the run that wrote it -- consumers must treat a
    loaded one as "unknown", not as "N seconds ago" (mission_supervisor
    does; see its _crossing_last_pass).
"""

import os
import time
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray

from risk_perception.risk_visualization import label_category, parse_class_id
from risk_perception.debug_log import open_debug_csv, open_latency_csv, log_latency

# Every per-category array that is persisted, in one place, so _save and
# _load cannot drift apart (and so a future channel is one line, not three
# edits). The first three are the original S/F schema; the last four are
# WP4's headway channels, absent from every file written before it.
GRID_ARRAYS = ("s", "fx", "fy",
               "last_pass_time", "headway_mean", "headway_count",
               "last_pass_id")

# Sentinel for "no pass ever recorded here" / "no track has deposited here".
# -1 rather than 0 because 0 is a perfectly ordinary sim timestamp.
NO_PASS = -1.0


class CategoryFlowGrid:
    """S[i,j] (occupancy) + F[i,j]=(fx,fy) (heading, raw-velocity EMA) +
    the WP4 headway channels for ONE label category, sharing one decay
    clock and one deposit gain -- see this module's docstring. Allocated
    lazily by spatial_prior_node the first time a track of that category
    clears both Behavioral gates."""

    def __init__(self, rows: int, cols: int) -> None:
        self.s = np.zeros((rows, cols), dtype=np.float32)
        self.fx = np.zeros((rows, cols), dtype=np.float32)
        self.fy = np.zeros((rows, cols), dtype=np.float32)
        # --- headway (WP4) -------------------------------------------
        self.last_pass_time = np.full((rows, cols), NO_PASS, dtype=np.float32)
        self.headway_mean = np.zeros((rows, cols), dtype=np.float32)
        self.headway_count = np.zeros((rows, cols), dtype=np.int32)
        self.last_pass_id = np.full((rows, cols), -1, dtype=np.int32)

    def decay(self, factor: float) -> None:
        """S and F only. The headway channels are COUNTS AND TIMESTAMPS,
        not EMAs over a decay clock: halving a mean gap every two hours
        would invent traffic that never happened, and decaying a count
        would erode exactly the confidence the blind-crossing rule leans
        on. They age out by being overwritten by newer passes."""
        self.s *= factor
        self.fx *= factor
        self.fy *= factor


def is_frozen(learn_rate: float, frozen_param: bool) -> bool:
    """The frozen-prior contract (README section 8 / CLAUDE.md's "the
    spatial prior must be frozen during a study run" gotcha): True when
    either `learn_rate` is non-positive -- the existing knob every study
    run already pins to 0.0 -- or the explicit `frozen` param forces it
    regardless of `learn_rate`'s value. Deposits and headway samples were
    already gated on k/learn_rate <= 0 (see deposit_into_grid and
    record_pass_into_grid); this is what additionally gates DECAY (via
    decay_factor below) and SAVE (via save_prior below), which previously
    fired unconditionally and faded, then overwrote, a frozen study run's
    warm-up prior."""
    return bool(frozen_param) or float(learn_rate) <= 0.0


def reload_decay_factor(saved_at: Optional[float], now: float,
                        half_life_s: float, frozen: bool) -> Tuple[float, str]:
    """Decay to apply to a prior being RELOADED from disk, for the wall-clock
    seconds that elapsed while nothing was running (2026-09-13).

    Before this existed, _load() read the saved arrays verbatim: S and F
    decayed only while a node was actually up and learning, so "persists
    across sessions" meant "is preserved perfectly across sessions,"
    however long the gap. A lane learned last month read exactly as strong
    as one learned an hour ago. That is not what a half-life claims.

    WALL CLOCK, deliberately, via time.time() at save and at load -- not the
    node clock. Under use_sim_time the node clock restarts with each Isaac
    session and would make an overnight gap read as a few minutes.

    `frozen` short-circuits this exactly as it does the live decay, and for
    a stronger reason: a study sweep loads one warm-up prior into arm after
    arm over several hours, and decaying on load would hand the last arm a
    measurably weaker prior than the first -- the conditions would no longer
    be reading the same place statistic, which is the entire point of
    freezing. A frozen node loads verbatim.

    Returns (factor, reason) -- `reason` is for the caller's log line and is
    "" when the factor is 1.0 for an uninteresting reason.
    """
    if frozen:
        return 1.0, "frozen: loaded verbatim, no reload decay"
    if half_life_s <= 0.0 or saved_at is None:
        return 1.0, ""
    elapsed = float(now) - float(saved_at)
    if elapsed <= 0.0:
        # Clock skew, or a file written by a machine ahead of this one.
        # Decaying by a negative interval would AMPLIFY the prior.
        return 1.0, ""
    factor = float(0.5 ** (elapsed / half_life_s))
    return factor, (f"aged {elapsed / 3600.0:.2f} h since save "
                    f"-> S/F x{factor:.3f} at a {half_life_s:.0f}s half-life")


def decay_factor(dt: float, half_life_s: float, frozen: bool) -> float:
    """1.0 (a no-op multiplier) when frozen or no half-life is configured;
    otherwise the standard half-life decay factor for `dt` seconds. Split
    out from the _update loop's call into CategoryFlowGrid.decay so the
    frozen short-circuit is Tier-0 testable without a Node."""
    if frozen or half_life_s <= 0.0:
        return 1.0
    return float(0.5 ** (dt / half_life_s))


def track_id_int(track_id) -> int:
    """A stable NON-NEGATIVE int32 for a vision_msgs track id (a string).

    The headway bookkeeping stores "which track was last here" in an int32
    grid rather than a per-cell Python object, so ids have to become
    integers. Numeric ids (what object_tracker actually emits) map to
    themselves; anything else hashes. Always >= 0, so it can never collide
    with the -1 "nothing has been here" sentinel.
    """
    try:
        return abs(int(track_id)) & 0x7FFFFFFF
    except (TypeError, ValueError):
        return int(zlib.crc32(str(track_id).encode("utf-8")) & 0x7FFFFFFF)


def grid_payload(category: str, grid: CategoryFlowGrid) -> Dict[str, np.ndarray]:
    """One category's arrays, keyed for the shared .npz. Pure -- see
    test_spatial_prior_headway.py's round-trip test."""
    return {f"{name}__{category}": getattr(grid, name) for name in GRID_ARRAYS}


def load_grid_arrays(payload, category: str, rows: int,
                     cols: int) -> CategoryFlowGrid:
    """Rebuild one category's grid from a loaded .npz (or any mapping).

    A MISSING KEY IS NOT AN ERROR -- it leaves that channel at its empty
    default. That is what lets every pre-WP4 warm-up file on disk keep
    loading: it carries s/fx/fy and nothing else, and the headway channels
    simply start empty (count 0 -> the supervisor's blind rule falls back to
    its bounded wait rather than trusting a mean it does not have). A key
    whose shape does not match the current grid is skipped the same way;
    the caller has already checked the geometry, so this only fires on a
    genuinely corrupt file, where empty beats wrong.
    """
    grid = CategoryFlowGrid(rows, cols)
    for name in GRID_ARRAYS:
        key = f"{name}__{category}"
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        if arr.shape != (rows, cols):
            continue
        setattr(grid, name, arr.astype(getattr(grid, name).dtype).copy())
    return grid


def footprint_inside_mask(
    cx: float, cy: float, radius: float, resolution: float,
    origin_x: float, origin_y: float, rows: int, cols: int,
) -> Tuple[int, int, int, int, Optional[np.ndarray]]:
    """Shared per-cell footprint geometry: the (i0:i1, j0:j1) bounding box
    of `radius` around (cx, cy), in grid-index space, and the boolean mask
    (in that box's LOCAL coordinates) of which cells are actually inside
    the circle -- vs. just inside its bounding box. Factored out of
    deposit_into_grid so record_pass_into_grid and the WP-C
    deposit_activity_into_grid share the exact same geometry rather than a
    second (and now third) hand-copied version drifting from it.

    Returns `inside=None` whenever the footprint doesn't touch the grid at
    all (off-grid center, or a radius so small no cell center falls inside
    it) -- every caller's own early-return-on-empty-footprint behavior is
    unchanged by this refactor, just moved in here."""
    r_cells = radius / resolution
    gx = (cx - origin_x) / resolution
    gy = (cy - origin_y) / resolution
    i0 = max(0, int(np.floor(gx - r_cells)))
    i1 = min(cols, int(np.ceil(gx + r_cells)) + 1)
    j0 = max(0, int(np.floor(gy - r_cells)))
    j1 = min(rows, int(np.ceil(gy + r_cells)) + 1)
    if i0 >= i1 or j0 >= j1:
        return i0, i1, j0, j1, None
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
    wx = origin_x + (ii + 0.5) * resolution
    wy = origin_y + (jj + 0.5) * resolution
    inside = np.hypot(wx - cx, wy - cy) <= radius
    if not inside.any():
        return i0, i1, j0, j1, None
    return i0, i1, j0, j1, inside


def ema_deposit_clipped(region: np.ndarray, inside: np.ndarray, k: float,
                        lo: float = 0.0, hi: float = 1.0) -> None:
    """EMA-toward-`hi` deposit, in place, over `region[inside]`, then
    clipped to [lo, hi]. Factored out of deposit_into_grid's S update
    (`S_cell += k*(1-S_cell)`) so the WP-C activity channel
    (deposit_activity_into_grid) shares the identical formula on its own
    grid (`A_cell += k*(1-A_cell)`) instead of a second copy of the same
    three lines. See test_spatial_prior_grid.py / test_spatial_prior_activity.py."""
    region[inside] += k * (hi - region[inside])
    np.clip(region, lo, hi, out=region)


def deposit_into_grid(
    grid: CategoryFlowGrid, cx: float, cy: float, radius: float, k: float,
    vx: float, vy: float, resolution: float, origin_x: float, origin_y: float,
    rows: int, cols: int,
) -> None:
    """Pure geometry: EMA-deposit (S, F) into every cell within `radius` of
    (cx, cy). No ROS, no Node -- unit-testable directly, no rclpy context
    needed, same rationale as relation_matching.py / encounter_geometry.py /
    mask_relation.py. See test_spatial_prior_grid.py."""
    if k <= 0.0:
        return
    i0, i1, j0, j1, inside = footprint_inside_mask(
        cx, cy, radius, resolution, origin_x, origin_y, rows, cols)
    if inside is None:
        return

    s_region = grid.s[j0:j1, i0:i1]
    ema_deposit_clipped(s_region, inside, k)

    # F is a raw-velocity EMA, not a probability -- no [0,1] clip. The
    # clamp below is defensive only (guards against a corrupted vx/vy
    # reading ever producing an unbounded cell), not expected to bind
    # under normal operation.
    fx_region = grid.fx[j0:j1, i0:i1]
    fy_region = grid.fy[j0:j1, i0:i1]
    fx_region[inside] += k * (vx - fx_region[inside])
    fy_region[inside] += k * (vy - fy_region[inside])
    np.clip(fx_region, -10.0, 10.0, out=fx_region)
    np.clip(fy_region, -10.0, 10.0, out=fy_region)


def deposit_activity_into_grid(
    activity: np.ndarray, cx: float, cy: float, radius: float, k: float,
    resolution: float, origin_x: float, origin_y: float,
    rows: int, cols: int,
) -> None:
    """WP-C: EMA-deposit into the class-agnostic activity channel A -- same
    footprint geometry (footprint_inside_mask) and the same EMA-toward-1
    maths as deposit_into_grid's S update (ema_deposit_clipped), applied to
    ONE shared grid instead of one grid per category, and with no F/heading
    companion channel (A answers "how much recent traffic", not "which
    way" -- see module docstring's ACTIVITY section). Pure -- no ROS, no
    Node, unit-testable directly. See test_spatial_prior_activity.py."""
    if k <= 0.0:
        return
    i0, i1, j0, j1, inside = footprint_inside_mask(
        cx, cy, radius, resolution, origin_x, origin_y, rows, cols)
    if inside is None:
        return
    region = activity[j0:j1, i0:i1]
    ema_deposit_clipped(region, inside, k)


def record_pass_into_grid(
    grid: CategoryFlowGrid, cx: float, cy: float, radius: float,
    track_id: int, now: float, learn_rate: float, headway_min_s: float,
    headway_alpha: float, resolution: float, origin_x: float, origin_y: float,
    rows: int, cols: int,
) -> int:
    """Record one track's presence at (cx, cy) in the headway channels.

    Same footprint geometry as deposit_into_grid (every cell within `radius`)
    and the same "frozen when there is no learning" contract: `learn_rate`
    <= 0 is a no-op, exactly as k <= 0 is for S/F, so a study run with the
    prior frozen never touches these grids either.

    Per cell in the footprint:
      * a DIFFERENT track id than the one last seen here, with a recorded
        previous pass at least `headway_min_s` ago, yields one headway
        sample `now - last_pass_time`, folded into headway_mean by EMA
        (the first sample IS the mean -- starting an EMA from 0 would say
        "this lane runs bumper to bumper" until it converged) and counted;
      * the id is then remembered and `last_pass_time` refreshed, for the
        same track or a new one. Refreshing on the same track is what makes
        the next sample a real gap (tail of one vehicle to nose of the next)
        rather than a nose-to-nose period.

    Returns the number of CELLS that took a sample (one pass covers a whole
    footprint, so this is ~30 at the shipped radii), which is what the node
    accumulates into its `headway samples` save-time log line.
    """
    if learn_rate <= 0.0 or resolution <= 0.0:
        return 0
    i0, i1, j0, j1, inside = footprint_inside_mask(
        cx, cy, radius, resolution, origin_x, origin_y, rows, cols)
    if inside is None:
        return 0

    ids = grid.last_pass_id[j0:j1, i0:i1]
    times = grid.last_pass_time[j0:j1, i0:i1]
    means = grid.headway_mean[j0:j1, i0:i1]
    counts = grid.headway_count[j0:j1, i0:i1]

    tid = int(track_id)
    new_user = inside & (ids != tid)
    gaps = float(now) - times.astype(np.float64)
    sampled = new_user & (times >= 0.0) & (gaps >= float(headway_min_s))
    first = sampled & (counts <= 0)
    later = sampled & (counts > 0)
    means[first] = gaps[first].astype(np.float32)
    means[later] += (float(headway_alpha)
                     * (gaps[later].astype(np.float32) - means[later]))
    counts[sampled] += 1

    ids[new_user] = tid
    times[inside] = np.float32(now)
    return int(sampled.sum())


def parse_flow_groups(flow_groups: List[str]) -> List[Tuple[str, List[str]]]:
    """Parse `flow_groups` param entries ("cat1,cat2,...") into
    (group_name, members) pairs. `group_name` is the FIRST member named in
    the spec -- the group's merged topics publish under `<group_name>_group`
    (see module docstring). Blank/whitespace-only specs are skipped; a spec
    naming only one category is a legal (degenerate) group of one."""
    out: List[Tuple[str, List[str]]] = []
    for spec in flow_groups:
        members = [c.strip() for c in str(spec).split(",") if c.strip()]
        if not members:
            continue
        out.append((members[0], members))
    return out


def merge_group_flow(
    grids: List[Optional[CategoryFlowGrid]], rows: int, cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge N member categories' (S, F) into one group channel -- pure,
    Tier 0 testable (no Node). S is the elementwise MAX across members
    (same max-combine convention as the aggregate /risk_perception/
    spatial_prior topic); F is NOT averaged -- it is the (fx, fy) of
    whichever member actually holds that cell's max S, because a group
    merges categories the tracker cannot reliably tell apart (e.g.
    "robot,wheeled" -- a Carter mislabeled either way should still surface
    its own observed heading, not a blend with an unrelated category's
    heading at the same cell). A `None` member (never allocated -- no
    deposits yet for that category) contributes an all-zero S/F, so a
    group with one empty member reduces to exactly the other member's
    grid."""
    zeros = np.zeros((rows, cols), dtype=np.float32)
    s_stack = np.stack([g.s if g is not None else zeros for g in grids], axis=0)
    fx_stack = np.stack([g.fx if g is not None else zeros for g in grids], axis=0)
    fy_stack = np.stack([g.fy if g is not None else zeros for g in grids], axis=0)
    winner = np.argmax(s_stack, axis=0)[np.newaxis, :, :]
    s_out = np.max(s_stack, axis=0)
    fx_out = np.take_along_axis(fx_stack, winner, axis=0)[0]
    fy_out = np.take_along_axis(fy_stack, winner, axis=0)[0]
    return (s_out.astype(np.float32), fx_out.astype(np.float32),
            fy_out.astype(np.float32))


def merge_group_headway(
    grids: List[Optional[CategoryFlowGrid]], rows: int, cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge N member categories' headway channels into one group channel,
    same rationale as merge_group_flow: pick the member with the larger
    `headway_count` per cell (the better-attested statistic) rather than
    averaging means computed from different sample sizes, and take that
    SAME member's `last_pass_time`/`headway_mean` too so the three
    published values always describe one member's observation. A `None`
    member contributes zero count / NO_PASS, so a group with one empty
    member reduces to exactly the other member's grid."""
    zeros_i = np.zeros((rows, cols), dtype=np.int32)
    zeros_f = np.zeros((rows, cols), dtype=np.float32)
    no_pass = np.full((rows, cols), NO_PASS, dtype=np.float32)
    count_stack = np.stack(
        [g.headway_count if g is not None else zeros_i for g in grids], axis=0)
    mean_stack = np.stack(
        [g.headway_mean if g is not None else zeros_f for g in grids], axis=0)
    last_stack = np.stack(
        [g.last_pass_time if g is not None else no_pass for g in grids], axis=0)
    winner = np.argmax(count_stack, axis=0)[np.newaxis, :, :]
    count_out = np.take_along_axis(count_stack, winner, axis=0)[0]
    mean_out = np.take_along_axis(mean_stack, winner, axis=0)[0]
    last_out = np.take_along_axis(last_stack, winner, axis=0)[0]
    return (mean_out.astype(np.float32), count_out.astype(np.int32),
            last_out.astype(np.float32))


def save_prior(
    path: str,
    category_grids: Dict[str, CategoryFlowGrid],
    resolution: float, origin_x: float, origin_y: float,
    rows: int, cols: int,
    deposits: int, headway_samples: int,
    frozen: bool, persist_blocked: bool,
) -> Tuple[bool, Optional[str]]:
    """Write `category_grids` to `path` as the shared .npz -- pulled out of
    SpatialPriorNode so the frozen/blocked short-circuits are Tier-0
    testable (no Node, no rclpy context; see test_spatial_prior_grid.py).
    Returns (wrote, message); `message` is None exactly when there is
    nothing worth logging (the silent persist_blocked path, unchanged from
    before this refactor).

    `frozen` is checked FIRST and unconditionally: a study run loads a
    warm-up prior specifically so it is NOT rewritten mid-run or on exit
    (the "frozen prior" contract -- README section 8 / CLAUDE.md's
    spatial-prior gotcha). This closes the bug where autosave and
    save-on-shutdown still fired under a frozen node and persisted a
    prior whose S channel had also been silently faded by
    forget_half_life_s decay applied every update regardless of
    learn_rate."""
    if frozen:
        return False, (
            "spatial prior FROZEN: refusing to save "
            f"({path} left untouched)")
    if persist_blocked:
        return False, None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        payload = dict(
            resolution=resolution, origin_x=origin_x, origin_y=origin_y,
            width=cols, height=rows,
            categories=np.array(list(category_grids.keys())),
            # WALL clock, so _load can age the prior by the real gap between
            # sessions (reload_decay_factor). Not the node clock: under
            # use_sim_time that restarts every Isaac session.
            saved_at=np.array(time.time()),
        )
        for category, grid in category_grids.items():
            payload.update(grid_payload(category, grid))
        np.savez_compressed(tmp, **payload)
        # np.savez appends .npz if the name lacks it
        src = tmp if os.path.exists(tmp) else tmp + ".npz"
        os.replace(src, path)   # atomic: never a half-written prior
        cats = ", ".join(category_grids.keys()) or "none"
        msg = (f"saved prior -> {path} "
               f"(categories: {cats}, {deposits} deposits, "
               f"{headway_samples} headway samples)")
        return True, msg
    except Exception as exc:                     # noqa: BLE001 - never fatal
        return False, f"could not save {path}: {exc}"


def occupancy_grid_data(values: np.ndarray) -> np.ndarray:
    """[0, 1] float grid -> int8 0-100 `nav_msgs/OccupancyGrid` payload
    (still 2-D; the caller flattens row-major, matching `OccupancyGrid`'s
    own `data[row*W + col]` convention). Pure, no ROS -- factored out of
    `_publish()` so the aggregate S topic and the WP-C activity topic
    (`_publish_activity()`) share one packing rule instead of two
    hand-copied `np.clip(np.round(...))` lines that could silently drift
    apart. See test_spatial_prior_activity.py."""
    return np.clip(np.round(values * 100.0), 0, 100).astype(np.int8)


class SpatialPriorNode(Node):
    def __init__(self) -> None:
        super().__init__("spatial_prior_node")

        self.declare_parameter("input_topic", "/risk_perception/world_objects")
        self.declare_parameter("output_topic", "/risk_perception/spatial_prior")
        self.declare_parameter("map_frame", "map")

        # geometry -- keep identical to the costmap nodes unless you have a
        # reason not to; predictive_risk_costmap_node resamples if it differs.
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width_m", 12.0)
        self.declare_parameter("height_m", 12.0)
        self.declare_parameter("origin_x", -6.0)
        self.declare_parameter("origin_y", -6.0)

        # learning
        # Gate 1 (Behavioral, class-level pre-filter) -- see module
        # docstring for why its exact value is not correctness-critical.
        # 0.5 reuses the same dynamic/static split point object_tracker_node
        # already uses for confidence half-life and eviction timeout, so
        # "movable" means the same thing everywhere in the system.
        self.declare_parameter("p_movable_min", 0.5)
        self.declare_parameter("p_motion_min", 0.5)
        self.declare_parameter("min_confidence", 0.15)
        self.declare_parameter("learn_rate", 0.20)          # per second, at p_motion=1
        # Explicit override of the frozen contract below -- normally
        # learn_rate <= 0 alone is what a study run pins, but a caller can
        # force it regardless of learn_rate's value. See is_frozen().
        self.declare_parameter("frozen", False)
        self.declare_parameter("forget_half_life_s", 7200.0)  # 2 h
        self.declare_parameter("min_deposit_radius_m", 0.25)
        self.declare_parameter("max_deposit_radius_m", 1.00)
        self.declare_parameter("update_rate", 5.0)
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("input_timeout_sec", 1.0)

        # persistence
        self.declare_parameter("persist_path", "~/.panoptex/spatial_prior.npz")
        self.declare_parameter("load_on_start", True)
        self.declare_parameter("autosave_period_sec", 60.0)

        # Per-category Spatial-Flow export -- see module docstring. Fixed,
        # configured list rather than "whatever categories happen to be
        # live": predictive_risk_costmap_node subscribes to exactly this
        # set at startup, so it must be known up front, not discovered.
        # furniture/unknown excluded by default -- they essentially never
        # clear the p_movable_min gate, so publishing them would just be an
        # always-empty topic.
        self.declare_parameter(
            "flow_categories", ["person", "robot", "wheeled"])

        # Research logging (from user-a/sandbox) -- see module docstring.
        # Off unless one of these is set.
        self.declare_parameter("debug_log_path", "")
        self.declare_parameter("debug_log_dir", "")

        # Category groups -- see module docstring's "MERGED" topic section.
        # Each entry is "cat1,cat2,..."; the group's merged topics publish
        # under the FIRST named category's name suffixed "_group". Default
        # matches object_tracker_node's association_groups so a mislabeled
        # Carter (robot <-> wheeled) is one merged channel, not two
        # half-populated ones.
        self.declare_parameter("flow_groups", ["robot,wheeled"])

        # Headway statistics (WP4) -- see this module's "HEADWAY" section.
        # Learned only while learn_rate > 0, exactly like S and F.
        self.declare_parameter("headway_min_s", 2.0)
        self.declare_parameter("headway_alpha", 0.2)

        # Activity (WP-C) -- see this module's "ACTIVITY" section. A single
        # class-agnostic 5-min-half-life channel, NEVER gated by
        # frozen/learn_rate above and NEVER persisted -- entirely separate
        # knobs from the lifelong S/F/headway statistics.
        self.declare_parameter("activity_enabled", False)
        self.declare_parameter("activity_half_life_s", 300.0)   # 5 min
        self.declare_parameter("activity_learn_rate", 0.5)      # per s at pmot=1
        self.declare_parameter("activity_pmot_min", 0.5)
        self.declare_parameter(
            "activity_topic", "/risk_perception/activity_prior")
        self.declare_parameter("activity_publish_rate", 2.0)
        # "" (default) = no second publisher. The primary activity_topic
        # above is already an ordinary OccupancyGrid and renders with
        # Nav2's usual costmap colour scheme in RViz without this; set it
        # only to give A its own distinctly-named layer alongside other
        # OccupancyGrid topics in the same view.
        self.declare_parameter("activity_viz_topic", "")

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.res = float(gp("resolution").value)
        self.ox = float(gp("origin_x").value)
        self.oy = float(gp("origin_y").value)
        self.cols = max(1, int(round(float(gp("width_m").value) / self.res)))
        self.rows = max(1, int(round(float(gp("height_m").value) / self.res)))
        self.p_movable_min = float(gp("p_movable_min").value)
        self.p_motion_min = float(gp("p_motion_min").value)
        self.min_conf = float(gp("min_confidence").value)
        self.learn_rate = float(gp("learn_rate").value)
        self.frozen = is_frozen(self.learn_rate, bool(gp("frozen").value))
        self.half_life = float(gp("forget_half_life_s").value)
        self.r_min = float(gp("min_deposit_radius_m").value)
        self.r_max = float(gp("max_deposit_radius_m").value)
        self.input_timeout = float(gp("input_timeout_sec").value)

        self.persist_path = os.path.expanduser(str(gp("persist_path").value))
        self.autosave_period = float(gp("autosave_period_sec").value)
        self.flow_categories: List[str] = [
            str(c) for c in gp("flow_categories").value]
        self.flow_groups: List[Tuple[str, List[str]]] = parse_flow_groups(
            [str(g) for g in gp("flow_groups").value])
        self.headway_min_s = float(gp("headway_min_s").value)
        self.headway_alpha = float(gp("headway_alpha").value)
        self.headway_samples = 0     # how many gaps this session measured
        self.bridge = CvBridge()

        # Activity (WP-C) -- see __init__'s declare_parameter block above
        # and the module docstring's ACTIVITY section.
        self.activity_enabled = bool(gp("activity_enabled").value)
        self.activity_half_life = float(gp("activity_half_life_s").value)
        self.activity_learn_rate = float(gp("activity_learn_rate").value)
        self.activity_pmot_min = float(gp("activity_pmot_min").value)

        # category (str) -> CategoryFlowGrid, allocated lazily -- see
        # module docstring. Typically only 1-2 categories are ever live.
        self.category_grids: Dict[str, CategoryFlowGrid] = {}
        self.deposits = 0            # how much evidence this session has seen
        # Set when the file on disk holds a grid we refused to load. Saving
        # then would overwrite weeks of learning with this session's empty
        # grid -- one wrong origin_x on one launch would silently destroy it.
        self.persist_blocked = False
        self.latest: Optional[Detection3DArray] = None
        self.latest_t = -1.0
        self.last_update = self._now()

        # WP-C: ONE shared, class-agnostic grid -- not per-category like
        # CategoryFlowGrid's S -- allocated up front (not lazily) since it
        # has no per-category identity to defer. Never loaded/saved: see
        # module docstring's ACTIVITY section for why this channel is
        # deliberately absent from GRID_ARRAYS / save_prior / load_grid_arrays.
        self.activity = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.activity_deposits = 0   # how much evidence this session has seen

        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(gp("debug_log_dir").value))

        self.log_writer, self.log_file, _ = open_debug_csv(
            self, str(gp("debug_log_path").value), str(gp("debug_log_dir").value),
            "spatial_prior",
            ["t", "kind", "track_id", "label", "category",
             "p_movable", "p_movable_min", "gate1_pass",
             "p_motion", "p_motion_min", "confidence", "min_confidence",
             "gate2_pass", "bbox_size", "radius", "k", "vx", "vy",
             "cx", "cy", "deposited",
             "s_max", "s_mean", "s_nnz", "f_speed_max", "f_speed_mean",
             "deposits_total"])

        if bool(gp("load_on_start").value):
            self._load()

        self.create_subscription(Detection3DArray, str(gp("input_topic").value),
                                 self._objects_cb, 10)
        self.pub = self.create_publisher(
            OccupancyGrid, str(gp("output_topic").value), 1)

        # WP-C activity publisher(s) -- only created when activity_enabled,
        # so a disabled channel costs nothing (no topic, no timer, no
        # deposit/decay work in _update). See module docstring.
        self.activity_pub = None
        self.activity_viz_pub = None
        if self.activity_enabled:
            activity_topic = str(gp("activity_topic").value)
            self.activity_pub = self.create_publisher(
                OccupancyGrid, activity_topic, 1)
            activity_viz_topic = str(gp("activity_viz_topic").value)
            if activity_viz_topic and activity_viz_topic != activity_topic:
                self.activity_viz_pub = self.create_publisher(
                    OccupancyGrid, activity_viz_topic, 1)

        self.flow_pubs = {
            category: self.create_publisher(
                Image, f"/risk_perception/spatial_flow/{category}", 1)
            for category in self.flow_categories
        }
        self.flow_group_pubs = {
            group_name: self.create_publisher(
                Image, f"/risk_perception/spatial_flow/{group_name}_group", 1)
            for group_name, _members in self.flow_groups
        }

        # Headway (WP4) is TRANSIENT_LOCAL, unlike flow: it is read by
        # mission_supervisor, which is started by a different launch file
        # long after this node, and a statistic that is only meaningful
        # after minutes of traffic must not depend on the subscriber having
        # been up for the publish that carried it.
        headway_qos = QoSProfile(depth=1,
                                 history=HistoryPolicy.KEEP_LAST,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.headway_pubs = {
            category: self.create_publisher(
                Image, f"/risk_perception/spatial_headway/{category}",
                headway_qos)
            for category in self.flow_categories
        }
        self.headway_group_pubs = {
            group_name: self.create_publisher(
                Image, f"/risk_perception/spatial_headway/{group_name}_group",
                headway_qos)
            for group_name, _members in self.flow_groups
        }

        self.create_timer(1.0 / max(0.1, float(gp("update_rate").value)), self._update)
        self.create_timer(1.0 / max(0.1, float(gp("publish_rate").value)), self._publish)
        self.create_timer(1.0 / max(0.1, float(gp("publish_rate").value)), self._publish_flow)
        self.create_timer(1.0 / max(0.1, float(gp("publish_rate").value)),
                          self._publish_headway)
        # FROZEN: no autosave timer at all -- see save_prior's frozen
        # short-circuit for the belt-and-suspenders guard on the
        # save-on-shutdown path too (destroy_node always calls _save()).
        if self.autosave_period > 0 and not self.frozen:
            self.create_timer(self.autosave_period, self._save)
        # WP-C: its own publish-rate timer, independent of publish_rate
        # above and of `frozen` -- see module docstring's ACTIVITY section.
        if self.activity_enabled:
            self.create_timer(
                1.0 / max(0.1, float(gp("activity_publish_rate").value)),
                self._publish_activity)

        self.get_logger().info(
            f"spatial_prior {self.rows}x{self.cols} @ {self.res} m/cell, "
            f"origin ({self.ox}, {self.oy}), half-life {self.half_life:.0f}s, "
            f"p_movable_min {self.p_movable_min:.2f}, persist -> {self.persist_path}")
        if self.frozen:
            self.get_logger().info(
                "spatial prior FROZEN (learn_rate=0): no decay, no deposits, no save")
        if self.activity_enabled:
            self.get_logger().info(
                f"activity channel ON: half-life {self.activity_half_life:.0f}s, "
                f"pmot_min {self.activity_pmot_min:.2f}, never frozen, never saved "
                f"-> {str(gp('activity_topic').value)}")
        else:
            self.get_logger().info("activity channel OFF (activity_enabled=false)")

    # ------------------------------------------------------------------ inputs

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _objects_cb(self, msg: Detection3DArray) -> None:
        self.latest = msg
        self.latest_t = self._now()

    # ------------------------------------------------------------------ learning

    def _update(self) -> None:
        now = self._now()
        dt = now - self.last_update
        self.last_update = now
        if dt <= 0.0 or dt > 5.0:      # clock jump / bag seek: skip this step
            return
        _t0 = time.perf_counter()

        decay = decay_factor(dt, self.half_life, self.frozen)
        if decay != 1.0:
            for grid in self.category_grids.values():
                grid.decay(decay)

        # WP-C: the activity channel decays on ITS OWN half-life and is
        # NEVER gated by `frozen`/`learn_rate` -- see module docstring's
        # ACTIVITY section and decay_factor()'s own docstring. `frozen=False`
        # here is not a bug: this is the one decay call in this node that
        # must never see the node's frozen state.
        if self.activity_enabled:
            a_decay = decay_factor(dt, self.activity_half_life, frozen=False)
            if a_decay != 1.0:
                self.activity *= a_decay

        if self.latest is None or (now - self.latest_t) > self.input_timeout:
            return

        # FROZEN forces the effective learn rate to 0 regardless of the
        # `learn_rate` param's own value -- deposit_into_grid (k<=0) and
        # record_pass_into_grid (learn_rate<=0) already no-op on this, so
        # routing both through one variable is what makes "frozen" and
        # "learn_rate==0.0" behave identically everywhere below. The
        # activity channel below deliberately does NOT use this variable --
        # it has its own activity_learn_rate, ungated by frozen.
        effective_learn_rate = 0.0 if self.frozen else self.learn_rate

        for det in self.latest.detections:
            if not det.results:
                continue
            h = det.results[0].hypothesis
            label, kv = parse_class_id(str(h.class_id))
            p_motion = kv.get("pmot", 0.0)
            conf = float(h.score)
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            radius = min(self.r_max, max(
                self.r_min,
                max(float(det.bbox.size.x), float(det.bbox.size.y)) / 2.0))

            # WP-C activity channel (A): class-agnostic, ignores
            # p_movable_min AND category entirely -- deliberately computed
            # BEFORE the Gate-1 `continue` below, which is specific to
            # S/F. See module docstring's ACTIVITY section.
            if (self.activity_enabled and p_motion >= self.activity_pmot_min
                    and conf >= self.min_conf):
                k_a = self.activity_learn_rate * dt * p_motion * min(1.0, conf)
                deposit_activity_into_grid(
                    self.activity, cx, cy, radius, k_a,
                    self.res, self.ox, self.oy, self.rows, self.cols)
                if k_a > 0.0:
                    self.activity_deposits += 1

            category = label_category(label)

            # Category scope (ported from user-a/sandbox, 2026-09-11): only
            # categories we publish a flow topic for may deposit into S/F at
            # all. furniture / unknown are static by nature -- excluding them
            # keeps the aggregate S grid (and therefore lane_layer and the
            # corridor lane band) from being painted by parked chairs and
            # shelving. The WP-C activity channel above is deliberately NOT
            # scoped this way: A is class-agnostic by definition.
            in_scope = category in self.flow_categories

            # Gate 1 (Behavioral, class-level) -- cheap pre-filter, not
            # correctness-critical. See module docstring. S/F ONLY -- the
            # activity deposit above deliberately ignores this gate.
            p_movable = kv.get("pmov", 0.0)
            gate1 = p_movable >= self.p_movable_min

            # Gate 2 (Behavioral, instant-level) -- this is the gate
            # correctness actually depends on.
            gate2 = p_motion >= self.p_motion_min and conf >= self.min_conf

            if not gate1 or not in_scope:
                # Keeps the perf pre-filter when we are not logging; when we
                # ARE, the rejected row is still recorded (that is the whole
                # point of the "why is nothing being learned" log).
                self._log_deposit(now, det, label, p_movable, gate1,
                                  p_motion, conf, gate2, 0.0, 0.0, False)
                continue

            k = effective_learn_rate * dt * p_motion * min(1.0, conf)
            if gate2:
                grid = self.category_grids.setdefault(
                    category, CategoryFlowGrid(self.rows, self.cols))
                self._deposit(grid, cx, cy, radius, k,
                              kv.get("vx", 0.0), kv.get("vy", 0.0))
                if k > 0.0:
                    self.deposits += 1
                # WP4: the same footprint, the same gates, the same
                # frozen-when-learn_rate-0 rule -- a pass is exactly a
                # deposit, seen from the time axis instead of the
                # occupancy axis.
                self.headway_samples += record_pass_into_grid(
                    grid, cx, cy, radius, track_id_int(det.id), now,
                    effective_learn_rate, self.headway_min_s,
                    self.headway_alpha,
                    self.res, self.ox, self.oy, self.rows, self.cols)
            self._log_deposit(now, det, label, p_movable, gate1, p_motion,
                              conf, gate2, radius, k, gate2)
        log_latency(self.lat_writer, self.lat_file, time.perf_counter() - _t0)

    def _deposit(self, grid: CategoryFlowGrid, cx: float, cy: float,
                 radius: float, k: float, vx: float, vy: float) -> None:
        deposit_into_grid(grid, cx, cy, radius, k, vx, vy,
                          self.res, self.ox, self.oy, self.rows, self.cols)

    # ------------------------------------------------------------------ logging

    def _log_deposit(self, now, det, label, p_movable, gate1, p_motion, conf,
                     gate2, radius, k, deposited) -> None:
        """One `kind=deposit` row per confirmed track per update tick, with
        both Behavioral gates and the category scope spelled out. Ported
        from user-a/sandbox; consumed by tools/prior_report.py."""
        if not self.log_writer:
            return
        _, kv = parse_class_id(str(det.results[0].hypothesis.class_id))
        self.log_writer.writerow([
            f"{now:.3f}", "deposit", det.id, label, label_category(label),
            f"{p_movable:.3f}", f"{self.p_movable_min:.2f}", int(gate1),
            f"{p_motion:.3f}", f"{self.p_motion_min:.2f}",
            f"{conf:.4f}", f"{self.min_conf:.2f}", int(gate2),
            f"{max(float(det.bbox.size.x), float(det.bbox.size.y)):.3f}",
            f"{radius:.3f}", f"{k:.5f}",
            f"{kv.get('vx', 0.0):.4f}", f"{kv.get('vy', 0.0):.4f}",
            f"{det.bbox.center.position.x:.3f}",
            f"{det.bbox.center.position.y:.3f}", int(deposited),
            "", "", "", "", "", self.deposits])
        self.log_file.flush()

    def _log_grid(self) -> None:
        """One `kind=grid` row per category per publish -- how much the
        Spatial-Flow prior has actually accumulated. Ported from
        user-a/sandbox."""
        if not self.log_writer:
            return
        now = self._now()
        for category, grid in self.category_grids.items():
            nnz = int(np.count_nonzero(grid.s > 1e-4))
            fspeed = np.hypot(grid.fx, grid.fy)
            # 16 empty fields between `category` and `s_max` -- must match
            # the header written in open_debug_csv (p_movable ... deposited).
            self.log_writer.writerow([
                f"{now:.3f}", "grid", "", "", category,
                *([""] * 16),
                f"{float(grid.s.max()):.4f}", f"{float(grid.s.mean()):.6f}",
                nnz, f"{float(fspeed.max()):.4f}",
                f"{float(fspeed[grid.s > 1e-4].mean()) if nnz else 0.0:.4f}",
                self.deposits])
        self.log_file.flush()

    # ------------------------------------------------------------------ output

    def _grid_info(self) -> MapMetaData:
        """This node's shared grid geometry as a MapMetaData -- factored out
        of _publish() so _publish_activity() (WP-C) builds its
        OccupancyGrid on the exact same info block rather than a second
        hand-copied one that could silently drift out of sync with S's."""
        info = MapMetaData()
        info.resolution = self.res
        info.width = self.cols
        info.height = self.rows
        info.origin.position.x = self.ox
        info.origin.position.y = self.oy
        info.origin.orientation.w = 1.0
        return info

    def _publish(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info = self._grid_info()

        # Aggregate across categories via max-combine, matching the
        # max-combine convention used everywhere else in this codebase for
        # merging multiple risk sources -- identical to the pre-category
        # behavior whenever only one category is currently live. F is not
        # published here; see module docstring.
        combined = np.zeros((self.rows, self.cols), dtype=np.float32)
        for grid in self.category_grids.values():
            np.maximum(combined, grid.s, out=combined)

        data = occupancy_grid_data(combined)
        msg.data = data.flatten(order="C").tolist()
        self.pub.publish(msg)
        self._log_grid()

    def _publish_activity(self) -> None:
        """WP-C: publish the class-agnostic 5-min activity channel A as its
        own OccupancyGrid, same header/info code path as S's _publish()
        above -- see module docstring's ACTIVITY section. A is never
        persisted and never gated by `frozen`; this topic (and, if
        configured, its `activity_viz_topic` mirror) is the only way to
        observe it. Only created/timed at all when `activity_enabled`."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info = self._grid_info()
        data = occupancy_grid_data(self.activity)
        msg.data = data.flatten(order="C").tolist()
        self.activity_pub.publish(msg)
        if self.activity_viz_pub is not None:
            self.activity_viz_pub.publish(msg)

    def _publish_flow(self) -> None:
        """Per-category (s, fx, fy) as a 3-channel float image -- see module
        docstring for why s rides along with f. Published for every
        configured category, even ones with no deposits yet (an all-zero
        image), so subscribers see a stable topic from startup."""
        stamp = self.get_clock().now().to_msg()
        for category, publisher in self.flow_pubs.items():
            grid = self.category_grids.get(category)
            if grid is None:
                packed = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
            else:
                packed = np.stack([grid.s, grid.fx, grid.fy], axis=-1).astype(np.float32)
            image_msg = self.bridge.cv2_to_imgmsg(packed, encoding="32FC3")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.map_frame
            publisher.publish(image_msg)

        for group_name, members in self.flow_groups:
            publisher = self.flow_group_pubs.get(group_name)
            if publisher is None:
                continue
            grids = [self.category_grids.get(m) for m in members]
            s, fx, fy = merge_group_flow(grids, self.rows, self.cols)
            packed = np.stack([s, fx, fy], axis=-1).astype(np.float32)
            image_msg = self.bridge.cv2_to_imgmsg(packed, encoding="32FC3")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.map_frame
            publisher.publish(image_msg)

    def _publish_headway(self) -> None:
        """Per-category (headway_mean, headway_count, last_pass_time) as a
        3-channel float image -- WP4, same geometry contract and the same
        "publish even when empty" rule as _publish_flow, so a subscriber
        sees a stable topic from startup and can tell "no data" (count 0,
        last_pass -1) from "quiet lane" (count high, last_pass old).

        The count channel is float32 here purely because the image is one
        32FC3 buffer; it holds exact integers up to 2^24, which is about
        six orders of magnitude more passes than a warehouse produces.
        """
        stamp = self.get_clock().now().to_msg()
        for category, publisher in self.headway_pubs.items():
            grid = self.category_grids.get(category)
            if grid is None:
                packed = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
                packed[:, :, 2] = np.float32(NO_PASS)
            else:
                packed = np.stack([grid.headway_mean,
                                   grid.headway_count.astype(np.float32),
                                   grid.last_pass_time],
                                  axis=-1).astype(np.float32)
            image_msg = self.bridge.cv2_to_imgmsg(packed, encoding="32FC3")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.map_frame
            publisher.publish(image_msg)

        for group_name, members in self.flow_groups:
            publisher = self.headway_group_pubs.get(group_name)
            if publisher is None:
                continue
            grids = [self.category_grids.get(m) for m in members]
            mean, count, last_pass = merge_group_headway(
                grids, self.rows, self.cols)
            packed = np.stack([mean, count.astype(np.float32), last_pass],
                              axis=-1).astype(np.float32)
            image_msg = self.bridge.cv2_to_imgmsg(packed, encoding="32FC3")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.map_frame
            publisher.publish(image_msg)

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        if not os.path.exists(self.persist_path):
            self.get_logger().info(
                f"no saved prior at {self.persist_path}; starting empty")
            return
        try:
            z = np.load(self.persist_path)
            same = (abs(float(z["resolution"]) - self.res) < 1e-9
                    and int(z["width"]) == self.cols
                    and int(z["height"]) == self.rows
                    and abs(float(z["origin_x"]) - self.ox) < 1e-6
                    and abs(float(z["origin_y"]) - self.oy) < 1e-6)
            if not same:
                self.persist_blocked = True
                self.get_logger().warning(
                    f"{self.persist_path} was saved on a different grid "
                    f"({int(z['width'])}x{int(z['height'])} @ "
                    f"{float(z['resolution'])} m, origin "
                    f"({float(z['origin_x'])}, {float(z['origin_y'])})); "
                    "starting empty rather than misplacing it, and NOT saving "
                    "over it -- fix the geometry params or point persist_path "
                    "somewhere else, then restart")
                return
            if "categories" not in z:
                # Pre-Spatial-Flow file: one undifferentiated "grid" array,
                # no category split, no F channel. Schema changed -- start
                # empty under the new schema rather than guessing which
                # category the old data belonged to. Does NOT set
                # persist_blocked: a fresh save under the new schema is
                # exactly what should happen next.
                self.get_logger().warning(
                    f"{self.persist_path} uses the pre-Spatial-Flow schema "
                    "(single grid, no category split, no F channel); "
                    "starting empty under the new schema")
                return
            # Scope filter (from user-a/sandbox): a file written before
            # `flow_categories` scoped the deposits can carry furniture /
            # unknown grids that nothing would ever refresh again; drop
            # them on load rather than publishing stale S forever.
            categories = [str(c) for c in z["categories"]
                          if str(c) in self.flow_categories]
            for category in categories:
                # Tolerant by construction: a file written before the WP4
                # headway channels existed carries only s/fx/fy, and those
                # channels simply start empty. See load_grid_arrays.
                self.category_grids[category] = load_grid_arrays(
                    z, category, self.rows, self.cols)
            # Age the reloaded prior by the wall-clock gap since it was
            # saved (2026-09-13). `saved_at` is written by save_prior; a
            # file predating that key falls back to the file's own mtime,
            # which is the same quantity measured a different way, so an
            # existing warm-up prior does not have to be regenerated to
            # pick this up. See reload_decay_factor for why this is gated
            # on `frozen` and why it uses wall clock.
            saved_at = float(z["saved_at"]) if "saved_at" in z else None
            if saved_at is None:
                try:
                    saved_at = os.path.getmtime(self.persist_path)
                except OSError:
                    saved_at = None
            reload_decay, reload_reason = reload_decay_factor(
                saved_at, time.time(), self.half_life, self.frozen)
            if reload_decay != 1.0:
                for grid in self.category_grids.values():
                    grid.decay(reload_decay)
            if reload_reason:
                self.get_logger().info(f"reload decay: {reload_reason}")

            learned = sum(int((g.headway_count > 0).sum())
                          for g in self.category_grids.values())
            self.get_logger().info(
                f"loaded prior from {self.persist_path} "
                f"(categories: {', '.join(categories) or 'none'}; "
                f"{learned} cells carry a learned headway)")
        except Exception as exc:                     # noqa: BLE001 - never fatal
            self.get_logger().error(f"could not load {self.persist_path}: {exc}")

    def _save(self) -> bool:
        """Thin Node wrapper around save_prior() -- see that function for
        the frozen/blocked short-circuits, pulled out so they are Tier-0
        testable without a Node. Called from the autosave timer (never
        created at all while self.frozen, see __init__) and from
        destroy_node's save-on-exit; save_prior's own `frozen` check is the
        belt-and-suspenders guard for the latter."""
        wrote, msg = save_prior(
            self.persist_path, self.category_grids,
            self.res, self.ox, self.oy, self.rows, self.cols,
            self.deposits, self.headway_samples,
            self.frozen, self.persist_blocked)
        if msg is not None:
            try:
                self.get_logger().info(msg)
            except Exception:                        # noqa: BLE001
                # shutdown path: rosout's context is already gone. The file
                # is written either way, which is the part that matters.
                print(msg, flush=True)
        return wrote

    def destroy_node(self):
        self._save()
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:                            # noqa: BLE001
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpatialPriorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ExternalShutdownException is what rclpy raises when the context is
        # torn down under us; without catching it the `finally` below still
        # runs, but the traceback buries the save message.
        pass
    finally:
        node.destroy_node()          # -> _save()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
