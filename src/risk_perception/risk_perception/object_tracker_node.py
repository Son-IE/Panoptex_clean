#!/usr/bin/env python3
"""
object_tracker_node.py  --  STAGE 1: probabilistic movability

Replaces the boolean `dynamic_capable` latch with TWO separate beliefs:

  p_movable  -- "could this ever move?"  Slow memory. Seeded by a semantic prior
                (person 0.9, unknown 0.3, wall 0.05); pushed UP when motion is
                observed; decays only WEAKLY toward the prior. A cart parked for
                30 s is still very movable.

  p_motion   -- "is it moving right now?"  Fast. Driven by the Mahalanobis length
                of the velocity, decays quickly to ~0 on stillness. Stage 2's
                future-occupancy mixture uses THIS weight, not p_movable -- a
                standing person must not smear forward.

motion_score = sqrt(v^T Sigma_v^-1 v): velocity in units of its own uncertainty.
A static object's jitter-velocity is small relative to its covariance, so the
score stays low. Set `motion_threshold` above the 99th percentile of a static
bag's motion_score histogram -- that plot is the threshold's justification.
Research logging (all off unless a path/dir is given):
  motion_log_path  -- explicit CSV path (kept for backward compat with
                      tools/motion_threshold_histogram.py).
  debug_log_dir    -- a DIRECTORY; the node writes
                      <dir>/object_tracker_<UTC timestamp>.csv itself, so
                      successive runs never clobber each other. Wins over
                      motion_log_path if both are set.
  log_unconfirmed  -- also log tracks below min_hits / min_confidence (the
                      rows those gates are throwing away), each tagged
                      confirmed=0 / alive=0. Default False.
One row per track per tick, covering every prior's inputs and outputs --
see _log(). Feed the CSV to tools/prior_report.py.

Output contract on /risk_perception/world_objects (everything downstream
parses this, see risk_visualization.parse_class_id):

    class_id        "label|pmov=0.90|pmot=0.05|vx=0.12|vy=-0.03|relbonus=0.00
                     |hits=12|age=3.40|disp=0.62|promoted=0"
    pose.covariance cov[0]=Pxx  cov[7]=Pyy  cov[21]=Pvxvx  cov[28]=Pvyvy

`label` above is Track.published_label()'s output (`mover_promote_label`,
"moving object", while a track is promoted -- see the 2026-09-10 follow-up
paragraph below -- else the raw label), NOT necessarily Track.label
itself; `promoted` (0/1) makes that substitution visible to an auditor
without having to already know a track's raw label to notice it changed.

Track maturity + speed plausibility (2026-09-10 finding): with real
perception, label-agnostic `lidar_cluster` tracks spawned off shelf edges
(static_margin_m alone let AMCL error + map quantisation through) jump
between shelf segments on consecutive scans; the Kalman filter reads that
jump as a velocity of several m/s, `motion_speed_mps`'s absolute test then
marks the track "moving", and the predictive stack paints a multi-second
comet along a path the object never took. `hits`/`age` above let a
downstream consumer down-weight a barely-observed track outright; the
`max_speed_*_mps` / `jump_reset_factor` params below clamp (or, past
`jump_reset_factor`x the cap, zero out) any track's post-update velocity so
a single bad association can no longer be read as physically-real motion.

2026-09-10 phantom velocity (WP1): the speed-plausibility clamp above
catches a single bad *association*; it does nothing about a PARKED object's
own steady-state Kalman velocity, which is not zero -- `Track.predict` adds
`process_noise * dt` to the velocity variance every tick, so with ~0.3 m
inter-camera ground-plane offsets between overhead cams the filter reads a
standing Carter's jitter as sigma_v ~ 0.5-0.9 m/s, comfortably clearing the
old `motion_speed_mps`/`motion_speed_score` absolute test and the old
`process_noise 0.5` / `P_v` init `1.0` made that velocity noisy enough that
a 6 s predictive rollout painted a 10-13 m comet around a track that never
moved (see `~/.claude/plans/context-perception-panoptex-real-cameras-cuddly-coral.md`
for the full bag analysis). The fix moves the SAME idea
`panoptex_nav/gt_tracks_node.py` already uses for oracle Carter tracks --
window displacement, not instantaneous velocity, decides "is this moving"
-- onto the real tracker: `Track.meas` keeps a `motion_window_s` deque of
raw `(stamp, x, y, cov)` measurements; `Track.displacement_evidence`
(`risk_perception.motion_evidence.displacement_evidence`) tests net
displacement over that window against a noise-scaled threshold AND a
path-straightness ratio (camera hand-over jitter zigzags without going
anywhere; a real mover's path stays close to a straight line), so
`motion_require_displacement: true` (default) makes `update_beliefs`'s
`moving` decision `disp_ok` alone -- the legacy Mahalanobis/speed OR-test
only runs when `motion_require_displacement: false`. `p_motion` below
`motion_vel_damp_below` also now exponentially damps the Kalman velocity
state itself (`motion_vel_damp_tau_s`), so a track that stops predicting
forward with a decaying-but-nonzero jitter velocity, and the *published*
`vx, vy` default to the displacement-window velocity (`velocity_source:
displacement`) rather than the KF's, with the raw displacement magnitude
appended to `class_id` as `|disp=..` for downstream diagnostics
(`tools/track_audit.py`, `tools/retrack_bag.py`). Retuned alongside this:
`process_noise` 0.5 -> 0.05, the Kalman velocity-variance seed (was
hardcoded 1.0) -> `velocity_var_init` 0.25, and the association gate
(`_gate_for`/`_revive`) now grows at `gate_speed_mps` (1.0 -> 0.6) capped at
`gate_max_m` (2.5) instead of growing unbounded -- the old uncapped gate is
what let a long-coasting track re-associate 10-25 m away ("speed jump
reset" log lines) in the first place.

2026-09-10 follow-up, "motion overrides class" (`unit2_pan_2` confirmation
run finding): fixing phantom velocity exposed a different failure the same
shape as it -- in 5/7 carter2 contacts on that run, the Carter WAS tracked
correctly as moving (pmot 1.0, v ~0.5 m/s, real displacement) but under a
mislabelled furniture category ("table"/"chair", GroundingDINO phrase
noise), and furniture is excluded from the risk stack, the speed governor
and mission_supervisor's corridor-yield users -- all three key off
`label_category()`, not a raw label, so a genuinely moving object with the
wrong label was invisible to every one of them. `Track.update_promotion`
sets `promoted = True` once a furniture/unknown-category track's p_motion
has stayed >= `mover_promote_pmot_min` continuously for `mover_promote_min_s`
with enough displacement (`mover_promote_min_disp_m`) and hits
(`mover_promote_min_hits`) to trust it, sticky until `mover_demote_s` of no
further motion (a real table that briefly stops moving mid-crossing must
not instantly fall back out of the risk stack). Promotion NEVER touches
`self.label`/category -- only `Track.published_label()` (what
`_publish_objects`/`_publish_markers` actually emit) swaps to
`mover_promote_label` ("moving object", mapped to category "wheeled" /
consequence 0.65 in `risk_visualization.py`) while promoted -- so a "table"
detection keeps associating with the (now promoted) track exactly as
before, and every category-gated downstream consumer picks up the
override for free through the existing label_category() plumbing.
"""

import math
import colorsys
import time
from collections import deque, Counter
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Quaternion
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from risk_perception.risk_visualization import (
    label_category, risk_score_from_label)
from risk_perception.debug_log import open_debug_csv, open_latency_csv, log_latency
from risk_perception.motion_evidence import displacement_evidence, ema_update

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def id_color(track_id):
    return colorsys.hsv_to_rgb((track_id * 0.61803398875) % 1.0, 0.65, 0.95)


def yaw_quat(vx, vy):
    q = Quaternion()
    yaw = math.atan2(vy, vx)
    q.z, q.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    return q


class Track:
    _next_id = 0

    def __init__(self, x, y, label, score, cov_xy, size, stamp, p_movable_prior,
                 relconf=0.0, velocity_var_init=0.25, confirm_m=5,
                 label_vote_half_life_s=30.0):
        Track._next_id += 1
        self.id = Track._next_id
        self.label = label
        self.size = size
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        # velocity_var_init (WP1, 2026-09-10 "phantom velocity" fix): was
        # hardcoded 1.0 -- the filter's own prior, not evidence -- which by
        # itself inflated a 6 s predictive rollout's sigma to ~6 m on a
        # freshly spawned track (see module docstring). 0.25 (sigma 0.5 m/s)
        # matches predictive_risk_costmap_node's own sigma_v_max_mps clamp.
        self.P = np.diag([cov_xy, cov_xy, velocity_var_init, velocity_var_init])
        self.confidence = float(score)
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.last_seen = stamp
        # First-observation stamp -- published as `age` (see
        # ObjectTrackerNode._publish_objects) so a downstream consumer can
        # tell "just spawned" apart from "tracked for a while" without
        # needing its own bookkeeping. Never touched again after __init__:
        # _revive() reuses the SAME Track object (age keeps accumulating
        # across the graveyard gap), only _spawn()/_spawn_agnostic() create
        # a fresh one with a fresh first_seen.
        self.first_seen = stamp
        self.p_movable = float(p_movable_prior)
        self.p_movable_prior = float(p_movable_prior)
        self.p_motion = 0.0
        self.motion_score = 0.0
        # relation prior (proposed) -- event-driven, from gdino_detector's
        # compound-phrase match ("person on forklift"), embedded as
        # relconf=<score> on this object's OWN incoming class_id. Rises and
        # decays smoothly rather than snapping to 1.0 on a hit: the relation
        # prompt only samples at ~2 Hz, so a hard snap would show as a visible
        # jump in the published costmap every ~0.5s, unlike p_motion's snap
        # which is invisible at its 10 Hz rate.
        self.relation_bonus = 0.0
        # latest relconf seen, consumed (and zeroed) by the next
        # update_relation() tick -- see that method's docstring.
        self.last_relconf = float(relconf)
        # What update_relation() consumed on the most recent tick (0.0 if
        # no relation hit that tick) -- kept so the research CSV can log the
        # relation prior's INPUT, which last_relconf has already been zeroed
        # for by the time _log() runs. Ported from user-a/sandbox.
        self.relconf_event = float(relconf)
        # WP1 displacement evidence (2026-09-10 "phantom velocity" fix, see
        # module docstring): raw (stamp, x, y, cov) measurement history,
        # oldest first, pruned in update() to the newest motion_window_s
        # seconds -- what displacement_evidence() tests for real net
        # displacement instead of trusting the Kalman filter's own jittery
        # instantaneous velocity.
        self.meas = deque()
        # EMA-smoothed displacement-window velocity (see update_beliefs) --
        # what _publish_objects reports when velocity_source is
        # "displacement" (the default). (0.0, 0.0) until enough measurements
        # accumulate.
        self.disp_v = (0.0, 0.0)
        # Most recent net displacement over the window, in metres --
        # published as class_id's |disp=.. field regardless of
        # motion_require_displacement, purely diagnostic (see
        # tools/track_audit.py / tools/retrack_bag.py).
        self.net_disp_m = 0.0
        # WP1 "motion overrides class" (2026-09-10 follow-up finding,
        # unit2_pan_2): a track whose RAW label/category is furniture or
        # unknown (GroundingDINO mislabelling a moving Carter as "table"/
        # "chair") is excluded from the risk stack, the speed governor and
        # mission_supervisor's corridor-yield users -- all category-gated
        # -- even once its OWN displacement evidence has clearly shown it
        # moving. `promoted` (see update_promotion) does not touch
        # `self.label`/category at all (association/graveyard/merge still
        # need the raw label a "table" detection keeps arriving as); it
        # only changes what published_label() reports. `_promote_high_since`
        # is the sim/wall-clock timestamp p_motion most recently rose above
        # mover_promote_pmot_min and has stayed there continuously since
        # (reset to None the instant it drops back below); `_last_motion_ge_half`
        # is the timestamp of the most recent tick with p_motion >= 0.5,
        # what the sticky demote timer (mover_demote_s) counts from.
        self.promoted = False
        self._promote_high_since = None
        self._last_motion_ge_half = None

        # WP-B (2026-09-11) category-agnostic, stable association --
        # agnostic-mode-only state; harmless/unused under legacy mode.
        # `recent`: N-of-M confirmation window (see
        # ObjectTrackerNode._tick's association_mode=="agnostic" branch) --
        # 1 appended on every update() hit, 0 on every mark_missed() miss.
        self.recent = deque(maxlen=max(1, int(confirm_m)))
        # WP-B miss-charging fix (2026-09-11): last time `_tick` charged
        # this track a miss, or None if it hasn't gone stale since its last
        # hit. update() resets this to None on every hit (a fresh detection
        # means the track is no longer silent, regardless of source) --
        # see ObjectTrackerNode._tick's grace-gated mark_missed() call.
        self._last_miss_charge_t = None
        # `label_votes`: score-weighted, half-life-decayed Counter keyed by
        # RISK CATEGORY (not raw phrase) -- see register_label_vote. Seeded
        # here with this track's own first observation so a track that is
        # only ever updated once still has a well-defined "winning
        # category" (itself) rather than an empty Counter.
        self.label_votes = Counter()
        # category -> most recent RAW label seen carrying that category --
        # what self.label gets set to once that category wins the vote
        # (register_label_vote's docstring), so consequence/category
        # lookups (label_category(self.label)) keep working exactly as
        # before even though the field is no longer "whatever the last
        # detection said" but "whatever the last detection of the WINNING
        # category said".
        self._label_last_raw = {}
        self._vote_last_t = None
        # Most recent raw label seen from ANY category (not gated by which
        # category is currently winning) -- purely diagnostic, published as
        # class_id's `|raw=..` field in agnostic mode so an auditor can see
        # a label flip-flop even while self.label itself stays stable on
        # the winning category. Distinct from self.label, which can lag
        # this by design.
        self.last_raw_label = label
        self.register_label_vote(label, score, label_category(label), stamp,
                                 label_vote_half_life_s)

    def predict(self, dt, q):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.x = F @ self.x
        Q = np.diag([0.0, 0.0, q * dt, q * dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, zx, zy, cov_xy, score, size, stamp, relconf=0.0,
               motion_window_s=2.0):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.diag([cov_xy, cov_xy])
        z = np.array([zx, zy])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.confidence = max(self.confidence, float(score))
        self.size = 0.7 * self.size + 0.3 * size
        self.hits += 1
        self.misses = 0
        self.last_seen = stamp
        if relconf > 0.0:
            self.last_relconf = float(relconf)
        # WP1 displacement evidence: raw measurement (not the filtered
        # state) -- see module docstring and motion_evidence.py. Pruned
        # relative to the NEWEST measurement's own stamp, not wall/sim
        # "now", so a track that stops receiving updates (a coast) keeps
        # whatever evidence it already had instead of losing it to a
        # clock-driven prune -- update_beliefs's motion_hold_s is what
        # decides how long a coast still counts as "moving".
        self.meas.append((float(stamp), float(zx), float(zy), float(cov_xy)))
        newest_t = self.meas[-1][0]
        # 1e-6 s tolerance: a sample landing exactly on the window boundary
        # (e.g. observations spaced exactly motion_window_s / (min_samples-1)
        # apart) must not be pruned away by float accumulation error alone.
        while len(self.meas) > 1 and (newest_t - self.meas[0][0]) > motion_window_s + 1e-6:
            self.meas.popleft()
        # WP-B N-of-M confirmation (agnostic mode only, see ObjectTrackerNode
        # ._tick) -- a hit. Harmless under legacy mode: `recent` just
        # accumulates unread.
        self.recent.append(1)
        # A hit means the track is no longer silent -- clear any in-progress
        # miss-charging state (see mark_missed()/_last_miss_charge_t) so the
        # next stale period starts its own fresh grace window instead of
        # picking up where an old, now-irrelevant one left off.
        self._last_miss_charge_t = None

    def mark_missed(self):
        """A miss for N-of-M confirmation (WP-B). Historically called from
        ObjectTrackerNode._detections_cb on every unmatched detections
        MESSAGE -- fixed 2026-09-11 ("Status 2026-09-11 13:00", WP-B
        regression): 3 overhead cameras + 10 Hz lidar share one callback,
        so a track seen by only one slow camera collected a miss on every
        OTHER source's message too (~5 misses per real hit) and almost
        never held enough hits in its window to confirm. Under agnostic
        mode this is now called from ObjectTrackerNode._tick instead, at
        most once per confirm_miss_grace_s of continuous silence
        (_last_miss_charge_t) -- completely decoupled from message
        arrival rate. Legacy mode still calls this once per unmatched
        message, unchanged (see _detections_cb's `mode == "legacy"`
        guard) -- `misses` keeps its old meaning there."""
        self.misses += 1
        self.recent.append(0)

    def register_label_vote(self, raw_label, score, category, now, half_life_s):
        """WP-B label voting (2026-09-11): score-weighted Counter over
        CATEGORIES (not raw phrases), decayed at `half_life_s` seconds
        since the last vote on THIS track -- so a GroundingDINO phrase
        flicker ("table"/"chair"/"mobile robot" on one physical Carter)
        settles into whichever category has actually accumulated the most
        confidence-weighted evidence recently, instead of `self.label`
        snapping to whatever the single most recent (or most confident,
        under the old _adopt_label rule) detection happened to say.

        `self.label` is then set to the most recent RAW label seen carrying
        the WINNING category (`_label_last_raw[winner]`) -- not the winning
        category's name itself -- so every existing consequence/category
        lookup (`label_category(self.label)`, `risk_score_from_label`, ...)
        keeps working unchanged; only WHICH raw label wins is now
        vote-stabilised rather than latest-wins.

        `last_raw_label` (unconditional, not gated by which category wins)
        is the purely diagnostic `|raw=..` field for class_id -- see
        ObjectTrackerNode._publish_objects."""
        if half_life_s > 0.0 and self._vote_last_t is not None:
            dt = max(0.0, now - self._vote_last_t)
            if dt > 0.0:
                decay = 0.5 ** (dt / half_life_s)
                for k in list(self.label_votes):
                    self.label_votes[k] *= decay
        self._vote_last_t = now
        self.label_votes[category] += max(0.0, float(score))
        self._label_last_raw[category] = raw_label
        self.last_raw_label = raw_label
        if self.label_votes:
            winner = max(self.label_votes.items(), key=lambda kv: kv[1])[0]
            self.label = self._label_last_raw.get(winner, raw_label)

    def position_mahalanobis_d2(self, zx, zy, cov_xy):
        """WP-B association cost primitive: d^2 = r^T (H P H^T + R)^-1 r for
        a candidate measurement (zx, zy, cov_xy) against this track's
        CURRENT state/covariance -- same H/R shape as update()'s own
        Kalman gain computation, just read-only (no state mutation). The
        caller is responsible for any age/capture-time shift of (zx, zy)
        (see ObjectTrackerNode._assoc_cost_agnostic: it shifts the
        DETECTION forward by track-velocity * age rather than shifting the
        track backward, mathematically the same residual either way).
        Returns (d2, euclidean_residual_m) -- the second value is what
        callers additionally gate on `‖r‖ <= _gate_for(...)` (a hard metric
        cap alongside the statistical one, since a very tight P can make an
        implausibly large jump read as a small d2)."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.diag([cov_xy, cov_xy])
        S = H @ self.P @ H.T + R
        r = np.array([zx - self.x[0], zy - self.x[1]])
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.eye(2) / max(cov_xy, 1e-6)
        d2 = float(r @ Sinv @ r)
        r_norm = float(np.hypot(r[0], r[1]))
        return d2, r_norm

    def predicted_covariance(self, dt, q):
        """Read-only twin of predict(): F @ P @ F.T + Q for elapsed `dt`,
        WITHOUT mutating self.P or self.x -- used by _revive_agnostic to
        gate a graveyard track (which predict() never ran on while dead)
        against a fresh detection without corrupting the track's actual
        state before deciding whether to revive it at all."""
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Q = np.diag([0.0, 0.0, q * dt, q * dt])
        return F @ self.P @ F.T + Q

    def enforce_speed_cap(self, cap_mps, jump_reset_factor, pre_x, pre_y):
        """Speed-plausibility clamp, run right after a Kalman update() (see
        the module docstring's 2026-09-10 finding). `cap_mps` is this
        track's category speed cap; `pre_x`/`pre_y` are its position
        immediately BEFORE the update just applied (used only to report the
        displacement that produced an implausible velocity -- this method
        does not otherwise need it).

        speed <= cap: no-op.
        cap < speed <= jump_reset_factor * cap: clamp the velocity STATE to
            cap_mps, keeping direction -- a genuine but slightly-too-fast
            estimate (sensor noise, a short KF transient) is trusted at the
            cap rather than discarded outright.
        speed > jump_reset_factor * cap: treated as an association jump
            (e.g. a lidar_cluster track that just re-associated with a
            different shelf edge), not real motion -- velocity and p_motion
            are both reset to 0 rather than merely clamped, since the
            direction itself is meaningless here.

        Returns (was_capped, was_jump, pre_clamp_speed_mps, displacement_m)
        so the caller (which owns the ROS logger) can log a jump event --
        Track stays pure/ROS-free, same as every other method here.
        """
        pre_clamp_speed = self.speed
        if pre_clamp_speed <= cap_mps:
            return False, False, pre_clamp_speed, 0.0
        displacement = float(np.hypot(self.x[0] - pre_x, self.x[1] - pre_y))
        if pre_clamp_speed > jump_reset_factor * cap_mps:
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.p_motion = 0.0
            return True, True, pre_clamp_speed, displacement
        scale = cap_mps / pre_clamp_speed
        self.x[2] *= scale
        self.x[3] *= scale
        return True, False, pre_clamp_speed, displacement

    def motion_mahalanobis(self):
        v = self.x[2:4]
        Sv = self.P[2:4, 2:4] + np.eye(2) * 1e-6
        try:
            m2 = float(v @ np.linalg.inv(Sv) @ v)
        except np.linalg.LinAlgError:
            m2 = 0.0
        self.motion_score = math.sqrt(max(0.0, m2))
        return self.motion_score

    def displacement_evidence(self, window_s, min_disp_m, k_sigma, min_samples,
                              straightness_min, method="half_median"):
        """Track-owned wrapper around motion_evidence.displacement_evidence
        -- see that function's docstring for the maths. Reads self.meas
        (raw measurement history, see update()); does not mutate it."""
        return displacement_evidence(
            list(self.meas), window_s, min_disp_m, k_sigma, min_samples,
            straightness_min, method=method)

    def update_beliefs(self, threshold, p_motion_decay, movable_gain,
                       movable_decay, dt, now, speed_mps=0.0, speed_score=1.0,
                       motion_window_s=2.0, motion_min_displacement_m=0.3,
                       motion_disp_k_sigma=1.5, motion_min_samples=2,
                       motion_straightness_min=0.6, motion_hold_s=3.0,
                       require_displacement=True,
                       motion_vel_damp_below=0.5, motion_vel_damp_tau_s=0.5,
                       motion_disp_method="half_median"):
        score = self.motion_mahalanobis()
        speed = float(np.hypot(self.x[2], self.x[3]))

        # WP1 displacement evidence (2026-09-10 "phantom velocity" fix, see
        # module docstring): does this track's own measurement history show
        # real net displacement, not just an instantaneous KF velocity that
        # is mostly camera jitter? net_disp_m/disp_v are stored/published
        # regardless of require_displacement -- purely diagnostic when the
        # legacy test is selected. motion_disp_method (2026-09-10
        # follow-up): "half_median" (default) is robust to a persistent
        # lidar-vs-camera position offset that the original "endpoints"
        # statistic misread as jitter -- see motion_evidence.py's
        # displacement_evidence docstring.
        net_disp_m, disp_vx, disp_vy, disp_moving = self.displacement_evidence(
            motion_window_s, motion_min_displacement_m, motion_disp_k_sigma,
            motion_min_samples, motion_straightness_min, method=motion_disp_method)
        self.net_disp_m = net_disp_m
        # Same EMA time-constant convention as gt_tracks_node.py: alpha is
        # this tick's dt as a fraction of the displacement window, so the
        # smoothing time constant IS motion_window_s. self.disp_v is None
        # is never true after __init__ ((0.0, 0.0) seed), matching
        # ema_update's "first sample" branch only on a genuinely fresh Track.
        alpha = (dt / motion_window_s) if motion_window_s > 0.0 else 1.0
        self.disp_v = ema_update(self.disp_v, (disp_vx, disp_vy), alpha)

        # disp_ok additionally requires the evidence to still be "fresh":
        # moving from displacement_evidence() only changes when a new
        # measurement arrives, so without this a track that stopped being
        # updated at all would read as moving forever off its last window.
        # motion_hold_s is deliberately the SAME kind of hold the pre-WP1
        # coasting-through-a-gap behaviour relied on (see
        # test_track_coasts_through_camera_handover_gap_with_velocity_intact).
        disp_ok = disp_moving and (now - self.last_seen) <= motion_hold_s

        if require_displacement:
            moving = disp_ok
        else:
            # Legacy composition (pre-2026-09-10): statistical test
            # (velocity distinguishable from zero at `threshold` sigmas) OR
            # an absolute one (speed_mps <= 0 keeps the statistical-only
            # test). Selectable via motion_require_displacement: false so
            # the lab tracker can fall back if the displacement test proves
            # too conservative on real hardware (see WP1 risk notes).
            moving = score > threshold or (
                speed_mps > 0.0 and speed >= speed_mps and score >= speed_score)

        if moving:
            self.p_motion = 1.0
        else:
            self.p_motion *= math.exp(-p_motion_decay * dt)
        # Both branches are now scaled by dt -- previously movable_gain was
        # applied once per TICK (not per second) while movable_decay already
        # was per-second, a 10x asymmetry at the default 10 Hz rate that let a
        # single noisy motion blip latch a static object into the "movable"
        # bucket for ~1 minute.
        if moving:
            self.p_movable += movable_gain * (1.0 - self.p_movable) * dt
        else:
            self.p_movable += movable_decay * (self.p_movable_prior - self.p_movable) * dt
        self.p_movable = float(min(1.0, max(0.0, self.p_movable)))

        # Velocity damping (WP1): a low p_motion means the KF velocity is
        # jitter, not real motion -- decay the STATE itself toward zero so
        # predict() stops walking a parked track along a jitter velocity
        # during a coast (and so CPA/TTC, which read Track.velocity, don't
        # score a phantom closing speed). Continuous exponential decay, not
        # a hard zero, so a track crossing back above the threshold keeps a
        # sensible velocity to resume from. Gated on require_displacement
        # (unlike net_disp_m/disp_v above, which are always computed) so
        # motion_require_displacement: false is a genuine, full fallback to
        # the pre-WP1 tracker -- not the new displacement test PLUS a novel
        # damping behaviour the legacy composition was never tuned against.
        if require_displacement and motion_vel_damp_tau_s > 0.0 \
                and self.p_motion < motion_vel_damp_below:
            damp = math.exp(-dt / motion_vel_damp_tau_s)
            self.x[2] *= damp
            self.x[3] *= damp

    def update_promotion(self, now, enabled, promote_pmot_min, promote_min_s,
                         promote_min_disp_m, promote_min_hits, demote_s):
        """WP1 "motion overrides class" (2026-09-10 follow-up, see
        `promoted`'s __init__ comment). Call once per tick, AFTER
        update_beliefs (needs this tick's p_motion/net_disp_m).

        Promotion (False -> True): only for a track whose RAW category
        (label_category(self.label), untouched by this method) is
        "furniture" or "unknown" -- a track already categorised as
        person/robot/wheeled is already actionable, nothing to override.
        Requires ALL of: p_motion >= promote_pmot_min CONTINUOUSLY for the
        last promote_min_s seconds (`_promote_high_since` tracks the start
        of the current streak, reset the instant p_motion dips below the
        threshold -- a single low tick restarts the clock), net_disp_m >=
        promote_min_disp_m (the same displacement-evidence diagnostic
        `update_beliefs` already computed this tick), and hits >=
        promote_min_hits (a barely-observed track's motion belief is the
        tracker's own seeded prior, not yet evidence -- same reasoning as
        predictive_risk_costmap_node.is_track_mature, independently
        re-checked here since this method has no access to that module).

        Demotion (True -> False): STICKY, not instant -- a real table
        that stops moving must not instantly fall back out of the risk
        stack/governor/corridor-yield the moment it parks (that is
        exactly the false-negative promotion exists to prevent), but a
        FALSE promotion (mislabelled clutter that briefly jittered above
        the promote thresholds) must not stay promoted forever either.
        `_last_motion_ge_half` is the timestamp of the most recent tick
        with p_motion >= 0.5 (independent of promote_pmot_min, a lower
        bar so the demote clock doesn't restart on ordinary p_motion
        noise around the promote threshold); demotion fires once
        (now - _last_motion_ge_half) exceeds demote_s, or immediately if
        p_motion has never once reached 0.5 (a promoted track that
        somehow never set the timestamp -- defensive, should not happen
        via the normal promotion path above, which itself requires
        p_motion >= promote_pmot_min >= 0.5 in practice)."""
        if not enabled:
            return
        if self.p_motion >= promote_pmot_min:
            if self._promote_high_since is None:
                self._promote_high_since = now
        else:
            self._promote_high_since = None
        if self.p_motion >= 0.5:
            self._last_motion_ge_half = now

        if not self.promoted:
            category = label_category(self.label)
            if category not in ("furniture", "unknown"):
                return
            if (self._promote_high_since is not None
                    and (now - self._promote_high_since) >= promote_min_s
                    and self.net_disp_m >= promote_min_disp_m
                    and self.hits >= promote_min_hits):
                self.promoted = True
        else:
            if (self._last_motion_ge_half is None
                    or (now - self._last_motion_ge_half) > demote_s):
                self.promoted = False

    def published_label(self, mover_promote_label="moving object"):
        """What _publish_objects/_publish_markers report as this track's
        label -- `mover_promote_label` while promoted (see
        update_promotion), else the raw `self.label` unchanged. Never
        touches self.label itself -- association/graveyard/merge (and any
        future _adopt_label/_upgrade_track call) all keep reading/writing
        the raw label exactly as before this feature."""
        return mover_promote_label if self.promoted else self.label

    def update_relation(self, rise, decay, dt):
        """Update relation_bonus for the relation prior (proposed).

        `last_relconf` is set by update() whenever a matched detection this
        cycle carried a relconf=.. tag, and consumed (zeroed) here -- so a
        hit is whatever came in since the last tick, not "this exact tick".

        On a hit: relation_bonus moves a fraction `rise` of the way toward
        last_relconf (never jumps straight to it -- `rise=1.0` would).
        Otherwise: exponential decay at rate `decay`, same shape as
        p_motion_decay above.
        """
        self.relconf_event = float(self.last_relconf)
        if self.last_relconf > 0.0:
            self.relation_bonus += rise * (self.last_relconf - self.relation_bonus)
            self.last_relconf = 0.0
        else:
            self.relation_bonus *= math.exp(-decay * dt)
        self.relation_bonus = float(min(1.0, max(0.0, self.relation_bonus)))

    def decay_confidence(self, dt, half_life):
        # dt is the elapsed TICK interval, not the age since last_seen -- this
        # is called once per tick, so using age-since-last-seen would multiply
        # a growing exponent back in every tick and compound quadratically
        # (a configured 120s half-life would then delete a track in ~8s).
        if dt > 0 and half_life > 0:
            self.confidence *= 0.5 ** (dt / half_life)

    @property
    def position(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self):
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


class ObjectTrackerNode(Node):
    def __init__(self, **kwargs):
        # **kwargs passthrough (WP4, 2026-09-10): lets
        # tools/retrack_bag.py construct this node headlessly with
        # parameter_overrides=[Parameter(k, value=v), ...] on top of the
        # yaml object_tracker: block, the same rclpy.node.Node constructor
        # kwarg any node already supports -- no other behaviour changes for
        # the normal no-kwargs launch-file construction.
        super().__init__("object_tracker", **kwargs)
        self.declare_parameter("input_topic", "/risk_perception/detections_3d_map")
        self.declare_parameter("output_topic", "/risk_perception/world_objects")
        self.declare_parameter("marker_topic", "/risk_perception/world_markers")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("gate_distance_m", 0.6)
        # Association gate grows with the time a track has gone unobserved:
        # gate = min(gate_max_m, gate_distance_m + gate_speed_mps *
        # (now - last_seen)). With detections arriving at ~0.5-1.5 Hz per
        # camera, a 0.6 m/s AMR moves farther than a fixed 0.6 m gate between
        # two sightings and was being re-spawned as a fresh (zero-velocity)
        # track every time (44 distinct tracks for one Carter in the
        # 2026-09-08 sim run). 0.0 = fixed gate. Lowered 1.0 -> 0.6 (WP1,
        # 2026-09-10): paired with the new gate_max_m cap below -- the old
        # UNCAPPED gate is what let a long-coasting track re-associate
        # 10-25 m away after a multi-second gap ("speed jump reset" log
        # lines, module docstring's 2026-09-10 finding); 0.6 m/s still
        # covers the sim Carter's own patrol speed within gate_max_m.
        self.declare_parameter("gate_speed_mps", 0.6)
        # Hard ceiling on the association gate regardless of how long a
        # track has gone unobserved (WP1, 2026-09-10) -- also bounds
        # _revive's graveyard-match gate, since _revive calls _gate_for the
        # same way. Without this, a track unseen for tens of seconds (a
        # long occlusion, or simply a slow update_rate during startup) could
        # associate with a detection many metres away and read the jump as
        # real velocity (see enforce_speed_cap's "speed jump reset").
        self.declare_parameter("gate_max_m", 2.5)
        # "label": exact class_id match (legacy). "category": match on
        # risk_visualization.label_category (person/robot/wheeled/furniture),
        # so GroundingDINO phrase churn ("mobile robot" vs "ground mobile
        # robot" vs "cart") no longer splits one object into many tracks.
        # Unknown-category labels still require an exact match.
        self.declare_parameter("association_key", "category")
        # Categories that may associate with each other even though they are
        # not equal (association_key == "category" only). Built for the
        # 2026-09-09 bag finding: GroundingDINO alternates "ground mobile
        # robot" (category robot) with "cart"/"forklift" (category wheeled)
        # on the SAME physical Carter, and category-keyed association hard-
        # blocked the cross-category match, spawning a second track that
        # never aged out (38 distinct ids on one Carter, 326 nearest-id
        # switches). Each entry is a comma-separated set of categories that
        # may all associate with each other; _same_object_class (category
        # mode) returns true when the two categories are equal OR share a
        # group. Track label adoption is unaffected (_adopt_label still
        # picks by confidence) -- a track can legitimately move from label
        # "cart" to "ground mobile robot", and its category (derived from
        # the label via risk_visualization.label_category, never cached)
        # follows automatically.
        self.declare_parameter("association_groups", ["robot,wheeled"])
        # Label-agnostic measurement mode (WP1): detections whose bare label
        # is in this list (case-insensitive) never go through the labeled
        # association pass above -- instead they run a SECOND pass
        # (_associate_agnostic) that matches the nearest existing LIVE track
        # of ANY category, never adopts a label, never revives a graveyard
        # track, and never changes a track's category. Built for the lidar
        # cluster detector (scan_cluster_detector_node.py, class_id
        # "lidar_cluster") but keyed off the label so any future
        # label-agnostic source can opt in the same way. An unmatched
        # label-agnostic detection spawns a fresh track carrying its own
        # label -- risk_visualization.label_category has no entry for
        # "lidar_cluster", so it already resolves to "unknown" there, and
        # _prior_for() then seeds it from prior_unknown.
        self.declare_parameter("label_agnostic_labels", ["lidar_cluster"])
        # Upgrade path (lidar-fusion probe, 2026-09-09): the labeled pass
        # above used to be closed to any track whose label is in
        # label_agnostic_labels (category "unknown") -- a camera detection
        # of the SAME physical object could never claim it, so the lidar
        # kept feeding an orphaned "unknown" track while a second, labeled
        # track got spawned alongside it for every subsequent camera
        # detection. Now the labeled pass also matches a label-agnostic
        # track against a labeled detection of ANY category (see
        # _detections_cb) and _upgrade_track()s it in place (same id,
        # position, velocity, covariance, p_motion -- only label/category/
        # p_movable_prior change). upgrade_penalty_m is added to the
        # Hungarian cost of such a cross-category "upgrade" candidate so a
        # genuine same-category match is preferred whenever both are within
        # the gate of the same detection.
        self.declare_parameter("upgrade_penalty_m", 0.3)
        # _associate_agnostic (the lidar pass) companion: when a lidar
        # cluster's gate contains both a labeled (non-"unknown") track and
        # an "unknown" one -- typically the leftover lidar-spawned track an
        # upgrade above hasn't consumed yet -- prefer the labeled track so
        # the unknown duplicate stops getting fed and ages out instead of
        # both being kept alive by the same lidar returns.
        self.declare_parameter("unknown_penalty_m", 0.3)
        # WP-B (2026-09-11) category-agnostic, stable association: "legacy"
        # is the pre-WP-B path above verbatim (_same_object_class hard
        # gate, Euclidean-distance cost, hits->confirmed, category-gated
        # graveyard/revive/merge). "agnostic" (default) replaces the
        # labeled pass' and _associate_agnostic's cost with a Mahalanobis-
        # on-position + size + soft-label-penalty cost (no hard category
        # gate -- a mover is a mover, category only nudges Hungarian's
        # choice, never blocks a candidate outright), adds score-weighted
        # label voting instead of latest/most-confident-wins label
        # adoption, N-of-M confirmation, a label-blind graveyard revive,
        # and a looser (Mahalanobis-or-distance) merge that tolerates one
        # unconfirmed member. See _assoc_cost_agnostic, register_label_vote,
        # _revive_agnostic, _merge_duplicates_agnostic below.
        self.declare_parameter("association_mode", "agnostic")
        # Chi-square gate for a 2-DOF Mahalanobis association test at 99%
        # confidence (scipy.stats.chi2.ppf(0.99, df=2) ~= 9.21) -- the
        # STATISTICAL half of the agnostic-mode association gate; ‖r‖ <=
        # _gate_for(...) (the existing metric cap) is still required too,
        # so an implausibly tight P can't let a large jump through on d2
        # alone. Tune this first on unit2_pan_1 against the ~0.3 m
        # lidar/camera position offset (plan's Risks section) if the gate
        # proves too tight/loose there.
        self.declare_parameter("assoc_chi2_gate", 9.21)
        # Added to the agnostic-mode association cost per metre of |size_a
        # - size_b| -- a genuinely different-sized object (a person vs. a
        # forklift) nudges Hungarian away from a coincidentally
        # position-close mismatch even when the label penalty doesn't apply
        # (e.g. both "unknown").
        self.declare_parameter("assoc_size_weight", 2.0)
        # Added to the agnostic-mode association cost when the track's and
        # detection's categories mismatch, NEITHER is "unknown", and they
        # do not share an association_groups entry -- a soft nudge, not a
        # hard gate (see association_mode's declare_parameter comment): a
        # phrase-churn mismatch still associates if the motion evidence is
        # strong enough, it just costs more than a same-category match
        # would, so Hungarian still prefers the correct match when one is
        # available.
        self.declare_parameter("assoc_label_penalty", 2.0)
        # register_label_vote's decay half-life (seconds) -- how fast an
        # old category's accumulated vote weight fades once a track stops
        # being detected under it, so a persistent relabel (not just a
        # single-frame phrase flicker) can still win within a bounded time.
        self.declare_parameter("label_vote_half_life_s", 30.0)
        # N-of-M confirmation (agnostic mode only): a track confirms once
        # at least confirm_n of its most recent confirm_m association
        # outcomes (Track.recent, hit=1/miss=0) were hits -- looser than
        # legacy's "confirmed once hits >= min_hits ever", so a track that
        # was briefly well-observed then went quiet doesn't confirm on
        # stale evidence, and a track hit intermittently (occlusion,
        # camera hand-over) can still confirm without needing min_hits
        # CONSECUTIVE good ticks.
        #
        # WP-B regression + fix (2026-09-11, "Status 2026-09-11 13:00" /
        # "Remaining work" #1 in the plan): the 4/8 ratio below was first
        # tuned against a DIFFERENT bug than the one that actually mattered
        # -- `recent` used to accumulate one hit/miss per INCOMING
        # detections message (3 overhead cameras + 10 Hz lidar all publish
        # onto the same topic and share one callback), not per system
        # tick. A parked Carter seen by only one 0.5 Hz camera collected a
        # miss on every OTHER source's message too -- roughly 5 misses per
        # real hit -- so it almost never held >= confirm_n hits in its last
        # confirm_m outcomes and simply never confirmed (never published:
        # only CONFIRMED tracks reach _publish_objects). That read as
        # carter2's tracked-% falling from 57% (legacy) to 21% (agnostic)
        # on unit2_pan_1, and no N/M ratio fixed it because the window was
        # being fed by message arrival rate, not by whether the track was
        # actually still being seen.
        #
        # Fix: misses are no longer charged in `_detections_cb` at all
        # under agnostic mode (see the `mode == "legacy"` guard around the
        # old `mark_missed()` call site below) -- they're charged from
        # `_tick` instead, at most once per `confirm_miss_grace_s` seconds
        # of continuous silence (`Track._last_miss_charge_t`), completely
        # decoupled from how many messages of any kind arrive in between.
        # A single-camera track whose real inter-detection gap stays under
        # `confirm_miss_grace_s` now accumulates ZERO misses; one at or
        # beyond a 0.5 Hz camera's ~2 s period costs at most one miss per
        # grace interval, not one per unrelated message. Re-tuned 4/8 ->
        # confirm_n/confirm_m below alongside `confirm_miss_grace_s` on
        # unit2_pan_1 after the fix (see retrack_bag numbers in the
        # tracker-fix branch's report) -- carter1/audit_after's 2-id result
        # held.
        self.declare_parameter("confirm_n", 3)
        self.declare_parameter("confirm_m", 6)
        # Grace period (seconds) a track may go unmatched before `_tick`
        # charges it a miss at all -- longer than one overhead-camera
        # period (0.5 Hz -> 2 s worst case) so a single-camera track's
        # normal inter-detection gap never crosses it and costs zero
        # misses; only a genuine gap (occlusion, hand-over, camera drop)
        # starts accumulating them, at most one per grace interval of
        # continued silence (see Track._last_miss_charge_t). Agnostic mode
        # only -- legacy mode keeps charging a miss per unmatched message,
        # unchanged (see the `mode == "legacy"` guard in `_detections_cb`).
        self.declare_parameter("confirm_miss_grace_s", 1.5)
        # Agnostic mode only: an UNCONFIRMED track that ages out still goes
        # to the graveyard (instead of being dropped outright, legacy's
        # behaviour) once it has at least this many hits -- enough real
        # observations to be worth reviving later (see _revive_agnostic),
        # not just the association noise floor a fresh 1-hit spawn would be.
        self.declare_parameter("graveyard_min_hits", 2)
        # Agnostic mode's duplicate-track merge pair test (see
        # _merge_duplicates_agnostic): two live tracks merge once their
        # position Mahalanobis distance clears merge_chi2 OR they are
        # within merge_distance_m (an OR, not legacy's AND-style
        # distance-AND-speed test) for merge_confirm_ticks consecutive
        # ticks, and at most one of the pair may be unconfirmed (legacy
        # requires BOTH confirmed). Tuned 4.0 -> 1.0 (WP-B, 2026-09-11,
        # unit2_pan_1 sweep): at 4.0, a barely-observed/wide-covariance
        # track (P has not converged yet -- e.g. a sparse "cable" clutter
        # detection) could clear the Mahalanobis test against an unrelated
        # object almost a metre away purely because its own uncertainty
        # was still large, observed live merging a "cable" track into a
        # "person" track at 0.83 m. 1.0 keeps the OR's Mahalanobis branch
        # for genuinely tight, converged duplicates while leaving
        # merge_distance_m (0.5 m) as the practical bound for everything
        # else -- confirmed on unit2_pan_1 to remove the cross-category
        # merges above while still merging real same-object splits.
        self.declare_parameter("merge_chi2", 1.0)
        # Detections carry the CAPTURE stamp (the projectors copy the image
        # stamp) but arrive 1-2 s later (GroundingDINO ~0.5 Hz per camera).
        # With this on, a detection of age `a` is compared against where the
        # track WAS at capture time and, once associated, shifted forward by
        # the track's own velocity * a before the Kalman update -- otherwise a
        # 0.6 m/s AMR is tracked 0.6-1.2 m behind where it actually is.
        self.declare_parameter("measurement_time_correction", True)
        self.declare_parameter("max_measurement_age_sec", 3.0)
        self.declare_parameter("min_hits", 3)
        self.declare_parameter("min_confidence", 0.15)
        # Lowered 0.5 -> 0.05 (WP1, 2026-09-10 "phantom velocity" fix): at
        # 0.5, steady-state KF velocity sigma from ~0.3 m inter-camera
        # jitter alone was ~0.9 m/s -- enough to clear the old absolute
        # motion test on a PARKED object. At 0.05, steady-state sigma_v is
        # ~0.3 m/s, closer to the jitter it is actually measuring; motion
        # detection itself now leans on displacement_evidence (see
        # motion_require_displacement below), not this covariance. Alters
        # the lab tracker too, not just sim -- see module docstring.
        self.declare_parameter("process_noise", 0.05)
        self.declare_parameter("default_cov_m2", 0.04)
        # Kalman velocity-variance SEED for a freshly spawned track (WP1,
        # 2026-09-10 finding, root cause 1 of the "SRM fills half the map"
        # bug): was hardcoded 1.0 (m/s)^2 in Track.__init__ -- the filter's
        # own prior, not evidence -- which by itself inflated a 6 s
        # predictive rollout's sigma to ~6 m on a track only a tick old.
        # 0.25 (sigma 0.5 m/s) matches predictive_risk_costmap_node's own
        # sigma_v_max_mps clamp, so the two stay consistent even though they
        # guard different stages of the same pipeline.
        self.declare_parameter("velocity_var_init", 0.25)
        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("marker_lifetime_sec", 0.5)
        # Was hardcoded white at 0.22m with a two-line "#id label\nmov.. mot.."
        # string -- oversized and unreadable against a white map. This overlay
        # is now the primary "obstacles on the map" view (see
        # rviz/panoptex.rviz), so its legibility matters more than the raw
        # projector's debug-only markers.
        self.declare_parameter("marker_text_height_m", 0.13)
        self.declare_parameter("marker_text_color", [0.05, 0.05, 0.08])
        self.declare_parameter("marker_show_beliefs", False)
        self.declare_parameter("motion_threshold", 3.0)
        # Second, absolute way to declare motion: speed >= motion_speed_mps
        # AND Mahalanobis score >= motion_speed_score. The pure Mahalanobis
        # test alone never fired for the sim Carter (|v| ~0.6 m/s against a
        # velocity sigma of 0.6-0.9 m/s -> score ~1-2 < 3.0). 0.0 disables.
        # As of WP1 (2026-09-10) this pair is the LEGACY test -- only used
        # when motion_require_displacement is false, see below.
        self.declare_parameter("motion_speed_mps", 0.3)
        self.declare_parameter("motion_speed_score", 0.5)
        # WP1 displacement evidence (2026-09-10 "phantom velocity" fix, see
        # module docstring and risk_perception/motion_evidence.py): whether
        # a track counts as "moving" is decided by real net displacement
        # over a measurement window, not the Kalman filter's own
        # instantaneous velocity, which on a parked object is mostly
        # inter-camera jitter (~0.3 m ground-plane offset between overhead
        # cams). See Track.displacement_evidence / update_beliefs.
        self.declare_parameter("motion_window_s", 2.0)
        # A genuine displacement has to clear this floor; must exceed the
        # ~0.3 m inter-camera offset or ordinary hand-over between two
        # cameras reads as motion on its own. Tuned 0.5 -> 0.3 (WP4,
        # 2026-09-10 follow-up, tools/retrack_bag.py --mode retrack --assert
        # against unit2_pan_1 AND audit_after with motion_disp_method:
        # half_median): 0.5 missed 76-97% of carter2's own genuine movement
        # samples even after the half_median fix, because it (along with
        # k_sigma/min_samples below) was still sized for the old, noisier
        # "endpoints" statistic -- half_median's median-of-half positions
        # are far less noisy, so it needs a lower floor to catch a real
        # 0.6 m/s mover within ~1 s.
        self.declare_parameter("motion_min_displacement_m", 0.3)
        # ... OR k_sigma * sqrt(cov_oldest + cov_newest), whichever is
        # larger -- a track fused mostly from noisy global-cam ground-plane
        # projections needs a bigger displacement to trust than one fused
        # from tight lidar-cluster returns. Tuned 2.0 -> 1.5 (WP4, same
        # sweep): phantom rate on unit2_pan_1 crosses the 5% hard cap
        # around k_sigma 1.6-1.7 and keeps climbing above that as k_sigma
        # drops further (2.4% at 1.5, 5.3% at 1.0) -- 1.5 is close to the
        # <=2% target with a safety margin, and reproduces cleanly on
        # audit_after (phantom 0.6%, see the WP1 tuning report).
        self.declare_parameter("motion_disp_k_sigma", 1.5)
        # Tuned 3 -> 2 (WP4, same sweep): a genuine 2-sample net
        # displacement (already past both thresholds above) was being
        # discarded purely for lacking a 3rd point during the sim Carter's
        # slower, ~1 Hz camera-only stretches.
        self.declare_parameter("motion_min_samples", 2)
        # "endpoints"-method-only: net displacement / path length (see
        # motion_disp_method below) -- a no-op under the default
        # "half_median" method, which has no path to compute a
        # straightness ratio over. Kept at the original WP1 value for
        # when motion_disp_method: endpoints is selected explicitly.
        self.declare_parameter("motion_straightness_min", 0.6)
        # 2026-09-10 follow-up finding (WP4 retrack sweep on unit2_pan_1):
        # a track fused from BOTH a 10 Hz lidar-cluster stream and a
        # slower camera stream has a real, PERSISTENT (not jittery) offset
        # between the two sources' own position estimates. The original
        # "endpoints" statistic (newest-vs-oldest displacement, gated by a
        # path-length straightness ratio) reads that offset as a zigzag --
        # observed straightness ~0.1-0.3 for a track moving in a dead
        # straight line at 0.6 m/s -- and rejected most of a genuine
        # mover's own evidence. "half_median" (default) instead splits the
        # measurement window in half BY TIME and compares the
        # coordinate-wise MEDIAN position of each half: a per-source
        # offset that's roughly constant within each half cancels out of
        # the median-to-median difference instead of alternating into
        # inflated path length. See motion_evidence.displacement_evidence.
        # "endpoints" is kept selectable as an explicit fallback (a single-
        # source track, or a preference for the original design) -- not
        # just dead code.
        self.declare_parameter("motion_disp_method", "half_median")
        # How long displacement evidence keeps counting as "moving" after
        # the track's last actual measurement (a coast: predict()-only
        # ticks with no update()) -- same purpose as the pre-WP1 confidence
        # half-life surviving a camera hand-over gap, just for p_motion's
        # own "moving" decision instead.
        self.declare_parameter("motion_hold_s", 3.0)
        # true (default): update_beliefs's `moving` decision is
        # displacement_evidence ALONE (disp_ok). false: the legacy
        # Mahalanobis-OR-speed test above, unchanged -- an explicit
        # fallback switch, not just an unused code path, in case the
        # displacement test proves too conservative on real (non-sim)
        # hardware (see the plan's risk notes).
        self.declare_parameter("motion_require_displacement", True)
        # Below this p_motion, damp the Kalman velocity STATE itself toward
        # zero (exp(-dt/motion_vel_damp_tau_s)) so predict() stops walking a
        # parked track along a jitter velocity during a coast, and CPA/TTC
        # (which read Track.velocity, not disp_v) don't score a phantom
        # closing speed. 0 disables damping.
        self.declare_parameter("motion_vel_damp_below", 0.5)
        self.declare_parameter("motion_vel_damp_tau_s", 0.5)
        # What _publish_objects reports as vx/vy: "displacement" (default)
        # is the EMA-smoothed displacement-window velocity (Track.disp_v,
        # what downstream risk-painting should trust); "kf" is the
        # (now velocity-damped) Kalman state, the pre-WP1 behaviour.
        self.declare_parameter("velocity_source", "displacement")
        self.declare_parameter("p_motion_decay", 3.0)
        self.declare_parameter("movable_gain", 0.5)
        self.declare_parameter("movable_decay", 0.01)
        # Raised from 2.0 (lab+sim default, User B's decision 2026-09-08): a
        # patrolling mover only gets covered by ONE overhead camera at a
        # time, and hand-over between cameras (or a single camera's own
        # occlusion by a shelf/doorway) is a 2-3 s gap, not the ~0.1-0.3 s a
        # single-camera dropout implies. At the old 2.0 s half-life,
        # confidence *= 0.5**(2.5/2.0) = 0.42 over a 2.5 s gap -- close
        # enough to min_confidence (0.15) that a mover with an
        # already-modest GDINO score got pruned and re-spawned as a fresh
        # (zero-velocity) track mid-gap; carter1 in the 2026-09-08
        # panoptex_nav sim runs was actually tracked only ~60% of the time
        # for exactly this reason. At 4.0 s, the same 2.5 s gap only decays
        # confidence to 0.5**(2.5/4.0) = 0.65 -- see
        # test_object_tracker_motion.py's coasting-through-a-gap test.
        self.declare_parameter("dynamic_half_life_s", 4.0)
        self.declare_parameter("static_half_life_s", 120.0)
        # Explicit deletion timeout -- confidence decay alone weights risk
        # down for stale observations but, on its own, only deletes a track
        # once confidence drops under min_confidence, which for a static
        # object at static_half_life_s can take minutes. This makes
        # "how long an unseen object lingers" a directly stated number.
        # Split by mobility (same p_movable > 0.5 split as half_life below):
        # a flat timeout was silently overriding static_half_life_s's own
        # intent -- an intermittent perception gap (occlusion, a missed sam2
        # frame) longer than 20s deleted a still-confident static track, and
        # the next detection spawned a brand new id instead of resuming the
        # old one. See the graveyard below for surviving a gap past even this.
        # Already well above the 3.0 s overhead-camera hand-over gap this
        # timeout needs to survive (dynamic_half_life_s above is what
        # actually needed raising for that case) -- kept at 20.0, not
        # lowered, so a genuinely longer dropout still gets one full
        # min_hits-worth of re-association room before the id is lost.
        self.declare_parameter("max_unseen_sec_dynamic", 20.0)
        self.declare_parameter("max_unseen_sec_static", 300.0)
        # A track pruned above still keeps its identity for a while: a
        # CONFIRMED track that ages out goes here instead of vanishing
        # outright, and a later same-label detection within the gate distance
        # revives the same Track object (same id, same accumulated
        # p_movable/p_motion/confidence) via update() -- see _revive() --
        # instead of _spawn()'ing a new one. Bounded by graveyard_ttl_sec so a
        # genuinely-gone object eventually stops being a candidate.
        self.declare_parameter("graveyard_ttl_sec", 300.0)
        self.declare_parameter("prior_person", 0.9)
        self.declare_parameter("prior_robot", 0.9)
        self.declare_parameter("prior_wheeled", 0.7)
        self.declare_parameter("prior_furniture", 0.1)
        self.declare_parameter("prior_unknown", 0.3)
        # Research logging (from user-a/sandbox) -- all off unless set.
        #   motion_log_path -- explicit CSV path (backward compat with
        #                      tools/motion_threshold_histogram.py).
        #   debug_log_dir   -- a DIRECTORY; the node names the file
        #                      <dir>/object_tracker_<UTC stamp>.csv itself so
        #                      successive runs never clobber each other.
        #                      Wins over motion_log_path if both are set.
        #   log_unconfirmed -- also log the tracks below min_hits /
        #                      min_confidence (the rows those gates throw
        #                      away), tagged confirmed=0 / alive=0.
        self.declare_parameter("motion_log_path", "")
        self.declare_parameter("debug_log_dir", "")
        self.declare_parameter("log_unconfirmed", False)
        # relation prior (proposed) -- see Track.update_relation. Starting
        # points, not tuned: validate with tools/scenario_publisher.py --rel-pulse
        # before trusting these numbers on real data.
        self.declare_parameter("relation_rise", 0.6)
        self.declare_parameter("relation_decay", 1.0)
        # Duplicate-track merge (same 2026-09-09 bag finding as
        # association_groups above): even with cross-category association
        # closing the gap for FUTURE detections, a second track that already
        # spawned before this fix landed (or that spawns anyway because a
        # category-mismatched detection happened to fall outside the other
        # track's gate) needs an explicit merge, not just prevention. Every
        # tick, live+confirmed track pairs that are category-compatible (see
        # merge_enabled's usage below), within merge_distance_m and
        # merge_speed_diff_mps of each other for merge_confirm_ticks
        # consecutive ticks get folded into one -- the older (smaller id)
        # survives, the younger is dropped outright (no graveyard entry: it
        # was never a separate object to revive).
        self.declare_parameter("merge_enabled", True)
        self.declare_parameter("merge_distance_m", 0.5)
        self.declare_parameter("merge_speed_diff_mps", 0.5)
        self.declare_parameter("merge_confirm_ticks", 3)
        # Speed plausibility (2026-09-10 finding, see module docstring): a
        # per-category cap on a track's post-update speed. A label-agnostic
        # lidar_cluster track re-associating with the WRONG shelf edge
        # between scans (static_margin_m alone let this through) reads as
        # tens of m/s to the Kalman filter -- nothing physical moves that
        # fast in this warehouse, camera-labeled or not. Categories not
        # listed here (e.g. furniture) fall to max_speed_default_mps.
        self.declare_parameter("max_speed_person_mps", 2.0)
        self.declare_parameter("max_speed_robot_mps", 1.5)
        self.declare_parameter("max_speed_wheeled_mps", 2.0)
        self.declare_parameter("max_speed_unknown_mps", 1.2)
        self.declare_parameter("max_speed_default_mps", 2.5)
        # Above cap_mps but at or below jump_reset_factor * cap_mps: clamp
        # velocity to the cap (direction kept) -- a plausible, merely
        # slightly-too-fast estimate. Above that: reset velocity/p_motion to
        # 0 instead -- see Track.enforce_speed_cap's docstring for why the
        # distinction exists (a genuine association jump has no meaningful
        # direction to preserve).
        self.declare_parameter("jump_reset_factor", 2.0)
        # WP1 "motion overrides class" (2026-09-10 follow-up, unit2_pan_2
        # finding): in 5/7 carter2 contacts on that confirmation run, the
        # Carter was tracked correctly as MOVING (pmot 1.0, v~0.5 m/s,
        # disp 0.6 m) but under a furniture label ("table"/"chair") --
        # furniture is excluded from the risk stack (stack_categories),
        # the speed governor (_ACTIONABLE_CATEGORIES) and
        # mission_supervisor's corridor-yield users (CORRIDOR_CATEGORIES),
        # so no comet, no yield, contact. See Track.update_promotion/
        # published_label -- promotion changes only the PUBLISHED label,
        # never self.label/category, so a "table" detection still
        # associates with the (now promoted) track exactly as before.
        self.declare_parameter("mover_promote_enabled", True)
        # p_motion floor the promotion streak timer requires CONTINUOUSLY
        # -- higher than update_beliefs's own moving-decision floor
        # (p_motion snaps to 1.0 or decays, it isn't usually sitting
        # near 0.5) so a promotion means "this has read as clearly and
        # persistently moving," not "flickered across 0.5 once."
        self.declare_parameter("mover_promote_pmot_min", 0.8)
        self.declare_parameter("mover_promote_min_s", 1.0)
        # Same displacement-evidence diagnostic update_beliefs already
        # computes (net_disp_m/class_id's |disp=..) -- re-checked here as
        # an independent floor so a track that only cleared
        # promote_pmot_min via the legacy (non-displacement) motion test
        # (motion_require_displacement: false) still needs real measured
        # displacement before being promoted, not just a Mahalanobis score.
        self.declare_parameter("mover_promote_min_disp_m", 0.5)
        self.declare_parameter("mover_promote_min_hits", 5)
        # Sticky demotion window: a promoted track reverts only after this
        # many seconds with no tick at p_motion >= 0.5 -- long enough that
        # a real mislabelled Carter parking mid-crossing doesn't instantly
        # drop back out of the risk stack/governor/corridor-yield (the
        # exact failure this feature exists to prevent), short enough that
        # a one-off false promotion (mislabelled clutter that briefly
        # jittered above the promote thresholds) doesn't stay promoted
        # for the rest of the run.
        self.declare_parameter("mover_demote_s", 60.0)
        # Published label for a promoted track -- risk_visualization.py's
        # LABEL_CATEGORIES maps this to "wheeled" and CLASS_BASE_RISK to
        # 0.65 (cart's value), so every category-gated consumer (stack
        # categories, governor, corridor users, extent floors) treats a
        # promoted track as an actionable mover with no changes of its
        # own -- they already key off label_category(), not a hardcoded
        # label list.
        self.declare_parameter("mover_promote_label", "moving object")

        gp = self.get_parameter
        self.map_frame = str(gp("map_frame").value)
        self.gate = float(gp("gate_distance_m").value)
        self.gate_speed = float(gp("gate_speed_mps").value)
        self.gate_max_m = float(gp("gate_max_m").value)
        self.association_key = str(gp("association_key").value).strip().lower()
        self.meas_time_corr = bool(gp("measurement_time_correction").value)
        self.max_meas_age = float(gp("max_measurement_age_sec").value)
        self._meas_ages = []
        if self.association_key not in ("label", "category"):
            raise ValueError("association_key must be 'label' or 'category'")
        self.association_groups = []
        for entry in gp("association_groups").value:
            group = frozenset(
                s.strip().lower() for s in str(entry).split(",") if s.strip())
            if group:
                self.association_groups.append(group)
        self.label_agnostic_labels = {
            str(x).strip().lower() for x in gp("label_agnostic_labels").value
        }
        self.upgrade_penalty_m = float(gp("upgrade_penalty_m").value)
        self.unknown_penalty_m = float(gp("unknown_penalty_m").value)
        self.association_mode = str(gp("association_mode").value).strip().lower()
        if self.association_mode not in ("legacy", "agnostic"):
            raise ValueError("association_mode must be 'legacy' or 'agnostic'")
        self.assoc_chi2_gate = float(gp("assoc_chi2_gate").value)
        self.assoc_size_weight = float(gp("assoc_size_weight").value)
        self.assoc_label_penalty = float(gp("assoc_label_penalty").value)
        self.label_vote_half_life_s = float(gp("label_vote_half_life_s").value)
        self.confirm_n = int(gp("confirm_n").value)
        self.confirm_m = int(gp("confirm_m").value)
        self.confirm_miss_grace_s = float(gp("confirm_miss_grace_s").value)
        self.graveyard_min_hits = int(gp("graveyard_min_hits").value)
        self.merge_chi2 = float(gp("merge_chi2").value)
        self.motion_speed_mps = float(gp("motion_speed_mps").value)
        self.motion_speed_score = float(gp("motion_speed_score").value)
        self.motion_window_s = float(gp("motion_window_s").value)
        self.motion_min_displacement_m = float(gp("motion_min_displacement_m").value)
        self.motion_disp_k_sigma = float(gp("motion_disp_k_sigma").value)
        self.motion_min_samples = int(gp("motion_min_samples").value)
        self.motion_straightness_min = float(gp("motion_straightness_min").value)
        self.motion_disp_method = str(gp("motion_disp_method").value).strip().lower()
        if self.motion_disp_method not in ("half_median", "endpoints"):
            raise ValueError("motion_disp_method must be 'half_median' or 'endpoints'")
        self.motion_hold_s = float(gp("motion_hold_s").value)
        self.motion_require_displacement = bool(gp("motion_require_displacement").value)
        self.motion_vel_damp_below = float(gp("motion_vel_damp_below").value)
        self.motion_vel_damp_tau_s = float(gp("motion_vel_damp_tau_s").value)
        self.velocity_source = str(gp("velocity_source").value).strip().lower()
        if self.velocity_source not in ("displacement", "kf"):
            raise ValueError("velocity_source must be 'displacement' or 'kf'")
        self.min_hits = int(gp("min_hits").value)
        self.min_conf = float(gp("min_confidence").value)
        self.process_noise = float(gp("process_noise").value)
        self.default_cov = float(gp("default_cov_m2").value)
        self.velocity_var_init = float(gp("velocity_var_init").value)
        self.marker_life = float(gp("marker_lifetime_sec").value)
        self.marker_text_height = float(gp("marker_text_height_m").value)
        text_color = [float(c) for c in gp("marker_text_color").value]
        self.marker_text_color = (text_color + [0.0, 0.0, 0.0])[:3]
        self.marker_show_beliefs = bool(gp("marker_show_beliefs").value)
        self.motion_threshold = float(gp("motion_threshold").value)
        self.p_motion_decay = float(gp("p_motion_decay").value)
        self.movable_gain = float(gp("movable_gain").value)
        self.movable_decay = float(gp("movable_decay").value)
        self.dyn_hl = float(gp("dynamic_half_life_s").value)
        self.stat_hl = float(gp("static_half_life_s").value)
        self.max_unseen_dynamic = float(gp("max_unseen_sec_dynamic").value)
        self.max_unseen_static = float(gp("max_unseen_sec_static").value)
        self.graveyard_ttl = float(gp("graveyard_ttl_sec").value)
        self.relation_rise = float(gp("relation_rise").value)
        self.relation_decay = float(gp("relation_decay").value)
        self.merge_enabled = bool(gp("merge_enabled").value)
        self.merge_distance_m = float(gp("merge_distance_m").value)
        self.merge_speed_diff_mps = float(gp("merge_speed_diff_mps").value)
        self.merge_confirm_ticks = int(gp("merge_confirm_ticks").value)
        self.jump_reset_factor = float(gp("jump_reset_factor").value)
        self.mover_promote_enabled = bool(gp("mover_promote_enabled").value)
        self.mover_promote_pmot_min = float(gp("mover_promote_pmot_min").value)
        self.mover_promote_min_s = float(gp("mover_promote_min_s").value)
        self.mover_promote_min_disp_m = float(gp("mover_promote_min_disp_m").value)
        self.mover_promote_min_hits = int(gp("mover_promote_min_hits").value)
        self.mover_demote_s = float(gp("mover_demote_s").value)
        self.mover_promote_label = str(gp("mover_promote_label").value)
        # Category -> speed cap lookup, mirroring category_priors below --
        # see _speed_cap_for. "default" is the fallback for any category
        # label_category can return that isn't one of the four explicit
        # ones (currently just "furniture").
        self.speed_caps = {
            "person": float(gp("max_speed_person_mps").value),
            "robot": float(gp("max_speed_robot_mps").value),
            "wheeled": float(gp("max_speed_wheeled_mps").value),
            "unknown": float(gp("max_speed_unknown_mps").value),
            "default": float(gp("max_speed_default_mps").value),
        }
        # (id_low, id_high) -> consecutive ticks the pair has met the merge
        # distance/speed test. Track ids are never reused (Track._next_id is
        # a monotonically increasing class counter), so a key always refers
        # to the same two Track objects for as long as it's in this dict.
        self._merge_streak = {}
        # Semantic prior: a fixed, class-conditioned seed for p_movable, not
        # updated online (that's the Behavioral prior's job -- see Track).
        # The label -> category mapping itself lives in
        # risk_visualization.label_category, shared with mask_relation's
        # person/operable-machine split so the two do not each keep their
        # own copy of "what kind of thing is a forklift".
        self.category_priors = {
            "person": float(gp("prior_person").value),
            "robot": float(gp("prior_robot").value),
            "wheeled": float(gp("prior_wheeled").value),
            "furniture": float(gp("prior_furniture").value),
        }
        self.prior_unknown = float(gp("prior_unknown").value)

        self.tracks: List[Track] = []
        self._graveyard: List[Track] = []
        self.last_predict: Optional[float] = None
        self.last_tick: Optional[float] = None

        self.lat_writer, self.lat_file = open_latency_csv(
            self, str(gp("debug_log_dir").value))

        self.log_unconfirmed = bool(gp("log_unconfirmed").value)
        self.log_writer, self.log_file, self.log_path = open_debug_csv(
            self, str(gp("motion_log_path").value), str(gp("debug_log_dir").value),
            "object_tracker",
            # kalman state ......... prior INPUTS ............ prior OUTPUTS ...... gates
            ["t", "tick_dt", "track_id", "label", "category",
             "x", "y", "vx", "vy", "speed", "Pxx", "Pyy", "Pvx", "Pvy",
             "motion_score", "motion_threshold", "moving",
             "p_motion", "p_movable", "p_movable_prior",
             "relconf_in", "relation_bonus",
             "confidence", "consequence",
             "hits", "misses", "age_s", "confirmed", "alive"])

        self.create_subscription(Detection3DArray, str(gp("input_topic").value),
                                 self._detections_cb, 10)
        self.obj_pub = self.create_publisher(
            Detection3DArray, str(gp("output_topic").value), 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, str(gp("marker_topic").value), 10)
        self.create_timer(1.0 / float(gp("update_rate").value), self._tick)

        if not HAVE_SCIPY:
            self.get_logger().warning("scipy missing -- greedy association.")
        self.get_logger().info("object_tracker STAGE1 ready (p_movable / p_motion)")

    def _prior_for(self, label):
        return self.category_priors.get(label_category(label), self.prior_unknown)

    def _speed_cap_for(self, label):
        return self.speed_caps.get(label_category(label), self.speed_caps["default"])

    def _apply_speed_cap(self, tr, pre_x, pre_y):
        """Run Track.enforce_speed_cap for `tr` right after a Kalman
        update() and, on a jump reset, log it -- see the module docstring's
        2026-09-10 finding and Track.enforce_speed_cap's own docstring.
        Called from every place update() runs on a live (non-graveyard-only)
        track: the labeled association pass, the label-agnostic
        (_associate_agnostic) pass, and _revive.
        """
        cap = self._speed_cap_for(tr.label)
        _, was_jump, pre_speed, displacement = tr.enforce_speed_cap(
            cap, self.jump_reset_factor, pre_x, pre_y)
        if was_jump:
            self.get_logger().warning(
                "object_tracker: speed jump reset on track #%d (%s): "
                "%.2f m/s pre-clamp (cap %.2f m/s), displacement %.2fm -- "
                "velocity/p_motion reset to 0"
                % (tr.id, tr.label, pre_speed, cap, displacement),
                throttle_duration_sec=2.0)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _measurement_age(self, msg, now):
        """Seconds between the detections' capture stamp and now (0 when the
        stamp is unset or correction is disabled), clamped."""
        if not self.meas_time_corr:
            return 0.0
        st = msg.header.stamp
        meas_t = float(st.sec) + float(st.nanosec) * 1e-9
        if meas_t <= 0.0:
            return 0.0
        age = max(0.0, min(self.max_meas_age, now - meas_t))
        self._meas_ages.append(age)
        if len(self._meas_ages) >= 50:
            self.get_logger().info(
                "measurement age (capture -> tracker): mean %.2fs max %.2fs over %d msgs"
                % (sum(self._meas_ages) / len(self._meas_ages), max(self._meas_ages),
                   len(self._meas_ages)), throttle_duration_sec=5.0)
            self._meas_ages = []
        return age

    def _detections_cb(self, msg):
        now = self._now()
        age = self._measurement_age(msg, now)
        dets = []
        for d in msg.detections:
            label, score, cov, relconf = "object", 0.0, self.default_cov, 0.0
            if d.results:
                h = d.results[0].hypothesis
                # gdino_detector may tag an object's class_id as
                # "label|relconf=0.82" (relation prior, proposed) -- the
                # label used for the association gate below must be the bare
                # label only, or a track's relation state flipping on/off
                # between frames looks like a label change and breaks
                # re-association. See risk_visualization.parse_class_id for
                # the same convention.
                parts = str(h.class_id).split("|")
                label = parts[0]
                score = float(h.score)
                for part in parts[1:]:
                    if part.startswith("relconf="):
                        try:
                            relconf = float(part.split("=", 1)[1])
                        except ValueError:
                            pass
                c = d.results[0].pose.covariance
                cov = float(c[0]) if c[0] > 1e-9 else self.default_cov
            dets.append({"x": float(d.bbox.center.position.x),
                         "y": float(d.bbox.center.position.y),
                         "label": label, "score": score, "cov": cov,
                         "relconf": relconf,
                         "size": max(float(d.bbox.size.x), float(d.bbox.size.y))})
        if not dets:
            return
        self._predict_to(now)

        # Split BEFORE any association: label-agnostic detections (lidar
        # clusters) never enter the labeled pass below, so a "mobile robot"
        # camera track's label/category can never be touched by them -- see
        # label_agnostic_labels' declare_parameter comment.
        labeled_dets = [d for d in dets if d["label"].lower() not in self.label_agnostic_labels]
        agnostic_dets = [d for d in dets if d["label"].lower() in self.label_agnostic_labels]

        # Old (pre-this-message) track count -- misses are only ever counted
        # against tracks that existed before this callback ran, never
        # against a track spawned by it (see the loop at the bottom). Track
        # indices below old_track_count stay stable across every _spawn*()
        # call in this method since spawning only ever appends.
        old_track_count = len(self.tracks)
        matched_t = set()

        mode = getattr(self, "association_mode", "legacy")
        if self.tracks:
            BIG = 1e6
            cost = np.full((len(self.tracks), len(labeled_dets)), BIG)
            for i, tr in enumerate(self.tracks):
                tx, ty = tr.position
                vx, vy = tr.velocity
                # Compare against where the track was at CAPTURE time.
                tx -= vx * age
                ty -= vy * age
                gate = self._gate_for(tr, now)
                # A lidar-spawned track (label in label_agnostic_labels,
                # category "unknown") is matchable by a labeled detection of
                # ANY category -- see upgrade_penalty_m's declare_parameter
                # comment -- so it stops coexisting with a second, labeled
                # track spawned for the same physical object. Same-category
                # matches are still preferred: an "upgrade" candidate's cost
                # carries the extra upgrade_penalty_m.
                track_is_agnostic = tr.label.lower() in self.label_agnostic_labels
                if mode == "agnostic":
                    # WP-B: no hard _same_object_class gate at all -- every
                    # track is a candidate for every detection, gated only
                    # by _assoc_cost_agnostic's own Mahalanobis+metric test;
                    # category only affects the COST (soft), never whether a
                    # pair is even considered. Folding upgrade_penalty_m in
                    # here preserves the old bias toward a genuine
                    # same-category match over "claiming" an agnostic track.
                    bias = self.upgrade_penalty_m if track_is_agnostic else 0.0
                    for j, d in enumerate(labeled_dets):
                        c = self._assoc_cost_agnostic(tr, d, age, now, extra_label_bias=bias)
                        if c is not None:
                            cost[i, j] = c
                else:
                    for j, d in enumerate(labeled_dets):
                        same_class = self._same_object_class(tr.label, d["label"])
                        upgrade = track_is_agnostic and not same_class
                        if not same_class and not upgrade:
                            continue
                        dist = np.hypot(tx - d["x"], ty - d["y"])
                        if dist <= gate:
                            cost[i, j] = dist + (self.upgrade_penalty_m if upgrade else 0.0)
            matched_d = set()
            for i, j in self._assign(cost, BIG):
                tr = self.tracks[i]
                vx, vy = tr.velocity
                was_agnostic = tr.label.lower() in self.label_agnostic_labels
                pre_x, pre_y = tr.position
                # Shift the stale measurement forward to "now" along the
                # track's own velocity before the update (see
                # measurement_time_correction).
                tr.update(labeled_dets[j]["x"] + vx * age, labeled_dets[j]["y"] + vy * age,
                          labeled_dets[j]["cov"], labeled_dets[j]["score"],
                          labeled_dets[j]["size"], now, relconf=labeled_dets[j]["relconf"],
                          motion_window_s=getattr(self, "motion_window_s", 2.0))
                if mode == "agnostic":
                    self._vote_label_and_reseed(tr, labeled_dets[j], now, was_agnostic)
                elif was_agnostic:
                    self._upgrade_track(tr, labeled_dets[j])
                else:
                    self._adopt_label(tr, labeled_dets[j])
                # Cap AFTER any label change above so an upgrade's new
                # category (e.g. unknown -> robot) picks the right cap.
                # hasattr guard: test_object_tracker_lidar.py /
                # test_object_tracker_merge.py bind _detections_cb onto a
                # minimal stub that predates this feature and never sets
                # speed_caps -- skip cleanly there rather than
                # AttributeError; a real ObjectTrackerNode always has it
                # (set in __init__).
                if hasattr(self, "speed_caps"):
                    self._apply_speed_cap(tr, pre_x, pre_y)
                matched_t.add(i)
                matched_d.add(j)
        else:
            matched_d = set()

        for j, d in enumerate(labeled_dets):
            if j not in matched_d:
                self._spawn(d, now)

        if agnostic_dets:
            self._associate_agnostic(agnostic_dets, matched_t, now, age)

        # WP-B miss-charging fix (2026-09-11): a miss used to be charged
        # here on EVERY unmatched detections message from ANY source (3
        # overhead cameras + 10 Hz lidar all share this callback) -- for a
        # track seen by only one slow camera that meant a miss on almost
        # every other source's message too, starving agnostic mode's N-of-M
        # confirmation window regardless of how well the track was actually
        # being tracked (see the "Status 2026-09-11 13:00" WP-B regression
        # and Track.mark_missed's docstring). Under agnostic mode misses are
        # now charged from _tick instead, gated by confirm_miss_grace_s and
        # decoupled from message arrival rate entirely -- do nothing here.
        # Legacy mode is intentionally UNCHANGED: it never reads `recent`
        # for confirmation, so charging a miss per unmatched message keeps
        # its exact pre-WP-B behaviour.
        if mode == "legacy":
            for i in range(old_track_count):
                if i not in matched_t:
                    self.tracks[i].mark_missed()

    def _associate_agnostic(self, agnostic_dets, matched_t, now, age):
        """Second association pass for label-agnostic (e.g. lidar) detections.

        Nearest existing LIVE track of ANY category within `_gate_for`,
        greedy nearest-neighbour (see `_greedy_assign` -- deliberately not
        the labeled pass's Hungarian-optimal `_assign`, since this pass must
        not reshuffle an otherwise-good labeled association just to shave a
        few centimetres off a lidar match), at most one lidar measurement
        per track per message. A matched track gets a normal sequential KF
        update (same measurement-time correction as the labeled pass) but
        NEVER `_adopt_label` -- identity/class stays whatever the labeled
        sources decided. Unmatched detections spawn a fresh (never revived)
        track via `_spawn_agnostic`.
        """
        matched_d = set()
        mode = getattr(self, "association_mode", "legacy")
        if self.tracks:
            BIG = 1e6
            cost = np.full((len(self.tracks), len(agnostic_dets)), BIG)
            for i, tr in enumerate(self.tracks):
                tx, ty = tr.position
                vx, vy = tr.velocity
                tx -= vx * age
                ty -= vy * age
                gate = self._gate_for(tr, now)
                # When a lidar cluster's gate contains both a labeled track
                # and an "unknown"-category one (typically the leftover
                # lidar-spawned track an upgrade in the labeled pass hasn't
                # consumed yet), prefer the labeled track -- see
                # unknown_penalty_m's declare_parameter comment -- so the
                # duplicate unknown track stops being fed and ages out.
                track_is_unknown = label_category(tr.label) == "unknown"
                penalty = self.unknown_penalty_m if track_is_unknown else 0.0
                if mode == "agnostic":
                    for j, d in enumerate(agnostic_dets):
                        c = self._assoc_cost_agnostic(tr, d, age, now,
                                                       extra_label_bias=penalty)
                        if c is not None:
                            cost[i, j] = c
                else:
                    for j, d in enumerate(agnostic_dets):
                        dist = np.hypot(tx - d["x"], ty - d["y"])
                        if dist <= gate:
                            cost[i, j] = dist + penalty
            for i, j in self._greedy_assign(cost, BIG):
                tr = self.tracks[i]
                vx, vy = tr.velocity
                pre_x, pre_y = tr.position
                tr.update(agnostic_dets[j]["x"] + vx * age, agnostic_dets[j]["y"] + vy * age,
                          agnostic_dets[j]["cov"], agnostic_dets[j]["score"],
                          agnostic_dets[j]["size"], now, relconf=agnostic_dets[j]["relconf"],
                          motion_window_s=getattr(self, "motion_window_s", 2.0))
                # See the labeled pass' hasattr guard comment above.
                if hasattr(self, "speed_caps"):
                    self._apply_speed_cap(tr, pre_x, pre_y)
                matched_t.add(i)
                matched_d.add(j)

        for j, d in enumerate(agnostic_dets):
            if j not in matched_d:
                self._spawn_agnostic(d, now)

    def _assign(self, cost, big):
        if HAVE_SCIPY:
            r, c = linear_sum_assignment(cost)
            return [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] < big]
        pairs, work = [], cost.copy()
        # A label-agnostic labels split can leave the labeled pass with a
        # non-empty track list but zero labeled detections this message --
        # np.argmin on an empty array raises, so bail out first.
        if work.size == 0:
            return pairs
        while True:
            i, j = np.unravel_index(np.argmin(work), work.shape)
            if work[i, j] >= big:
                break
            pairs.append((int(i), int(j)))
            work[i, :] = big
            work[:, j] = big
        return pairs

    @staticmethod
    def _greedy_assign(cost, big):
        """Greedy nearest-neighbour assignment: repeatedly take the globally
        smallest remaining cost cell, then remove its row and column, until
        nothing left is under `big`. Used for the label-agnostic (lidar)
        association pass -- see _associate_agnostic's docstring for why
        greedy rather than the labeled pass's Hungarian-optimal _assign()."""
        pairs, work = [], cost.copy()
        if work.size == 0:
            return pairs
        while True:
            i, j = np.unravel_index(np.argmin(work), work.shape)
            if work[i, j] >= big:
                break
            pairs.append((int(i), int(j)))
            work[i, :] = big
            work[:, j] = big
        return pairs

    def _same_object_class(self, track_label, det_label):
        """Association gate on identity: exact label, or (association_key ==
        "category") the same non-unknown risk category OR two categories
        that share an association_groups entry (e.g. "robot"/"wheeled" --
        GroundingDINO's "ground mobile robot" vs. "cart" phrase churn on the
        same physical Carter)."""
        a, b = track_label.lower(), det_label.lower()
        if a == b:
            return True
        if self.association_key != "category":
            return False
        ca, cb = label_category(a), label_category(b)
        if ca == "unknown" or cb == "unknown":
            return False
        return ca == cb or self._in_same_group(ca, cb)

    def _in_same_group(self, category_a, category_b):
        """True if category_a/category_b (already non-unknown) co-occur in
        any association_groups entry."""
        for group in self.association_groups:
            if category_a in group and category_b in group:
                return True
        return False

    def _gate_for(self, tr, now):
        """Distance gate for this track: widens with time since last update
        so an object that moved between sparse sightings still associates,
        capped at gate_max_m (WP1, 2026-09-10) so a long-coasting track
        cannot re-associate tens of metres away and read the jump as real
        velocity -- see enforce_speed_cap's "speed jump reset" and this
        param's own declare_parameter comment. _revive uses this same
        method, so the cap applies to graveyard revival too."""
        return min(self.gate_max_m,
                   self.gate + self.gate_speed * max(0.0, now - tr.last_seen))

    @staticmethod
    def _adopt_label(tr, d):
        """Category-level association can pair a "cart" track with a "mobile
        robot" detection; keep the label whose detection is currently more
        confident so consequence weighting downstream follows the evidence."""
        if d["label"].lower() != tr.label.lower() and d["score"] >= tr.confidence:
            tr.label = d["label"]

    def _upgrade_track(self, tr, d):
        """Upgrade a lidar-spawned "unknown" track to a labeled category the
        first time a labeled detection claims it (see upgrade_penalty_m's
        declare_parameter comment). Unlike _adopt_label this is
        unconditional -- any claiming detection wins, there is no
        confidence gate, since the track had no real category before.

        id/position/velocity/covariance/p_motion are untouched here -- the
        caller's tr.update() just ran the Kalman measurement update before
        calling this. p_movable is re-seeded to the NEW category's prior
        only if it is still sitting exactly at the unknown prior it was
        spawned with; if motion evidence already pushed it up (or down),
        that evidence is kept rather than being overwritten by a fresh
        prior. p_movable_prior always moves to the new category's prior so
        future belief decay pulls toward the right target.
        """
        prior = self._prior_for(d["label"])
        if abs(tr.p_movable - tr.p_movable_prior) < 1e-9:
            tr.p_movable = prior
        tr.p_movable_prior = prior
        tr.label = d["label"]

    def _spawn(self, d, now):
        if getattr(self, "association_mode", "legacy") == "agnostic":
            revived = self._revive_agnostic(d, now)
        else:
            revived = self._revive(d, now)
        self.tracks.append(revived if revived is not None else Track(
            d["x"], d["y"], d["label"], d["score"], d["cov"], d["size"], now,
            self._prior_for(d["label"]), relconf=d["relconf"],
            velocity_var_init=getattr(self, "velocity_var_init", 0.25),
            confirm_m=getattr(self, "confirm_m", 5),
            label_vote_half_life_s=getattr(self, "label_vote_half_life_s", 30.0)))

    def _spawn_agnostic(self, d, now):
        """Spawn a fresh track for an unmatched label-agnostic detection.

        Unlike _spawn(), this NEVER checks the graveyard (in EITHER
        association_mode) -- a label-agnostic source (lidar) has no class
        identity to match a graveyard entry by, and reviving a
        camera-labeled track's old id from a bare lidar return would
        silently hand it a category it was never actually re-observed as
        -- true even under WP-B's label-blind _revive_agnostic, since that
        still needs a labeled detection to eventually re-claim/re-vote the
        revived track's category; a lidar-only revival would leave it
        permanently unclaimed. category resolves to "unknown" via
        risk_visualization.label_category (no "lidar_cluster" entry
        there), so _prior_for() seeds p_movable from prior_unknown.
        """
        self.tracks.append(Track(
            d["x"], d["y"], d["label"], d["score"], d["cov"], d["size"], now,
            self._prior_for(d["label"]), relconf=d["relconf"],
            velocity_var_init=getattr(self, "velocity_var_init", 0.25),
            confirm_m=getattr(self, "confirm_m", 5),
            label_vote_half_life_s=getattr(self, "label_vote_half_life_s", 30.0)))

    def _revive(self, d, now):
        """Same-label graveyard track within the gate distance -> update() it
        back to life instead of spawning fresh, so its id (and accumulated
        p_movable/p_motion/confidence) survives the perception gap."""
        best_i, best_dist = None, None
        for i, tr in enumerate(self._graveyard):
            if not self._same_object_class(tr.label, d["label"]):
                continue
            tx, ty = tr.position
            dist = np.hypot(tx - d["x"], ty - d["y"])
            if dist <= self._gate_for(tr, now) and (best_dist is None or dist < best_dist):
                best_i, best_dist = i, dist
        if best_i is None:
            return None
        tr = self._graveyard.pop(best_i)
        pre_x, pre_y = tr.position
        tr.update(d["x"], d["y"], d["cov"], d["score"], d["size"], now,
                  relconf=d["relconf"],
                  motion_window_s=getattr(self, "motion_window_s", 2.0))
        # See _detections_cb's hasattr guard comment above.
        if hasattr(self, "speed_caps"):
            self._apply_speed_cap(tr, pre_x, pre_y)
        return tr

    def _revive_agnostic(self, d, now):
        """WP-B (2026-09-11) label-blind graveyard revival: unlike _revive,
        this checks EVERY graveyard track regardless of category (a mover
        is a mover -- the whole point of association_mode: agnostic) and
        gates purely on motion plausibility. Position is extrapolated
        forward by the track's own last velocity, clamped to its category
        speed cap (so a long-dead track's stale/jittery velocity can't
        extrapolate it implausibly far); covariance for the Mahalanobis
        test is `P + Q*dt` on a COPY (Track.predicted_covariance) -- a
        graveyard track never ran predict() while dead, so its raw P
        understates the uncertainty accumulated over the gap. Best
        (lowest-d2) match within both the chi-square gate and the existing
        metric _gate_for cap wins; ties are not expected (d2 is
        continuous) so no tie-break beyond "first seen at the minimum"."""
        best_i, best_d2 = None, None
        chi2_gate = getattr(self, "assoc_chi2_gate", 9.21)
        process_noise = getattr(self, "process_noise", 0.05)
        for i, tr in enumerate(self._graveyard):
            dt = max(0.0, now - tr.last_seen)
            vx, vy = tr.velocity
            speed = math.hypot(vx, vy)
            cap = self._speed_cap_for(tr.label)
            if speed > cap and speed > 1e-9:
                scale = cap / speed
                vx, vy = vx * scale, vy * scale
            px = tr.x[0] + vx * dt
            py = tr.x[1] + vy * dt
            P_pred = tr.predicted_covariance(dt, process_noise)
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
            S = H @ P_pred @ H.T + np.diag([d["cov"], d["cov"]])
            r = np.array([d["x"] - px, d["y"] - py])
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.eye(2) / max(d["cov"], 1e-6)
            d2 = float(r @ Sinv @ r)
            r_norm = float(np.hypot(r[0], r[1]))
            gate = self._gate_for(tr, now)
            if d2 <= chi2_gate and r_norm <= gate:
                if best_d2 is None or d2 < best_d2:
                    best_i, best_d2 = i, d2
        if best_i is None:
            return None
        tr = self._graveyard.pop(best_i)
        pre_x, pre_y = tr.position
        tr.update(d["x"], d["y"], d["cov"], d["score"], d["size"], now,
                  relconf=d["relconf"],
                  motion_window_s=getattr(self, "motion_window_s", 2.0))
        if hasattr(self, "speed_caps"):
            self._apply_speed_cap(tr, pre_x, pre_y)
        return tr

    def _assoc_cost_agnostic(self, tr, d, age, now, extra_label_bias=0.0):
        """WP-B unified association cost, used by BOTH the labeled pass
        (_detections_cb) and the label-agnostic/lidar pass
        (_associate_agnostic) under association_mode == "agnostic":

            d2 + assoc_size_weight * |size_a - size_b| + label_penalty

        where d2/‖r‖ come from Track.position_mahalanobis_d2 (the
        detection shifted forward by track-velocity * age, same
        measurement-time-correction convention as the legacy cost's
        `tx -= vx * age` -- just applied to the detection instead of the
        track so Track's own H/R-based helper can be reused unchanged),
        and label_penalty is `_category_mismatch_penalty` (a SOFT nudge,
        not a hard gate: category never blocks a candidate here, only
        costs more) plus `extra_label_bias` -- the caller-supplied fold-in
        of upgrade_penalty_m (labeled pass, claiming an agnostic/lidar-
        spawned track) or unknown_penalty_m (lidar pass, preferring a
        labeled track over an unknown duplicate), preserving the old
        biases' intent under the new cost shape.

        Returns None if the pair fails EITHER gate (d2 <= assoc_chi2_gate
        AND ‖r‖ <= _gate_for(tr, now)), else the scalar cost."""
        vx, vy = tr.velocity
        d2, r_norm = tr.position_mahalanobis_d2(d["x"] + vx * age, d["y"] + vy * age, d["cov"])
        gate = self._gate_for(tr, now)
        chi2_gate = getattr(self, "assoc_chi2_gate", 9.21)
        if d2 > chi2_gate or r_norm > gate:
            return None
        size_w = getattr(self, "assoc_size_weight", 2.0)
        size_term = size_w * abs(float(tr.size) - float(d["size"]))
        label_term = self._category_mismatch_penalty(tr, d["label"]) + extra_label_bias
        return d2 + size_term + label_term

    def _category_mismatch_penalty(self, tr, det_label):
        """assoc_label_penalty if the track's and detection's categories
        mismatch, NEITHER is "unknown", and they don't share an
        association_groups entry -- else 0.0. See association_mode's
        declare_parameter comment: this is a soft cost term, never a hard
        association gate, under WP-B."""
        cat_a, cat_b = label_category(tr.label), label_category(det_label)
        if cat_a == "unknown" or cat_b == "unknown":
            return 0.0
        if cat_a == cat_b or self._in_same_group(cat_a, cat_b):
            return 0.0
        return getattr(self, "assoc_label_penalty", 2.0)

    def _vote_label_and_reseed(self, tr, d, now, was_agnostic):
        """WP-B labeled-pass post-match step (association_mode ==
        "agnostic"), replacing _adopt_label/_upgrade_track: register this
        detection's (raw label, score, category) as a vote
        (Track.register_label_vote -- see its docstring for why this
        stabilises against GroundingDINO phrase churn better than
        latest/most-confident-wins), then, if the track was label-agnostic
        (lidar-spawned "unknown") BEFORE this update, re-seed its
        p_movable prior toward the (now voted) label's category -- same
        p_movable-reseed rule as the legacy _upgrade_track, just decoupled
        from the unconditional tr.label assignment that method also did."""
        half_life = getattr(self, "label_vote_half_life_s", 30.0)
        tr.register_label_vote(d["label"], d["score"], label_category(d["label"]), now, half_life)
        if was_agnostic:
            self._reseed_prior_if_still_default(tr, tr.label)

    def _reseed_prior_if_still_default(self, tr, new_label):
        """See _upgrade_track's docstring -- same rule, factored out so
        both the legacy upgrade path and WP-B's vote-based one share it."""
        prior = self._prior_for(new_label)
        if abs(tr.p_movable - tr.p_movable_prior) < 1e-9:
            tr.p_movable = prior
        tr.p_movable_prior = prior

    def _predict_to(self, now):
        if self.last_predict is None:
            self.last_predict = now
            return
        dt = now - self.last_predict
        if dt <= 0:
            return
        for tr in self.tracks:
            tr.predict(dt, self.process_noise)
        self.last_predict = now

    def _mergeable_categories(self, label_a, label_b):
        """Category test for the duplicate-track merge: equal categories,
        an association_groups match, or either side "unknown" (covers both
        a genuinely unclassified object and a label-agnostic lidar-spawned
        track, which risk_visualization.label_category also resolves to
        "unknown") -- an unknown-category track carries no label evidence
        against the merge, so it must not block one."""
        ca, cb = label_category(label_a), label_category(label_b)
        if ca == "unknown" or cb == "unknown":
            return True
        return ca == cb or self._in_same_group(ca, cb)

    @staticmethod
    def _pair_key(tr_a, tr_b):
        return tuple(sorted((tr_a.id, tr_b.id)))

    @staticmethod
    def _fold_track(older, younger):
        """Merge `younger` into `older` for the duplicate-track merge: older
        keeps its id and its Kalman state (position/velocity/covariance,
        last_seen) untouched -- only the belief/label fields that should
        reflect the BETTER of the two observations are combined. Label
        adoption mirrors _adopt_label's confidence-wins-ties rule."""
        if younger.confidence >= older.confidence:
            older.label = younger.label
        older.confidence = max(older.confidence, younger.confidence)
        older.p_motion = max(older.p_motion, younger.p_motion)
        older.p_movable = max(older.p_movable, younger.p_movable)

    def _merge_duplicates(self):
        """Dispatch to the legacy or WP-B agnostic merge pass -- see
        _merge_duplicates_legacy/_merge_duplicates_agnostic. getattr
        guard: pre-WP-B test stubs (test_object_tracker_merge.py) bind
        this method directly onto a minimal stub that never sets
        association_mode -- defaults to "legacy", the exact pre-WP-B
        behaviour those tests exercise."""
        if not self.merge_enabled or len(self.tracks) < 2:
            return
        if getattr(self, "association_mode", "legacy") == "agnostic":
            self._merge_duplicates_agnostic()
        else:
            self._merge_duplicates_legacy()

    def _merge_duplicates_legacy(self):
        """Fold duplicate tracks of the same physical object into one.

        Runs once per tick over live, CONFIRMED tracks only (an unconfirmed
        track is <min_hits observations -- not enough evidence to fold
        anything into, or to fold away). A pair is a merge candidate once
        it is category-compatible (_mergeable_categories) and its centres
        are within merge_distance_m with a velocity difference within
        merge_speed_diff_mps; the streak has to hold for merge_confirm_ticks
        CONSECUTIVE ticks (any miss resets it) before the fold actually
        happens, so a single close pass of two genuinely different objects
        can't merge them. The younger track (larger id) is dropped outright
        -- not sent to the graveyard, since it was never a distinct object
        to revive later.
        """
        live = [t for t in self.tracks if t.confirmed]
        to_drop = set()
        seen_pairs = set()
        for a in range(len(live)):
            tr_a = live[a]
            if tr_a.id in to_drop:
                continue
            for b in range(a + 1, len(live)):
                tr_b = live[b]
                if tr_b.id in to_drop:
                    continue
                key = self._pair_key(tr_a, tr_b)
                seen_pairs.add(key)
                if not self._mergeable_categories(tr_a.label, tr_b.label):
                    self._merge_streak.pop(key, None)
                    continue
                ax, ay = tr_a.position
                bx, by = tr_b.position
                dist = math.hypot(ax - bx, ay - by)
                avx, avy = tr_a.velocity
                bvx, bvy = tr_b.velocity
                speed_diff = math.hypot(avx - bvx, avy - bvy)
                if dist > self.merge_distance_m or speed_diff > self.merge_speed_diff_mps:
                    self._merge_streak.pop(key, None)
                    continue
                streak = self._merge_streak.get(key, 0) + 1
                self._merge_streak[key] = streak
                if streak < self.merge_confirm_ticks:
                    continue
                older, younger = (tr_a, tr_b) if tr_a.id < tr_b.id else (tr_b, tr_a)
                self._fold_track(older, younger)
                to_drop.add(younger.id)
                self._merge_streak.pop(key, None)
                self.get_logger().info(
                    "object_tracker: merged duplicate track #%d (%s) into "
                    "#%d (%s), dist=%.2fm speed_diff=%.2fm/s"
                    % (younger.id, younger.label, older.id, older.label,
                       dist, speed_diff))
        # Drop stale streaks for pairs that no longer coexist (one already
        # merged/pruned elsewhere this tick) so _merge_streak can't grow
        # unbounded across a long run.
        stale = [k for k in self._merge_streak if k not in seen_pairs]
        for k in stale:
            del self._merge_streak[k]
        if to_drop:
            self.tracks = [t for t in self.tracks if t.id not in to_drop]

    @staticmethod
    def _pair_mahalanobis_d2(tr_a, tr_b):
        """Position Mahalanobis distance between two LIVE tracks' own
        Kalman estimates: r = pos_a - pos_b, S = P_a[:2,:2] + P_b[:2,:2]
        (their position covariances summed, same idea as combining two
        independent Gaussian position estimates). Used by
        _merge_duplicates_agnostic's pair test. Returns (d2, ‖r‖)."""
        r = np.array([tr_a.x[0] - tr_b.x[0], tr_a.x[1] - tr_b.x[1]])
        S = tr_a.P[0:2, 0:2] + tr_b.P[0:2, 0:2]
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.eye(2) / 1e-6
        d2 = float(r @ Sinv @ r)
        r_norm = float(np.hypot(r[0], r[1]))
        return d2, r_norm

    @staticmethod
    def _fold_track_agnostic(older, younger):
        """WP-B version of _fold_track: same confidence/p_motion/p_movable
        combination, plus (a) label_votes summed (Counter + Counter, so
        the surviving track's category vote reflects BOTH tracks' evidence
        rather than just the older one's) with `younger`'s last-raw-label
        map merged in (ties favour `younger` only where `older` had no
        entry for that category yet -- update() applies `younger`'s dict
        on top), and (b) `meas` (raw measurement history) extended with
        `younger`'s and re-sorted by stamp, so displacement evidence after
        the fold sees the full combined history rather than restarting
        from just `older`'s own window. After combining, `older.label` is
        re-derived from the COMBINED vote (not just left at whichever of
        the two happened to win before the fold)."""
        if younger.confidence >= older.confidence:
            older.last_raw_label = younger.last_raw_label
        older.confidence = max(older.confidence, younger.confidence)
        older.p_motion = max(older.p_motion, younger.p_motion)
        older.p_movable = max(older.p_movable, younger.p_movable)
        older.label_votes.update(younger.label_votes)
        older._label_last_raw.update(younger._label_last_raw)
        combined = sorted(list(older.meas) + list(younger.meas), key=lambda m: m[0])
        older.meas = deque(combined)
        if older.label_votes:
            winner = max(older.label_votes.items(), key=lambda kv: kv[1])[0]
            older.label = older._label_last_raw.get(winner, older.label)

    def _merge_duplicates_agnostic(self):
        """WP-B duplicate-track merge: label-blind (no _mergeable_categories
        gate -- a mover is a mover, same reasoning as the association cost
        above) and looser than legacy in two ways: (1) the pair test is
        Mahalanobis d2 <= merge_chi2 OR euclidean distance <= merge_distance_m
        (an OR, not legacy's AND-style distance-and-speed test), and (2) at
        most one of the pair may be UNCONFIRMED (legacy requires both
        confirmed) -- a genuinely duplicate track pair where one side just
        hasn't accumulated enough hits yet should still fold rather than
        wait. Still requires merge_confirm_ticks CONSECUTIVE ticks before
        folding, same anti-chatter reasoning as legacy."""
        live = self.tracks
        to_drop = set()
        seen_pairs = set()
        chi2 = getattr(self, "merge_chi2", 4.0)
        for a in range(len(live)):
            tr_a = live[a]
            if tr_a.id in to_drop:
                continue
            for b in range(a + 1, len(live)):
                tr_b = live[b]
                if tr_b.id in to_drop:
                    continue
                if not tr_a.confirmed and not tr_b.confirmed:
                    continue
                key = self._pair_key(tr_a, tr_b)
                seen_pairs.add(key)
                ax, ay = tr_a.position
                bx, by = tr_b.position
                dist = math.hypot(ax - bx, ay - by)
                d2, _r = self._pair_mahalanobis_d2(tr_a, tr_b)
                if d2 > chi2 and dist > self.merge_distance_m:
                    self._merge_streak.pop(key, None)
                    continue
                streak = self._merge_streak.get(key, 0) + 1
                self._merge_streak[key] = streak
                if streak < self.merge_confirm_ticks:
                    continue
                older, younger = (tr_a, tr_b) if tr_a.id < tr_b.id else (tr_b, tr_a)
                self._fold_track_agnostic(older, younger)
                to_drop.add(younger.id)
                self._merge_streak.pop(key, None)
                self.get_logger().info(
                    "object_tracker: merged duplicate track #%d (%s) into "
                    "#%d (%s), dist=%.2fm d2=%.2f"
                    % (younger.id, younger.label, older.id, older.label, dist, d2))
        stale = [k for k in self._merge_streak if k not in seen_pairs]
        for k in stale:
            del self._merge_streak[k]
        if to_drop:
            self.tracks = [t for t in self.tracks if t.id not in to_drop]

    def _charge_stale_misses(self, now):
        """WP-B miss-charging fix (2026-09-11, "Status 2026-09-11 13:00" /
        "Remaining work" #1 in the plan). Charges Track.mark_missed() here,
        from _tick, instead of _detections_cb -- and only once per
        confirm_miss_grace_s seconds of CONTINUOUS silence per track (see
        Track._last_miss_charge_t) -- completely decoupled from how many
        detection messages of any kind (3 overhead cameras + 10 Hz lidar,
        all sharing one callback) arrive while a track goes unmatched.

        Before this fix, _detections_cb charged a miss to every unmatched
        pre-existing track on EVERY incoming message from ANY source: a
        parked Carter seen by only one 0.5 Hz camera collected a miss on
        nearly every other source's message too (~5 misses per real hit),
        so it almost never held enough hits in Track.recent's window to
        confirm (unconfirmed tracks are never published -- see
        _publish_objects's caller). That's what actually sank carter2's
        tracked-% under agnostic mode (57% legacy -> 21% agnostic) on
        unit2_pan_1, not the confirm_n/confirm_m ratio itself.

        A track whose gap since last_seen stays <= confirm_miss_grace_s
        costs nothing here; one that's been silent longer gets charged at
        most once per grace interval of continued silence, regardless of
        _tick's own rate (10 Hz) or of unrelated messages arriving in the
        meantime. Agnostic mode only -- legacy mode never reads `recent`
        for confirmation and keeps its old per-message mark_missed() call
        in _detections_cb unchanged (see that method's `mode == "legacy"`
        guard)."""
        if self.association_mode != "agnostic":
            return
        for tr in self.tracks:
            gap = now - tr.last_seen
            if gap <= self.confirm_miss_grace_s:
                continue
            last_charge = (tr._last_miss_charge_t
                           if tr._last_miss_charge_t is not None
                           else tr.last_seen)
            if (now - last_charge) >= self.confirm_miss_grace_s:
                tr.mark_missed()
                tr._last_miss_charge_t = now

    def _tick(self):
        _t0 = time.perf_counter()
        now = self._now()
        self._predict_to(now)

        # Measured tick interval, not a hardcoded 1/10 -- that hardcode used
        # to silently desync from update_rate (misfeeding p_motion_decay /
        # movable_decay) whenever update_rate was changed. Clamp away zero/
        # negative deltas (clock quirks) and cap the top end so a stalled
        # process (e.g. paused under a debugger) doesn't apply one enormous
        # decay/belief step on resume.
        if self.last_tick is None:
            dt = 1.0 / max(1e-3, float(self.get_parameter("update_rate").value))
        else:
            dt = min(1.0, max(1e-3, now - self.last_tick))
        self.last_tick = now

        self._charge_stale_misses(now)

        for tr in self.tracks:
            tr.update_beliefs(self.motion_threshold, self.p_motion_decay,
                              self.movable_gain, self.movable_decay, dt, now,
                              speed_mps=self.motion_speed_mps,
                              speed_score=self.motion_speed_score,
                              motion_window_s=self.motion_window_s,
                              motion_min_displacement_m=self.motion_min_displacement_m,
                              motion_disp_k_sigma=self.motion_disp_k_sigma,
                              motion_min_samples=self.motion_min_samples,
                              motion_straightness_min=self.motion_straightness_min,
                              motion_disp_method=self.motion_disp_method,
                              motion_hold_s=self.motion_hold_s,
                              require_displacement=self.motion_require_displacement,
                              motion_vel_damp_below=self.motion_vel_damp_below,
                              motion_vel_damp_tau_s=self.motion_vel_damp_tau_s)
            tr.update_promotion(now, self.mover_promote_enabled,
                                self.mover_promote_pmot_min, self.mover_promote_min_s,
                                self.mover_promote_min_disp_m, self.mover_promote_min_hits,
                                self.mover_demote_s)
            tr.update_relation(self.relation_rise, self.relation_decay, dt)
            half_life = self.dyn_hl if tr.p_movable > 0.5 else self.stat_hl
            tr.decay_confidence(dt, half_life)
            if not tr.confirmed:
                # WP-B N-of-M confirmation (agnostic mode): confirmed once
                # at least confirm_n of the last confirm_m association
                # outcomes were hits (Track.recent) -- looser than legacy's
                # "hits >= min_hits ever" (see confirm_n/confirm_m's
                # declare_parameter comment).
                if self.association_mode == "agnostic":
                    if sum(tr.recent) >= self.confirm_n:
                        tr.confirmed = True
                elif tr.hits >= self.min_hits:
                    tr.confirmed = True

        # Duplicate-track merge (see _merge_duplicates' docstring) runs
        # after this tick's belief update / confirmation, before the
        # alive/graveyard prune below -- a track folded away here must never
        # reach the graveyard as a second, separately-revivable identity.
        self._merge_duplicates()

        # Log BEFORE eviction so the rows min_hits / min_confidence throw
        # away are visible too (log_unconfirmed) -- ported from
        # user-a/sandbox, which placed this call for the same reason.
        self._log(now, dt, self.tracks)

        alive = []
        for t in self.tracks:
            timeout = (self.max_unseen_dynamic if t.p_movable > 0.5
                       else self.max_unseen_static)
            if t.confidence >= self.min_conf and (now - t.last_seen) <= timeout:
                alive.append(t)
            elif t.confirmed:
                # unconfirmed tracks are <min_hits observations -- exactly the
                # noise floor _revive() would otherwise start reviving.
                self._graveyard.append(t)
            elif self.association_mode == "agnostic" and t.hits >= self.graveyard_min_hits:
                # WP-B: an unconfirmed-but-not-noise track still gets a
                # graveyard entry instead of being dropped outright, so
                # _revive_agnostic (label-blind) has a chance to recover it
                # -- legacy behaviour (drop) is unchanged.
                self._graveyard.append(t)
        self.tracks = alive
        self._graveyard = [
            t for t in self._graveyard
            if (now - t.last_seen) <= self.graveyard_ttl
        ]
        confirmed = [t for t in self.tracks if t.confirmed]
        self._publish_objects(confirmed, now)
        self._publish_markers(confirmed)
        log_latency(self.lat_writer, self.lat_file, time.perf_counter() - _t0)

    def _log(self, now, dt, tracks):
        """Research CSV, one row per track per tick (ported from
        user-a/sandbox). Columns cover every prior's inputs and outputs:
        semantic (category, p_movable_prior, consequence), behavioral
        (motion_score vs motion_threshold, moving, p_motion, p_movable),
        relation (relconf_in, relation_bonus), plus the raw Kalman state
        and the confirm/evict gates. Feed it to tools/prior_report.py (or
        the bare tools/motion_threshold_histogram.py). Ours uses
        `first_seen` where hers used `born` -- same first-observation
        stamp under the name this tree already publishes `age` from."""
        if not self.log_writer:
            return
        for tr in tracks:
            timeout = (self.max_unseen_dynamic if tr.p_movable > 0.5
                       else self.max_unseen_static)
            alive = (tr.confidence >= self.min_conf
                     and (now - tr.last_seen) <= timeout)
            if not self.log_unconfirmed and not (tr.confirmed and alive):
                continue
            vx, vy = tr.velocity
            self.log_writer.writerow([
                f"{now:.3f}", f"{dt:.4f}", tr.id, tr.label,
                label_category(tr.label),
                f"{tr.x[0]:.3f}", f"{tr.x[1]:.3f}", f"{vx:.4f}", f"{vy:.4f}",
                f"{tr.speed:.4f}",
                f"{tr.P[0, 0]:.5f}", f"{tr.P[1, 1]:.5f}",
                f"{tr.P[2, 2]:.5f}", f"{tr.P[3, 3]:.5f}",
                f"{tr.motion_score:.4f}", f"{self.motion_threshold:.3f}",
                int(tr.motion_score > self.motion_threshold),
                f"{tr.p_motion:.3f}", f"{tr.p_movable:.3f}",
                f"{tr.p_movable_prior:.3f}",
                f"{tr.relconf_event:.3f}", f"{tr.relation_bonus:.3f}",
                f"{tr.confidence:.4f}",
                f"{risk_score_from_label(tr.label, tr.confidence):.3f}",
                tr.hits, tr.misses, f"{now - tr.first_seen:.2f}",
                int(tr.confirmed), int(alive),
            ])
        self.log_file.flush()

    def _publish_objects(self, tracks, now):
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for tr in tracks:
            d = Detection3D()
            d.header = msg.header
            d.id = str(tr.id)
            x, y = tr.position
            # velocity_source (WP1, 2026-09-10): "displacement" (default)
            # publishes the EMA-smoothed displacement-window velocity
            # (Track.disp_v) instead of the raw (now velocity-damped)
            # Kalman state -- see module docstring. getattr guards the
            # test_object_tracker_lidar.py/_merge.py stubs, which predate
            # this attribute and bind _publish_objects's caller directly
            # onto Track objects without going through a real node.
            if getattr(self, "velocity_source", "displacement") == "kf":
                vx, vy = tr.velocity
            else:
                vx, vy = tr.disp_v
            d.bbox.center.position.x = x
            d.bbox.center.position.y = y
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = tr.size
            r = ObjectHypothesisWithPose()
            # velocity is embedded here, not just in the marker arrow: Stage 2
            # rolls the moving hypothesis out from (x, y, vx, vy), and without
            # these two fields it rolls out at zero velocity -- i.e. the whole
            # prediction collapses onto the stationary hypothesis. Stage 4's
            # CPA/TTC needs them too. Keep the key names in sync with
            # risk_visualization.parse_class_id.
            # hits/age (2026-09-10 finding, see module docstring): track
            # maturity, so a downstream consumer can down-weight a
            # barely-observed track (e.g. a lidar_cluster spawned last scan)
            # instead of trusting it as much as a long-lived one.
            age = max(0.0, now - tr.first_seen)
            # published_label (WP1 "motion overrides class", 2026-09-10
            # follow-up): tr.label unchanged -- only what's PUBLISHED here
            # swaps to mover_promote_label while promoted. getattr guards
            # the same pre-this-feature stubs as velocity_source above.
            promote_label = getattr(self, "mover_promote_label", "moving object")
            published_label = (tr.published_label(promote_label)
                               if hasattr(tr, "published_label") else tr.label)
            # WP-B (2026-09-11): `|raw=..` is the diagnostic "most recent
            # raw label seen, any category" field -- only appended in
            # agnostic mode (where self.label/published_label can lag the
            # single most recent detection by design, see
            # Track.register_label_vote) so legacy mode's class_id string
            # is byte-for-byte unchanged.
            raw_suffix = (f"|raw={tr.last_raw_label}"
                         if getattr(self, "association_mode", "legacy") == "agnostic"
                         else "")
            r.hypothesis.class_id = (
                f"{published_label}|pmov={tr.p_movable:.2f}|pmot={tr.p_motion:.2f}"
                f"|vx={vx:.3f}|vy={vy:.3f}|relbonus={tr.relation_bonus:.3f}"
                f"|hits={tr.hits}|age={age:.2f}"
                f"|disp={getattr(tr, 'net_disp_m', 0.0):.3f}"
                f"|promoted={1 if getattr(tr, 'promoted', False) else 0}"
                f"{raw_suffix}")
            r.hypothesis.score = float(tr.confidence)
            r.pose.pose = d.bbox.center
            cov = [0.0] * 36
            cov[0] = float(tr.P[0, 0])     # Pxx
            cov[7] = float(tr.P[1, 1])     # Pyy
            cov[21] = float(tr.P[2, 2])    # Pvxvx
            cov[28] = float(tr.P[3, 3])    # Pvyvy
            r.pose.covariance = cov
            d.results.append(r)
            msg.detections.append(d)
        self.obj_pub.publish(msg)

    def _publish_markers(self, tracks):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        life = DurationMsg(sec=int(self.marker_life),
                           nanosec=int((self.marker_life % 1) * 1e9))
        stamp = self.get_clock().now().to_msg()
        for tr in tracks:
            x, y = tr.position
            vx, vy = tr.velocity
            r, g, b = id_color(tr.id)
            alpha = float(max(0.35, min(1.0, tr.confidence)))
            footprint = max(0.22, min(float(tr.size), 0.7))

            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = self.map_frame
            disc.ns = "world_objects"
            disc.id = tr.id * 3
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position = Point(x=x, y=y, z=0.03)
            disc.pose.orientation.w = 1.0
            disc.scale.x = disc.scale.y = footprint
            disc.scale.z = 0.06
            disc.color = ColorRGBA(r=r, g=g, b=b, a=alpha)
            disc.lifetime = life
            arr.markers.append(disc)

            ring = Marker()
            ring.header = disc.header
            ring.ns = "world_object_rings"
            ring.id = tr.id * 3 + 1
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position = Point(x=x, y=y, z=0.01)
            ring.pose.orientation.w = 1.0
            ring.scale.x = ring.scale.y = footprint + 0.08
            ring.scale.z = 0.02
            ring.color = ColorRGBA(r=0.95, g=0.35, b=0.1,
                                   a=float(0.1 + 0.85 * tr.p_movable))
            ring.lifetime = life
            arr.markers.append(ring)

            text = Marker()
            text.header = disc.header
            text.ns = "world_object_labels"
            text.id = tr.id * 3 + 2
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=x, y=y, z=0.55)
            text.pose.orientation.w = 1.0
            text.scale.z = self.marker_text_height
            text.color = ColorRGBA(r=self.marker_text_color[0],
                                   g=self.marker_text_color[1],
                                   b=self.marker_text_color[2], a=1.0)
            text.lifetime = life
            # published_label (WP1 "motion overrides class"): show what's
            # actually published/scored, not the raw (possibly furniture-
            # mislabelled) tr.label -- see _publish_objects's own comment.
            _promote_label = getattr(self, "mover_promote_label", "moving object")
            _shown_label = (tr.published_label(_promote_label)
                            if hasattr(tr, "published_label") else tr.label)
            text.text = f"#{tr.id} {_shown_label}"
            if self.marker_show_beliefs:
                text.text += f"\nmov{tr.p_movable:.1f} mot{tr.p_motion:.1f}"
            arr.markers.append(text)

            if tr.p_motion > 0.5:
                a = Marker()
                a.header = disc.header
                a.ns = "world_object_velocity"
                a.id = tr.id * 3
                a.type = Marker.ARROW
                a.action = Marker.ADD
                a.pose.position = Point(x=x, y=y, z=0.1)
                a.pose.orientation = yaw_quat(vx, vy)
                a.scale.x = min(1.0, 0.3 + tr.speed)
                a.scale.y = a.scale.z = 0.06
                a.color = ColorRGBA(r=0.95, g=0.2, b=0.2, a=0.9)
                a.lifetime = life
                arr.markers.append(a)

        self.marker_pub.publish(arr)

    def destroy_node(self):
        if self.log_file:
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerNode()
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
