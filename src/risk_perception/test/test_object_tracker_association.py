"""
WP-B tests (plan: ~/.claude/plans/context-perception-panoptex-real-cameras-
cuddly-coral.md) for object_tracker_node.py's category-agnostic, stable
association mode (`association_mode: agnostic`, default): Mahalanobis-
on-position association cost with a SOFT (never hard-gating) category
penalty, score-weighted label voting, N-of-M confirmation, a label-blind
graveyard revive, and a looser duplicate-track merge that tolerates one
unconfirmed member.

Same style as test_object_tracker_lidar.py/test_object_tracker_merge.py:
exercises ObjectTrackerNode's pure/instance methods bound onto a plain
stub (types.SimpleNamespace), no rclpy.init() and no real ROS messages.
`association_mode` defaults to "legacy" when ABSENT from a stub (see every
mode-branch's `getattr(self, "association_mode", "legacy")` in
object_tracker_node.py) -- every pre-WP-B test file's stub never sets it,
which is what keeps those tests passing unchanged; this file's stub sets
it to "agnostic" explicitly.
"""
import types

import pytest

from risk_perception.object_tracker_node import ObjectTrackerNode, Track
from risk_perception.risk_visualization import label_category

_BOUND_METHODS = (
    "_detections_cb", "_associate_agnostic", "_assign", "_same_object_class",
    "_in_same_group", "_gate_for", "_spawn", "_spawn_agnostic", "_revive",
    "_revive_agnostic", "_prior_for", "_predict_to", "_measurement_age",
    "_upgrade_track", "_vote_label_and_reseed", "_reseed_prior_if_still_default",
    "_assoc_cost_agnostic", "_category_mismatch_penalty", "_apply_speed_cap",
    "_speed_cap_for", "_mergeable_categories", "_merge_duplicates",
    "_merge_duplicates_legacy", "_merge_duplicates_agnostic",
    "_charge_stale_misses",
)
_STATIC_METHODS = ("_greedy_assign", "_adopt_label", "_fold_track",
                   "_fold_track_agnostic", "_pair_key", "_pair_mahalanobis_d2")


def make_tracker(association_mode="agnostic", gate=0.6, gate_speed=0.6, gate_max_m=2.5,
                 association_key="category", association_groups=("robot,wheeled",),
                 assoc_chi2_gate=9.21, assoc_size_weight=2.0, assoc_label_penalty=2.0,
                 label_vote_half_life_s=30.0, confirm_n=3, confirm_m=5,
                 confirm_miss_grace_s=1.5,
                 graveyard_min_hits=2, merge_chi2=4.0, merge_distance_m=0.5,
                 merge_speed_diff_mps=0.5, merge_confirm_ticks=3, merge_enabled=True,
                 process_noise=0.05, default_cov=0.04, upgrade_penalty_m=0.3,
                 unknown_penalty_m=0.3):
    stub = types.SimpleNamespace(
        association_mode=association_mode,
        association_key=association_key,
        gate=gate, gate_speed=gate_speed, gate_max_m=gate_max_m,
        association_groups=[
            frozenset(s.strip().lower() for s in g.split(",") if s.strip())
            for g in association_groups
        ],
        motion_window_s=2.0,
        meas_time_corr=False,   # age always 0 -- deterministic
        max_meas_age=3.0,
        _meas_ages=[],
        label_agnostic_labels={"lidar_cluster"},
        upgrade_penalty_m=upgrade_penalty_m,
        unknown_penalty_m=unknown_penalty_m,
        assoc_chi2_gate=assoc_chi2_gate,
        assoc_size_weight=assoc_size_weight,
        assoc_label_penalty=assoc_label_penalty,
        label_vote_half_life_s=label_vote_half_life_s,
        confirm_n=confirm_n, confirm_m=confirm_m,
        confirm_miss_grace_s=confirm_miss_grace_s,
        graveyard_min_hits=graveyard_min_hits,
        merge_chi2=merge_chi2, merge_distance_m=merge_distance_m,
        merge_speed_diff_mps=merge_speed_diff_mps,
        merge_confirm_ticks=merge_confirm_ticks, merge_enabled=merge_enabled,
        category_priors={"person": 0.9, "robot": 0.9, "wheeled": 0.7, "furniture": 0.1},
        prior_unknown=0.3,
        process_noise=process_noise,
        default_cov=default_cov,
        jump_reset_factor=2.0,
        speed_caps={"person": 2.0, "robot": 1.5, "wheeled": 2.0, "unknown": 1.2,
                   "default": 2.5},
        tracks=[],
        _graveyard=[],
        _merge_streak={},
        last_predict=None,
        _now_value=[0.0],
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    stub._now = lambda: stub._now_value[0]
    for name in _BOUND_METHODS:
        setattr(stub, name, types.MethodType(getattr(ObjectTrackerNode, name), stub))
    for name in _STATIC_METHODS:
        setattr(stub, name, getattr(ObjectTrackerNode, name))
    return stub


def make_msg(dets):
    """dets: list of {label, score, x, y, covariance0, size}."""
    detections = []
    for d in dets:
        covariance = [0.0] * 36
        covariance[0] = d.get("covariance0", d.get("cov", 0.0))
        detections.append(types.SimpleNamespace(
            results=[types.SimpleNamespace(
                hypothesis=types.SimpleNamespace(class_id=d["label"], score=d["score"]),
                pose=types.SimpleNamespace(covariance=covariance))],
            bbox=types.SimpleNamespace(
                center=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=d["x"], y=d["y"])),
                size=types.SimpleNamespace(x=d["size"], y=d["size"])),
        ))
    return types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0)),
        detections=detections)


def mk_det(x, y, label="mobile robot", score=0.9, cov=0.04, size=0.5, relconf=0.0):
    return {"x": x, "y": y, "label": label, "score": score, "cov": cov, "size": size,
           "relconf": relconf}


# --------------------------------------------------------- phrase churn keeps one id

def test_agnostic_mode_phrase_churn_across_categories_keeps_one_id():
    """The core WP-B claim: under legacy's hard _same_object_class gate, a
    detection sequence that alternates between categories with NO shared
    association_groups entry ("mobile robot" -> "table" -> "chair", robot
    -> furniture -> furniture) would fragment into a new track every time
    the category changes -- see the module docstring's 2026-09-08 finding
    and test_object_tracker_merge.py's own header comment for the
    real-bag version of this failure. Under association_mode: agnostic,
    category mismatch only costs (assoc_label_penalty), it never blocks
    the candidate outright, so motion continuity alone keeps one id."""
    stub = make_tracker()
    stub._now_value[0] = 0.0
    stub._detections_cb(make_msg([mk_det(1.00, 0.0, label="mobile robot", cov=0.10)]))
    assert len(stub.tracks) == 1
    track_id = stub.tracks[0].id

    stub._now_value[0] = 0.5
    stub._detections_cb(make_msg([mk_det(1.05, 0.0, label="table", cov=0.10)]))
    assert len(stub.tracks) == 1
    assert stub.tracks[0].id == track_id

    stub._now_value[0] = 1.0
    stub._detections_cb(make_msg([mk_det(1.10, 0.0, label="chair", cov=0.10)]))
    assert len(stub.tracks) == 1
    assert stub.tracks[0].id == track_id
    assert stub.tracks[0].hits == 3


def test_legacy_mode_same_phrase_churn_fragments_into_separate_tracks():
    """Contrast case: the exact same detection sequence, association_mode:
    legacy -- category mismatch with no shared group hard-blocks
    association, so each phrase change spawns a NEW track (the pre-WP-B,
    pre-association_groups failure mode)."""
    stub = make_tracker(association_mode="legacy")
    stub._now_value[0] = 0.0
    stub._detections_cb(make_msg([mk_det(1.00, 0.0, label="mobile robot", cov=0.10)]))
    assert len(stub.tracks) == 1

    stub._now_value[0] = 0.5
    stub._detections_cb(make_msg([mk_det(1.05, 0.0, label="table", cov=0.10)]))
    assert len(stub.tracks) == 2   # "table" (furniture) can't join a "robot" track


# ------------------------------------------------------- category penalty is soft

def test_category_mismatch_penalty_zero_when_either_side_unknown():
    stub = make_tracker()
    unknown_tr = Track(x=0.0, y=0.0, label="lidar_cluster", score=0.5, cov_xy=0.02,
                       size=0.3, stamp=0.0, p_movable_prior=0.3)
    assert label_category(unknown_tr.label) == "unknown"
    assert stub._category_mismatch_penalty(unknown_tr, "table") == 0.0

    labeled_tr = Track(x=0.0, y=0.0, label="table", score=0.5, cov_xy=0.02,
                       size=0.3, stamp=0.0, p_movable_prior=0.1)
    assert stub._category_mismatch_penalty(labeled_tr, "lidar_cluster") == 0.0


def test_category_mismatch_penalty_full_when_both_known_no_shared_group():
    stub = make_tracker()
    tr = Track(x=0.0, y=0.0, label="table", score=0.5, cov_xy=0.02, size=0.3,
              stamp=0.0, p_movable_prior=0.1)
    assert stub._category_mismatch_penalty(tr, "mobile robot") == pytest.approx(
        stub.assoc_label_penalty)


def test_category_mismatch_penalty_zero_within_association_group():
    stub = make_tracker()
    tr = Track(x=0.0, y=0.0, label="cart", score=0.5, cov_xy=0.02, size=0.3,
              stamp=0.0, p_movable_prior=0.7)
    assert stub._category_mismatch_penalty(tr, "mobile robot") == 0.0  # robot,wheeled group


# ------------------------------------------------------------------- chi2 gate

def test_assoc_cost_agnostic_rejects_tight_covariance_jump_within_metric_gate():
    """Isolates the STATISTICAL half of the agnostic gate from the metric
    half: gate_distance_m is loosened to 2.0 m (so a 1 m jump easily
    clears the metric cap), but a TIGHT covariance (0.001 m^2 on both
    sides) makes that same 1 m jump wildly implausible statistically --
    d2 = 1.0^2 / (0.001+0.001) = 500 >> assoc_chi2_gate (9.21)."""
    stub = make_tracker(gate=2.0, gate_speed=0.0)
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.001, size=0.5,
              stamp=0.0, p_movable_prior=0.9)
    det = mk_det(1.0, 0.0, cov=0.001)
    assert stub._assoc_cost_agnostic(tr, det, age=0.0, now=0.0) is None


def test_assoc_cost_agnostic_accepts_plausible_motion():
    stub = make_tracker(gate=2.0, gate_speed=0.0)
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.1, size=0.5,
              stamp=0.0, p_movable_prior=0.9)
    det = mk_det(0.3, 0.0, cov=0.1)
    cost = stub._assoc_cost_agnostic(tr, det, age=0.0, now=0.0)
    assert cost is not None
    assert cost >= 0.0


def test_assoc_cost_agnostic_folds_upgrade_penalty_for_agnostic_track():
    """An agnostic (lidar-spawned, "unknown") track claimed by a labeled
    detection carries an extra upgrade_penalty_m-equivalent bias over an
    otherwise-identical already-labeled track at the same distance -- see
    _detections_cb's `bias = self.upgrade_penalty_m if track_is_agnostic
    else 0.0` and object_tracker_node.py's module docstring on
    upgrade_penalty_m."""
    stub = make_tracker(gate=2.0, gate_speed=0.0)
    lidar_tr = Track(x=0.0, y=0.0, label="lidar_cluster", score=0.5, cov_xy=0.1,
                     size=0.5, stamp=0.0, p_movable_prior=0.3)
    labeled_tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.1,
                       size=0.5, stamp=0.0, p_movable_prior=0.9)
    det = mk_det(0.2, 0.0, label="mobile robot", cov=0.1)
    cost_lidar = stub._assoc_cost_agnostic(lidar_tr, det, age=0.0, now=0.0,
                                           extra_label_bias=stub.upgrade_penalty_m)
    cost_labeled = stub._assoc_cost_agnostic(labeled_tr, det, age=0.0, now=0.0)
    assert cost_lidar == pytest.approx(cost_labeled + stub.upgrade_penalty_m)


# --------------------------------------------------------------- label voting

def test_label_vote_mode_survives_a_short_flicker():
    """A few low-score furniture blips must not flip a track whose
    dominant, higher-score evidence is "robot" -- register_label_vote's
    score-weighted Counter, not latest-wins."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
              stamp=0.0, p_movable_prior=0.9)   # seeds one robot vote @ 0.9
    for i in range(3):
        tr.register_label_vote("table", 0.2, "furniture", 0.1 * (i + 1), half_life_s=30.0)
    tr.register_label_vote("mobile robot", 0.9, "robot", 0.5, half_life_s=30.0)
    assert label_category(tr.label) == "robot"
    assert tr.label == "mobile robot"


def test_label_vote_mode_flips_after_sustained_relabel():
    """Sustained, repeated furniture evidence with NO further robot votes
    (a genuine relabel, not a flicker) does eventually win -- vote mode is
    stabilising, not permanently stuck on the first label. Short
    half_life_s here so the flip is visible within a handful of calls."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
              stamp=0.0, p_movable_prior=0.9)
    for i in range(1, 4):
        tr.register_label_vote("chair", 0.9, "furniture", float(i), half_life_s=1.0)
    assert label_category(tr.label) == "furniture"
    assert tr.label == "chair"


def test_last_raw_label_tracks_most_recent_detection_regardless_of_winner():
    """last_raw_label (the diagnostic |raw=.. field) is NOT gated by which
    category is currently winning the vote -- it always reflects the most
    recent detection, even a losing-category flicker."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
              stamp=0.0, p_movable_prior=0.9)
    tr.register_label_vote("table", 0.1, "furniture", 0.5, half_life_s=30.0)
    assert tr.last_raw_label == "table"
    assert tr.label == "mobile robot"   # vote winner unchanged (robot still dominant)


# ------------------------------------------------------------- label-blind revive

def test_revive_agnostic_recovers_track_across_a_category_change():
    """A graveyard track spawned/last seen as "table" (furniture) is
    revived by a detection carrying a totally different raw label/category
    ("mobile robot") purely on motion continuity -- _revive (legacy) would
    never even consider this pair (_same_object_class gate)."""
    stub = make_tracker()
    tr = Track(x=1.0, y=0.0, label="table", score=0.5, cov_xy=0.05, size=0.5,
              stamp=0.0, p_movable_prior=0.1)
    tr.x[2], tr.x[3] = 0.2, 0.0   # was drifting east at 0.2 m/s
    tr.last_seen = 0.0
    stub._graveyard = [tr]

    det = mk_det(1.2, 0.0, label="mobile robot", cov=0.05)  # near the extrapolated position
    revived = stub._revive_agnostic(det, now=1.0)
    assert revived is tr
    assert len(stub._graveyard) == 0


def test_revive_agnostic_rejects_implausibly_distant_candidate():
    stub = make_tracker()
    tr = Track(x=0.0, y=0.0, label="table", score=0.5, cov_xy=0.02, size=0.5,
              stamp=0.0, p_movable_prior=0.1)
    tr.last_seen = 0.0
    stub._graveyard = [tr]
    det = mk_det(20.0, 20.0, label="mobile robot", cov=0.02)
    assert stub._revive_agnostic(det, now=1.0) is None
    assert len(stub._graveyard) == 1   # left untouched


# --------------------------------------------------------------- N-of-M confirm

def test_n_of_m_confirmation_confirms_within_window():
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
              stamp=0.0, p_movable_prior=0.9, confirm_m=5)
    tr.mark_missed()
    tr.mark_missed()
    for _ in range(3):
        tr.update(0.0, 0.0, 0.04, 0.9, 0.5, 0.0)
    assert sum(tr.recent) == 3
    assert sum(tr.recent) >= 3   # confirm_n default


def test_n_of_m_confirmation_forgets_stale_hits_outside_window():
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
              stamp=0.0, p_movable_prior=0.9, confirm_m=5)
    for _ in range(3):
        tr.update(0.0, 0.0, 0.04, 0.9, 0.5, 0.0)
    assert sum(tr.recent) == 3
    for _ in range(5):
        tr.mark_missed()
    assert sum(tr.recent) == 0   # the maxlen=5 window has fully rolled over


# --------------------------------------- WP-B miss-charging fix (2026-09-11)
#
# "Status 2026-09-11 13:00" / "Remaining work" #1 in the plan: a miss used
# to be charged in _detections_cb on EVERY unmatched detections message
# from ANY source (3 overhead cameras + 10 Hz lidar share one callback) --
# a parked Carter seen by only one 0.5 Hz camera collected a miss on
# nearly every OTHER source's message too (~5 misses per real hit) and
# essentially never held enough hits in its confirm_m window to confirm
# (unconfirmed tracks are never published). Fixed by moving miss-charging
# to _tick (_charge_stale_misses), gated by confirm_miss_grace_s and
# invoked at most once per grace interval of continuous silence --
# completely decoupled from message arrival rate/count.

def test_single_camera_half_hz_track_confirms_and_stays_published():
    """The exact unit2_pan_1 carter2 regression, reproduced directly: a
    track seen by only ONE 0.5 Hz camera (a hit every 2 s) must accumulate
    enough hits in `recent` to confirm under agnostic mode.
    _charge_stale_misses is invoked every 0.1 s (update_rate) across each
    2 s gap -- not just once per gap -- to prove the grace gate, not
    merely infrequent polling, is what keeps a single-camera track from
    being miss-starved."""
    stub = make_tracker(confirm_n=3, confirm_m=6, confirm_miss_grace_s=1.5)
    stub._now_value[0] = 0.0
    stub._detections_cb(make_msg([mk_det(0.0, 0.0)]))
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]

    for cycle in range(5):   # 5 more camera hits, 2 s apart (0.5 Hz)
        for step in range(1, 21):   # _tick at 10 Hz across the 2 s gap
            stub._charge_stale_misses(cycle * 2.0 + step * 0.1)
        t = (cycle + 1) * 2.0
        stub._now_value[0] = t
        stub._detections_cb(make_msg([mk_det(0.0, 0.0)]))

    assert len(stub.tracks) == 1
    assert stub.tracks[0].id == tr.id      # never fragmented into a new id
    assert sum(tr.recent) >= stub.confirm_n   # confirms -- would publish


def test_charge_stale_misses_charges_at_most_one_per_grace_interval():
    """Isolates the grace gate itself: even when _charge_stale_misses is
    called every 0.1 s (as _tick would, at 10 Hz) across a long silence,
    at most one miss is charged per confirm_miss_grace_s -- not one per
    call. This is what makes the fix "at most once per tick, and only
    when now - last_seen > confirm_miss_grace_s" rather than "once per
    tick, unconditionally" (which would still starve a sparse track, just
    at 10 Hz instead of message rate)."""
    stub = make_tracker(confirm_miss_grace_s=1.5)
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
              size=0.5, stamp=0.0, p_movable_prior=0.9, confirm_m=50)
    tr.update(0.0, 0.0, 0.04, 0.9, 0.5, 0.0)   # last_seen = 0.0, recent=[1]
    stub.tracks = [tr]

    for step in range(1, 61):   # 6 s of 10 Hz ticks, no further detections
        stub._charge_stale_misses(step * 0.1)

    # Grace-gated cadence: silence duration 6 s / grace 1.5 s -> at most
    # floor(6 / 1.5) = 4 misses, never 59 (one per tick call).
    misses_charged = sum(1 for v in tr.recent if v == 0)
    assert 1 <= misses_charged <= 4


def test_other_sources_messages_do_not_starve_a_sparse_track():
    """A track associated only by a slow-camera label must not lose
    confirmation progress just because OTHER, unrelated detection
    messages (a different object entirely, far away) keep arriving in
    between -- the pre-fix bug charged this track a miss on every one of
    those unrelated messages too. Interleaves 10 unrelated-object messages
    into each 2 s gap between this track's own 0.5 Hz hits and confirms
    exactly as in the isolated case above."""
    stub = make_tracker(confirm_n=3, confirm_m=6, confirm_miss_grace_s=1.5)
    stub._now_value[0] = 0.0
    stub._detections_cb(make_msg([mk_det(0.0, 0.0)]))
    tr = stub.tracks[0]

    for cycle in range(5):
        for step in range(1, 21):
            t = cycle * 2.0 + step * 0.1
            # An unrelated object far away, on every "tick" -- exercises
            # _detections_cb's per-message path without ever matching tr.
            stub._now_value[0] = t
            stub._detections_cb(make_msg([mk_det(50.0, 50.0, label="pallet")]))
            stub._charge_stale_misses(t)
        t = (cycle + 1) * 2.0
        stub._now_value[0] = t
        stub._detections_cb(make_msg([mk_det(0.0, 0.0)]))

    tracked = [t for t in stub.tracks if t.id == tr.id]
    assert len(tracked) == 1
    assert sum(tr.recent) >= stub.confirm_n   # still confirms -- unstarved


def test_legacy_mode_still_charges_a_miss_per_unmatched_message():
    """Contrast case, required by the plan ("legacy mode unchanged"):
    legacy mode never reads Track.recent for confirmation, and keeps the
    exact pre-WP-B behaviour of charging Track.misses on every unmatched
    detections message -- _charge_stale_misses is a no-op under legacy
    mode (see its own `if self.association_mode != "agnostic": return`)."""
    stub = make_tracker(association_mode="legacy")
    stub._now_value[0] = 0.0
    stub._detections_cb(make_msg([mk_det(0.0, 0.0)]))
    tr = stub.tracks[0]
    assert tr.misses == 0

    for i in range(4):
        stub._now_value[0] = 0.1 * (i + 1)
        # An unrelated, far-away detection -- tr itself goes unmatched.
        stub._detections_cb(make_msg([mk_det(50.0, 50.0, label="pallet")]))
    assert tr.misses == 4   # one per unmatched message, exactly as before

    # _charge_stale_misses is inert under legacy mode -- confirms nothing
    # extra was charged through the new tick-based path either.
    stub._charge_stale_misses(10.0)
    assert tr.misses == 4


# --------------------------------------------------------- merge with unconfirmed

def test_merge_agnostic_allows_one_unconfirmed_member():
    stub = make_tracker()
    tr_a = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
                stamp=0.0, p_movable_prior=0.9)
    tr_a.confirmed = True
    tr_b = Track(x=0.05, y=0.0, label="cart", score=0.5, cov_xy=0.04, size=0.5,
                stamp=0.0, p_movable_prior=0.7)
    tr_b.confirmed = False
    stub.tracks = [tr_a, tr_b]
    for _ in range(stub.merge_confirm_ticks):
        stub._merge_duplicates_agnostic()
    assert len(stub.tracks) == 1
    assert stub.tracks[0].id == min(tr_a.id, tr_b.id)


def test_merge_agnostic_rejects_pair_both_unconfirmed():
    stub = make_tracker()
    tr_a = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
                stamp=0.0, p_movable_prior=0.9)
    tr_a.confirmed = False
    tr_b = Track(x=0.05, y=0.0, label="cart", score=0.5, cov_xy=0.04, size=0.5,
                stamp=0.0, p_movable_prior=0.7)
    tr_b.confirmed = False
    stub.tracks = [tr_a, tr_b]
    for _ in range(5):
        stub._merge_duplicates_agnostic()
    assert len(stub.tracks) == 2


def test_fold_track_agnostic_sums_votes_and_extends_meas():
    older = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04, size=0.5,
                  stamp=0.0, p_movable_prior=0.9)
    older.update(0.0, 0.0, 0.04, 0.9, 0.5, 0.0)
    younger = Track(x=0.05, y=0.0, label="cart", score=0.5, cov_xy=0.04, size=0.5,
                    stamp=1.0, p_movable_prior=0.7)
    younger.update(0.05, 0.0, 0.04, 0.5, 0.5, 1.0)
    older_votes_before = dict(older.label_votes)
    ObjectTrackerNode._fold_track_agnostic(older, younger)
    for cat, w in older_votes_before.items():
        assert older.label_votes[cat] >= w   # never loses vote weight
    assert older.label_votes["wheeled"] >= 0.5   # younger's vote folded in
    assert len(older.meas) == 2   # both tracks' single measurement kept
