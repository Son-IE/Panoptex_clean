"""
Tests for the 2026-09-09 duplicate-track fixes, both from the same bag
finding: a patrolling Carter alternates GroundingDINO phrases across risk
categories ("ground mobile robot" == robot, "cart"/"forklift" == wheeled),
and the old category-keyed association hard-blocked the cross-category
match -- 74.6% of /risk_perception/world_objects messages carried >=2
tracks within 1m of one Carter simultaneously (38 distinct ids on one
Carter, 326 nearest-id switches).

1. association_groups: _same_object_class (association_key == "category")
   now also matches two DIFFERENT categories that share a group (default
   ["robot,wheeled"]) -- exercised through the full _detections_cb pipeline
   so association, not just the raw predicate, is covered.
2. Duplicate-track merge (_merge_duplicates): live, confirmed track pairs
   that are category-compatible and stay close (position + velocity) for
   merge_confirm_ticks consecutive ticks fold into the older id.

Same style as test_object_tracker_lidar.py: exercises ObjectTrackerNode's
pure/instance methods bound onto a plain stub (types.SimpleNamespace), no
rclpy.init() and no real ROS messages.
"""
import types

import pytest

from risk_perception.object_tracker_node import ObjectTrackerNode, Track

_BOUND_METHODS = (
    "_detections_cb", "_associate_agnostic", "_assign", "_same_object_class",
    "_in_same_group", "_gate_for", "_spawn", "_spawn_agnostic", "_revive",
    "_prior_for", "_predict_to", "_measurement_age", "_upgrade_track",
    "_mergeable_categories", "_merge_duplicates", "_merge_duplicates_legacy",
    # WP-B (2026-09-11): _merge_duplicates now DISPATCHES to
    # _merge_duplicates_legacy/_merge_duplicates_agnostic based on
    # association_mode (absent on this stub -> defaults to "legacy" via
    # getattr, see that method's docstring) -- both need to be bound for
    # the dispatch itself to resolve on this stub.
    "_merge_duplicates_agnostic",
)
# @staticmethod on the real class -- take no `self`.
_STATIC_METHODS = ("_greedy_assign", "_adopt_label", "_fold_track", "_pair_key",
                   "_fold_track_agnostic", "_pair_mahalanobis_d2")


def make_tracker(association_groups=("robot,wheeled",), gate=0.6, gate_speed=1.0,
                 association_key="category", process_noise=0.05, default_cov=0.04,
                 merge_enabled=True, merge_distance_m=0.5, merge_speed_diff_mps=0.5,
                 merge_confirm_ticks=3):
    stub = types.SimpleNamespace(
        association_key=association_key,
        association_groups=[
            frozenset(s.strip().lower() for s in g.split(",") if s.strip())
            for g in association_groups
        ],
        gate=gate,
        gate_speed=gate_speed,
        gate_max_m=2.5,
        motion_window_s=2.0,
        meas_time_corr=False,   # age always 0 -- deterministic
        max_meas_age=3.0,
        _meas_ages=[],
        label_agnostic_labels={"lidar_cluster"},
        upgrade_penalty_m=0.3,
        unknown_penalty_m=0.3,
        category_priors={"person": 0.9, "robot": 0.9, "wheeled": 0.7, "furniture": 0.1},
        prior_unknown=0.3,
        process_noise=process_noise,
        default_cov=default_cov,
        merge_enabled=merge_enabled,
        merge_distance_m=merge_distance_m,
        merge_speed_diff_mps=merge_speed_diff_mps,
        merge_confirm_ticks=merge_confirm_ticks,
        _merge_streak={},
        tracks=[],
        _graveyard=[],
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
        covariance[0] = d["covariance0"]
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


def make_confirmed_track(x, y, vx, vy, label, confidence, p_motion=0.0, p_movable=0.5):
    tr = Track(x=x, y=y, label=label, score=confidence, cov_xy=0.04, size=0.5,
               stamp=0.0, p_movable_prior=p_movable)
    tr.x[2], tr.x[3] = vx, vy
    tr.confidence = confidence
    tr.p_motion = p_motion
    tr.p_movable = p_movable
    tr.hits = 5
    tr.confirmed = True
    return tr


# ---------------------------------------------------------------------------
# 1. association_groups: cross-category association
# ---------------------------------------------------------------------------

def test_cross_category_detection_updates_existing_track_under_default_group():
    """A "cart" (wheeled) detection 0.3m from a live "ground mobile robot"
    (robot) track updates that track under the default association_groups
    -- no new id."""
    stub = make_tracker()
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "ground mobile robot", "score": 0.9, "x": 1.0, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    track_id = tr.id

    stub._now_value[0] = 0.1
    cart_msg = make_msg([{"label": "cart", "score": 0.5, "x": 1.3, "y": 0.0,
                          "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cart_msg)

    assert len(stub.tracks) == 1
    assert stub.tracks[0] is tr
    assert tr.id == track_id
    assert tr.hits == 2


def test_cross_category_detection_spawns_new_track_when_no_groups_configured():
    """Same scenario, but with association_groups: [] -- the old hard-block
    behaviour must be preserved."""
    stub = make_tracker(association_groups=())
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "ground mobile robot", "score": 0.9, "x": 1.0, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    assert len(stub.tracks) == 1

    stub._now_value[0] = 0.1
    cart_msg = make_msg([{"label": "cart", "score": 0.5, "x": 1.3, "y": 0.0,
                          "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cart_msg)

    assert len(stub.tracks) == 2


# ---------------------------------------------------------------------------
# 2. Duplicate-track merge
# ---------------------------------------------------------------------------

def test_close_same_group_tracks_merge_after_confirm_ticks_keeping_higher_confidence():
    stub = make_tracker()
    older = make_confirmed_track(0.0, 0.0, 0.2, 0.0, "ground mobile robot", 0.6)
    younger = make_confirmed_track(0.3, 0.0, 0.2, 0.0, "cart", 0.8)
    assert older.id < younger.id
    stub.tracks = [older, younger]

    # merge_confirm_ticks defaults to 3 -- must not fire before the 3rd tick.
    stub._merge_duplicates()
    assert len(stub.tracks) == 2
    stub._merge_duplicates()
    assert len(stub.tracks) == 2
    stub._merge_duplicates()

    assert len(stub.tracks) == 1
    survivor = stub.tracks[0]
    assert survivor is older
    assert survivor.id == older.id
    assert survivor.confidence == pytest.approx(0.8)   # higher of the two
    assert survivor.label == "cart"                    # from the higher-confidence track
    assert survivor.position == pytest.approx((0.0, 0.0))  # older's state kept


def test_opposite_velocity_tracks_do_not_merge():
    stub = make_tracker()
    older = make_confirmed_track(0.0, 0.0, 0.5, 0.0, "ground mobile robot", 0.6)
    younger = make_confirmed_track(0.3, 0.0, -0.5, 0.0, "cart", 0.8)
    stub.tracks = [older, younger]

    for _ in range(5):
        stub._merge_duplicates()

    assert len(stub.tracks) == 2


def test_person_next_to_robot_does_not_merge():
    stub = make_tracker()
    older = make_confirmed_track(0.0, 0.0, 0.0, 0.0, "person", 0.9)
    younger = make_confirmed_track(0.2, 0.0, 0.0, 0.0, "ground mobile robot", 0.9)
    stub.tracks = [older, younger]

    for _ in range(5):
        stub._merge_duplicates()

    assert len(stub.tracks) == 2


def test_merge_disabled_never_folds():
    stub = make_tracker(merge_enabled=False)
    older = make_confirmed_track(0.0, 0.0, 0.0, 0.0, "ground mobile robot", 0.6)
    younger = make_confirmed_track(0.1, 0.0, 0.0, 0.0, "cart", 0.8)
    stub.tracks = [older, younger]

    for _ in range(5):
        stub._merge_duplicates()

    assert len(stub.tracks) == 2


def test_streak_resets_when_pair_drifts_apart_between_ticks():
    stub = make_tracker()
    older = make_confirmed_track(0.0, 0.0, 0.0, 0.0, "ground mobile robot", 0.6)
    younger = make_confirmed_track(0.1, 0.0, 0.0, 0.0, "cart", 0.8)
    stub.tracks = [older, younger]

    stub._merge_duplicates()
    stub._merge_duplicates()
    # Drift apart for one tick -- streak must reset, not just pause.
    younger.x[0] = 5.0
    stub._merge_duplicates()
    assert len(stub.tracks) == 2
    younger.x[0] = 0.1
    stub._merge_duplicates()
    stub._merge_duplicates()
    assert len(stub.tracks) == 2   # only 2 consecutive ticks so far
    stub._merge_duplicates()
    assert len(stub.tracks) == 1   # 3rd consecutive tick


def test_unconfirmed_tracks_are_never_merge_candidates():
    stub = make_tracker()
    older = make_confirmed_track(0.0, 0.0, 0.0, 0.0, "ground mobile robot", 0.6)
    younger = make_confirmed_track(0.1, 0.0, 0.0, 0.0, "cart", 0.8)
    younger.confirmed = False
    stub.tracks = [older, younger]

    for _ in range(5):
        stub._merge_duplicates()

    assert len(stub.tracks) == 2
