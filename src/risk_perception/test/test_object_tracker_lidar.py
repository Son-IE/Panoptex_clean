"""
Tests for object_tracker_node.py's WP1 label-agnostic measurement mode
(label_agnostic_labels, default ["lidar_cluster"]): a second association
pass, after the existing labeled pass, that lets scan_cluster_detector_node's
lidar clusters tighten an existing track's Kalman covariance or spawn an
"unknown"-category track, but never adopt a label, never revive a graveyard
track and never change a track's category.

Same style as test_object_tracker_motion.py: exercises ObjectTrackerNode's
pure/instance methods bound onto a plain stub (types.SimpleNamespace), no
rclpy.init() and no real ROS messages -- msg/detection objects are the
minimal SimpleNamespace shape _detections_cb actually reads.
"""
import types

import pytest

from risk_perception.object_tracker_node import ObjectTrackerNode, Track
from risk_perception.risk_visualization import label_category

_BOUND_METHODS = (
    "_detections_cb", "_associate_agnostic", "_assign", "_same_object_class",
    "_gate_for", "_spawn", "_spawn_agnostic", "_revive", "_prior_for",
    "_predict_to", "_measurement_age", "_upgrade_track",
)
# Both @staticmethod on the real class -- take no `self`, so the plain
# underlying function can be stored straight onto the stub instance.
_STATIC_METHODS = ("_greedy_assign", "_adopt_label")


def make_tracker(label_agnostic_labels=("lidar_cluster",), gate=0.6, gate_speed=1.0,
                 association_key="category", process_noise=0.05, default_cov=0.04,
                 upgrade_penalty_m=0.3, unknown_penalty_m=0.3):
    stub = types.SimpleNamespace(
        association_key=association_key,
        gate=gate,
        gate_speed=gate_speed,
        gate_max_m=2.5,
        motion_window_s=2.0,
        meas_time_corr=False,   # age always 0 -- deterministic, matches the
                                # existing motion tests' simplification
        max_meas_age=3.0,
        _meas_ages=[],
        label_agnostic_labels={s.lower() for s in label_agnostic_labels},
        upgrade_penalty_m=upgrade_penalty_m,
        unknown_penalty_m=unknown_penalty_m,
        category_priors={"person": 0.9, "robot": 0.9, "wheeled": 0.7, "furniture": 0.1},
        prior_unknown=0.3,
        process_noise=process_noise,
        default_cov=default_cov,
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


def test_lidar_measurement_tightens_position_and_velocity_covariance():
    stub = make_tracker()
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "mobile robot", "score": 0.9, "x": 1.0, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    p_xx_before = tr.P[0, 0]
    p_vxvx_before = tr.P[2, 2]
    assert p_xx_before == pytest.approx(0.10)
    # velocity_var_init (WP1, 2026-09-10): was hardcoded 1.0, now the
    # configurable default 0.25 -- see object_tracker_node.py's module
    # docstring and Track.__init__.
    assert p_vxvx_before == pytest.approx(0.25)

    # A lidar cluster close to the track, one second later.
    stub._now_value[0] = 1.0
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 1.05, "y": 0.0,
                           "covariance0": 0.025, "size": 0.3}])
    stub._detections_cb(lidar_msg)

    assert len(stub.tracks) == 1          # matched the existing track, not a new one
    assert stub.tracks[0] is tr
    assert tr.P[0, 0] < p_xx_before       # position covariance tightened
    assert tr.P[1, 1] < p_xx_before
    assert tr.P[2, 2] < p_vxvx_before     # velocity covariance tightened too
    assert tr.P[3, 3] < p_vxvx_before
    assert tr.label == "mobile robot"     # never adopted the lidar label


def test_unmatched_lidar_detection_spawns_unknown_track_labelled_lidar_cluster():
    stub = make_tracker()
    stub._now_value[0] = 0.0
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 5.0, "y": 5.0,
                           "covariance0": 0.025, "size": 0.3}])
    stub._detections_cb(lidar_msg)

    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    assert tr.label == "lidar_cluster"
    assert label_category(tr.label) == "unknown"
    assert tr.p_movable_prior == pytest.approx(stub.prior_unknown)


def test_lidar_detection_near_mobile_robot_track_never_changes_its_label():
    stub = make_tracker()
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "cart", "score": 0.5, "x": 2.0, "y": 2.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    tr = stub.tracks[0]
    assert tr.label == "cart"

    # A high-score lidar detection would flip the label under the labeled
    # pass's _adopt_label rule (score >= confidence) -- must NOT here.
    stub._now_value[0] = 0.5
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.95, "x": 2.02, "y": 2.0,
                           "covariance0": 0.02, "size": 0.3}])
    stub._detections_cb(lidar_msg)

    assert len(stub.tracks) == 1
    assert stub.tracks[0] is tr
    assert tr.label == "cart"


def test_labeled_association_unchanged_without_lidar_detections():
    """Regression: with no label-agnostic detections in the message, the
    labeled pass (association, adopt_label, hit counting) must behave
    exactly as it did before this feature existed."""
    stub = make_tracker()
    stub._now_value[0] = 0.0
    msg1 = make_msg([{"label": "cart", "score": 0.5, "x": 0.0, "y": 0.0,
                      "covariance0": 0.04, "size": 0.5}])
    stub._detections_cb(msg1)
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    assert tr.label == "cart"
    assert tr.hits == 1

    # "forklift" is a different phrase but the same risk category
    # (wheeled) as "cart" -- category-level association should still match
    # it to the same track and adopt the higher-confidence label, exactly
    # as test_object_tracker_motion.py's test_adopt_label_follows_confidence
    # exercises directly on _adopt_label.
    stub._now_value[0] = 0.5
    msg2 = make_msg([{"label": "forklift", "score": 0.9, "x": 0.05, "y": 0.0,
                      "covariance0": 0.04, "size": 0.5}])
    stub._detections_cb(msg2)

    assert len(stub.tracks) == 1
    assert stub.tracks[0] is tr
    assert tr.label == "forklift"
    assert tr.hits == 2


# ---------------------------------------------------------------------------
# Upgrade path (lidar-fusion probe follow-up, 2026-09-09): a lidar-spawned
# "unknown" track can now be claimed by a labeled detection of ANY category
# in the labeled pass, instead of coexisting with a second, separately
# spawned labeled track for the same physical object.
# ---------------------------------------------------------------------------

def test_upgrade_track_reseeds_p_movable_when_still_at_unknown_prior():
    """Direct _upgrade_track unit test: id/position/velocity/p_motion are
    untouched (the caller's tr.update() already ran the KF measurement
    update before this is called); p_movable == p_movable_prior (no motion
    evidence yet) gets re-seeded to the NEW category's prior."""
    stub = make_tracker()
    tr = Track(x=1.0, y=2.0, label="lidar_cluster", score=0.5, cov_xy=0.02,
               size=0.3, stamp=0.0, p_movable_prior=stub.prior_unknown)
    tr.p_movable = stub.prior_unknown   # untouched by motion -- still at the seed prior
    tr.x[2], tr.x[3] = 0.4, -0.1        # arbitrary pre-existing velocity
    tr.p_motion = 0.8
    track_id = tr.id

    stub._upgrade_track(tr, {"label": "mobile robot", "score": 0.9})

    assert tr.id == track_id
    assert tr.label == "mobile robot"
    assert label_category(tr.label) == "robot"
    assert tr.p_movable == pytest.approx(stub.category_priors["robot"])
    assert tr.p_movable_prior == pytest.approx(stub.category_priors["robot"])
    assert tr.p_motion == pytest.approx(0.8)                 # untouched
    assert tr.velocity == pytest.approx((0.4, -0.1))          # untouched
    assert tr.position == pytest.approx((1.0, 2.0))           # untouched


def test_upgrade_track_keeps_p_movable_if_motion_already_raised_it():
    """Once motion evidence has already pushed p_movable above the unknown
    prior it was seeded with, an upgrade must NOT regress it back down to a
    lower category prior."""
    stub = make_tracker()
    tr = Track(x=0.0, y=0.0, label="lidar_cluster", score=0.5, cov_xy=0.02,
               size=0.3, stamp=0.0, p_movable_prior=stub.prior_unknown)
    tr.p_movable = 0.75   # motion evidence already raised this above prior_unknown

    stub._upgrade_track(tr, {"label": "chair", "score": 0.5})  # label_category -> "furniture"

    assert tr.label == "chair"
    assert tr.p_movable == pytest.approx(0.75)                          # kept
    assert tr.p_movable_prior == pytest.approx(stub.category_priors["furniture"])


def test_labeled_detection_upgrades_lidar_spawned_unknown_track():
    stub = make_tracker()
    stub._now_value[0] = 0.0
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 3.0, "y": 0.0,
                           "covariance0": 0.025, "size": 0.3}])
    stub._detections_cb(lidar_msg)
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    track_id = tr.id
    assert tr.label == "lidar_cluster"
    assert label_category(tr.label) == "unknown"

    # "mobile robot" 0.3 m away claims the unknown track -- same id, not a
    # second track.
    stub._now_value[0] = 0.2
    cam_msg = make_msg([{"label": "mobile robot", "score": 0.9, "x": 3.3, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)

    assert len(stub.tracks) == 1
    assert stub.tracks[0] is tr
    assert tr.id == track_id
    assert tr.label == "mobile robot"
    assert label_category(tr.label) == "robot"


def test_labeled_detection_prefers_same_category_track_over_upgrade_candidate():
    """When both an existing same-category track and a lidar-spawned unknown
    track sit within gate of an incoming labeled detection, the labeled pass
    must prefer the same-category match (upgrade_penalty_m) -- the unknown
    track is left alone (not upgraded, not touched)."""
    stub = make_tracker()
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "mobile robot", "score": 0.9, "x": 5.0, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    robot_tr = stub.tracks[0]

    # Far enough from robot_tr (0.9 m > the 0.6 m agnostic gate at t=0) that
    # this lidar detection spawns its OWN unknown track instead of updating
    # robot_tr.
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 5.9, "y": 0.0,
                           "covariance0": 0.025, "size": 0.3}])
    stub._detections_cb(lidar_msg)
    assert len(stub.tracks) == 2
    unknown_tr = next(t for t in stub.tracks if t is not robot_tr)
    assert unknown_tr.label == "lidar_cluster"

    # 0.45 m from BOTH tracks' current positions -- same-category cost 0.45
    # vs. upgrade cost 0.45 + upgrade_penalty_m(0.3) = 0.75.
    stub._now_value[0] = 0.1
    cam_msg2 = make_msg([{"label": "mobile robot", "score": 0.9, "x": 5.45, "y": 0.0,
                          "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg2)

    assert len(stub.tracks) == 2
    assert robot_tr.hits == 2          # matched by the same-category track
    assert unknown_tr.hits == 1        # untouched
    assert unknown_tr.label == "lidar_cluster"   # not upgraded


def test_lidar_cluster_prefers_labeled_track_over_unknown_track_when_equidistant():
    """_associate_agnostic companion: an unknown-category track's cost
    carries unknown_penalty_m, so a lidar cluster equidistant from a labeled
    track and an unknown one updates the labeled track."""
    stub = make_tracker()
    stub._now_value[0] = 0.0
    cam_msg = make_msg([{"label": "mobile robot", "score": 0.9, "x": 5.0, "y": 0.0,
                         "covariance0": 0.10, "size": 0.5}])
    stub._detections_cb(cam_msg)
    robot_tr = stub.tracks[0]

    other_lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 6.0, "y": 0.0,
                                 "covariance0": 0.025, "size": 0.3}])
    stub._detections_cb(other_lidar_msg)
    assert len(stub.tracks) == 2
    unknown_tr = next(t for t in stub.tracks if t is not robot_tr)

    # (5.5, 0) is exactly 0.5 m from both robot_tr (5.0, 0) and unknown_tr
    # (6.0, 0).
    stub._now_value[0] = 0.1
    lidar_msg = make_msg([{"label": "lidar_cluster", "score": 0.5, "x": 5.5, "y": 0.0,
                           "covariance0": 0.02, "size": 0.3}])
    stub._detections_cb(lidar_msg)

    assert len(stub.tracks) == 2
    assert robot_tr.hits == 2          # the labeled track absorbed the lidar hit
    assert unknown_tr.hits == 1        # unchanged
