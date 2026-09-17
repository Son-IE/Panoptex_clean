"""
WP5 tests for WP1's displacement-based motion evidence (object_tracker_node
.py's module docstring, "2026-09-10 phantom velocity"): a parked object's
Kalman velocity is mostly inter-camera jitter, so `Track.update_beliefs`'s
`moving` decision -- by default (`motion_require_displacement: true`) --
comes from real net displacement over a measurement window
(`risk_perception.motion_evidence.displacement_evidence`), not the filter's
own instantaneous velocity estimate.

Same style as test_object_tracker_motion.py: drives `Track`'s own methods
directly (predict/update/update_beliefs), or binds `ObjectTrackerNode`'s
pure/instance methods onto a plain stub, so none of this needs rclpy.init()
or a real node.
"""
import math
import types

from risk_perception.object_tracker_node import ObjectTrackerNode, Track


def _make_gate_stub(gate=0.6, gate_speed=1.0, gate_max=2.5,
                    association_key="category",
                    association_groups=("robot,wheeled",)):
    stub = types.SimpleNamespace(
        association_key=association_key, gate=gate, gate_speed=gate_speed,
        gate_max_m=gate_max,
        association_groups=[
            frozenset(s.strip().lower() for s in g.split(",") if s.strip())
            for g in association_groups
        ])
    stub._same_object_class = types.MethodType(
        ObjectTrackerNode._same_object_class, stub)
    stub._in_same_group = types.MethodType(ObjectTrackerNode._in_same_group, stub)
    stub._gate_for = types.MethodType(ObjectTrackerNode._gate_for, stub)
    return stub


def test_camera_jitter_zigzag_never_triggers_motion():
    """Parked object as seen by two overhead cams whose ground-plane
    calibration disagrees by ~0.3 m: the reported centroid alternates
    between the two estimates at 2 Hz for 20 s. Net displacement over any
    2 s window stays ~0 under the default "half_median" statistic (each
    half's coordinate-wise median lands on whichever side has more samples
    in that half, and an even split cancels outright) -- see
    test_two_source_offset_parked_object_does_not_fire below for the
    fused TWO-SOURCE-at-different-rates case this was generalised from."""
    tr = Track(x=0.3, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    hz = 2.0
    next_obs = 1.0 / hz
    t = 0.0
    toggle = True
    while t < 20.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_obs:
            x = 0.3 if toggle else -0.3
            toggle = not toggle
            tr.update(x, 0.0, 0.04, 0.9, 0.5, t)
            next_obs += 1.0 / hz
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
    # An early transient (before the window fills with a full symmetric
    # alternating set) may trigger `moving` for a tick or two; p_motion
    # decays exponentially afterward and never reaches EXACTLY 0.0 in
    # floating point, so this checks "negligible by the end of a 20 s
    # run" rather than bit-exact zero.
    assert tr.p_motion < 1e-6
    assert math.hypot(*tr.disp_v) < 0.15


def test_two_source_offset_parked_object_does_not_fire():
    """Coordinator follow-up (2026-09-10): the real bug wasn't a single
    source's jitter, it was TWO sources -- a 10 Hz lidar-cluster stream
    and a slower ~2 Hz camera stream -- with a PERSISTENT (not jittery)
    ~0.3 m ground-plane offset between them, fused into the same track.
    Under "half_median" (default), each half's coordinate-wise median is
    dominated by whichever source contributes more samples to that half
    (5x more lidar than camera points per half here), so both halves'
    medians land on the lidar side and net displacement stays ~0 --
    unlike the old "endpoints" statistic, which read this as a genuine
    net displacement plus (depending on sample order) a confusing
    straightness ratio. See test_two_source_fused_mover_fires_within_1p5s
    below for the same two sources on a track that is ACTUALLY moving."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    hz_lidar, hz_cam = 10.0, 2.0
    next_lidar, next_cam = 1.0 / hz_lidar, 1.0 / hz_cam
    t = 0.0
    while t < 20.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_lidar:
            tr.update(0.15, 0.0, 0.02, 0.9, 0.5, t)    # lidar: +0.15 m offset
            next_lidar += 1.0 / hz_lidar
        if t + 1e-9 >= next_cam:
            tr.update(-0.15, 0.0, 0.04, 0.9, 0.5, t)   # camera: -0.15 m offset
            next_cam += 1.0 / hz_cam
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
    assert tr.p_motion < 1e-6
    assert math.hypot(*tr.disp_v) < 0.15


def test_two_source_fused_mover_fires_within_1p5s():
    """Coordinator follow-up (2026-09-10): a genuine 0.6 m/s straight-line
    mover seen by BOTH the same two sources as the parked-object test
    above (10 Hz lidar at +0.15 m, ~2 Hz camera at -0.15 m, riding on top
    of the real motion) must still read as moving within 1.5 s -- the
    per-source offsets cancel out of the half-to-half median comparison
    the same way they do for a parked object, leaving the real net
    displacement intact."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    speed = 0.6
    hz_lidar, hz_cam = 10.0, 2.0
    next_lidar, next_cam = 1.0 / hz_lidar, 1.0 / hz_cam
    t = 0.0
    became_moving_at = None
    while t < 6.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_lidar:
            tr.update(speed * t + 0.15, 0.0, 0.02, 0.9, 0.5, t)
            next_lidar += 1.0 / hz_lidar
        if t + 1e-9 >= next_cam:
            tr.update(speed * t - 0.15, 0.0, 0.04, 0.9, 0.5, t)
            next_cam += 1.0 / hz_cam
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
        if became_moving_at is None and tr.p_motion == 1.0:
            became_moving_at = t
    assert became_moving_at is not None
    assert became_moving_at <= 1.5


def test_lidar_only_track_with_small_jitter_stays_static():
    """10 Hz lidar cluster, +-0.05 m (0.1 m peak-to-peak) jitter -- well
    under motion_min_displacement_m (0.3 m default), so this never trips
    even though it updates every tick."""
    tr = Track(x=0.05, y=0.0, label="lidar_cluster", score=0.5, cov_xy=0.02,
               size=0.3, stamp=0.0, p_movable_prior=0.3)
    dt_tick = 0.1
    t = 0.0
    toggle = True
    while t < 10.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        x = 0.05 if toggle else -0.05
        toggle = not toggle
        tr.update(x, 0.0, 0.02, 0.5, 0.3, t)
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
    assert tr.p_motion == 0.0


def test_displacement_evidence_fires_within_1p5s_of_real_motion_onset():
    """Parked 5 s, then a genuine 0.6 m/s mover sampled at 1 Hz: p_motion
    must reach 1.0 within 1.5 s of the object actually starting to move."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    next_obs = 1.0
    t = 0.0
    motion_onset = 5.0
    became_moving_at = None
    while t < 8.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_obs:
            x = 0.6 * max(0.0, t - motion_onset)
            tr.update(x, 0.0, 0.04, 0.9, 0.5, t)
            next_obs += 1.0
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
        if became_moving_at is None and tr.p_motion == 1.0:
            became_moving_at = t
    assert became_moving_at is not None
    assert became_moving_at - motion_onset <= 1.5


def test_displacement_evidence_holds_through_a_2p5s_gap():
    """A track observed moving at 0.6 m/s then unseen for 2.5 s (predict()
    only, no update() calls -- an overhead-camera hand-over gap) must still
    read p_motion == 1 throughout, via motion_hold_s (default 3.0 s)."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.5, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    dt_tick = 0.1
    speed = 0.6
    t = 0.0
    next_obs = 1.0
    while t < 4.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_obs:
            tr.update(speed * t, 0.0, 0.04, 0.5, 0.5, t)
            next_obs += 1.0
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
    assert tr.p_motion == 1.0

    unseen_for = 0.0
    while unseen_for < 2.5:
        unseen_for += dt_tick
        tr.predict(dt_tick, 0.05)
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=4.0 + unseen_for,
                          require_displacement=True)
    assert tr.p_motion == 1.0, "should still read as moving through a 2.5 s gap"


def test_velocity_damping_settles_kf_speed_within_2s_of_stopping():
    """Once no further displacement evidence supports "moving" (an empty
    self.meas here stands in for an object that has genuinely stopped and
    stopped generating fresh measurements), p_motion decays below
    motion_vel_damp_below and the Kalman velocity STATE itself is damped
    toward zero -- so predict() stops walking a parked track forward on a
    stale jitter velocity. Directly seeds a "was moving" Track (p_motion=1,
    KF velocity 0.6 m/s) rather than replaying a full detection stream --
    this isolates the damping mechanism itself, already covered end-to-end
    by the hold-through-a-gap test above."""
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    tr.p_motion = 1.0
    tr.x[2], tr.x[3] = 0.6, 0.0
    dt_tick = 0.1
    t = 0.0
    while t < 2.0:
        t += dt_tick
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True,
                          motion_vel_damp_below=0.5, motion_vel_damp_tau_s=0.5)
    assert tr.speed < 0.05


def test_gate_for_caps_at_gate_max_after_60s():
    stub = _make_gate_stub(gate=0.6, gate_speed=1.0, gate_max=2.5)
    tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.9)
    assert stub._gate_for(tr, now=60.0) == 2.5


def test_revive_rejects_a_20m_match():
    """_revive must not resurrect a graveyard track for a detection 20 m
    away, however long it has been unseen -- gate_max_m (2.5 m default)
    caps _gate_for, and _revive uses the same method (see
    object_tracker_node.py's module docstring)."""
    stub = types.SimpleNamespace(
        association_key="category", gate=0.6, gate_speed=1.0, gate_max_m=2.5,
        association_groups=[frozenset({"robot", "wheeled"})],
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    stub._same_object_class = types.MethodType(
        ObjectTrackerNode._same_object_class, stub)
    stub._in_same_group = types.MethodType(ObjectTrackerNode._in_same_group, stub)
    stub._gate_for = types.MethodType(ObjectTrackerNode._gate_for, stub)
    stub._revive = types.MethodType(ObjectTrackerNode._revive, stub)

    graveyard_tr = Track(x=0.0, y=0.0, label="mobile robot", score=0.9,
                         cov_xy=0.04, size=0.5, stamp=0.0, p_movable_prior=0.9)
    # Unseen for a long time -- even at gate_speed=1.0 this would be a huge
    # gate without the cap (0.6 + 1.0*100 = 100.6 m); with the cap it's 2.5 m.
    stub._graveyard = [graveyard_tr]

    d = {"x": 20.0, "y": 0.0, "label": "mobile robot", "score": 0.9,
         "cov": 0.04, "size": 0.5, "relconf": 0.0}
    assert stub._revive(d, now=100.0) is None
    assert len(stub._graveyard) == 1   # untouched -- nothing was popped


# ---------------------------------------------------------------------------
# WP1 "motion overrides class" (2026-09-10 follow-up, unit2_pan_2 finding):
# Track.update_promotion / published_label. See object_tracker_node.py's
# module docstring for the full motivation.
# ---------------------------------------------------------------------------

def test_furniture_labelled_mover_promotes_and_publishes_moving_object():
    """A "table"-labelled track that is actually moving in a straight line
    at 0.6 m/s must eventually get promoted (p_motion >= mover_promote_pmot_min
    continuously for mover_promote_min_s, with real displacement and enough
    hits), and its published class_id label must become "moving object" --
    NOT its raw "table" label."""
    tr = Track(x=0.0, y=0.0, label="table", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.1)
    dt_tick = 0.1
    speed = 0.6
    hz = 10.0
    next_obs = 1.0 / hz
    t = 0.0
    promoted_at = None
    while t < 6.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_obs:
            tr.update(speed * t, 0.0, 0.02, 0.9, 0.5, t)
            next_obs += 1.0 / hz
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
        tr.update_promotion(t, True, 0.8, 1.0, 0.5, 5, 60.0)
        if promoted_at is None and tr.promoted:
            promoted_at = t
    assert promoted_at is not None
    # p_motion itself needs up to ~1.5 s to reach the promote_pmot_min
    # streak start (see test_displacement_evidence_fires_within_1p5s_of_
    # real_motion_onset), plus mover_promote_min_s (1.0 s) of that streak
    # holding continuously before promotion fires -- ~2.5 s total is the
    # expected order of magnitude, comfortably under half this loop's 6 s.
    assert promoted_at <= 3.0
    assert tr.label == "table"                                  # raw label untouched
    assert tr.published_label("moving object") == "moving object"


def test_jittery_parked_table_never_promotes():
    """The same two-source (+-0.15 m lidar/camera offset, 10 Hz + 2 Hz)
    parked-object scenario as test_two_source_offset_parked_object_does_
    not_fire, run through update_promotion too -- net_disp_m never clears
    mover_promote_min_disp_m (0.5 m), so promotion never fires regardless
    of whatever p_motion transiently does."""
    tr = Track(x=0.0, y=0.0, label="table", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.1)
    dt_tick = 0.1
    hz_lidar, hz_cam = 10.0, 2.0
    next_lidar, next_cam = 1.0 / hz_lidar, 1.0 / hz_cam
    t = 0.0
    while t < 20.0:
        t += dt_tick
        tr.predict(dt_tick, 0.05)
        if t + 1e-9 >= next_lidar:
            tr.update(0.15, 0.0, 0.02, 0.9, 0.5, t)
            next_lidar += 1.0 / hz_lidar
        if t + 1e-9 >= next_cam:
            tr.update(-0.15, 0.0, 0.04, 0.9, 0.5, t)
            next_cam += 1.0 / hz_cam
        tr.update_beliefs(threshold=3.0, p_motion_decay=3.0, movable_gain=0.5,
                          movable_decay=0.01, dt=dt_tick, now=t,
                          require_displacement=True)
        tr.update_promotion(t, True, 0.8, 1.0, 0.5, 5, 60.0)
    assert tr.promoted is False
    assert tr.published_label("moving object") == "table"


def test_promoted_track_demotes_after_mover_demote_s_of_no_motion():
    """Sticky demotion: a promoted track must stay promoted while it is
    still within mover_demote_s of its last p_motion >= 0.5 tick, and
    revert once that window has fully elapsed with no further motion."""
    tr = Track(x=0.0, y=0.0, label="table", score=0.9, cov_xy=0.04,
               size=0.5, stamp=0.0, p_movable_prior=0.1)
    tr.promoted = True
    tr.p_motion = 0.0
    tr._last_motion_ge_half = 0.0   # last seen moving at t=0
    demote_s = 5.0
    dt_tick = 0.1
    t = 0.0
    while t < demote_s - 0.5:
        t += dt_tick
        tr.update_promotion(t, True, 0.8, 1.0, 0.5, 5, demote_s)
    assert tr.promoted is True, "must stay promoted (sticky) before demote_s elapses"
    while t < demote_s + 1.0:
        t += dt_tick
        tr.update_promotion(t, True, 0.8, 1.0, 0.5, 5, demote_s)
    assert tr.promoted is False, "must demote once demote_s of no motion has passed"


def _make_promotion_association_stub():
    stub = types.SimpleNamespace(
        association_key="category",
        gate=0.6, gate_speed=1.0, gate_max_m=2.5,
        meas_time_corr=False, max_meas_age=3.0, _meas_ages=[],
        motion_window_s=2.0,
        label_agnostic_labels=set(),
        upgrade_penalty_m=0.3, unknown_penalty_m=0.3,
        association_groups=[frozenset({"robot", "wheeled"})],
        category_priors={"person": 0.9, "robot": 0.9, "wheeled": 0.7, "furniture": 0.1},
        prior_unknown=0.3,
        process_noise=0.05, default_cov=0.04,
        tracks=[], _graveyard=[], last_predict=None,
        _now_value=[0.0],
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    stub._now = lambda: stub._now_value[0]
    for name in ("_detections_cb", "_associate_agnostic", "_assign",
                "_same_object_class", "_in_same_group", "_gate_for",
                "_spawn", "_spawn_agnostic", "_revive",
                "_prior_for", "_predict_to", "_measurement_age", "_upgrade_track"):
        setattr(stub, name, types.MethodType(getattr(ObjectTrackerNode, name), stub))
    for name in ("_greedy_assign", "_adopt_label"):
        setattr(stub, name, getattr(ObjectTrackerNode, name))
    return stub


def _make_table_msg(x, y):
    covariance = [0.0] * 36
    covariance[0] = 0.04
    det = types.SimpleNamespace(
        results=[types.SimpleNamespace(
            hypothesis=types.SimpleNamespace(class_id="table", score=0.9),
            pose=types.SimpleNamespace(covariance=covariance))],
        bbox=types.SimpleNamespace(
            center=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y)),
            size=types.SimpleNamespace(x=0.5, y=0.5)))
    return types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0)),
        detections=[det])


def test_table_detection_still_associates_with_promoted_track():
    """WP1 "motion overrides class": promotion changes only the PUBLISHED
    label (published_label) -- self.label/category stay "table", so a
    later "table" detection keeps associating with the SAME track id
    instead of spawning a duplicate once the track is promoted."""
    stub = _make_promotion_association_stub()
    stub._now_value[0] = 0.0
    stub._detections_cb(_make_table_msg(2.0, 0.0))
    assert len(stub.tracks) == 1
    tr = stub.tracks[0]
    tid = tr.id

    # Force promotion directly -- the promotion mechanics themselves are
    # covered by the tests above; this test is only about association.
    tr.p_motion = 0.9
    tr.net_disp_m = 0.6
    tr.hits = 10
    tr.promoted = True
    assert tr.label == "table"
    assert tr.published_label("moving object") == "moving object"

    stub._now_value[0] = 1.0
    stub._detections_cb(_make_table_msg(2.05, 0.0))
    assert len(stub.tracks) == 1, "the table detection must associate, not re-spawn"
    assert stub.tracks[0].id == tid
    assert stub.tracks[0].label == "table"       # raw label still untouched
    assert stub.tracks[0].promoted is True        # promotion state survived the update
