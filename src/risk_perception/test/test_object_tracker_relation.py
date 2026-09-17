"""
Direct test of Track.update_relation's rise/decay math -- no ROS graph, no
scenario_publisher. scenario_publisher.py cannot exercise this code path at
all: it publishes straight to world_objects (object_tracker's OUTPUT), so
Track.update_relation (which lives strictly upstream, inside object_tracker
itself) never runs in that data path. This is a plain instantiation of the
Track class instead -- needs rclpy (for the ROS message types Track's
neighbors use) but not torch, so it runs in a system-python environment
with no GPU and no model checkpoints, same as test_relation_matching.py.
"""

import math

from risk_perception.object_tracker_node import Track


def make_track():
    return Track(x=0.0, y=0.0, label="forklift", score=0.9, cov_xy=0.04,
                 size=0.5, stamp=0.0, p_movable_prior=0.7)


def test_relation_bonus_starts_at_zero():
    tr = make_track()
    assert tr.relation_bonus == 0.0


def test_no_hit_ever_stays_zero():
    tr = make_track()
    for _ in range(20):
        tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
    assert tr.relation_bonus == 0.0


def test_rise_moves_toward_relconf_not_a_hard_snap():
    """rise=1.0 would jump straight to relconf; rise<1.0 must not."""
    tr = make_track()
    tr.last_relconf = 0.8
    tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
    assert 0.0 < tr.relation_bonus < 0.8
    assert abs(tr.relation_bonus - 0.48) < 1e-9  # 0 + 0.6*(0.8-0) = 0.48


def test_rise_converges_toward_target_over_repeated_hits():
    tr = make_track()
    prev = 0.0
    for _ in range(10):
        tr.last_relconf = 0.8  # re-armed each tick, as update() would do on a hit
        tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
        assert tr.relation_bonus > prev  # strictly increasing toward the target
        assert tr.relation_bonus <= 0.8 + 1e-9
        prev = tr.relation_bonus
    assert abs(tr.relation_bonus - 0.8) < 0.01  # converged close to target


def test_decay_matches_p_motion_decay_shape():
    """Same exp(-rate*dt) shape as p_motion_decay elsewhere in this file --
    after 1s at decay=1.0, ~37% (1/e) should remain."""
    tr = make_track()
    tr.relation_bonus = 1.0
    for _ in range(10):  # 10 * 0.1s = 1.0s, no hits in between
        tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
    assert abs(tr.relation_bonus - math.exp(-1.0)) < 1e-6


def test_hit_consumes_last_relconf_exactly_once():
    """A hit should only apply for the next tick, not linger and re-apply
    on a later tick with no new detection."""
    tr = make_track()
    tr.last_relconf = 0.8
    tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
    after_hit = tr.relation_bonus
    assert tr.last_relconf == 0.0  # consumed
    tr.update_relation(rise=0.6, decay=1.0, dt=0.1)  # no new hit -> must decay
    assert tr.relation_bonus < after_hit


def test_relation_bonus_stays_within_zero_one():
    tr = make_track()
    for _ in range(50):
        tr.last_relconf = 1.5  # out-of-range input should still clamp the result
        tr.update_relation(rise=0.6, decay=1.0, dt=0.1)
        assert 0.0 <= tr.relation_bonus <= 1.0
