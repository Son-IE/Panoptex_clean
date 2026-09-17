"""
Tier 0 of the GDINO-free relation-prior fallback: pure geometry, no ROS, no
camera, no GPU, no GDINO. Runs in a second with plain pytest -- see
risk_perception/mask_relation.py for why this replaces relation-phrase
grounding (empirically found not to work) with segmentation containment.
"""

import numpy as np

from risk_perception.mask_relation import (
    best_operator_match,
    dilate_mask,
    mask_containment_ratio,
    tag_operator_relations,
    within_3d_proximity,
)


class FakeHypothesis:
    """Duck-typed stand-in for vision_msgs' ObjectHypothesisWithPose.hypothesis
    -- only needs a mutable .class_id, same as tag_operator_relations expects."""

    def __init__(self, class_id):
        self.class_id = class_id


def rect_mask(shape, r0, r1, c0, c1):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_person_fully_inside_forklift_mask_is_full_containment():
    """The core case: a small person mask entirely inside a much larger
    forklift mask -- symmetric IoU would be tiny, containment must be 1.0."""
    shape = (100, 100)
    forklift = rect_mask(shape, 10, 90, 10, 90)   # big
    person = rect_mask(shape, 40, 60, 40, 60)     # small, fully inside
    ratio = mask_containment_ratio(person, forklift)
    assert ratio == 1.0


def test_disjoint_masks_have_zero_containment():
    shape = (100, 100)
    forklift = rect_mask(shape, 0, 20, 0, 20)
    person = rect_mask(shape, 80, 100, 80, 100)
    assert mask_containment_ratio(person, forklift) == 0.0


def test_partial_overlap_containment_is_fraction_of_inner_only():
    """Half the person's mask overlaps the machine's -- containment should
    read ~0.5 regardless of how much bigger the machine mask is."""
    shape = (100, 100)
    machine = rect_mask(shape, 0, 100, 0, 50)       # left half of the image
    person = rect_mask(shape, 40, 60, 40, 60)       # straddles the boundary at col 50
    ratio = mask_containment_ratio(person, machine)
    assert abs(ratio - 0.5) < 1e-9


def test_empty_person_mask_is_zero_not_a_crash():
    shape = (50, 50)
    empty = np.zeros(shape, dtype=bool)
    machine = rect_mask(shape, 0, 50, 0, 50)
    assert mask_containment_ratio(empty, machine) == 0.0


def test_dilation_closes_a_small_gap():
    """Person standing just outside the machine's mask by a couple pixels --
    should read zero with no dilation, positive once dilated enough."""
    shape = (50, 50)
    machine = rect_mask(shape, 0, 20, 0, 50)
    person = rect_mask(shape, 22, 30, 20, 30)  # 2px gap at row 20-22
    assert mask_containment_ratio(person, machine, dilate_outer_px=0) == 0.0
    assert mask_containment_ratio(person, machine, dilate_outer_px=3) > 0.0


def test_dilate_mask_grows_by_exactly_the_margin():
    shape = (21, 21)
    m = np.zeros(shape, dtype=bool)
    m[10, 10] = True
    grown = dilate_mask(m, margin_px=2)
    # a diamond of Chebyshev... actually Manhattan radius 2 around (10,10)
    assert grown[10, 10]
    assert grown[8, 10] and grown[12, 10] and grown[10, 8] and grown[10, 12]
    assert not grown[7, 10]  # outside the radius


def test_dilate_zero_margin_is_a_noop():
    shape = (10, 10)
    m = rect_mask(shape, 2, 4, 2, 4)
    assert np.array_equal(dilate_mask(m, 0), m)


# ------------------------------------------------------------------ winner selection

def test_best_operator_match_picks_highest_containment():
    shape = (100, 100)
    person = rect_mask(shape, 40, 50, 40, 50)
    candidates = {
        1: rect_mask(shape, 0, 100, 0, 100),   # fully contains -> 1.0
        2: rect_mask(shape, 40, 46, 40, 50),   # partial overlap only
    }
    result = best_operator_match(person, candidates, containment_threshold=0.5)
    assert result is not None
    winner_index, ratio = result
    assert winner_index == 1
    assert ratio == 1.0


def test_best_operator_match_below_threshold_returns_none():
    shape = (100, 100)
    person = rect_mask(shape, 40, 50, 40, 50)
    candidates = {1: rect_mask(shape, 40, 43, 40, 50)}  # ~30% containment
    assert best_operator_match(person, candidates, containment_threshold=0.8) is None


def test_best_operator_match_no_candidates_returns_none():
    shape = (20, 20)
    person = rect_mask(shape, 5, 10, 5, 10)
    assert best_operator_match(person, {}, containment_threshold=0.5) is None


# ------------------------------------------------------------------ 3D gate

def test_within_3d_proximity_true_when_close():
    assert within_3d_proximity((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), max_distance_m=1.0)


def test_within_3d_proximity_false_when_far_despite_2d_overlap():
    """The case this gate exists for: person and machine silhouettes touch
    in the image but are actually meters apart in the real world."""
    assert not within_3d_proximity((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), max_distance_m=1.0)


# ------------------------------------------------------------------ orchestration

def test_tag_operator_relations_tags_the_winning_machine():
    shape = (50, 50)
    person_mask = rect_mask(shape, 20, 30, 20, 30)
    forklift_mask = rect_mask(shape, 0, 50, 0, 50)  # fully contains
    forklift_hyp = FakeHypothesis("forklift")
    person_entries = [(person_mask, (0.0, 0.0, 0.0))]
    machine_entries = [(forklift_mask, (0.3, 0.0, 0.0), forklift_hyp)]

    tag_operator_relations(
        person_entries, machine_entries,
        containment_threshold=0.5, dilate_outer_px=0, max_distance_m=2.0)

    assert forklift_hyp.class_id == "forklift|relconf=1.00"


def test_tag_operator_relations_leaves_untagged_when_too_far_in_3d():
    """Silhouettes overlap in 2D, but the 3D gate should block the tag."""
    shape = (50, 50)
    person_mask = rect_mask(shape, 20, 30, 20, 30)
    forklift_mask = rect_mask(shape, 0, 50, 0, 50)
    forklift_hyp = FakeHypothesis("forklift")
    person_entries = [(person_mask, (0.0, 0.0, 0.0))]
    machine_entries = [(forklift_mask, (10.0, 0.0, 0.0), forklift_hyp)]

    tag_operator_relations(
        person_entries, machine_entries,
        containment_threshold=0.5, dilate_outer_px=0, max_distance_m=2.0)

    assert forklift_hyp.class_id == "forklift"


def test_tag_operator_relations_leaves_untagged_below_containment_threshold():
    shape = (50, 50)
    person_mask = rect_mask(shape, 20, 30, 20, 30)
    cart_mask = rect_mask(shape, 20, 23, 20, 30)  # ~30% containment only
    cart_hyp = FakeHypothesis("cart")
    person_entries = [(person_mask, (0.0, 0.0, 0.0))]
    machine_entries = [(cart_mask, (0.1, 0.0, 0.0), cart_hyp)]

    tag_operator_relations(
        person_entries, machine_entries,
        containment_threshold=0.8, dilate_outer_px=0, max_distance_m=2.0)

    assert cart_hyp.class_id == "cart"


def test_tag_operator_relations_picks_best_among_multiple_machines():
    shape = (50, 50)
    person_mask = rect_mask(shape, 20, 30, 20, 30)
    near_but_partial = FakeHypothesis("cart")
    full_containment = FakeHypothesis("forklift")
    person_entries = [(person_mask, (0.0, 0.0, 0.0))]
    machine_entries = [
        (rect_mask(shape, 20, 24, 20, 30), (0.1, 0.0, 0.0), near_but_partial),
        (rect_mask(shape, 0, 50, 0, 50), (0.2, 0.0, 0.0), full_containment),
    ]

    tag_operator_relations(
        person_entries, machine_entries,
        containment_threshold=0.5, dilate_outer_px=0, max_distance_m=2.0)

    assert near_but_partial.class_id == "cart"
    assert full_containment.class_id == "forklift|relconf=1.00"
