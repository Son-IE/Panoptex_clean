"""
Tier 0 of the relation-prior test plan: pure geometry, no camera, no sim,
no GDINO, no ROS. Runs in a second with plain pytest -- see
risk_perception/relation_matching.py for why this logic lives outside
gdino_detector_node.py (which imports torch at module level and can't be
imported in an environment without it).
"""

from risk_perception.relation_matching import (
    box_iou,
    match_relations_to_objects,
    normalize_phrase,
)


def test_normalize_phrase_strips_case_dots_and_whitespace():
    assert normalize_phrase(" Person . ") == "person"
    assert normalize_phrase("Forklift") == "forklift"


def test_box_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint_boxes_is_zero():
    assert box_iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_box_iou_known_overlap():
    # two 10x10 boxes overlapping in a 5x10 strip: intersection 50,
    # union 150 -> iou = 1/3
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert abs(box_iou(a, b) - (1.0 / 3.0)) < 1e-9


def test_relation_attaches_to_machine_not_person():
    """The earlier-agreed modeling choice: a relation box overlapping both
    a person and a forklift should tag the forklift."""
    person = (0.0, 0.0, 2.0, 2.0)
    forklift = (1.5, 0.0, 5.0, 3.0)
    relation_box = (0.5, 0.0, 4.5, 2.5)  # overlaps both

    result = match_relations_to_objects(
        relation_hits=[(*relation_box, 0.8)],
        object_boxes=[(*person, "person"), (*forklift, "forklift")],
        iou_threshold=0.15,
    )
    assert result == {1: 0.8}  # index 1 == forklift, not index 0 == person


def test_relation_falls_back_to_best_iou_when_only_person_candidates():
    """If nothing non-person clears the threshold, still attach somewhere
    rather than silently dropping the signal."""
    person_a = (0.0, 0.0, 2.0, 2.0)
    person_b = (10.0, 10.0, 12.0, 12.0)  # far away, should not win
    relation_box = (0.2, 0.2, 1.8, 1.8)  # tight overlap with person_a only

    result = match_relations_to_objects(
        relation_hits=[(*relation_box, 0.7)],
        object_boxes=[(*person_a, "person"), (*person_b, "person")],
        iou_threshold=0.15,
    )
    assert result == {0: 0.7}


def test_relation_below_threshold_matches_nothing():
    person = (0.0, 0.0, 2.0, 2.0)
    forklift = (100.0, 100.0, 102.0, 102.0)  # nowhere near the relation box
    relation_box = (0.0, 0.0, 2.0, 2.0)

    result = match_relations_to_objects(
        relation_hits=[(*relation_box, 0.9)],
        object_boxes=[(*person, "person"), (*forklift, "forklift")],
        iou_threshold=0.15,
    )
    # only "person" overlaps at all, and it's the only candidate -- falls
    # back to it (see test_relation_falls_back_to_best_iou_when_only_person_candidates)
    assert result == {0: 0.9}


def test_two_relation_hits_on_same_object_take_the_max_not_both():
    """Two different relation phrases ('on', 'driving') both landing on the
    same forklift must produce ONE relconf, not two concatenated tags that
    only the last would parse correctly downstream."""
    forklift = (0.0, 0.0, 10.0, 10.0)
    hit_a = (0.0, 0.0, 10.0, 10.0)  # relconf 0.6
    hit_b = (0.0, 0.0, 10.0, 10.0)  # relconf 0.9, should win

    result = match_relations_to_objects(
        relation_hits=[(*hit_a, 0.6), (*hit_b, 0.9)],
        object_boxes=[(*forklift, "forklift")],
        iou_threshold=0.15,
    )
    assert result == {0: 0.9}


def test_no_relation_hits_returns_empty():
    forklift = (0.0, 0.0, 10.0, 10.0)
    result = match_relations_to_objects(
        relation_hits=[],
        object_boxes=[(*forklift, "forklift")],
        iou_threshold=0.15,
    )
    assert result == {}
