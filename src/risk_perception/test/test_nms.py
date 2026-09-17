"""
Pure-Python tests for nms.py -- no camera, no sim, no GDINO, no ROS. Runs in
a fraction of a second with plain pytest, same rationale as
test_relation_matching.py / test_encounter_geometry.py / test_mask_relation.py.
"""

from risk_perception.nms import class_aware_nms


def test_empty_input_returns_empty():
    assert class_aware_nms([], [], [], 0.5) == []


def test_single_box_is_kept():
    boxes = [(0.0, 0.0, 10.0, 10.0)]
    assert class_aware_nms(boxes, [0.9], ["person"], 0.5) == [0]


def test_non_overlapping_same_label_both_kept():
    boxes = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    result = class_aware_nms(boxes, [0.9, 0.8], ["person", "person"], 0.5)
    assert result == [0, 1]


def test_overlapping_same_label_keeps_only_higher_score():
    """The exact duplicate-detection case this module exists for: GDINO
    fires twice on one physical person."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (0.5, 0.5, 10.5, 10.5)]
    result = class_aware_nms(boxes, [0.6, 0.9], ["person", "person"], 0.5)
    assert result == [1]  # index 1 has the higher score (0.9)


def test_overlapping_different_labels_both_kept():
    """A person sitting on a chair: near-identical boxes, but this is a
    real simultaneous detection, not a duplicate -- must NOT suppress."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)]
    result = class_aware_nms(boxes, [0.9, 0.85], ["person", "chair"], 0.5)
    assert result == [0, 1]


def test_label_comparison_is_normalized():
    """'Person' vs 'person .' must still be treated as the same label for
    suppression -- GDINO's raw phrase output isn't consistently cased or
    trimmed, see relation_matching.normalize_phrase."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (0.5, 0.5, 10.5, 10.5)]
    result = class_aware_nms(boxes, [0.6, 0.9], ["Person", "person . "], 0.5)
    assert result == [1]


def test_three_overlapping_same_label_keeps_only_the_best():
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (0.3, 0.3, 10.3, 10.3),
        (0.6, 0.6, 10.6, 10.6),
    ]
    scores = [0.5, 0.95, 0.7]
    result = class_aware_nms(boxes, scores, ["cart", "cart", "cart"], 0.3)
    assert result == [1]


def test_iou_exactly_at_threshold_is_suppressed():
    """box_iou >= iou_threshold suppresses -- boundary is inclusive."""
    # Two 10x10 boxes overlapping in a 5x10 strip: IoU = 50 / 150 = 1/3,
    # see test_relation_matching.test_box_iou_known_overlap for the same
    # geometry's IoU value.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    result = class_aware_nms([a, b], [0.9, 0.8], ["table", "table"], 1.0 / 3.0)
    assert result == [0]


def test_below_threshold_both_kept():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    result = class_aware_nms([a, b], [0.9, 0.8], ["table", "table"], 0.5)
    assert result == [0, 1]


def test_result_indices_are_sorted_in_original_order():
    """Keep-list order is input order, not score order -- callers zip it
    back against their own parallel arrays (boxes/scores/labels), which
    would silently misalign if this returned score-sorted indices."""
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (100.0, 100.0, 110.0, 110.0),
        (200.0, 200.0, 210.0, 210.0),
    ]
    scores = [0.5, 0.99, 0.7]  # highest score is NOT first
    result = class_aware_nms(boxes, scores, ["chair", "chair", "chair"], 0.5)
    assert result == [0, 1, 2]
