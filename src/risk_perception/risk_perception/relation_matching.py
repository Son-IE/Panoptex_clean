#!/usr/bin/env python3
"""
relation_matching.py  --  pure geometry for the relation prior (proposed)

No ROS, no torch, no GDINO import -- deliberately separate from
gdino_detector_node.py (which imports torch/groundingdino at module level)
so this logic can be unit-tested without a GPU, a model checkpoint, or even
rclpy installed. See src/risk_perception/test/test_relation_matching.py.

Matches GroundingDINO's relation-phrase detections ("person on forklift")
against this same frame's object-phrase detections ("person", "forklift"),
in 2D image space, and decides which single object each relation hit
belongs to.
"""

from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def normalize_phrase(text: str) -> str:
    return str(text).lower().strip().strip(".").strip()


def box_iou(a: Box, b: Box) -> float:
    """Standard axis-aligned intersection-over-union."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_relations_to_objects(
    relation_hits: List[Tuple[float, float, float, float, float]],
    object_boxes: List[Tuple[float, float, float, float, str]],
    iou_threshold: float,
) -> Dict[int, float]:
    """
    relation_hits: list of (x1, y1, x2, y2, score).
    object_boxes: list of (x1, y1, x2, y2, normalized_label), index-aligned
                  with whatever the caller will tag by that same index.

    -> {object_index: relconf}, one entry per object that won at least one
    relation match, relconf = the MAX score across every relation hit that
    matched it (never multiple entries per object -- see gdino_detector_node.py
    for why concatenating would break downstream parsing).

    Preference rule: among candidates clearing iou_threshold, attach to the
    best-overlapping candidate whose label is NOT "person" (the operated
    machine, not the operator); fall back to best IoU if every candidate
    that cleared the threshold is a person.
    """
    best_relconf: Dict[int, float] = {}
    for rx1, ry1, rx2, ry2, rscore in relation_hits:
        candidates = []
        for obj_index, (ox1, oy1, ox2, oy2, obj_label) in enumerate(object_boxes):
            iou = box_iou((rx1, ry1, rx2, ry2), (ox1, oy1, ox2, oy2))
            if iou >= iou_threshold:
                candidates.append((obj_index, obj_label, iou))
        if not candidates:
            continue
        non_person = [c for c in candidates if c[1] != "person"]
        pool = non_person if non_person else candidates
        winner_index = max(pool, key=lambda c: c[2])[0]
        best_relconf[winner_index] = max(
            best_relconf.get(winner_index, 0.0), rscore)
    return best_relconf
