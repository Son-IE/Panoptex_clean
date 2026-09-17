#!/usr/bin/env python3
"""
nms.py  --  pure non-max suppression for GDINO's own 2D detections

GroundingDINO has no NMS built in -- checked directly against the installed
package: Model.predict_with_caption() -> predict() (box/text threshold
filtering only) -> post_process_result() (box-format conversion only), no
suppression step anywhere in that chain. Left alone, one physical object
routinely gets two or three overlapping boxes in a single frame -- each one
then gets its own SAM2 mask, its own 3D projection, and object_tracker's
per-frame Hungarian assignment is one-to-one (see object_tracker_node.py),
so only ONE of the duplicates can match an existing track; the rest spawn
as brand-new tracks sitting right on top of the real one.

No ROS, no torch -- same rationale as relation_matching.py /
encounter_geometry.py / mask_relation.py: unit-testable without a GPU or a
model checkpoint. Reuses relation_matching.box_iou/normalize_phrase rather
than reimplementing IoU a second time.
"""

from typing import List, Sequence

from risk_perception.relation_matching import Box, box_iou, normalize_phrase


def class_aware_nms(
    boxes: Sequence[Box],
    scores: Sequence[float],
    labels: Sequence[str],
    iou_threshold: float,
) -> List[int]:
    """Greedy NMS, scoped per (normalized) label.

    Suppression is per-label on purpose: two overlapping boxes of
    DIFFERENT labels are a real simultaneous detection (a person on a
    chair, a cart next to a table) and must not suppress each other --
    only two boxes competing to describe the SAME physical object should.

    Returns the indices to KEEP, in their original input order.
    """
    n = len(boxes)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    norm_labels = [normalize_phrase(label) for label in labels]
    suppressed = [False] * n
    keep = []

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if norm_labels[j] != norm_labels[i]:
                continue
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                suppressed[j] = True

    keep.sort()
    return keep
