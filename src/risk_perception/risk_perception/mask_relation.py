#!/usr/bin/env python3
"""
mask_relation.py  --  pure geometry for the relation prior, GDINO-free

Replaces gdino_detector_node's relation-phrase grounding (empirically found
to not work -- GDINO parses a compound phrase like "person operating
forklift" more like independent per-noun grounding than a bound relation,
see the relation-prior playbook's decision gate). This module answers the
same question -- "is this person on/in/operating that machine" -- with pure
segmentation geometry instead of language grounding: does the person's own
mask sit mostly INSIDE the candidate machine's mask.

Deliberately no ROS, no torch, no cv2 -- numpy only -- so it is unit
testable with plain pytest, same rationale as relation_matching.py and
encounter_geometry.py (see those modules' docstrings).

Containment, not IoU: a person standing in a forklift's operator position
has a MUCH smaller mask than the forklift's, so symmetric IoU (intersection
over union) stays low even for a true positive -- most of the union is
forklift the person isn't standing on. What matters is what fraction of the
SMALLER mask (the person) falls inside the LARGER one (the machine),
dilated by a small margin so the two silhouettes touching is not required
to be pixel-perfect.
"""

from typing import Optional

import numpy as np


def dilate_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Binary dilation by a square structuring element, no cv2/scipy.

    margin_px <= 0 returns the mask unchanged. Implemented as `margin_px`
    single-pixel shifts in each of 4 directions per iteration (a diamond
    growing into a rounded square) -- fine for the small margins (a handful
    of pixels) this is used with; not meant for large-kernel morphology.
    """
    if margin_px <= 0:
        return mask
    out = mask.copy()
    for _ in range(margin_px):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def mask_containment_ratio(
    inner_mask: np.ndarray, outer_mask: np.ndarray, dilate_outer_px: int = 0,
) -> float:
    """Fraction of inner_mask's true pixels that fall within outer_mask
    (optionally dilated first). 0.0 if inner_mask is empty."""
    inner_count = int(inner_mask.sum())
    if inner_count == 0:
        return 0.0
    outer = dilate_mask(outer_mask, dilate_outer_px) if dilate_outer_px > 0 else outer_mask
    overlap = int(np.logical_and(inner_mask, outer).sum())
    return overlap / inner_count


def best_operator_match(
    person_mask: np.ndarray,
    machine_masks: dict,
    containment_threshold: float,
    dilate_outer_px: int = 0,
) -> Optional[tuple]:
    """machine_masks: {machine_index: mask}. -> (machine_index, ratio) for
    the highest-containment machine clearing containment_threshold, or None.

    Winner-take-all per person, same shape as
    relation_matching.match_relations_to_objects -- one person can only be
    "operating" one machine at a time.
    """
    best_index, best_ratio = None, 0.0
    for index, machine_mask in machine_masks.items():
        ratio = mask_containment_ratio(person_mask, machine_mask, dilate_outer_px)
        if ratio > best_ratio:
            best_index, best_ratio = index, ratio
    if best_index is not None and best_ratio >= containment_threshold:
        return best_index, best_ratio
    return None


# ------------------------------------------------------------------ 3D gate

def within_3d_proximity(
    person_xyz: tuple, machine_xyz: tuple, max_distance_m: float,
) -> bool:
    """Optional second gate alongside mask containment, using the 3D
    centroids the same projector node already computes from depth --
    catches the case where two silhouettes overlap in the 2D image purely
    from camera perspective (person standing well behind a machine, not
    actually near it) but are far apart in the real world."""
    dx = person_xyz[0] - machine_xyz[0]
    dy = person_xyz[1] - machine_xyz[1]
    dz = person_xyz[2] - machine_xyz[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 <= max_distance_m


# ------------------------------------------------------------------ orchestration

def tag_operator_relations(
    person_entries: list,
    machine_entries: list,
    containment_threshold: float,
    dilate_outer_px: int,
    max_distance_m: float,
) -> None:
    """Mutates the winning machine's hypothesis.class_id in place, appending
    "|relconf=<ratio>" -- the same convention gdino_detector_node used to
    produce via language-phrase grounding (replaced; see the relation-prior
    playbook for why). Callers are the projector nodes, which already have
    both cameras' masks and 3D centroids synced at this exact point.

    person_entries:  list of (mask: np.ndarray[bool], xyz: tuple)
    machine_entries: list of (mask: np.ndarray[bool], xyz: tuple, hypothesis)
                      -- hypothesis is duck-typed (any object with a mutable
                      string .class_id attribute), so this module still has
                      no ROS message import of its own.

    3D proximity is applied as a pre-filter before the mask check, not
    inside best_operator_match -- a candidate too far away in the real world
    is never even offered to the containment comparison, regardless of how
    much its 2D silhouette happens to overlap the person's from this
    camera's viewpoint.
    """
    for person_mask, person_xyz in person_entries:
        candidates = {
            index: mask
            for index, (mask, xyz, _hyp) in enumerate(machine_entries)
            if within_3d_proximity(person_xyz, xyz, max_distance_m)
        }
        match = best_operator_match(
            person_mask, candidates, containment_threshold, dilate_outer_px)
        if match is None:
            continue
        winner_index, ratio = match
        hypothesis = machine_entries[winner_index][2]
        hypothesis.class_id = f"{hypothesis.class_id}|relconf={ratio:.2f}"
