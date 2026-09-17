#!/usr/bin/env python3
"""
encounter_geometry.py  --  pure geometry for Stage 4 (consequence + CPA/TTC)

Separate from predictive_risk_costmap_node.py (a rclpy Node, needs a live context to
instantiate) so this logic can be unit-tested with plain pytest, no ROS
graph and no GPU. Same rationale as relation_matching.py being split out of
gdino_detector_node.py; see that module's docstring.

cpa_geometry() is a straight port of
PredictiveRiskCostmapNode._encounter's math -- see that method's docstring
for the derivation. combine_severity() is the three-line fusion at the
bottom of _predict_and_splat (C = consequence * factor + relation_bonus),
pulled out because the ORDER of those operations is a specific modeling
choice (relation_bonus is additive and outside the CPA multiplier, so a
person boarding a still-stationary machine registers before any relative
motion exists to amplify) and a silent reordering during a refactor would
not otherwise be caught by anything.
"""

import math
from typing import Tuple


def cpa_geometry(
    obj_x: float, obj_y: float, obj_vx: float, obj_vy: float,
    robot_x: float, robot_y: float, robot_vx: float, robot_vy: float,
    cpa_gain: float, cpa_scale_m: float, ttc_scale_s: float,
    min_rel_speed: float,
) -> Tuple[float, float, float]:
    """(factor, t_cpa, d_cpa) for one object against the robot's own motion.

    factor == 1.0 exactly whenever Stage 4b is inert: relative speed below
    min_rel_speed, or the track is diverging / already past its closest
    approach (t_cpa <= 0). Otherwise factor > 1.0, growing as the closest
    approach (d_cpa) gets nearer and sooner (t_cpa).
    """
    prx = obj_x - robot_x
    pry = obj_y - robot_y
    rvx = obj_vx - robot_vx
    rvy = obj_vy - robot_vy
    rel_speed = math.hypot(rvx, rvy)

    if rel_speed < min_rel_speed:
        return 1.0, float("inf"), math.hypot(prx, pry)

    t_cpa = -(prx * rvx + pry * rvy) / (rel_speed * rel_speed)
    if t_cpa <= 0.0:
        return 1.0, t_cpa, math.hypot(prx, pry)

    d_cpa = math.hypot(prx + rvx * t_cpa, pry + rvy * t_cpa)
    factor = 1.0 + cpa_gain * math.exp(-d_cpa / cpa_scale_m) \
        * math.exp(-t_cpa / ttc_scale_s)
    return factor, t_cpa, d_cpa


def combine_severity(
    consequence: float, encounter_factor: float, relation_bonus: float,
) -> float:
    """C = min(1.0, consequence * encounter_factor + relation_bonus).

    relation_bonus is added AFTER the CPA/TTC multiplier, not before --
    it is a flat "someone is operating this" bump, not a directional
    closing-speed signal, and must not be amplified by encounter geometry
    that has nothing to do with it.

    Clipped at 1.0 (WP-A, 2026-09-11): consequence * encounter_factor alone
    can exceed 1.0 -- a high-consequence track (person 0.90, or the WP-A
    class-agnostic 0.75/0.90) on a close collision course can see
    encounter_factor > 1.4 (cpa_gain 1.5's own ceiling), and relation_bonus
    stacks on top of that. An UNCLIPPED value > 1.0 reaches the STAGE 5
    stack and, from there, the SRM's raw-occupancy read (stack_to_srm
    always also reads a cell's own raw value directly, not just the
    highest srm_level it clears -- see predictive_risk_costmap_node.py's
    module docstring, SRM topic contract) -- so an uncapped C would read as
    "more than fully occupied," which every consumer downstream (the SRM's
    [0, 1] risk field, the MPPI critic's collision_threshold, the collapsed
    grid's 0-100 occupancy encoding) already assumes cannot happen and
    silently saturates on its own terms anyway. Clipping HERE, once, keeps
    every one of those consumers honest about what 1.0 means instead of
    each one re-deriving its own implicit ceiling.
    """
    return min(1.0, consequence * encounter_factor + relation_bonus)
