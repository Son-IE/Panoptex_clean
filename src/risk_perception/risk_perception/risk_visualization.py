from typing import Tuple


# Every value here is the pre-2026-09-11 table divided by 1.5 (person
# 0.90 -> 0.60, forklift 0.75 -> 0.50, ...), relative ordering unchanged.
# Reason: with motion_gate: speed, w_stat = C = consequence*factor is
# painted at FULL weight even at rest (factor=1.0), so person alone already
# read as solid red (0.90*~0.85 confidence ~ 76/100) before any encounter
# amplification -- there was no headroom left to show CPA/TTC escalating
# risk as the robot actually closes in. At 0.60, an at-rest person reads
# ~51/100 and climbs toward ~76/100 only on a genuine closing encounter
# (factor up to 1.5x, see predictive_risk_costmap_node.cpa_gain); other
# classes scale the same way, so a forklift still outranks a chair, etc.
# Multiply everything back by 1.5 to restore the old table if this turns
# out to be too timid.
CLASS_BASE_RISK = {
    "person": 0.90,
    "forklift": 0.95,
    "mobile robot": 0.75,
    "industrial machine": 0.75,
    "cart": 0.65,
    "cable": 0.85,
    "spill": 0.90,
    "chair": 0.35,
    "table": 0.40,
    "monitor": 0.20,
    "object": 0.30,
    "door": 0.35,
    # object_tracker_node's WP1 "motion overrides class" promotion
    # (2026-09-10 follow-up): a track whose RAW label is furniture/unknown
    # (e.g. GroundingDINO mislabelling a moving Carter as "table"/"chair")
    # but whose own displacement evidence has clearly shown it moving gets
    # PUBLISHED under this label instead -- same value as "cart" (0.65),
    # the CATEGORY_BASE_RISK["wheeled"] fallback below would already
    # resolve to this exact number even without this entry, but an exact
    # CLASS_BASE_RISK hit is kept explicit rather than relying on the
    # fallback silently agreeing with it.
    "moving object": 0.65,
    # New labels from user-a/sandbox (2026-09-11 merge). That table was still
    # on the pre-2026-09-10 "timid" scale (person 0.60), so only the LABELS
    # are taken; the numbers are rescaled onto this table while keeping its
    # stated ordering -- pallet slightly above "table" (a hard, low-profile
    # floor obstacle with a less predictable footprint), and "rolling chair"
    # identical to "chair" since it is the same physical object (movability
    # lives in LABEL_CATEGORIES/prior_*, not here).
    "pallet": 0.45,
    "rolling chair": 0.35,
}


def normalize_label(label: str) -> str:
    return label.lower().strip().replace(".", "")


# Single shared source for "what kind of thing is this label" -- previously
# duplicated ad hoc inside object_tracker_node's Semantic-prior lookup, and
# needed again by mask_relation's person/operable-machine candidate split.
# Both now route through label_category() instead of each keeping their own
# label->group mapping.
LABEL_CATEGORIES = {
    "person": "person",
    "robot": "robot",
    "mobile robot": "robot",
    "ground mobile robot": "robot",
    "wheeled object": "wheeled",
    "cart": "wheeled",
    "forklift": "wheeled",
    "pallet jack": "furniture",
    "chair": "furniture",
    "table": "furniture",
    "desk": "furniture",
    "monitor": "furniture",
    # object_tracker_node's WP1 "motion overrides class" promotion label
    # (2026-09-10 follow-up) -- see the matching CLASS_BASE_RISK comment
    # above. Category "wheeled" so the risk stack (stack_categories),
    # risk_speed_governor, and mission_supervisor's corridor-yield users
    # all treat a promoted track as an actionable mover with no changes
    # of their own (they already key off label_category(), not a
    # hardcoded label list).
    "moving object": "wheeled",
    # From user-a/sandbox (2026-09-11 merge). "pallet" is a static floor
    # object, same bucket as table/chair. "rolling chair" needs its own key
    # because GDINO's returned phrase is matched verbatim (normalize_label
    # only lowercases/strips), so it would NOT fall back to "chair"; it is
    # bucketed "wheeled" because it actually rolls. NOT taken from that
    # branch: `"door": "uncertain"` (no such category exists here -- door
    # keeps falling through to "unknown", which is the intended
    # no-information bucket).
    "pallet": "furniture",
    "rolling chair": "wheeled",
}

# Categories a person can plausibly be "operating" -- excludes "person"
# itself and "furniture" (a chair or table is never an operator target).
OPERABLE_MACHINE_CATEGORIES = {"wheeled", "robot"}

# WP2 fallback (2026-09-10 evening): risk_score_from_label's exact-label
# lookup into CLASS_BASE_RISK missed labels like "ground mobile robot"
# (GroundingDINO's real-perception phrasing) that LABEL_CATEGORIES already
# maps to "robot" -- gt_tracks_node's oracle path always emits "mobile
# robot" verbatim, so the SAME physical object read consequence 0.75 under
# perception:=oracle and the generic default 0.40 under perception:=panoptex,
# a silent oracle/panoptex divergence (the plan's root-cause #4). Keyed by
# label_category()'s own category names, one representative CLASS_BASE_RISK
# entry per category -- NOT a new/duplicated number, so the two tables can
# never drift apart:
#   "person"  -> CLASS_BASE_RISK["person"]        (0.90)
#   "robot"   -> CLASS_BASE_RISK["mobile robot"]   (0.75, matches the oracle)
#   "wheeled" -> CLASS_BASE_RISK["cart"]           (0.65, the generic-cart case;
#                a "forklift" label still hits CLASS_BASE_RISK directly at
#                0.95 before this fallback is ever consulted)
# "furniture" and "unknown" have no entry -- CLASS_BASE_RISK already has
# several distinct furniture severities (chair 0.35, table 0.40, monitor
# 0.20) with no single representative value, and "unknown" covers anything
# LABEL_CATEGORIES has no opinion on at all -- both keep falling through to
# risk_score_from_label's existing generic 0.40 default, unchanged from
# before this feature.
CATEGORY_BASE_RISK = {
    "person": CLASS_BASE_RISK["person"],
    "robot": CLASS_BASE_RISK["mobile robot"],
    "wheeled": CLASS_BASE_RISK["cart"],
}


def label_category(label: str) -> str:
    return LABEL_CATEGORIES.get(normalize_label(label), "unknown")


# Canonical parser for the tracker's packed class_id string. Previously
# duplicated verbatim (same logic, different docstring) in
# predictive_risk_costmap_node.py and spatial_prior_node.py; both now import
# it from here instead of keeping their own copy.
def parse_class_id(class_id: str):
    """'label|pmov=..|pmot=..|vx=..|vy=..|relbonus=..|hits=..|age=..' ->
    (label, dict).

    Same contract as the tracker's output (object_tracker_node's packed
    class_id string): `label` is whatever precedes the first `|`, defaulting
    to "object" if the string is empty; every subsequent `|`-separated
    `key=value` token is parsed into `dict` (silently skipped if it has no
    `=` or the value doesn't parse, e.g. a malformed or truncated token).
    Callers typically read `pmov`, `pmot`, `vx`, `vy`, `relbonus` out of the
    returned dict via `.get(key, default)`.

    `hits` (track observation count) and `age` (seconds since the track was
    first seen) are track-maturity fields object_tracker_node started
    appending on 2026-09-10 (see its _publish_objects) so downstream
    consumers can down-weight a barely-observed track (e.g. a lidar cluster
    spawned off a shelf edge for one scan) instead of trusting it as much as
    a track with dozens of hits and several seconds of age. `hits` parses as
    `int`, `age` as `float`; both are simply ABSENT from the dict for any
    class_id that predates this change or comes from a source that never
    sets them (e.g. gt_tracks_node) -- every existing caller already reads
    this dict via `.get(key, default)`, so an absent key is indistinguishable
    from "not yet observed" and never raises.
    """
    parts = class_id.split("|")
    label = parts[0] if parts else "object"
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                kv[k] = int(v) if k == "hits" else float(v)
            except ValueError:
                pass
    return label, kv


def is_operable_machine(label: str) -> bool:
    return label_category(label) in OPERABLE_MACHINE_CATEGORIES


def risk_score_from_label(
    label: str,
    confidence: float = 1.0,
) -> float:
    """Class-severity lookup (see module comment on CATEGORY_BASE_RISK for
    the 2026-09-10 evening fallback this added): an exact CLASS_BASE_RISK
    hit for the normalized label always wins unchanged; failing that, the
    label's label_category() is looked up in CATEGORY_BASE_RISK (person/
    robot/wheeled only); failing THAT (furniture, unknown, or any label
    with no category at all), the pre-existing generic 0.40 default."""
    normalized = normalize_label(label)

    if normalized in CLASS_BASE_RISK:
        base = CLASS_BASE_RISK[normalized]
    else:
        base = CATEGORY_BASE_RISK.get(label_category(label), 0.40)

    score = base * max(0.0, min(1.0, confidence))

    return max(0.0, min(1.0, score))


def risk_to_bgr(risk: float) -> Tuple[int, int, int]:
    """
    Convert risk score [0, 1] to OpenCV BGR color.

    low    risk: blue/cyan
    medium risk: yellow/orange
    high   risk: red
    """

    risk = max(0.0, min(1.0, risk))

    if risk < 0.5:
        # blue -> yellow
        t = risk / 0.5
        b = int((1.0 - t) * 255)
        g = int(t * 255)
        r = int(t * 255)
    else:
        # yellow -> red
        t = (risk - 0.5) / 0.5
        b = 0
        g = int((1.0 - t) * 255)
        r = 255

    return b, g, r
