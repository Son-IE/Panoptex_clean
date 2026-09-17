"""
WP2 unit tests for risk_perception.risk_visualization's label -> consequence
lookup, specifically the 2026-09-10 evening category fallback
(CATEGORY_BASE_RISK) added to risk_score_from_label(). Pure functions, no
rclpy, no ROS graph.
"""

import pytest

from risk_perception.risk_visualization import (
    risk_score_from_label, CLASS_BASE_RISK, CATEGORY_BASE_RISK, label_category)


# ------------------------------------------------------------ exact-label hits unchanged

def test_exact_label_hit_unchanged_for_every_existing_entry():
    """Every label already in CLASS_BASE_RISK must keep reading its exact
    table value -- the new category fallback must never be consulted when
    an exact hit exists."""
    for label, base in CLASS_BASE_RISK.items():
        assert risk_score_from_label(label, 1.0) == pytest.approx(base)


def test_exact_label_hit_still_scales_by_confidence():
    assert risk_score_from_label("person", 0.5) == pytest.approx(0.45)


# ------------------------------------------------------------ category fallback

def test_ground_mobile_robot_falls_back_to_robot_category_like_oracle():
    """The plan's headline case: real perception's "ground mobile robot"
    phrasing is not an exact CLASS_BASE_RISK key, but label_category()
    already maps it to "robot" -- it must read the SAME 0.75 as the
    oracle's own "mobile robot" label, not the generic 0.40 default."""
    assert label_category("ground mobile robot") == "robot"
    assert "ground mobile robot" not in CLASS_BASE_RISK

    got = risk_score_from_label("ground mobile robot", 1.0)
    want = risk_score_from_label("mobile robot", 1.0)
    assert got == pytest.approx(want)
    assert got == pytest.approx(0.75)


def test_person_category_fallback_matches_person_label():
    """No non-exact "person" label exists in LABEL_CATEGORIES today, so
    this checks the fallback table value directly against the exact-label
    value it must mirror."""
    assert CATEGORY_BASE_RISK["person"] == pytest.approx(CLASS_BASE_RISK["person"])


def test_wheeled_category_fallback_uses_cart_value():
    """"wheeled object" isn't an exact CLASS_BASE_RISK key but maps to
    category "wheeled" -- falls back to the generic-cart severity (0.65),
    not the forklift-specific 0.95 (forklift is matched on the exact
    label directly, before this fallback is ever consulted)."""
    assert label_category("wheeled object") == "wheeled"
    got = risk_score_from_label("wheeled object", 1.0)
    assert got == pytest.approx(CLASS_BASE_RISK["cart"])
    assert got == pytest.approx(0.65)


def test_forklift_label_still_hits_exact_entry_not_the_wheeled_fallback():
    """"forklift" IS an exact CLASS_BASE_RISK key (0.95) -- it must never
    fall through to the wheeled category's cart-derived fallback (0.65),
    even though label_category("forklift") == "wheeled"."""
    assert label_category("forklift") == "wheeled"
    got = risk_score_from_label("forklift", 1.0)
    assert got == pytest.approx(0.95)


# ------------------------------------------------------------ no fallback for furniture/unknown

def test_furniture_category_has_no_fallback_entry():
    """CATEGORY_BASE_RISK deliberately has no "furniture" entry -- chair/
    table/monitor severities differ too much for one representative value.
    A furniture label NOT already in CLASS_BASE_RISK (e.g. "desk") must
    fall all the way through to the generic 0.40 default, unchanged from
    before this feature."""
    assert label_category("desk") == "furniture"
    assert "desk" not in CLASS_BASE_RISK
    assert "furniture" not in CATEGORY_BASE_RISK
    assert risk_score_from_label("desk", 1.0) == pytest.approx(0.40)


def test_unknown_category_label_falls_back_to_generic_default():
    assert label_category("lidar_cluster") == "unknown"
    assert risk_score_from_label("lidar_cluster", 1.0) == pytest.approx(0.40)
