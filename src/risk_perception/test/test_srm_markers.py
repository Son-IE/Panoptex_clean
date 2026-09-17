"""
WP3 unit tests for predictive_risk_costmap_node.srm_marker_mask() -- the
pure per-layer "comet" gate _srm_markers() calls (see that function's own
docstring and the module docstring's "Comet-vs-parked-disc fix" section).
Pure numpy, no rclpy, no ROS graph -- same Tier 0 style as test_risk_stack.py
and test_srm.py.
"""

import numpy as np

from risk_perception.predictive_risk_costmap_node import srm_marker_mask

SRM_MIN = 0.3
DELTA = 0.15
STRIDE = 3


def _srm(layer0_value, *layer_values):
    """(K, 3, 3) SRM stack: every layer is a flat array of one value at
    every cell (layer0_value for layer 0, layer_values[k-1] for layer k)."""
    K = 1 + len(layer_values)
    srm = np.zeros((K, 3, 3), dtype=np.float32)
    srm[0] = layer0_value
    for k, v in enumerate(layer_values, start=1):
        srm[k] = v
    return srm


# ------------------------------------------------------------ layer 0 ("now")

def test_layer0_gate_is_plain_srm_min_threshold():
    srm = _srm(0.5)
    mask = srm_marker_mask(srm, 0, SRM_MIN, DELTA, STRIDE)
    assert mask.all()

    srm_below = _srm(0.2)
    mask_below = srm_marker_mask(srm_below, 0, SRM_MIN, DELTA, STRIDE)
    assert not mask_below.any()


def test_layer0_is_never_stride_gated():
    """k==0 is always evaluated regardless of srm_marker_layer_stride --
    even a stride that would otherwise skip k==0 (stride=0, or any stride
    where 0 % stride would be undefined) must not blank layer 0."""
    srm = _srm(0.5)
    for stride in (1, 2, 3, 5, 0):
        mask = srm_marker_mask(srm, 0, SRM_MIN, DELTA, stride)
        assert mask.all(), f"stride={stride}"


# ------------------------------------------------------------ layers k>=1

def test_layer_k_requires_both_delta_and_absolute_floor():
    """A cell must clear BOTH srm[k]-srm[0] >= delta AND srm[k] >= srm_min
    -- delta alone is not enough (two near-zero values can satisfy delta
    trivially). stride=1 here so stride gating (its own dedicated tests
    below) can't interfere with isolating this logic at k=1."""
    # delta satisfied (0.20 - 0.01 = 0.19 >= 0.15) but srm[k] itself is
    # below srm_min (0.20 < 0.3) -- must NOT pass.
    srm_delta_only = _srm(0.01, 0.20)
    mask = srm_marker_mask(srm_delta_only, 1, SRM_MIN, DELTA, stride=1)
    assert not mask.any()

    # srm_min satisfied (0.5 >= 0.3) but delta is not (0.5 - 0.45 = 0.05 <
    # 0.15) -- must NOT pass either.
    srm_min_only = _srm(0.45, 0.50)
    mask2 = srm_marker_mask(srm_min_only, 1, SRM_MIN, DELTA, stride=1)
    assert not mask2.any()

    # both satisfied (0.10 -> 0.40: delta 0.30 >= 0.15, srm[k]=0.40 >= 0.3)
    srm_both = _srm(0.10, 0.40)
    mask3 = srm_marker_mask(srm_both, 1, SRM_MIN, DELTA, stride=1)
    assert mask3.all()


def test_parked_object_layer_equals_layer0_never_passes():
    """The stationary hypothesis is splatted into EVERY stack layer, so a
    parked object's srm[k] equals srm[0] exactly -- delta is exactly zero
    -- and must never pass the k>=1 gate no matter how high srm[0] itself
    reads. stride=1 isolates this from stride gating."""
    srm = _srm(0.9, 0.9)
    mask = srm_marker_mask(srm, 1, SRM_MIN, DELTA, stride=1)
    assert not mask.any()


# ------------------------------------------------------------ stride gating

def test_stride_skips_non_multiple_layers_entirely():
    """A layer whose k is not a multiple of stride returns all-False,
    REGARDLESS of how strongly the delta/srm_min gates would otherwise
    pass -- stride gating happens first, unconditionally."""
    srm = _srm(0.0, 1.0, 1.0)  # layers 1, 2 would both clear delta/srm_min easily
    mask_k1 = srm_marker_mask(srm, 1, SRM_MIN, DELTA, STRIDE)  # 1 % 3 != 0
    mask_k2 = srm_marker_mask(srm, 2, SRM_MIN, DELTA, STRIDE)  # 2 % 3 != 0
    assert not mask_k1.any()
    assert not mask_k2.any()


def test_stride_draws_multiples_of_stride():
    srm = _srm(0.0, 1.0, 1.0, 1.0)  # layers 1, 2, 3
    mask_k3 = srm_marker_mask(srm, 3, SRM_MIN, DELTA, STRIDE)  # 3 % 3 == 0
    assert mask_k3.all()


def test_stride_one_draws_every_layer():
    srm = _srm(0.0, 0.5, 0.5)
    for k in (1, 2):
        mask = srm_marker_mask(srm, k, SRM_MIN, DELTA, stride=1)
        assert mask.all(), f"k={k}"


def test_mask_shape_matches_one_layer():
    srm = _srm(0.5, 0.5)
    mask = srm_marker_mask(srm, 0, SRM_MIN, DELTA, STRIDE)
    assert mask.shape == srm[0].shape
    assert mask.dtype == bool
