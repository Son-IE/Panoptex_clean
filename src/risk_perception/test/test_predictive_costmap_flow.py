"""
Tier 0 of the Spatial-Flow rollout blend: pure geometry/arithmetic, no
ROS graph, no rclpy context needed for sample_flow_grid / blend_velocity
themselves (importing the module is fine without rclpy.init -- same as
test_predictive_costmap_multiobject.py's heavier Node-level tests).
"""

import numpy as np
import pytest

from risk_perception.predictive_risk_costmap_node import (
    blend_velocity,
    parse_flow_category_map,
    resolve_flow_channel,
    sample_flow_grid,
)

RES = 0.10
OX, OY = -6.0, -6.0
ROWS, COLS = 120, 120


def flow_array_with(row, col, s, fx, fy):
    arr = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    arr[row, col] = (s, fx, fy)
    return arr


# ------------------------------------------------------------------ sample_flow_grid

def test_sample_flow_grid_none_array_returns_zero():
    assert sample_flow_grid(None, 0.0, 0.0, RES, OX, OY, ROWS, COLS) == (0.0, 0.0, 0.0)


def test_sample_flow_grid_reads_the_right_cell():
    row, col = 60, 60  # near world origin on this grid
    arr = flow_array_with(row, col, 0.8, 1.2, -0.4)
    wx = OX + (col + 0.5) * RES
    wy = OY + (row + 0.5) * RES
    s, fx, fy = sample_flow_grid(arr, wx, wy, RES, OX, OY, ROWS, COLS)
    assert (s, fx, fy) == (pytest.approx(0.8), pytest.approx(1.2), pytest.approx(-0.4))


def test_sample_flow_grid_out_of_bounds_returns_zero():
    arr = np.zeros((ROWS, COLS, 3), dtype=np.float32)
    assert sample_flow_grid(arr, 1000.0, 1000.0, RES, OX, OY, ROWS, COLS) == (0.0, 0.0, 0.0)
    assert sample_flow_grid(arr, -1000.0, -1000.0, RES, OX, OY, ROWS, COLS) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------ blend_velocity

def test_blend_off_returns_track_velocity_unchanged():
    """flow_blend_weight <= 0 is the shipped default -- must be a pure
    pass-through regardless of what flow data is offered."""
    vx, vy = blend_velocity(1.5, -0.7, flow_s=1.0, flow_fx=99.0, flow_fy=99.0,
                            step=5, horizon_steps=10, flow_blend_weight=0.0)
    assert (vx, vy) == (1.5, -0.7)


def test_blend_with_zero_confidence_cell_ignores_flow():
    """flow_s=0 (no deposit history) must contribute nothing even with a
    nonzero configured weight -- this is the safety property the design
    hinges on: F=(0,0) at an unvisited cell must not be read as 'observed
    stationary'."""
    vx, vy = blend_velocity(1.5, -0.7, flow_s=0.0, flow_fx=99.0, flow_fy=99.0,
                            step=9, horizon_steps=10, flow_blend_weight=1.0)
    assert vx == pytest.approx(1.5)
    assert vy == pytest.approx(-0.7)


def test_blend_grows_with_horizon_step():
    kwargs = dict(vx=2.0, vy=0.0, flow_s=1.0, flow_fx=0.0, flow_fy=0.0,
                  horizon_steps=10, flow_blend_weight=1.0)
    vx_near, _ = blend_velocity(step=0, **kwargs)
    vx_far, _ = blend_velocity(step=9, **kwargs)
    # blending toward flow_fx=0.0 while the track's own vx=2.0 -> more
    # blend means vx pulled further toward 0.
    assert vx_near > vx_far >= 0.0


def test_blend_matches_hand_worked_formula():
    vx, vy = blend_velocity(vx=4.0, vy=0.0, flow_s=0.5, flow_fx=0.0, flow_fy=2.0,
                            step=4, horizon_steps=10, flow_blend_weight=1.0)
    # step_frac = 5/10 = 0.5 ; blend = 1.0 * 0.5 * 0.5 = 0.25
    assert vx == pytest.approx((1 - 0.25) * 4.0 + 0.25 * 0.0)
    assert vy == pytest.approx((1 - 0.25) * 0.0 + 0.25 * 2.0)


def test_blend_weight_is_clamped_to_one():
    """An aggressively large flow_blend_weight must not overshoot past
    fully trusting the flow cell."""
    vx, vy = blend_velocity(vx=10.0, vy=0.0, flow_s=1.0, flow_fx=1.0, flow_fy=1.0,
                            step=9, horizon_steps=10, flow_blend_weight=50.0)
    assert vx == pytest.approx(1.0)
    assert vy == pytest.approx(1.0)


# ------------------------------------------------------------------ flow_category_map

def test_a_wheeled_track_blends_with_the_group_channel_when_only_it_has_evidence():
    """The shipped default (["robot:robot_group", "wheeled:robot_group"])
    means a "wheeled" track never had its own /risk_perception/spatial_flow/
    wheeled topic populated in this scenario (e.g. every deposit this
    session happened to land under the "robot" label instead -- the
    tracker's association_groups fuses the two) -- only the merged
    robot_group channel has evidence. resolve_flow_channel must send a
    wheeled track's lookup to that channel, and the (s, fx, fy) sampled
    from it must be what blend_velocity actually uses, exactly reproducing
    what predictive_risk_costmap_node._predict_and_splat does with
    self.flow_grids.get(flow_channel)."""
    flow_category_map = parse_flow_category_map(
        ["robot:robot_group", "wheeled:robot_group"])
    channel = resolve_flow_channel("wheeled", flow_category_map)
    assert channel == "robot_group"

    # This session's per-category "wheeled" channel is absent entirely
    # (never subscribed evidence, or simply never arrived) -- only the
    # merged group channel exists.
    flow_grids = {
        "robot_group": np.zeros((ROWS, COLS, 3), dtype=np.float32),
    }
    row, col = 60, 60
    flow_grids["robot_group"][row, col] = (0.8, 1.2, -0.4)  # strong evidence
    wx = OX + (col + 0.5) * RES
    wy = OY + (row + 0.5) * RES

    flow_array = flow_grids.get(channel)  # what the Node actually does
    assert flow_array is not None
    s, fx, fy = sample_flow_grid(flow_array, wx, wy, RES, OX, OY, ROWS, COLS)
    assert (s, fx, fy) == (pytest.approx(0.8), pytest.approx(1.2), pytest.approx(-0.4))

    vx, vy = blend_velocity(vx=0.0, vy=0.0, flow_s=s, flow_fx=fx, flow_fy=fy,
                            step=9, horizon_steps=10, flow_blend_weight=1.0)
    # step_frac = 1.0, blend = min(1, 1.0 * 1.0 * 0.8) = 0.8
    assert vx == pytest.approx(0.8 * 1.2)
    assert vy == pytest.approx(0.8 * -0.4)

    # A category with no map entry (e.g. "person") is unaffected -- it
    # keeps sampling its own channel, which is simply absent here.
    person_channel = resolve_flow_channel("person", flow_category_map)
    assert person_channel == "person"
    assert flow_grids.get(person_channel) is None
