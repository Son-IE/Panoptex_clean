// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0
//
// Unit tests for the shared RiskStack lookup rules (WP2,
// include/panoptex_nav/risk_stack_lookup.hpp): layer indexing (incl. the
// stack-age time shift and end clamping), cell lookup through a grid origin
// that carries a yaw, out-of-grid handling and freshness.
//
// Everything here is a pure function -- no node, no costmap, no TF.

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"

#include "panoptex_nav/risk_stack_lookup.hpp"

using panoptex_nav::StackGrid;
using panoptex_nav::Transform2D;
using panoptex_nav::cellRisk;
using panoptex_nav::isStackFresh;
using panoptex_nav::layerIndex;
using panoptex_nav::quaternionYaw;

namespace
{

/// 4 m x 3 m grid at 0.5 m (8 x 6 cells), dt 0.3, 11 layers, origin (-1, -2).
panoptex_msgs::msg::RiskStack makeStack(double origin_yaw = 0.0)
{
  panoptex_msgs::msg::RiskStack s;
  s.header.frame_id = "map";
  s.info.resolution = 0.5;
  s.info.width = 8;
  s.info.height = 6;
  s.info.origin.position.x = -1.0;
  s.info.origin.position.y = -2.0;
  s.info.origin.orientation.z = std::sin(origin_yaw / 2.0);
  s.info.origin.orientation.w = std::cos(origin_yaw / 2.0);
  s.dt = 0.3f;
  s.steps = 11;
  s.horizon_start = 0.0f;
  s.data.assign(static_cast<size_t>(s.steps) * s.info.width * s.info.height, 0);
  return s;
}

void setCell(
  panoptex_msgs::msg::RiskStack & s, size_t k, size_t row, size_t col, int8_t value)
{
  s.data[k * s.info.width * s.info.height + row * s.info.width + col] = value;
}

}  // namespace

// ---------------------------------------------------------------- layerIndex

TEST(RiskStackLookup, LayerIndexBasic)
{
  // dt 0.3, 11 layers: t rounds to the nearest layer.
  EXPECT_EQ(layerIndex(0.0, 0.0, 0.0, 0.3, 11), 0u);
  EXPECT_EQ(layerIndex(0.3, 0.0, 0.0, 0.3, 11), 1u);
  EXPECT_EQ(layerIndex(0.44, 0.0, 0.0, 0.3, 11), 1u);   // 1.47 -> 1
  EXPECT_EQ(layerIndex(0.46, 0.0, 0.0, 0.3, 11), 2u);   // 1.53 -> 2
  EXPECT_EQ(layerIndex(3.0, 0.0, 0.0, 0.3, 11), 10u);
}

TEST(RiskStackLookup, LayerIndexAppliesTimeShift)
{
  // A 1.2 s old stack: "t = 0.3 s from now" is layer 5, not layer 1.
  EXPECT_EQ(layerIndex(0.3, 1.2, 0.0, 0.3, 11), 5u);
  // horizon_start offsets the whole axis the other way.
  EXPECT_EQ(layerIndex(0.9, 0.0, 0.6, 0.3, 11), 1u);
}

TEST(RiskStackLookup, LayerIndexClampsBothEnds)
{
  EXPECT_EQ(layerIndex(-5.0, 0.0, 0.0, 0.3, 11), 0u);       // before layer 0
  EXPECT_EQ(layerIndex(100.0, 0.0, 0.0, 0.3, 11), 10u);     // past the horizon
  EXPECT_EQ(layerIndex(0.3, 100.0, 0.0, 0.3, 11), 10u);     // shifted past it
  // Degenerate stacks never index out of bounds.
  EXPECT_EQ(layerIndex(1.0, 0.0, 0.0, 0.0, 11), 0u);
  EXPECT_EQ(layerIndex(1.0, 0.0, 0.0, 0.3, 0), 0u);
}

// ------------------------------------------------------------------ cellRisk

TEST(RiskStackLookup, CellRiskAxisAligned)
{
  auto s = makeStack();
  setCell(s, 3, 2, 4, 80);   // cell (row 2, col 4) of layer 3
  // Cell centre: x = -1 + (4 + 0.5) * 0.5 = 1.25, y = -2 + (2 + 0.5) * 0.5 = -0.75
  EXPECT_NEAR(cellRisk(s, 1.25, -0.75, 3), 0.8, 1e-9);
  // Same point, a different layer -> nothing there.
  EXPECT_NEAR(cellRisk(s, 1.25, -0.75, 4), 0.0, 1e-9);
  // Neighbouring cell -> nothing there.
  EXPECT_NEAR(cellRisk(s, 1.75, -0.75, 3), 0.0, 1e-9);
  // Unknown (-1) reads as 0, not as risk.
  setCell(s, 3, 2, 5, -1);
  EXPECT_NEAR(cellRisk(s, 1.75, -0.75, 3), 0.0, 1e-9);
}

TEST(RiskStackLookup, CellRiskHonoursOriginYaw)
{
  const double yaw = M_PI / 2.0;    // grid +x runs along world +y
  auto s = makeStack(yaw);
  setCell(s, 0, 1, 3, 60);
  ASSERT_NEAR(quaternionYaw(s.info.origin.orientation), yaw, 1e-9);

  // Grid-local centre of (row 1, col 3): (1.75, 0.75). Rotate by +90 deg and
  // add the origin: world = (-1 - 0.75, -2 + 1.75) = (-1.75, -0.25).
  EXPECT_NEAR(cellRisk(s, -1.75, -0.25, 0), 0.6, 1e-9);
  // The axis-aligned interpretation of the same cell must NOT hit it (that
  // point falls outside the rotated grid entirely, hence < 0.6 covers both).
  EXPECT_LT(cellRisk(s, 0.75, -1.25, 0), 0.6);
}

TEST(RiskStackLookup, CellRiskOutsideGridIsMinusOne)
{
  auto s = makeStack();
  EXPECT_LT(cellRisk(s, -1.01, 0.0, 0), 0.0);          // left of the origin
  EXPECT_LT(cellRisk(s, 0.0, -2.01, 0), 0.0);          // below the origin
  EXPECT_LT(cellRisk(s, 3.01, 0.0, 0), 0.0);           // past width  (-1 + 8*0.5)
  EXPECT_LT(cellRisk(s, 0.0, 1.01, 0), 0.0);           // past height (-2 + 6*0.5)
  EXPECT_LT(cellRisk(s, 0.0, 0.0, 11), 0.0);           // layer out of range
  // Inside is >= 0 even where the value is zero.
  EXPECT_GE(cellRisk(s, 0.0, 0.0, 0), 0.0);
}

TEST(RiskStackLookup, DegenerateStackIsInvalid)
{
  panoptex_msgs::msg::RiskStack s;   // all zeros: no resolution, no cells
  EXPECT_FALSE(StackGrid(s).valid());
  EXPECT_LT(cellRisk(s, 0.0, 0.0, 0), 0.0);

  auto truncated = makeStack();
  truncated.data.resize(10);         // fewer bytes than steps * W * H
  EXPECT_FALSE(StackGrid(truncated).valid());
}

// ----------------------------------------------------------------- transform

TEST(RiskStackLookup, Transform2DAppliesRotationAndTranslation)
{
  Transform2D tf;
  const double theta = M_PI / 2.0;
  tf.cos_theta = std::cos(theta);
  tf.sin_theta = std::sin(theta);
  tf.tx = 1.0;
  tf.ty = -2.0;
  double x = 0.0, y = 0.0;
  tf.apply(2.0, 0.0, x, y);
  EXPECT_NEAR(x, 1.0, 1e-9);
  EXPECT_NEAR(y, 0.0, 1e-9);
}

// ----------------------------------------------------------------- freshness

TEST(RiskStackLookup, FreshnessUsesStampThenReceiveTime)
{
  const rclcpp::Time now(1000, 0, RCL_ROS_TIME);
  const rclcpp::Time fresh(999, 500000000, RCL_ROS_TIME);
  const rclcpp::Time stale(995, 0, RCL_ROS_TIME);
  const rclcpp::Time zero(0, 0, RCL_ROS_TIME);

  double age = 0.0;
  EXPECT_TRUE(isStackFresh(fresh, now, now, 2.0, &age));
  EXPECT_NEAR(age, 0.5, 1e-6);
  EXPECT_FALSE(isStackFresh(stale, now, now, 2.0, &age));
  EXPECT_NEAR(age, 5.0, 1e-6);
  // Zero stamp falls back to the receive time; neither -> not fresh.
  EXPECT_TRUE(isStackFresh(zero, fresh, now, 2.0, &age));
  EXPECT_FALSE(isStackFresh(zero, stale, now, 2.0, &age));
  EXPECT_FALSE(isStackFresh(zero, zero, now, 2.0, &age));
  // A stamp slightly in the future is fresh, not an error.
  EXPECT_TRUE(isStackFresh(rclcpp::Time(1000, 500000000, RCL_ROS_TIME), now, now, 2.0, &age));
  EXPECT_LT(age, 0.0);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
