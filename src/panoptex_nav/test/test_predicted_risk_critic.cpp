// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0
//
// Unit tests for panoptex_nav::PredictedRiskCritic.
//
// The scoring core is a pure function (scoreStack), so the interesting
// behaviour is tested without spinning up a Costmap2DROS.  A separate test
// loads the critic through pluginlib to catch registration mistakes.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "dwb_core/exceptions.hpp"
#include "dwb_core/trajectory_critic.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"

#include "panoptex_nav/predicted_risk_critic.hpp"

using panoptex_nav::PredictedRiskCritic;
using panoptex_nav::PredictedRiskParams;
using panoptex_nav::Transform2D;
using panoptex_nav::scoreStack;

namespace
{

// ---- synthetic risk stack ---------------------------------------------------
// 10 m x 10 m at 0.1 m, origin (-5, -5), dt 0.3 s, 11 layers (t = 0 .. 3.0 s).
constexpr double kRes = 0.1;
constexpr uint32_t kW = 100;
constexpr uint32_t kH = 100;
constexpr double kOx = -5.0;
constexpr double kOy = -5.0;
constexpr double kDt = 0.3;
constexpr uint8_t kSteps = 11;

// Lethal blob: centre (0, 2.0) at layer 0, moving in -y at 1.0 m/s, radius 0.3 m.
constexpr double kBlobX = 0.0;
constexpr double kBlobY0 = 2.0;
constexpr double kBlobVy = -1.0;
constexpr double kBlobR = 0.3;

// Low background-risk lane: the horizontal band y in [-1.1, -0.9], value 5.
constexpr double kLaneYMin = -1.1;
constexpr double kLaneYMax = -0.9;
constexpr int8_t kLaneValue = 5;

inline double cellCenterX(uint32_t col) {return kOx + (col + 0.5) * kRes;}
inline double cellCenterY(uint32_t row) {return kOy + (row + 0.5) * kRes;}

panoptex_msgs::msg::RiskStack makeStack()
{
  panoptex_msgs::msg::RiskStack s;
  s.header.frame_id = "map";
  s.header.stamp.sec = 1000;
  s.header.stamp.nanosec = 0;
  s.info.resolution = kRes;
  s.info.width = kW;
  s.info.height = kH;
  s.info.origin.position.x = kOx;
  s.info.origin.position.y = kOy;
  s.info.origin.orientation.w = 1.0;
  s.dt = kDt;
  s.steps = kSteps;
  s.horizon_start = 0.0;
  s.data.assign(static_cast<size_t>(kSteps) * kW * kH, 0);

  for (uint8_t k = 0; k < kSteps; ++k) {
    const size_t base = static_cast<size_t>(k) * kW * kH;
    const double t = k * kDt;

    // low-risk lane, present in every layer
    for (uint32_t r = 0; r < kH; ++r) {
      const double cy = cellCenterY(r);
      if (cy < kLaneYMin || cy > kLaneYMax) {continue;}
      for (uint32_t c = 0; c < kW; ++c) {
        s.data[base + r * kW + c] = kLaneValue;
      }
    }

    // moving lethal blob (written last so it wins over the lane)
    const double bx = kBlobX;
    const double by = kBlobY0 + kBlobVy * t;
    for (uint32_t r = 0; r < kH; ++r) {
      const double cy = cellCenterY(r);
      if (std::fabs(cy - by) > kBlobR) {continue;}
      for (uint32_t c = 0; c < kW; ++c) {
        const double cx = cellCenterX(c);
        if (std::hypot(cx - bx, cy - by) <= kBlobR) {
          s.data[base + r * kW + c] = 100;
        }
      }
    }
  }
  return s;
}

dwb_msgs::msg::Trajectory2D makeStraightTraj(
  double x0, double y0, double vx, double vy, double t_max, double step = 0.1)
{
  dwb_msgs::msg::Trajectory2D traj;
  traj.velocity.x = vx;
  traj.velocity.y = vy;
  for (double t = 0.0; t <= t_max + 1e-9; t += step) {
    geometry_msgs::msg::Pose2D p;
    p.x = x0 + vx * t;
    p.y = y0 + vy * t;
    traj.poses.push_back(p);
    builtin_interfaces::msg::Duration d;
    d.sec = static_cast<int32_t>(t);
    d.nanosec = static_cast<uint32_t>(std::llround((t - std::floor(t)) * 1e9)) % 1000000000u;
    traj.time_offsets.push_back(d);
  }
  return traj;
}

/// Trajectory that stays at one point but is only evaluated at [t_start, t_end].
dwb_msgs::msg::Trajectory2D makeHoldTraj(
  double x, double y, double t_start, double t_end, double step = 0.1)
{
  dwb_msgs::msg::Trajectory2D traj;
  for (double t = t_start; t <= t_end + 1e-9; t += step) {
    geometry_msgs::msg::Pose2D p;
    p.x = x;
    p.y = y;
    traj.poses.push_back(p);
    builtin_interfaces::msg::Duration d;
    d.sec = static_cast<int32_t>(t);
    d.nanosec = static_cast<uint32_t>(std::llround((t - std::floor(t)) * 1e9)) % 1000000000u;
    traj.time_offsets.push_back(d);
  }
  return traj;
}

PredictedRiskParams defaultParams(double max_horizon_s = 3.0)
{
  PredictedRiskParams p;
  p.cost_power = 1.0;
  p.time_discount = 0.9;
  p.lethal_threshold = 0.85;
  p.max_horizon_s = max_horizon_s;
  p.critic_name = "PredictedRisk";
  p.skip_first_s = 0.0;      // legacy behaviour for the geometry tests
  p.escape_radius_m = 0.0;
  return p;
}

}  // namespace

// Case 1: straight along +x, clear of both the blob's x-corridor and the lane.
TEST(PredictedRiskCritic, ClearTrajectoryScoresZero)
{
  const auto stack = makeStack();
  const auto traj = makeStraightTraj(-2.0, 0.0, 0.26, 0.0, 3.0);
  panoptex_nav::ScoreStats stats;
  const double score = scoreStack(stack, Transform2D{}, traj, defaultParams(), &stats);
  EXPECT_NEAR(score, 0.0, 1e-9);
  EXPECT_GT(stats.scored, 0u);  // poses really were inside the grid
}

// Case 2: reaches the crossing cell at the same time as the blob -> illegal.
TEST(PredictedRiskCritic, CollisionWithMovingBlobThrows)
{
  const auto stack = makeStack();
  // y(t) = -1 + 0.5 t meets the blob y(t) = 2 - t at t = 2.0 s, both at y = 0.
  const auto traj = makeStraightTraj(0.0, -1.0, 0.0, 0.5, 3.0);
  EXPECT_THROW(
    scoreStack(stack, Transform2D{}, traj, defaultParams(), nullptr),
    dwb_core::IllegalTrajectoryException);
}

// Case 3: same geometry, much slower, evaluated over a shorter horizon: the
// blob never reaches it.  It sits in the low-risk lane, so the score is small
// but non-zero.
TEST(PredictedRiskCritic, SlowTrajectoryIsLegalAndCheap)
{
  const auto stack = makeStack();
  const auto traj = makeStraightTraj(0.0, -1.0, 0.0, 0.05, 1.5);
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, traj, defaultParams(1.5), nullptr));
  EXPECT_GT(score, 0.0);
  EXPECT_LT(score, 1.0);
}

// Case 4: the blob's *current* cell is legal once the blob has moved on.
// A static costmap critic would reject this trajectory.
TEST(PredictedRiskCritic, PastHazardCellIsLegalLater)
{
  const auto stack = makeStack();

  // Same cell, evaluated now -> lethal.
  const auto now_traj = makeHoldTraj(kBlobX, kBlobY0, 0.0, 0.5);
  EXPECT_THROW(
    scoreStack(stack, Transform2D{}, now_traj, defaultParams(), nullptr),
    dwb_core::IllegalTrajectoryException);

  // Same cell, evaluated at t >= 2.5 s -> the blob is long gone.
  const auto later_traj = makeHoldTraj(kBlobX, kBlobY0, 2.5, 3.0);
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, later_traj, defaultParams(), nullptr));
  EXPECT_NEAR(score, 0.0, 1e-9);
}

// A stale-by-age stack must be indexed by (t + age): with the blob's current
// cell evaluated at t in [0, 0.5] but a 2.5 s age shift, the layer read is the
// one for 2.5-3.0 s, where the blob has already passed -> legal, score 0.
TEST(PredictedRiskCritic, TimeShiftMovesLayerIndex)
{
  const auto stack = makeStack();
  const auto now_traj = makeHoldTraj(kBlobX, kBlobY0, 0.0, 0.5);

  auto params = defaultParams();
  params.time_shift_s = 2.5;
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, now_traj, params, nullptr));
  EXPECT_NEAR(score, 0.0, 1e-9);

  // And the converse: a pose that would be safe at its own time becomes lethal
  // once the shift lines it up with the blob's crossing.
  const auto early_traj = makeHoldTraj(kBlobX, kBlobY0, 0.0, 0.1);
  params.time_shift_s = 0.0;
  EXPECT_THROW(
    scoreStack(stack, Transform2D{}, early_traj, params, nullptr),
    dwb_core::IllegalTrajectoryException);
}

// Escape rule: a robot standing inside a lethal cell must still be able to
// choose a trajectory. With skip_first_s / escape_radius_m active, poses at
// the start of a candidate never throw; a lethal cell further along still does.
TEST(PredictedRiskCritic, EscapeRuleNeverRejectsTheRobotsOwnCell)
{
  const auto stack = makeStack();
  auto params = defaultParams();
  params.skip_first_s = 0.5;
  params.escape_radius_m = 0.3;
  // Sit in the blob's current cell, evaluated at t in [0, 0.5]: legal.
  const auto sit = makeHoldTraj(kBlobX, kBlobY0, 0.0, 0.5);
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, sit, params, nullptr));
  EXPECT_NEAR(score, 0.0, 1e-9);
  // Sit on the low-risk lane band (value 5, all layers) for 1.5 s: poses
  // after skip_first_s are within escape_radius of the start, so graded
  // cost accrues but nothing throws.
  const auto sit_long = makeHoldTraj(kBlobX, -1.0, 0.0, 1.5);
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, sit_long, params, nullptr));
  EXPECT_GT(score, 0.0);
  // Driving from outside INTO the blob's crossing later on still throws.
  const auto into = makeStraightTraj(kBlobX, kBlobY0 - 2.0 - 0.3, 0.0, 0.26, 3.0);
  bool threw = false;
  try { scoreStack(stack, Transform2D{}, into, params, nullptr); }
  catch (const dwb_core::IllegalTrajectoryException &) { threw = true; }
  EXPECT_TRUE(threw);
}

// Poses beyond max_horizon_s are ignored entirely.
TEST(PredictedRiskCritic, PosesBeyondHorizonAreSkipped)
{
  const auto stack = makeStack();
  const auto traj = makeStraightTraj(0.0, -1.0, 0.0, 0.5, 3.0);
  // Truncating the horizon to 1.0 s removes the collision (which happens ~1.8 s).
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, traj, defaultParams(1.0), nullptr));
  EXPECT_GE(score, 0.0);
}

// The costmap-frame -> risk-frame transform is applied to every pose.
TEST(PredictedRiskCritic, TransformIsApplied)
{
  const auto stack = makeStack();
  // In the costmap ("odom") frame the trajectory sits at y = -3; the odom->map
  // transform shifts it by +2 in y, putting it exactly on the case-2 collision.
  const auto traj = makeStraightTraj(0.0, -3.0, 0.0, 0.5, 3.0);
  Transform2D tf;
  tf.ty = 2.0;
  EXPECT_THROW(
    scoreStack(stack, tf, traj, defaultParams(), nullptr),
    dwb_core::IllegalTrajectoryException);

  // Without the shift the same trajectory is clear.
  double score = -1.0;
  ASSERT_NO_THROW(
    score = scoreStack(stack, Transform2D{}, traj, defaultParams(), nullptr));
  EXPECT_NEAR(score, 0.0, 1e-9);
}

// Poses outside the grid are skipped rather than treated as lethal.
TEST(PredictedRiskCritic, PosesOutsideGridAreSkipped)
{
  const auto stack = makeStack();
  const auto traj = makeStraightTraj(100.0, 100.0, 0.5, 0.0, 3.0);
  panoptex_nav::ScoreStats stats;
  double score = -1.0;
  ASSERT_NO_THROW(score = scoreStack(stack, Transform2D{}, traj, defaultParams(), &stats));
  EXPECT_NEAR(score, 0.0, 1e-9);
  EXPECT_EQ(stats.scored, 0u);
  EXPECT_GT(stats.skipped, 0u);
}

// cost_power / time_discount actually change the score.
TEST(PredictedRiskCritic, DiscountAndPowerAffectScore)
{
  const auto stack = makeStack();
  const auto traj = makeStraightTraj(0.0, -1.0, 0.0, 0.05, 1.5);

  auto p_flat = defaultParams(1.5);
  p_flat.time_discount = 1.0;
  const double flat = scoreStack(stack, Transform2D{}, traj, p_flat, nullptr);

  auto p_disc = defaultParams(1.5);
  p_disc.time_discount = 0.5;
  const double discounted = scoreStack(stack, Transform2D{}, traj, p_disc, nullptr);
  EXPECT_LT(discounted, flat);

  auto p_pow = defaultParams(1.5);
  p_pow.time_discount = 1.0;
  p_pow.cost_power = 2.0;
  const double squared = scoreStack(stack, Transform2D{}, traj, p_pow, nullptr);
  EXPECT_LT(squared, flat);  // risk 0.05 < 1 so squaring shrinks it
}

// Case 5: a stale stack deactivates the critic; nothing is scored and nothing throws.
TEST(PredictedRiskCritic, StaleStackIsRejected)
{
  const rclcpp::Time now(1000, 0, RCL_ROS_TIME);
  const rclcpp::Time fresh_stamp(999, 500000000, RCL_ROS_TIME);
  const rclcpp::Time stale_stamp(995, 0, RCL_ROS_TIME);
  const rclcpp::Time zero(0, 0, RCL_ROS_TIME);

  double age = 0.0;
  EXPECT_TRUE(PredictedRiskCritic::isStackFresh(fresh_stamp, now, now, 2.0, &age));
  EXPECT_NEAR(age, 0.5, 1e-6);

  EXPECT_FALSE(PredictedRiskCritic::isStackFresh(stale_stamp, now, now, 2.0, &age));
  EXPECT_NEAR(age, 5.0, 1e-6);

  // Zero stamp falls back to the receive time.
  EXPECT_TRUE(PredictedRiskCritic::isStackFresh(zero, fresh_stamp, now, 2.0, &age));
  EXPECT_FALSE(PredictedRiskCritic::isStackFresh(zero, stale_stamp, now, 2.0, &age));
  // Neither a stamp nor a receive time -> not fresh.
  EXPECT_FALSE(PredictedRiskCritic::isStackFresh(zero, zero, now, 2.0, &age));
}

// An inactive critic (no stack / stale / TF failure) scores 0 and never throws.
TEST(PredictedRiskCritic, InactiveCriticScoresZero)
{
  PredictedRiskCritic critic;
  ASSERT_FALSE(critic.isActive());
  const auto traj = makeStraightTraj(0.0, -1.0, 0.0, 0.5, 3.0);  // the lethal one
  double score = -1.0;
  ASSERT_NO_THROW(score = critic.scoreTrajectory(traj));
  EXPECT_NEAR(score, 0.0, 1e-9);
}

// The plugin is registered under its fully qualified name and is loadable.
TEST(PredictedRiskCritic, PluginlibRegistration)
{
  pluginlib::ClassLoader<dwb_core::TrajectoryCritic> loader(
    "dwb_core", "dwb_core::TrajectoryCritic");
  pluginlib::UniquePtr<dwb_core::TrajectoryCritic> critic;
  ASSERT_NO_THROW(
    critic = loader.createUniqueInstance("panoptex_nav::PredictedRiskCritic"));
  EXPECT_NE(critic, nullptr);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
