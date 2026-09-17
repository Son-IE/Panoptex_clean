// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0
//
// Unit tests for panoptex_nav::PredictedRiskMppiCritic (WP2, + WP-B's
// Spatiotemporal Risk Map rewrite -- the PredictedRiskSrmCritic suite below).
//
// score() itself is a thin shell: fetch the latest stack, check freshness,
// cache one TF, then hand the raw xtensor batch buffers to
// panoptex_nav::scoreStackBatch() (risk_stack_lookup.hpp). Standing a real
// critic up would need a LifecycleNode + mppi::ParametersHandler + a
// configured Costmap2DROS (map server, layer plugins, TF) just to reach that
// one call, so -- as the WP2 brief allows -- the scoring semantics are tested
// through the scoring cores directly (WP2's scoreStackBatch, still used by the
// DWB sibling, and WP-B's scoreSrmBatch, which the MPPI critic's score() now
// calls) and only the pluginlib registration is tested on the class itself.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "nav2_mppi_controller/critic_function.hpp"
#include "pluginlib/class_loader.hpp"

#include "panoptex_nav/predicted_risk_mppi_critic.hpp"
#include "panoptex_nav/risk_stack_lookup.hpp"

using panoptex_nav::MppiRiskParams;
using panoptex_nav::Transform2D;
using panoptex_nav::cellRisk;
using panoptex_nav::layerIndex;
using panoptex_nav::scoreStackBatch;

namespace
{

// 10 m x 10 m at 0.1 m, origin (-5, -5); dt 0.1 s, 61 layers (0 .. 6.0 s) --
// the same shape as the study arms' 60-step / 0.1 s MPPI horizon.
constexpr double kRes = 0.1;
constexpr uint32_t kW = 100;
constexpr uint32_t kH = 100;
constexpr double kOx = -5.0;
constexpr double kOy = -5.0;
constexpr double kDt = 0.1;
constexpr uint8_t kSteps = 61;
constexpr double kModelDt = 0.1;

panoptex_msgs::msg::RiskStack makeStack()
{
  panoptex_msgs::msg::RiskStack s;
  s.header.frame_id = "map";
  s.info.resolution = kRes;
  s.info.width = kW;
  s.info.height = kH;
  s.info.origin.position.x = kOx;
  s.info.origin.position.y = kOy;
  s.info.origin.orientation.w = 1.0;
  s.dt = static_cast<float>(kDt);
  s.steps = kSteps;
  s.horizon_start = 0.0f;
  s.data.assign(static_cast<size_t>(kSteps) * kW * kH, 0);
  return s;
}

size_t colOf(double x) {return static_cast<size_t>(std::floor((x - kOx) / kRes));}
size_t rowOf(double y) {return static_cast<size_t>(std::floor((y - kOy) / kRes));}

void setCellAt(panoptex_msgs::msg::RiskStack & s, size_t k, double x, double y, int8_t v)
{
  s.data[k * kW * kH + rowOf(y) * kW + colOf(x)] = v;
}

MppiRiskParams defaultParams()
{
  MppiRiskParams p;
  p.cost_weight = 5.0;
  p.cost_power = 1.0;
  p.time_discount = 0.95;
  p.lethal_threshold = 0.45;
  p.collision_cost = 5000.0;
  p.skip_first_s = 0.3;
  p.time_shift_s = 0.0;
  return p;
}

/// One straight rollout along +x at `speed`, `steps` samples at kModelDt.
void fillStraight(
  std::vector<float> & xs, std::vector<float> & ys, size_t b, size_t steps,
  double x0, double y0, double speed)
{
  for (size_t j = 0; j < steps; ++j) {
    xs[b * steps + j] = static_cast<float>(x0 + speed * static_cast<double>(j) * kModelDt);
    ys[b * steps + j] = static_cast<float>(y0);
  }
}

/// Straightforward per-pose reference implementation of the same scoring rule.
float referenceScore(
  const panoptex_msgs::msg::RiskStack & stack, const Transform2D & tf,
  const float * xs, const float * ys, size_t b, size_t steps,
  const MppiRiskParams & p)
{
  double accum = 0.0;
  double extra = 0.0;
  for (size_t j = 0; j < steps; ++j) {
    const double t = static_cast<double>(j) * kModelDt;
    if (t < p.skip_first_s) {continue;}
    double rx = 0.0, ry = 0.0;
    tf.apply(xs[b * steps + j], ys[b * steps + j], rx, ry);
    const size_t k = layerIndex(t, p.time_shift_s, stack.horizon_start, stack.dt, stack.steps);
    const double r = cellRisk(stack, rx, ry, k);
    if (r < 0.0) {continue;}
    if (r >= p.lethal_threshold) {
      extra = p.collision_cost;
      break;
    }
    if (r > 0.0) {
      accum += std::pow(p.time_discount, t) * std::pow(r, p.cost_power);
    }
  }
  return static_cast<float>(extra) + static_cast<float>(p.cost_weight * accum);
}

}  // namespace

// A lethal cell that exists only in layer 10 (t = 1.0 s) charges a rollout
// that is there at t = 1.0 s and leaves alone one that arrives at t = 3.0 s.
TEST(PredictedRiskMppiCritic, LethalCellIsChargedAtItsOwnLayerOnly)
{
  auto stack = makeStack();
  setCellAt(stack, 10, 1.05, 0.05, 100);   // cell centre, col 60 / row 50

  const size_t steps = 60;
  std::vector<float> xs(2 * steps, 0.0f), ys(2 * steps, 0.0f);
  fillStraight(xs, ys, 0, steps, 0.02, 0.02, 1.0);        // in that cell at t = 1.0 s
  fillStraight(xs, ys, 1, steps, 0.02, 0.02, 1.0 / 3.0);  // in that cell at t = 3.0 s

  std::vector<float> costs(2, 0.0f);
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 2, steps, kModelDt,
    defaultParams(), costs.data());

  EXPECT_GE(costs[0], 5000.0f);
  EXPECT_NEAR(costs[1], 0.0f, 1e-4f);
}

// The stack's age shifts which layer a step reads: the same rollout that hits
// the layer-10 hazard with a fresh stack misses it once the stack is 2 s old.
TEST(PredictedRiskMppiCritic, TimeShiftMovesTheLayer)
{
  auto stack = makeStack();
  setCellAt(stack, 10, 1.05, 0.05, 100);   // cell centre, col 60 / row 50

  const size_t steps = 60;
  std::vector<float> xs(steps, 0.0f), ys(steps, 0.0f);
  fillStraight(xs, ys, 0, steps, 0.02, 0.02, 1.0);

  auto params = defaultParams();
  std::vector<float> costs(1, 0.0f);
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 1, steps, kModelDt, params, costs.data());
  EXPECT_GE(costs[0], 5000.0f);

  params.time_shift_s = 2.0;      // t = 1.0 s now reads layer 30
  costs[0] = 0.0f;
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 1, steps, kModelDt, params, costs.data());
  EXPECT_NEAR(costs[0], 0.0f, 1e-4f);
}

// skip_first_s protects the steps the robot is practically already on: a
// hazard sitting on the rollout's own first cells must not charge it.
TEST(PredictedRiskMppiCritic, SkipFirstSecondsIgnoresTheEarlySteps)
{
  auto stack = makeStack();
  for (size_t k = 0; k < 3; ++k) {
    setCellAt(stack, k, 0.05, 0.05, 100);
  }
  const size_t steps = 60;
  std::vector<float> xs(steps, 0.0f), ys(steps, 0.0f);
  fillStraight(xs, ys, 0, steps, 0.02, 0.02, 1.0);

  auto params = defaultParams();
  std::vector<float> costs(1, 0.0f);
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 1, steps, kModelDt, params, costs.data());
  EXPECT_NEAR(costs[0], 0.0f, 1e-4f);

  params.skip_first_s = 0.0;
  costs[0] = 0.0f;
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 1, steps, kModelDt, params, costs.data());
  EXPECT_GE(costs[0], 5000.0f);
}

// Graded (sub-lethal) risk: the batch pass must equal a per-pose reference
// loop sample by sample, and it must ADD to whatever cost is already there.
TEST(PredictedRiskMppiCritic, BatchEqualsPerPoseReferenceLoop)
{
  auto stack = makeStack();
  for (size_t k = 0; k < kSteps; ++k) {
    for (size_t r = 0; r < kH; ++r) {
      for (size_t c = 0; c < kW; ++c) {
        // 0..39 -> risk 0.00..0.39, always below lethal_threshold (0.45).
        stack.data[k * kW * kH + r * kW + c] = static_cast<int8_t>((r + c + k) % 40);
      }
    }
  }
  // ... and one genuinely lethal patch, so the break path is exercised too.
  setCellAt(stack, 25, -1.95, 1.05, 90);

  const size_t batch = 64;
  const size_t steps = 60;
  std::vector<float> xs(batch * steps), ys(batch * steps);
  uint32_t seed = 12345u;
  auto rnd = [&seed]() {
      seed = seed * 1103515245u + 12345u;
      return static_cast<double>((seed >> 16) & 0x7fff) / 32767.0;
    };
  for (size_t b = 0; b < batch; ++b) {
    const double x0 = -4.0 + 8.0 * rnd();
    const double y0 = -4.0 + 8.0 * rnd();
    const double vx = -1.0 + 2.0 * rnd();
    const double vy = -1.0 + 2.0 * rnd();
    for (size_t j = 0; j < steps; ++j) {
      const double t = static_cast<double>(j) * kModelDt;
      xs[b * steps + j] = static_cast<float>(x0 + vx * t);
      ys[b * steps + j] = static_cast<float>(y0 + vy * t);
    }
  }

  Transform2D tf;                    // a non-identity odom -> map transform
  tf.cos_theta = std::cos(0.3);
  tf.sin_theta = std::sin(0.3);
  tf.tx = 0.4;
  tf.ty = -0.25;

  const auto params = defaultParams();
  std::vector<float> costs(batch, 7.5f);   // pre-existing cost from other critics
  scoreStackBatch(
    stack, tf, xs.data(), ys.data(), batch, steps, kModelDt, params, costs.data());

  size_t nonzero = 0;
  for (size_t b = 0; b < batch; ++b) {
    const float expected = 7.5f + referenceScore(stack, tf, xs.data(), ys.data(), b, steps, params);
    EXPECT_NEAR(costs[b], expected, 1e-3f) << "batch " << b;
    if (costs[b] > 7.5f + 1e-4f) {++nonzero;}
  }
  EXPECT_GT(nonzero, 0u);            // the field really was sampled
}

TEST(PredictedRiskMppiCritic, OutsideTheGridCostsNothing)
{
  auto stack = makeStack();
  for (size_t k = 0; k < kSteps; ++k) {
    for (size_t i = 0; i < kW * kH; ++i) {
      stack.data[k * kW * kH + i] = 100;   // the whole grid is lethal...
    }
  }
  const size_t steps = 60;
  std::vector<float> xs(steps, 0.0f), ys(steps, 0.0f);
  fillStraight(xs, ys, 0, steps, 100.0, 100.0, 0.5);   // ... but we are 100 m away

  std::vector<float> costs(1, 0.0f);
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 1, steps, kModelDt,
    defaultParams(), costs.data());
  EXPECT_NEAR(costs[0], 0.0f, 1e-4f);
}

TEST(PredictedRiskMppiCritic, DegenerateInputsAreNoOps)
{
  panoptex_msgs::msg::RiskStack empty;
  std::vector<float> xs(10, 0.0f), ys(10, 0.0f), costs(1, 3.0f);
  scoreStackBatch(
    empty, Transform2D{}, xs.data(), ys.data(), 1, 10, kModelDt,
    defaultParams(), costs.data());
  EXPECT_NEAR(costs[0], 3.0f, 1e-6f);

  auto stack = makeStack();
  scoreStackBatch(
    stack, Transform2D{}, nullptr, nullptr, 1, 10, kModelDt, defaultParams(), costs.data());
  EXPECT_NEAR(costs[0], 3.0f, 1e-6f);
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), 0, 0, kModelDt,
    defaultParams(), costs.data());
  EXPECT_NEAR(costs[0], 3.0f, 1e-6f);
}

// The study arms run batch 1000 x 60 steps at 10 Hz; the whole critic must
// stay a small fraction of one control period.
TEST(PredictedRiskMppiCritic, BatchScoringIsFastEnough)
{
  auto stack = makeStack();
  for (size_t k = 0; k < kSteps; ++k) {
    for (size_t i = 0; i < kW * kH; ++i) {
      stack.data[k * kW * kH + i] = static_cast<int8_t>((i + k) % 40);
    }
  }
  const size_t batch = 1000;
  const size_t steps = 60;
  std::vector<float> xs(batch * steps), ys(batch * steps), costs(batch, 0.0f);
  for (size_t b = 0; b < batch; ++b) {
    const double y0 = -4.0 + 8.0 * static_cast<double>(b) / static_cast<double>(batch);
    fillStraight(xs, ys, b, steps, -4.0, y0, 1.2);
  }

  const auto t0 = std::chrono::steady_clock::now();
  scoreStackBatch(
    stack, Transform2D{}, xs.data(), ys.data(), batch, steps, kModelDt,
    defaultParams(), costs.data());
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
  std::cout << "[ PERF     ] 1000 x 60 batch scored in " << ms << " ms" << std::endl;
  EXPECT_LT(ms, 25.0);   // budget is 10 ms; the slack absorbs a loaded CI box
}

// The critic is registered for nav2_mppi_controller's loader. The short name
// MPPI's CriticManager actually resolves is "mppi::critics::PredictedRiskCritic"
// (it prepends that namespace to every entry of `critics:`); the fully
// qualified alias exists for direct loads like this one.
TEST(PredictedRiskMppiCritic, PluginlibRegistration)
{
  pluginlib::ClassLoader<mppi::critics::CriticFunction> loader(
    "nav2_mppi_controller", "mppi::critics::CriticFunction");

  pluginlib::UniquePtr<mppi::critics::CriticFunction> by_alias;
  ASSERT_NO_THROW(
    by_alias = loader.createUniqueInstance("panoptex_nav::PredictedRiskMppiCritic"));
  EXPECT_NE(by_alias, nullptr);

  pluginlib::UniquePtr<mppi::critics::CriticFunction> by_critic_name;
  ASSERT_NO_THROW(
    by_critic_name = loader.createUniqueInstance("mppi::critics::PredictedRiskCritic"));
  EXPECT_NE(by_critic_name, nullptr);
}

// ===================================================================== //
// WP-B: the Spatiotemporal Risk Map (SRM) scoring core, scoreSrmBatch(). //
// ===================================================================== //
//
// The SRM (Thomas et al., 2021) is the same RiskStack wire type carrying a
// DISTANCE field instead of a class confidence: 1 on an occupied core,
// linearly down to 0 at d0. It arrives as a WINDOW around the robot, 21
// layers x 0.3 s, so these fixtures use that geometry rather than the
// 61-layer / 0.1 s one above.

namespace srm
{

constexpr double kRes = 0.1;
constexpr uint32_t kW = 80;      // x in [-2, 6)
constexpr uint32_t kH = 100;     // y in [-4, 6)
constexpr double kOx = -2.0;
constexpr double kOy = -4.0;
constexpr double kLayerDt = 0.3;
constexpr uint8_t kLayers = 21;  // 0 .. 6.0 s, the /risk_stack_srm contract
constexpr double kD0 = 1.5;
constexpr double kModelDt = 0.1;
constexpr size_t kHorizon = 60;  // MPPI time_steps

panoptex_msgs::msg::RiskStack makeSrmWindow()
{
  panoptex_msgs::msg::RiskStack s;
  s.header.frame_id = "map";
  s.info.resolution = kRes;
  s.info.width = kW;
  s.info.height = kH;
  s.info.origin.position.x = kOx;
  s.info.origin.position.y = kOy;
  s.info.origin.orientation.w = 1.0;
  s.dt = static_cast<float>(kLayerDt);
  s.steps = kLayers;
  s.horizon_start = 0.0f;
  s.data.assign(static_cast<size_t>(kLayers) * kW * kH, 0);
  return s;
}

/// Paint one occupied core at (cx, cy) into layer k: value 1 at the core,
/// falling linearly to 0 at d0 -- exactly the /risk_stack_srm contract.
/// Cells keep the max over cores, as the real SRM does.
void paintCore(panoptex_msgs::msg::RiskStack & s, size_t k, double cx, double cy)
{
  for (uint32_t row = 0; row < kH; ++row) {
    const double y = kOy + (row + 0.5) * kRes;
    for (uint32_t col = 0; col < kW; ++col) {
      const double x = kOx + (col + 0.5) * kRes;
      const double d = std::hypot(x - cx, y - cy);
      const double v = std::max(0.0, 1.0 - d / kD0);
      const size_t i = k * kW * kH + row * kW + col;
      s.data[i] = std::max(
        s.data[i], static_cast<int8_t>(std::lround(v * 100.0)));
    }
  }
}

/// A static core: the same position in every layer.
void paintStaticCore(panoptex_msgs::msg::RiskStack & s, double cx, double cy)
{
  for (size_t k = 0; k < kLayers; ++k) {paintCore(s, k, cx, cy);}
}

/// A mover crossing +y at `vy`, at `x_cross`, reaching y = 0 at `t_cross`.
void paintCrossingMover(
  panoptex_msgs::msg::RiskStack & s, double x_cross, double vy, double t_cross)
{
  for (size_t k = 0; k < kLayers; ++k) {
    const double t = static_cast<double>(k) * kLayerDt;
    paintCore(s, k, x_cross, vy * (t - t_cross));
  }
}

panoptex_nav::SrmRiskParams params()
{
  // The values config/nav2_x3_panoptex_mppi.yaml ships (WP-B).
  panoptex_nav::SrmRiskParams p;
  p.cost_weight = 30.0;
  p.cost_power = 2.0;
  p.time_discount = 0.97;
  p.collision_threshold = 0.90;
  p.collision_cost = 5000.0;
  p.skip_first_s = 0.3;
  p.escape_radius_m = 0.30;
  p.time_shift_s = 0.0;
  return p;
}

/// Score `n` candidate trajectories given as (x, y) sample vectors.
std::vector<float> score(
  const panoptex_msgs::msg::RiskStack & stack,
  const std::vector<std::vector<std::pair<double, double>>> & cands,
  const panoptex_nav::SrmRiskParams & p)
{
  const size_t batch = cands.size();
  const size_t steps = kHorizon;
  std::vector<float> xs(batch * steps, 0.0f), ys(batch * steps, 0.0f);
  for (size_t b = 0; b < batch; ++b) {
    EXPECT_EQ(cands[b].size(), steps);
    for (size_t j = 0; j < steps; ++j) {
      xs[b * steps + j] = static_cast<float>(cands[b][j].first);
      ys[b * steps + j] = static_cast<float>(cands[b][j].second);
    }
  }
  std::vector<float> costs(batch, 0.0f);
  panoptex_nav::scoreSrmBatch(
    stack, Transform2D{}, xs.data(), ys.data(), batch, steps, kModelDt, p, costs.data());
  return costs;
}

/// A candidate that sits at (hx, hy) for `dwell` steps then leaves the window.
std::vector<std::pair<double, double>> dwellThenLeave(double hx, double hy, size_t dwell)
{
  std::vector<std::pair<double, double>> c;
  for (size_t j = 0; j < kHorizon; ++j) {
    c.emplace_back(j < dwell ? hx : 50.0, j < dwell ? hy : 50.0);
  }
  return c;
}

/// A candidate on +x whose speed is `v(t)`, starting at (x0, 0).
template<typename F>
std::vector<std::pair<double, double>> alongX(double x0, F && speed)
{
  std::vector<std::pair<double, double>> c;
  double x = x0;
  for (size_t j = 0; j < kHorizon; ++j) {
    const double t = static_cast<double>(j) * kModelDt;
    c.emplace_back(x, 0.0);
    x += speed(t) * kModelDt;
  }
  return c;
}

}  // namespace srm

// (a) A linear field around a static core: the longer a candidate stays
//     inside it, the more it costs -- monotonically, with no cliff.
TEST(PredictedRiskSrmCritic, CostGrowsWithTimeSpentInTheField)
{
  auto stack = srm::makeSrmWindow();
  srm::paintStaticCore(stack, 2.0, 0.0);

  // 1.0 m from the core: srm = 1 - 1.0/1.5 = 0.33, well under collision.
  std::vector<std::vector<std::pair<double, double>>> cands;
  const std::vector<size_t> dwells = {5, 10, 20, 40, 60};
  for (size_t d : dwells) {cands.push_back(srm::dwellThenLeave(2.0, 1.0, d));}

  const auto costs = srm::score(stack, cands, srm::params());
  for (size_t i = 0; i + 1 < costs.size(); ++i) {
    EXPECT_LT(costs[i], costs[i + 1])
      << "dwell " << dwells[i] << " must cost less than dwell " << dwells[i + 1];
  }
  EXPECT_GT(costs.front(), 0.0f);
  EXPECT_LT(costs.back(), 5000.0f);   // graded, never a collision charge
}

// (b) A candidate that reaches the core (srm >= collision_threshold) is
//     charged collision_cost -- unless the core is where it already stands,
//     which is what escape_radius_m protects.
TEST(PredictedRiskSrmCritic, ReachingTheCoreChargesCollisionCost)
{
  auto stack = srm::makeSrmWindow();
  srm::paintStaticCore(stack, 2.0, 0.0);

  std::vector<std::vector<std::pair<double, double>>> cands;
  cands.push_back(srm::alongX(0.0, [](double) {return 0.5;}));   // at the core at t = 4 s
  cands.push_back(srm::alongX(0.0, [](double) {return 0.20;}));  // gets close, never in
  // Already standing on the core, shuffling inside the escape radius.
  cands.push_back(srm::alongX(2.0, [](double) {return 0.02;}));

  const auto costs = srm::score(stack, cands, srm::params());
  EXPECT_GE(costs[0], 5000.0f);
  EXPECT_LT(costs[1], 5000.0f);
  EXPECT_GT(costs[1], 0.0f);
  EXPECT_LT(costs[2], 5000.0f) << "escape_radius_m must not charge a robot "
                                  "for the cell it is already standing in";
  EXPECT_GT(costs[2], 0.0f) << "...but standing in it is not free either";
}

// (c) THE PASS-BEHIND PROPERTY. A mover crosses the robot's path at
//     x = 0.52 m going +y at 1 m/s, through the crossing at t = 2 s.
//
//     A "go now"   0.26 m/s (the arm's vx_max)  -> at the crossing at t = 2 s
//     B "wait"     0.05 m/s for 3 s, then 0.26  -> crosses at t ~ 4.4 s
//     C "dash"     0.60 m/s                     -> crosses at t ~ 0.87 s
//
//     A must cost more than B: that is the whole point of scoring in (x, y, t)
//     rather than against a collapsed "now" grid.
TEST(PredictedRiskSrmCritic, PassBehindCostsLessThanDrivingIntoTheCrossing)
{
  auto stack = srm::makeSrmWindow();
  srm::paintCrossingMover(stack, 0.52, 1.0, 2.0);

  std::vector<std::vector<std::pair<double, double>>> cands;
  cands.push_back(srm::alongX(0.0, [](double) {return 0.26;}));
  cands.push_back(srm::alongX(0.0, [](double t) {return t < 3.0 ? 0.05 : 0.26;}));
  cands.push_back(srm::alongX(0.0, [](double) {return 0.60;}));

  const auto costs = srm::score(stack, cands, srm::params());
  std::cout << "[ SRM      ] pass-behind costs:"
            << " A(go now, 0.26 m/s) = " << costs[0]
            << " | B(wait then cross behind) = " << costs[1]
            << " | C(dash across ahead, 0.60 m/s) = " << costs[2] << std::endl;

  // NOTE (WP-B bench finding): C < B here, NOT between B and A as the brief
  // guessed. The reason is geometric, not a scoring bug: at vx_max 0.26 m/s the
  // robot covers 1.56 m over the whole 6 s horizon while the SRM comet is
  // 2 * d0 = 3.0 m across, so "wait" cannot get the robot OUT of the field --
  // B loiters 0.42 m from the point the mover sweeps through (srm 0.72),
  // while C's closest approach is 0.59 m (srm 0.60). The SRM correctly
  // reports that, at these speeds, dashing keeps more clearance than
  // loitering just short of the crossing. Both are far below A's 5000.
  EXPECT_GT(costs[0], costs[1]) << "driving into the crossing must cost more "
                                   "than waiting and passing behind";
  EXPECT_GE(costs[0], 5000.0f) << "A drives into the core: that is a predicted collision";
  EXPECT_LT(costs[1], 5000.0f) << "B never enters the core";
  EXPECT_LT(costs[2], 5000.0f) << "C clears the crossing before the core arrives";
  EXPECT_GT(costs[0], costs[2]);
}

// Poses outside the SRM window score nothing: the window follows the robot,
// and "no window here" is not "no risk here" -- it is "no information".
TEST(PredictedRiskSrmCritic, OutsideTheWindowCostsNothing)
{
  auto stack = srm::makeSrmWindow();
  srm::paintStaticCore(stack, 2.0, 0.0);
  std::vector<std::vector<std::pair<double, double>>> cands;
  cands.push_back(srm::dwellThenLeave(50.0, 50.0, 60));
  const auto costs = srm::score(stack, cands, srm::params());
  EXPECT_NEAR(costs[0], 0.0f, 1e-4f);
}

// The stack's own age shifts which layer each step reads, exactly as before:
// a 1.5 s-old SRM makes the "go now" candidate read the mover 1.5 s further
// along its crossing, so it no longer meets it.
TEST(PredictedRiskSrmCritic, TimeShiftMovesTheLayer)
{
  auto stack = srm::makeSrmWindow();
  srm::paintCrossingMover(stack, 0.52, 1.0, 2.0);

  std::vector<std::vector<std::pair<double, double>>> cands;
  cands.push_back(srm::alongX(0.0, [](double) {return 0.26;}));

  auto p = srm::params();
  const auto fresh = srm::score(stack, cands, p);
  p.time_shift_s = 1.5;
  const auto stale = srm::score(stack, cands, p);
  EXPECT_GE(fresh[0], 5000.0f);
  EXPECT_LT(stale[0], fresh[0]);
}

// The 1000 x 60 batch budget is unchanged by the SRM rewrite.
TEST(PredictedRiskSrmCritic, BatchScoringIsFastEnough)
{
  auto stack = srm::makeSrmWindow();
  srm::paintCrossingMover(stack, 0.52, 1.0, 2.0);

  const size_t batch = 1000;
  const size_t steps = srm::kHorizon;
  std::vector<float> xs(batch * steps), ys(batch * steps), costs(batch, 0.0f);
  for (size_t b = 0; b < batch; ++b) {
    const double vy = -0.2 + 0.4 * static_cast<double>(b) / static_cast<double>(batch);
    for (size_t j = 0; j < steps; ++j) {
      const double t = static_cast<double>(j) * srm::kModelDt;
      xs[b * steps + j] = static_cast<float>(0.26 * t);
      ys[b * steps + j] = static_cast<float>(vy * t);
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  panoptex_nav::scoreSrmBatch(
    stack, Transform2D{}, xs.data(), ys.data(), batch, steps, srm::kModelDt,
    srm::params(), costs.data());
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
  std::cout << "[ PERF     ] SRM 1000 x 60 batch scored in " << ms << " ms" << std::endl;
  EXPECT_LT(ms, 25.0);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
