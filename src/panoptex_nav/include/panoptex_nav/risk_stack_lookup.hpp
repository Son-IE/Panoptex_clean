// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#ifndef PANOPTEX_NAV__RISK_STACK_LOOKUP_HPP_
#define PANOPTEX_NAV__RISK_STACK_LOOKUP_HPP_

/// \file
/// Shared, controller-agnostic helpers for reading a panoptex_msgs/RiskStack.
///
/// WP2 (2026-09-09): everything in here used to live inside the DWB critic
/// (src/predicted_risk_critic.cpp).  The MPPI critic
/// (panoptex_nav::PredictedRiskMppiCritic) needs exactly the same geometry,
/// freshness and layer-index rules, so they were lifted out verbatim.
///
/// Nothing here touches a node, a lifecycle, TF or a costmap -- the only ROS
/// types are the message itself and rclcpp::Time (used purely as an arithmetic
/// timestamp), so every rule below is unit-testable without a graph.  See
/// test/test_risk_stack_lookup.cpp.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "geometry_msgs/msg/quaternion.hpp"
#include "rclcpp/time.hpp"

#include "panoptex_msgs/msg/risk_stack.hpp"

namespace panoptex_nav
{

/// Yaw of a quaternion, assuming a (near) planar orientation.
double quaternionYaw(const geometry_msgs::msg::Quaternion & q);

/**
 * @brief A cached planar rigid transform: p_out = R(theta) * p_in + t.
 *
 * Used to move a candidate trajectory point from the local costmap global
 * frame (`odom` on the X3) into the risk grid frame (`map`).
 */
struct Transform2D
{
  double cos_theta{1.0};
  double sin_theta{0.0};
  double tx{0.0};
  double ty{0.0};

  inline void apply(double x, double y, double & out_x, double & out_y) const
  {
    out_x = cos_theta * x - sin_theta * y + tx;
    out_y = sin_theta * x + cos_theta * y + ty;
  }
};

/**
 * @brief Freshness/age test for a received stack.
 *
 * Prefers the message stamp (so it behaves under `use_sim_time`); falls back
 * to the receive time when the publisher left the stamp at zero.  A negative
 * age (stamp slightly in the future) counts as fresh.
 *
 * @param stamp      stack header stamp (may be zero when not stamped)
 * @param recv_time  time the message was received
 * @param now        current time
 * @param timeout_s  staleness threshold in seconds
 * @param age_out    optional output: the age used for the decision
 */
bool isStackFresh(
  const rclcpp::Time & stamp,
  const rclcpp::Time & recv_time,
  const rclcpp::Time & now,
  double timeout_s,
  double * age_out = nullptr);

/**
 * @brief Layer covering trajectory time @p t, shifted by the stack's own age.
 *
 * Layer k of the stack covers `header.stamp + horizon_start + k*dt`, while a
 * candidate's `t` is relative to the *current* control cycle; adding
 * @p time_shift_s (the stack's age) keeps "t seconds from now" meaning the
 * same instant whether the stack just arrived or is a little stale.
 *
 * The result is clamped into [0, steps-1] -- past the last layer the stack
 * simply has nothing newer to say, and clamping is what makes the far end of
 * a long MPPI horizon degrade gracefully instead of falling off a cliff.
 * Returns 0 for a degenerate stack (dt <= 0 or steps == 0).
 */
std::size_t layerIndex(
  double t, double time_shift_s, double horizon_start, double dt, std::size_t steps);

/**
 * @brief Precomputed planar geometry of a RiskStack grid, for hot lookups.
 *
 * Building it costs two trig calls; after that a cell lookup is a rotate,
 * two divisions and one byte load, which is what makes the MPPI critic's
 * batch x horizon inner loop (1000 x 60 cells) affordable.
 */
class StackGrid
{
public:
  StackGrid() = default;
  explicit StackGrid(const panoptex_msgs::msg::RiskStack & stack);

  /// False for a degenerate/truncated stack; every lookup then returns -1.
  bool valid() const {return valid_;}
  std::size_t steps() const {return steps_;}

  /**
   * @brief Risk in [0, 1] at (@p rx, @p ry) *in the risk frame* on layer @p k.
   * @return -1 when the point is outside the grid, the layer is out of range,
   *         or the stack is degenerate.  Unknown cells (raw value < 0) read 0.
   */
  inline double risk(double rx, double ry, std::size_t k) const
  {
    if (!valid_) {return -1.0;}
    // Risk frame -> grid-local (the grid may carry an origin yaw).
    const double dx = rx - ox_;
    const double dy = ry - oy_;
    const double local_x = oc_ * dx + os_ * dy;
    const double local_y = -os_ * dx + oc_ * dy;
    if (local_x < 0.0 || local_y < 0.0) {return -1.0;}
    const double fc = std::floor(local_x / resolution_);
    const double fr = std::floor(local_y / resolution_);
    if (fc < 0.0 || fr < 0.0) {return -1.0;}
    const auto col = static_cast<std::size_t>(fc);
    const auto row = static_cast<std::size_t>(fr);
    if (col >= width_ || row >= height_ || k >= steps_) {return -1.0;}
    const int8_t v = data_[k * layer_size_ + row * width_ + col];
    return (v < 0) ? 0.0 : static_cast<double>(v) / 100.0;
  }

private:
  bool valid_{false};
  double resolution_{0.0};
  double ox_{0.0}, oy_{0.0}, oc_{1.0}, os_{0.0};
  std::size_t width_{0}, height_{0}, steps_{0}, layer_size_{0};
  const int8_t * data_{nullptr};
};

/**
 * @brief One-shot cell lookup (builds a StackGrid internally).
 * @param x_map,y_map point in the *risk* frame
 * @param k           layer index
 * @return risk in [0, 1], or -1 outside the grid / for a degenerate stack.
 */
double cellRisk(
  const panoptex_msgs::msg::RiskStack & stack, double x_map, double y_map, std::size_t k);

/// Scoring parameters shared by the MPPI critic and its unit tests.
struct MppiRiskParams
{
  double cost_weight{5.0};
  double cost_power{1.0};
  double time_discount{0.95};
  double lethal_threshold{0.45};
  double collision_cost{5000.0};
  /// Steps earlier than this are not scored: the robot is practically already
  /// there and no control choice changes them (same escape rationale as the
  /// DWB critic's skip_first_s, which stopped 263 aborted goals on 2026-09-09).
  double skip_first_s{0.3};
  /// Age of the stack at scoring time; see layerIndex().
  double time_shift_s{0.0};
};

/**
 * @brief Score a whole MPPI trajectory batch against a risk stack.
 *
 * @p xs / @p ys are row-major [batch, steps] buffers (mppi::models::Trajectories'
 * xtensor data pointers) in the costmap global frame; @p tf moves them into the
 * risk frame.  For every batch b:
 *
 *   t = j * model_dt, skipping t < skip_first_s
 *   k = layerIndex(t, ...), r = cellRisk(...)  (outside the grid -> ignored)
 *   r >= lethal_threshold  -> costs[b] += collision_cost, stop this trajectory
 *   otherwise              -> accum += time_discount^t * r^cost_power
 *   costs[b] += cost_weight * accum
 *
 * Additive and exception-free, MPPI style: an unreachable-looking trajectory is
 * expensive, never illegal.  @p costs is *accumulated into*, not overwritten.
 */
void scoreStackBatch(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const float * xs,
  const float * ys,
  std::size_t batch,
  std::size_t steps,
  double model_dt,
  const MppiRiskParams & params,
  float * costs);

}  // namespace panoptex_nav

#endif  // PANOPTEX_NAV__RISK_STACK_LOOKUP_HPP_
