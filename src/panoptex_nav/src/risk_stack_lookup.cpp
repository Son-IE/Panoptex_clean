// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#include "panoptex_nav/risk_stack_lookup.hpp"

#include <algorithm>
#include <vector>

namespace panoptex_nav
{

double quaternionYaw(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

bool isStackFresh(
  const rclcpp::Time & stamp,
  const rclcpp::Time & recv_time,
  const rclcpp::Time & now,
  double timeout_s,
  double * age_out)
{
  // Prefer the message stamp (works under sim time); fall back to the receive
  // time when the publisher left the stamp at zero.
  const rclcpp::Time reference = (stamp.nanoseconds() > 0) ? stamp : recv_time;
  if (reference.nanoseconds() <= 0) {
    if (age_out) {*age_out = std::numeric_limits<double>::infinity();}
    return false;
  }
  const double age = (now - reference).seconds();
  if (age_out) {*age_out = age;}
  // Negative age (stamp slightly in the future) is fine.
  return age <= timeout_s;
}

std::size_t layerIndex(
  double t, double time_shift_s, double horizon_start, double dt, std::size_t steps)
{
  if (dt <= 0.0 || steps == 0) {
    return 0;
  }
  long k = std::lround((t + time_shift_s - horizon_start) / dt);
  k = std::max<long>(0, std::min<long>(k, static_cast<long>(steps) - 1));
  return static_cast<std::size_t>(k);
}

StackGrid::StackGrid(const panoptex_msgs::msg::RiskStack & stack)
{
  resolution_ = stack.info.resolution;
  width_ = stack.info.width;
  height_ = stack.info.height;
  steps_ = stack.steps;
  layer_size_ = width_ * height_;

  if (resolution_ <= 0.0 || width_ == 0 || height_ == 0 || steps_ == 0 ||
    stack.data.size() < steps_ * layer_size_)
  {
    valid_ = false;
    return;
  }

  // Grid pose in the risk frame (origin position + yaw), as nav2_risk_layer does.
  const double origin_yaw = quaternionYaw(stack.info.origin.orientation);
  oc_ = std::cos(origin_yaw);
  os_ = std::sin(origin_yaw);
  ox_ = stack.info.origin.position.x;
  oy_ = stack.info.origin.position.y;
  data_ = stack.data.data();
  valid_ = true;
}

double cellRisk(
  const panoptex_msgs::msg::RiskStack & stack, double x_map, double y_map, std::size_t k)
{
  const StackGrid grid(stack);
  return grid.risk(x_map, y_map, k);
}

void scoreStackBatch(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const float * xs,
  const float * ys,
  std::size_t batch,
  std::size_t steps,
  double model_dt,
  const MppiRiskParams & params,
  float * costs)
{
  if (batch == 0 || steps == 0 || xs == nullptr || ys == nullptr || costs == nullptr) {
    return;
  }
  const StackGrid grid(stack);
  if (!grid.valid()) {
    return;
  }

  // Layer index and time discount depend only on the step, not on the sample,
  // so they are hoisted out of the batch loop (60 entries for the study arms).
  std::vector<std::size_t> layer(steps, 0);
  std::vector<double> discount(steps, 0.0);
  std::vector<char> scored(steps, 0);
  for (std::size_t j = 0; j < steps; ++j) {
    const double t = static_cast<double>(j) * model_dt;
    if (t < params.skip_first_s) {
      continue;
    }
    scored[j] = 1;
    layer[j] = layerIndex(
      t, params.time_shift_s, static_cast<double>(stack.horizon_start),
      static_cast<double>(stack.dt), grid.steps());
    discount[j] = std::pow(params.time_discount, t);
  }

  const bool unit_power = std::fabs(params.cost_power - 1.0) < 1e-9;

  for (std::size_t b = 0; b < batch; ++b) {
    const float * xrow = xs + b * steps;
    const float * yrow = ys + b * steps;
    double accum = 0.0;

    for (std::size_t j = 0; j < steps; ++j) {
      if (!scored[j]) {
        continue;
      }
      double rx = 0.0, ry = 0.0;
      tf.apply(static_cast<double>(xrow[j]), static_cast<double>(yrow[j]), rx, ry);
      const double r = grid.risk(rx, ry, layer[j]);
      if (r < 0.0) {          // outside the grid: no information, not a hazard
        continue;
      }
      if (r >= params.lethal_threshold) {
        // MPPI style: additive, no exception. Charging it once is enough to
        // make the sample lose the softmax; the rest of the rollout is moot.
        costs[b] += static_cast<float>(params.collision_cost);
        break;
      }
      if (r > 0.0) {
        accum += discount[j] * (unit_power ? r : std::pow(r, params.cost_power));
      }
    }

    costs[b] += static_cast<float>(params.cost_weight * accum);
  }
}

}  // namespace panoptex_nav
