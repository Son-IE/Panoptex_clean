// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#include "panoptex_nav/predicted_risk_mppi_critic.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

namespace panoptex_nav
{

void PredictedRiskMppiCritic::initialize()
{
  auto node = parent_.lock();
  if (!node) {
    throw std::runtime_error(
            "PredictedRiskMppiCritic: failed to lock the parent lifecycle node");
  }
  clock_ = node->get_clock();
  recv_time_ = rclcpp::Time(0, 0, clock_->get_clock_type());

  // nav2's MPPI convention: every critic parameter is declared under
  // "<controller>.<critic name>." by the shared ParametersHandler, which is
  // also what makes them dynamically reconfigurable.
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(topic_, "topic", std::string("/risk_stack_srm"));
  getParam(params_.cost_weight, "cost_weight", 30.0);
  getParam(params_.cost_power, "cost_power", 2.0);
  getParam(params_.time_discount, "time_discount", 0.97);

  // WP-B: `lethal_threshold` (a class-confidence threshold on the old
  // class/confidence RiskStack) became `collision_threshold` (a distance
  // threshold on the SRM). The old name is still READ, once, as a deprecated
  // alias: it supplies collision_threshold's default, so a params file that
  // still sets only lethal_threshold keeps working, while one that sets
  // collision_threshold wins outright.
  double deprecated_lethal = -1.0;
  getParam(deprecated_lethal, "lethal_threshold", -1.0);
  const bool lethal_given = deprecated_lethal >= 0.0;
  getParam(
    params_.collision_threshold, "collision_threshold",
    lethal_given ? deprecated_lethal : 0.90);

  getParam(params_.collision_cost, "collision_cost", 5000.0);
  getParam(params_.skip_first_s, "skip_first_s", 0.3);
  getParam(params_.escape_radius_m, "escape_radius_m", 0.30);
  getParam(stale_timeout_s_, "stale_timeout_s", 2.0);
  getParam(risk_frame_, "risk_frame", std::string("map"));
  getParam(warn_period_s_, "warn_period_s", 5.0);

  if (warn_period_s_ <= 0.0) {
    warn_period_s_ = 5.0;
  }

  if (lethal_given) {
    RCLCPP_WARN(
      logger_,
      "PredictedRiskMppiCritic[%s]: `lethal_threshold` is DEPRECATED -- this critic now "
      "scores a Spatiotemporal Risk Map, where the threshold is a distance, not a class "
      "confidence. Rename it to `collision_threshold` (0.90 = within 0.15 m of an "
      "occupied core at d0 = 1.5 m). Using collision_threshold=%.2f for now.",
      name_.c_str(), params_.collision_threshold);
  }

  // The RiskStack publisher is RELIABLE + TRANSIENT_LOCAL + KeepLast(1); the
  // durability must match or the latched message is never delivered.
  sub_ = node->create_subscription<panoptex_msgs::msg::RiskStack>(
    topic_, rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&PredictedRiskMppiCritic::riskStackCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_,
    "PredictedRiskMppiCritic[%s]: SRM mode; topic=%s risk_frame=%s cost_weight=%.2f "
    "cost_power=%.2f time_discount=%.2f collision_threshold=%.2f collision_cost=%.1f "
    "skip_first_s=%.2f escape_radius_m=%.2f stale_timeout_s=%.2f",
    name_.c_str(), topic_.c_str(), risk_frame_.c_str(), params_.cost_weight,
    params_.cost_power, params_.time_discount, params_.collision_threshold,
    params_.collision_cost, params_.skip_first_s, params_.escape_radius_m,
    stale_timeout_s_);
}

void PredictedRiskMppiCritic::riskStackCallback(panoptex_msgs::msg::RiskStack::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  stack_ = std::move(msg);
  recv_time_ = clock_ ? clock_->now() : rclcpp::Time(0, 0, RCL_ROS_TIME);
}

bool PredictedRiskMppiCritic::updateTransform()
{
  const auto warn_ms = static_cast<int64_t>(warn_period_s_ * 1000.0);

  const std::string costmap_frame = costmap_ros_ ? costmap_ros_->getGlobalFrameID() : std::string();
  if (costmap_frame.empty()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskMppiCritic[%s]: costmap global frame is unknown; not scoring.",
      name_.c_str());
    return false;
  }

  if (costmap_frame == risk_frame_) {
    tf_costmap_to_risk_ = Transform2D{};
    return true;
  }

  try {
    const geometry_msgs::msg::TransformStamped tfs =
      costmap_ros_->getTfBuffer()->lookupTransform(
      risk_frame_, costmap_frame, tf2::TimePointZero);
    const double yaw = quaternionYaw(tfs.transform.rotation);
    tf_costmap_to_risk_.cos_theta = std::cos(yaw);
    tf_costmap_to_risk_.sin_theta = std::sin(yaw);
    tf_costmap_to_risk_.tx = tfs.transform.translation.x;
    tf_costmap_to_risk_.ty = tfs.transform.translation.y;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskMppiCritic[%s]: could not transform %s -> %s (%s); not scoring.",
      name_.c_str(), costmap_frame.c_str(), risk_frame_.c_str(), ex.what());
    return false;
  }
  return true;
}

void PredictedRiskMppiCritic::score(mppi::CriticData & data)
{
  active_ = false;
  if (!enabled_) {
    return;
  }

  panoptex_msgs::msg::RiskStack::ConstSharedPtr stack;
  rclcpp::Time recv_time(0, 0, RCL_ROS_TIME);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stack = stack_;          // shared_ptr copy: the callback may swap underneath us
    recv_time = recv_time_;
  }

  const auto warn_ms = static_cast<int64_t>(warn_period_s_ * 1000.0);

  if (!stack) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskMppiCritic[%s]: no RiskStack received on %s yet; not scoring.",
      name_.c_str(), topic_.c_str());
    return;
  }

  const rclcpp::Time now = clock_->now();
  const rclcpp::Time stamp(stack->header.stamp, now.get_clock_type());
  double age = 0.0;
  if (!isStackFresh(stamp, recv_time, now, stale_timeout_s_, &age)) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskMppiCritic[%s]: RiskStack is stale (age %.2fs > %.2fs); not scoring.",
      name_.c_str(), age, stale_timeout_s_);
    return;
  }

  if (!updateTransform()) {
    return;
  }

  const auto & xs = data.trajectories.x;
  const auto & ys = data.trajectories.y;
  if (xs.dimension() != 2 || ys.shape() != xs.shape()) {
    return;
  }
  const std::size_t batch = xs.shape(0);
  const std::size_t steps = xs.shape(1);
  if (batch == 0 || steps == 0 || data.costs.shape(0) != batch) {
    return;
  }

  // The stack's own age: a rollout step's t is relative to *this* control
  // cycle, the stack's layers are relative to its header stamp.
  params_.time_shift_s = std::max(0.0, age);
  active_ = true;

  scoreSrmBatch(
    *stack, tf_costmap_to_risk_, xs.data(), ys.data(), batch, steps,
    static_cast<double>(data.model_dt), params_, data.costs.data());
}

void scoreSrmBatch(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const float * xs,
  const float * ys,
  std::size_t batch,
  std::size_t steps,
  double model_dt,
  const SrmRiskParams & params,
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

  const bool square_power = std::fabs(params.cost_power - 2.0) < 1e-9;
  const bool unit_power = std::fabs(params.cost_power - 1.0) < 1e-9;
  const double escape_r2 = params.escape_radius_m * params.escape_radius_m;

  for (std::size_t b = 0; b < batch; ++b) {
    const float * xrow = xs + b * steps;
    const float * yrow = ys + b * steps;
    // The rollout's own first pose == the robot right now: the escape radius
    // is measured from here, in the costmap frame (tf is rigid, so distances
    // are the same either side of it -- no need to transform twice).
    const double x0 = static_cast<double>(xrow[0]);
    const double y0 = static_cast<double>(yrow[0]);
    double accum = 0.0;

    for (std::size_t j = 0; j < steps; ++j) {
      if (!scored[j]) {
        continue;
      }
      const double xj = static_cast<double>(xrow[j]);
      const double yj = static_cast<double>(yrow[j]);
      double rx = 0.0, ry = 0.0;
      tf.apply(xj, yj, rx, ry);
      const double s = grid.risk(rx, ry, layer[j]);
      if (s <= 0.0) {
        // Outside the SRM window (-1) or genuinely zero risk: nothing to add.
        continue;
      }
      if (s >= params.collision_threshold) {
        const double dx = xj - x0;
        const double dy = yj - y0;
        if (dx * dx + dy * dy > escape_r2) {
          // MPPI style: additive, no exception. Charging it once is enough to
          // make the sample lose the softmax; the rest of the rollout is moot.
          costs[b] += static_cast<float>(params.collision_cost);
          break;
        }
        // Inside the escape radius: no collision charge, but the graded term
        // below still applies, so "sit still in the hot cell" is not free.
      }
      accum += discount[j] *
        (square_power ? s * s : (unit_power ? s : std::pow(s, params.cost_power)));
    }

    costs[b] += static_cast<float>(params.cost_weight * accum);
  }
}

}  // namespace panoptex_nav

PLUGINLIB_EXPORT_CLASS(panoptex_nav::PredictedRiskMppiCritic, mppi::critics::CriticFunction)
