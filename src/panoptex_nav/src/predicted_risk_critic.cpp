// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#include "panoptex_nav/predicted_risk_critic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "dwb_core/exceptions.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

namespace panoptex_nav
{

double scoreStack(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const dwb_msgs::msg::Trajectory2D & traj,
  const PredictedRiskParams & params,
  ScoreStats * stats)
{
  ScoreStats local_stats;
  ScoreStats & st = (stats != nullptr) ? *stats : local_stats;

  // Geometry, bounds and layer indexing all come from risk_stack_lookup.hpp,
  // shared with the MPPI critic -- see that header for the exact rules.
  const StackGrid grid(stack);
  if (!grid.valid()) {
    return 0.0;
  }

  const std::size_t n = std::min(traj.poses.size(), traj.time_offsets.size());
  double score = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const double t =
      static_cast<double>(traj.time_offsets[i].sec) +
      1e-9 * static_cast<double>(traj.time_offsets[i].nanosec);

    if (t > params.max_horizon_s || t < params.skip_first_s) {
      ++st.skipped;
      continue;
    }
    // Escape rule: never make a pose the robot practically already
    // occupies illegal (see PredictedRiskParams::escape_radius_m).
    const bool near_start = params.escape_radius_m > 0.0 &&
      std::hypot(traj.poses[i].x - traj.poses[0].x, traj.poses[i].y - traj.poses[0].y) <=
      params.escape_radius_m;

    // Costmap global frame -> risk frame -> cell.
    double rx = 0.0, ry = 0.0;
    tf.apply(traj.poses[i].x, traj.poses[i].y, rx, ry);

    // Time offset -> layer index, shifted by the stack's age so a pose at
    // control-cycle time t reads the layer that covers the same instant.
    const std::size_t k = layerIndex(
      t, params.time_shift_s, static_cast<double>(stack.horizon_start),
      static_cast<double>(stack.dt), grid.steps());

    const double r = grid.risk(rx, ry, k);
    if (r < 0.0) {   // outside the grid
      ++st.skipped;
      continue;
    }
    ++st.scored;

    if (r >= params.lethal_threshold && !near_start) {
      throw dwb_core::IllegalTrajectoryException(
              params.critic_name,
              "predicted collision at t=" + std::to_string(t) + "s (risk " +
              std::to_string(r) + ")");
    }

    if (r > 0.0) {
      score += std::pow(params.time_discount, t) * std::pow(r, params.cost_power);
    }
  }

  return score;
}

void PredictedRiskCritic::onInit()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("PredictedRiskCritic: failed to lock the parent lifecycle node");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  recv_time_ = rclcpp::Time(0, 0, clock_->get_clock_type());

  const std::string prefix = dwb_plugin_name_ + "." + name_ + ".";

  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "topic", rclcpp::ParameterValue(std::string("/risk_stack")));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "cost_power", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "time_discount", rclcpp::ParameterValue(0.9));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "lethal_threshold", rclcpp::ParameterValue(0.85));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "stale_timeout_s", rclcpp::ParameterValue(2.0));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "max_horizon_s", rclcpp::ParameterValue(3.3));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "skip_first_s", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "escape_radius_m", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "risk_frame", rclcpp::ParameterValue(std::string("map")));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "warn_period_s", rclcpp::ParameterValue(5.0));

  node->get_parameter(prefix + "topic", topic_);
  node->get_parameter(prefix + "cost_power", params_.cost_power);
  node->get_parameter(prefix + "time_discount", params_.time_discount);
  node->get_parameter(prefix + "lethal_threshold", params_.lethal_threshold);
  node->get_parameter(prefix + "stale_timeout_s", stale_timeout_s_);
  node->get_parameter(prefix + "max_horizon_s", params_.max_horizon_s);
  node->get_parameter(prefix + "skip_first_s", params_.skip_first_s);
  node->get_parameter(prefix + "escape_radius_m", params_.escape_radius_m);
  node->get_parameter(prefix + "risk_frame", risk_frame_);
  node->get_parameter(prefix + "warn_period_s", warn_period_s_);
  params_.critic_name = name_;

  if (warn_period_s_ <= 0.0) {
    warn_period_s_ = 5.0;
  }

  // The RiskStack publisher is RELIABLE + TRANSIENT_LOCAL + KeepLast(1); the
  // durability must match or the latched message is never delivered.
  sub_ = node->create_subscription<panoptex_msgs::msg::RiskStack>(
    topic_, rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&PredictedRiskCritic::riskStackCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_,
    "PredictedRiskCritic[%s]: topic=%s risk_frame=%s cost_power=%.2f time_discount=%.2f "
    "lethal_threshold=%.2f max_horizon_s=%.2f stale_timeout_s=%.2f",
    name_.c_str(), topic_.c_str(), risk_frame_.c_str(), params_.cost_power,
    params_.time_discount, params_.lethal_threshold, params_.max_horizon_s, stale_timeout_s_);
}

void PredictedRiskCritic::riskStackCallback(panoptex_msgs::msg::RiskStack::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  stack_ = std::move(msg);
  recv_time_ = clock_ ? clock_->now() : rclcpp::Time(0, 0, RCL_ROS_TIME);
}

bool PredictedRiskCritic::prepare(
  const geometry_msgs::msg::Pose2D &,
  const nav_2d_msgs::msg::Twist2D &,
  const geometry_msgs::msg::Pose2D &,
  const nav_2d_msgs::msg::Path2D &)
{
  active_ = false;
  active_stack_.reset();
  cycle_illegal_ = 0;
  cycle_scored_ = 0;

  panoptex_msgs::msg::RiskStack::ConstSharedPtr stack;
  rclcpp::Time recv_time(0, 0, RCL_ROS_TIME);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stack = stack_;
    recv_time = recv_time_;
  }

  const auto warn_ms = static_cast<int64_t>(warn_period_s_ * 1000.0);

  if (!stack) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskCritic[%s]: no RiskStack received on %s yet; scoring 0.",
      name_.c_str(), topic_.c_str());
    return true;
  }

  const rclcpp::Time now = clock_->now();
  const rclcpp::Time stamp(stack->header.stamp, now.get_clock_type());
  double age = 0.0;
  if (!isStackFresh(stamp, recv_time, now, stale_timeout_s_, &age)) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskCritic[%s]: RiskStack is stale (age %.2fs > %.2fs); scoring 0.",
      name_.c_str(), age, stale_timeout_s_);
    return true;
  }

  // TF: risk_frame <- costmap global frame.
  const std::string costmap_frame = costmap_ros_ ? costmap_ros_->getGlobalFrameID() : std::string();
  if (costmap_frame.empty()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, warn_ms,
      "PredictedRiskCritic[%s]: costmap global frame is unknown; scoring 0.", name_.c_str());
    return true;
  }

  if (costmap_frame == risk_frame_) {
    tf_costmap_to_risk_ = Transform2D{};
  } else {
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
        "PredictedRiskCritic[%s]: could not transform %s -> %s (%s); scoring 0.",
        name_.c_str(), costmap_frame.c_str(), risk_frame_.c_str(), ex.what());
      return true;
    }
  }

  active_stack_ = stack;
  params_.time_shift_s = std::max(0.0, age);
  active_ = true;
  return true;
}

double PredictedRiskCritic::scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj)
{
  if (!active_ || !active_stack_) {
    return 0.0;
  }

  ScoreStats stats;
  try {
    const double score = scoreStack(*active_stack_, tf_costmap_to_risk_, traj, params_, &stats);
    cycle_scored_ += stats.scored;
    return score;
  } catch (const dwb_core::IllegalTrajectoryException &) {
    ++cycle_illegal_;
    throw;
  }
}

void PredictedRiskCritic::debrief(const nav_2d_msgs::msg::Twist2D &)
{
  RCLCPP_DEBUG(
    logger_,
    "PredictedRiskCritic[%s]: active=%d illegal_trajectories=%zu scored_poses=%zu",
    name_.c_str(), static_cast<int>(active_), cycle_illegal_, cycle_scored_);
  cycle_illegal_ = 0;
  cycle_scored_ = 0;
}

void PredictedRiskCritic::reset()
{
  active_ = false;
  active_stack_.reset();
  cycle_illegal_ = 0;
  cycle_scored_ = 0;
}

}  // namespace panoptex_nav

PLUGINLIB_EXPORT_CLASS(panoptex_nav::PredictedRiskCritic, dwb_core::TrajectoryCritic)
