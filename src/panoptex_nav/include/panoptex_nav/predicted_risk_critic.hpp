// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#ifndef PANOPTEX_NAV__PREDICTED_RISK_CRITIC_HPP_
#define PANOPTEX_NAV__PREDICTED_RISK_CRITIC_HPP_

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "dwb_core/trajectory_critic.hpp"
#include "dwb_msgs/msg/trajectory2_d.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "rclcpp/rclcpp.hpp"

#include "panoptex_msgs/msg/risk_stack.hpp"
// WP2: quaternionYaw / Transform2D / isStackFresh / layerIndex / cellRisk now
// live here, shared verbatim with the MPPI critic (predicted_risk_mppi_critic.hpp).
#include "panoptex_nav/risk_stack_lookup.hpp"

namespace panoptex_nav
{

/// Scoring parameters for scoreStack(); mirrors the ROS parameters of the critic.
struct PredictedRiskParams
{
  double cost_power{1.0};
  double time_discount{0.9};
  double lethal_threshold{0.85};
  double max_horizon_s{3.3};
  /// Age of the stack at scoring time (seconds). Layer 0 is "now" as of the
  /// publisher's header.stamp, while trajectory time offsets are relative to the
  /// control cycle; adding the age picks the layer that actually corresponds to
  /// the pose's wall/sim time. Set by prepare(); 0 for a fresh stack.
  double time_shift_s{0.0};
  /// Poses earlier than this along a candidate are not scored at all: the
  /// robot is already there and no control choice changes them (DWB's
  /// first pose is t=0 -- rejecting on it makes EVERY candidate illegal
  /// the moment the robot's own cell is lethal, which froze the X3 and
  /// aborted goals 263 times on 2026-09-09).
  double skip_first_s{0.5};
  /// Poses within this distance of the candidate's first pose (the robot's
  /// current position) add graded cost but never throw: leaving a lethal
  /// cell must stay possible.
  double escape_radius_m{0.3};
  /// Name reported in the IllegalTrajectoryException (the DWB critic instance name).
  std::string critic_name{"PredictedRisk"};
};

/// Bookkeeping filled in by scoreStack() for debug logging.
struct ScoreStats
{
  std::size_t scored{0};   ///< poses that contributed a (possibly zero) risk sample
  std::size_t skipped{0};  ///< poses beyond the horizon or outside the grid
};

/**
 * @brief Score one candidate trajectory against a time-layered risk stack.
 *
 * Pure function: no ROS state, so it is directly unit testable.
 *
 * For every pose i with time offset t:
 *  - t > max_horizon_s          -> skipped
 *  - pose transformed into the risk frame, converted to a cell using
 *    info.origin (position + yaw) and info.resolution; outside -> skipped
 *  - layer k = clamp(lround((t - horizon_start) / dt), 0, steps - 1)
 *  - r = data[k*H*W + row*W + col] / 100 (unknown, i.e. < 0, counts as 0)
 *  - r >= lethal_threshold      -> throws dwb_core::IllegalTrajectoryException
 *  - otherwise                  -> score += time_discount^t * r^cost_power
 *
 * @throws dwb_core::IllegalTrajectoryException on a predicted collision.
 */
double scoreStack(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const dwb_msgs::msg::Trajectory2D & traj,
  const PredictedRiskParams & params,
  ScoreStats * stats = nullptr);

/**
 * @class PredictedRiskCritic
 * @brief DWB trajectory critic scoring candidate trajectories against a
 *        time-layered predictive risk field (panoptex_msgs/RiskStack).
 *
 * Unlike a costmap based critic, this one indexes the risk field by the time
 * offset of each trajectory pose, so a trajectory may legally pass through a
 * cell that is occupied *now* as long as it arrives after the hazard has left.
 *
 * The critic fails soft: if no risk stack has been received, if the stack is
 * stale, or if the TF from the costmap frame into the risk frame is
 * unavailable, it scores every trajectory 0 and never blocks DWB.
 */
class PredictedRiskCritic : public dwb_core::TrajectoryCritic
{
public:
  PredictedRiskCritic() = default;

  void onInit() override;
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose,
    const nav_2d_msgs::msg::Twist2D & vel,
    const geometry_msgs::msg::Pose2D & goal,
    const nav_2d_msgs::msg::Path2D & global_plan) override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & traj) override;
  void debrief(const nav_2d_msgs::msg::Twist2D & cmd_vel) override;
  void reset() override;

  /**
   * @brief Freshness test used by prepare(). Thin alias for
   *        panoptex_nav::isStackFresh() (risk_stack_lookup.hpp), kept so the
   *        critic's own contract stays discoverable from this class.
   */
  static bool isStackFresh(
    const rclcpp::Time & stamp,
    const rclcpp::Time & recv_time,
    const rclcpp::Time & now,
    double timeout_s,
    double * age_out = nullptr)
  {
    return panoptex_nav::isStackFresh(stamp, recv_time, now, timeout_s, age_out);
  }

  /// Exposed for tests / introspection: true when prepare() found a usable stack + TF.
  bool isActive() const {return active_;}

protected:
  void riskStackCallback(panoptex_msgs::msg::RiskStack::ConstSharedPtr msg);

  // Parameters
  std::string topic_{"/risk_stack"};
  std::string risk_frame_{"map"};
  double stale_timeout_s_{2.0};
  double warn_period_s_{5.0};
  PredictedRiskParams params_;

  // Latest received stack
  std::mutex mutex_;
  panoptex_msgs::msg::RiskStack::ConstSharedPtr stack_;
  rclcpp::Time recv_time_{0, 0, RCL_ROS_TIME};

  // Per-cycle state produced by prepare()
  bool active_{false};
  panoptex_msgs::msg::RiskStack::ConstSharedPtr active_stack_;
  Transform2D tf_costmap_to_risk_;

  // Per-cycle debug counters
  std::size_t cycle_illegal_{0};
  std::size_t cycle_scored_{0};

  rclcpp::Subscription<panoptex_msgs::msg::RiskStack>::SharedPtr sub_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("PredictedRiskCritic")};
};

}  // namespace panoptex_nav

#endif  // PANOPTEX_NAV__PREDICTED_RISK_CRITIC_HPP_
