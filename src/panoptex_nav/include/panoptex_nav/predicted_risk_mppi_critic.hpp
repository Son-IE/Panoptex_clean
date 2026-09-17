// Copyright 2026 Panoptex
// Licensed under the Apache License, Version 2.0

#ifndef PANOPTEX_NAV__PREDICTED_RISK_MPPI_CRITIC_HPP_
#define PANOPTEX_NAV__PREDICTED_RISK_MPPI_CRITIC_HPP_

#include <memory>
#include <mutex>
#include <string>

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/critic_data.hpp"
#include "rclcpp/rclcpp.hpp"

#include "panoptex_msgs/msg/risk_stack.hpp"
#include "panoptex_nav/risk_stack_lookup.hpp"

namespace panoptex_nav
{

/// \brief Scoring parameters for the Spatiotemporal Risk Map (SRM) critic.
///
/// WP-B (2026-09-10): the critic no longer scores the raw class/confidence
/// RiskStack; it scores a *Spatiotemporal Risk Map* (Thomas et al., 2021) --
/// the same panoptex_msgs/RiskStack wire type, but every cell already carries
/// a DISTANCE-based risk in [0, 1]: 1 on an occupied core, falling linearly to
/// 0 at `d0` (1.5 m by default, set by the publisher, not here). Because the
/// value is a smooth distance field rather than a class probability, the two
/// scoring knobs change meaning:
///
///  * `cost_power` 2 (was 1) -- a squared distance field is what turns "keep
///    some clearance" into a gradient the softmax can actually follow;
///  * `collision_threshold` 0.90 (replaces `lethal_threshold` 0.45) -- with a
///    linear falloff, 0.90 is "within 0.15 m of an occupied core" at d0 = 1.5,
///    i.e. a geometric statement, not a confidence one.
struct SrmRiskParams
{
  double cost_weight{30.0};
  double cost_power{2.0};
  double time_discount{0.97};
  /// SRM value at or above which a pose counts as a predicted collision.
  double collision_threshold{0.90};
  double collision_cost{5000.0};
  /// Steps earlier than this are not scored (see skip_first_s in the README).
  double skip_first_s{0.3};
  /// Escape radius: a pose closer than this to the rollout's OWN first pose
  /// (i.e. to the robot right now) can never be charged `collision_cost`. A
  /// robot that is already standing inside a hot core must keep a way out --
  /// if every rollout is charged the same 5000, the softmax is flat and the
  /// robot freezes. `skip_first_s` does this in time, this does it in space
  /// (a rollout that barely moves stays cheap however long it lingers).
  double escape_radius_m{0.30};
  /// Age of the stack at scoring time; see layerIndex().
  double time_shift_s{0.0};
};

/**
 * @brief Score a whole MPPI trajectory batch against a Spatiotemporal Risk Map.
 *
 * @p xs / @p ys are row-major [batch, steps] buffers in the costmap global
 * frame; @p tf moves them into the risk frame. For every batch b:
 *
 *   t = j * model_dt, skipping t < skip_first_s
 *   k = layerIndex(t, ...), s = srm(x_t, y_t, k)  (outside the window -> 0)
 *   s >= collision_threshold, and the pose is farther than escape_radius_m
 *       from pose 0 -> costs[b] += collision_cost, stop this trajectory
 *   otherwise -> accum += time_discount^t * s^cost_power
 *   costs[b] += cost_weight * accum
 *
 * Additive and exception-free, MPPI style. @p costs is accumulated into.
 *
 * Lives here rather than in risk_stack_lookup.hpp because the SRM value
 * semantics (and therefore collision_threshold / escape_radius_m) belong to
 * this critic; the geometry helpers it uses (StackGrid, layerIndex) are still
 * the shared ones.
 */
void scoreSrmBatch(
  const panoptex_msgs::msg::RiskStack & stack,
  const Transform2D & tf,
  const float * xs,
  const float * ys,
  std::size_t batch,
  std::size_t steps,
  double model_dt,
  const SrmRiskParams & params,
  float * costs);

/**
 * @class PredictedRiskMppiCritic
 * @brief MPPI critic scoring a whole trajectory batch against the
 *        Spatiotemporal Risk Map (panoptex_msgs/RiskStack on /risk_stack_srm).
 *
 * The MPPI sibling of panoptex_nav::PredictedRiskCritic (DWB). Same contract --
 * step j of a rollout is scored against the stack layer covering that step's own
 * instant, so a rollout may legally pass through a cell that is lethal *now* as
 * long as it arrives after the hazard has moved on -- but MPPI-shaped:
 *
 *  - cost is ADDITIVE, never an exception: a predicted collision adds
 *    `collision_cost` (which the softmax then drowns) instead of invalidating
 *    the sample, so the optimizer always has a distribution to update;
 *  - the whole batch is scored in one pass over the raw xtensor buffers;
 *  - `data.fail_flag` is never set: this critic must never be the reason MPPI
 *    reports "no valid trajectories".
 *
 * The stack it reads is a WINDOW around the robot (its own info.origin and
 * size, 21 layers x 0.3 s); poses outside that window score 0, exactly as
 * poses outside the old full-map grid did.
 *
 * Fail-soft exactly like the DWB critic: no stack, a stale stack, an unknown
 * costmap frame or a missing TF -> inactive for this cycle, throttled warning,
 * costs left untouched.
 *
 * Params live under `<controller>.<critic name>.` (nav2's flat dotted-key
 * convention), e.g. `controller_server.FollowPath.PredictedRiskCritic.topic`.
 */
class PredictedRiskMppiCritic : public mppi::critics::CriticFunction
{
public:
  PredictedRiskMppiCritic() = default;

  void initialize() override;
  void score(mppi::CriticData & data) override;

  /// True when the last score() call found a usable stack + TF. For tests/logs.
  bool isActive() const {return active_;}

protected:
  void riskStackCallback(panoptex_msgs::msg::RiskStack::ConstSharedPtr msg);

  /// Refresh tf_costmap_to_risk_ for this cycle. False -> go inactive.
  bool updateTransform();

  // Parameters
  std::string topic_{"/risk_stack_srm"};
  std::string risk_frame_{"map"};
  double stale_timeout_s_{2.0};
  double warn_period_s_{5.0};
  SrmRiskParams params_;

  // Latest received stack (swapped under the mutex, read as a shared_ptr copy)
  std::mutex mutex_;
  panoptex_msgs::msg::RiskStack::ConstSharedPtr stack_;
  rclcpp::Time recv_time_{0, 0, RCL_ROS_TIME};

  bool active_{false};
  Transform2D tf_costmap_to_risk_;

  rclcpp::Subscription<panoptex_msgs::msg::RiskStack>::SharedPtr sub_;
  rclcpp::Clock::SharedPtr clock_;
};

}  // namespace panoptex_nav

#endif  // PANOPTEX_NAV__PREDICTED_RISK_MPPI_CRITIC_HPP_
