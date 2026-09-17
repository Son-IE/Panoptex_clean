#include "nav2_risk_layer/risk_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace nav2_risk_layer
{

namespace
{

std::string normalizeFrame(std::string frame)
{
  if (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

double quaternionYaw(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

RiskLayer::RiskLayer()
: enabled_(true), has_map_(false), max_cost_(200), min_risk_value_(1)
{
}

void RiskLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("RiskLayer failed to lock the parent lifecycle node");
  }

  node->declare_parameter(name_ + ".enabled", rclcpp::ParameterValue(true));
  node->declare_parameter(name_ + ".topic", rclcpp::ParameterValue("/risk_map"));
  node->declare_parameter(name_ + ".max_cost", rclcpp::ParameterValue(200));
  node->declare_parameter(name_ + ".min_risk_value", rclcpp::ParameterValue(1));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic_);
  node->get_parameter(name_ + ".max_cost", max_cost_);
  node->get_parameter(name_ + ".min_risk_value", min_risk_value_);

  max_cost_ = std::clamp(max_cost_, 1, 252);
  min_risk_value_ = std::clamp(min_risk_value_, 0, 100);

  const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
  subscription_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
    topic_, qos,
    std::bind(&RiskLayer::riskMapCallback, this, std::placeholders::_1));

  current_ = true;
  RCLCPP_INFO(
    node->get_logger(),
    "RiskLayer '%s' listening on '%s' (max_cost=%d, min_risk_value=%d)",
    name_.c_str(), topic_.c_str(), max_cost_, min_risk_value_);
}

void RiskLayer::riskMapCallback(nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  auto node = node_.lock();
  if (!node) { return; }

  const std::size_t expected_size =
    static_cast<std::size_t>(msg->info.width) *
    static_cast<std::size_t>(msg->info.height);

  if (msg->data.size() != expected_size) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Ignoring malformed risk map: data size=%zu, expected=%zu",
      msg->data.size(), expected_size);
    return;
  }

  const std::string incoming_frame = normalizeFrame(msg->header.frame_id);
  const std::string costmap_frame = normalizeFrame(layered_costmap_->getGlobalFrameID());

  if (incoming_frame != costmap_frame) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Ignoring risk map in frame '%s'; this costmap uses frame '%s'.",
      incoming_frame.c_str(), costmap_frame.c_str());
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    risk_map_ = std::move(msg);
    has_map_ = true;
  }
  current_ = true;
}

void RiskLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) { return; }

  nav_msgs::msg::OccupancyGrid::SharedPtr map;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_map_ || !risk_map_) { return; }
    map = risk_map_;
  }

  const double width_m = static_cast<double>(map->info.width) * map->info.resolution;
  const double height_m = static_cast<double>(map->info.height) * map->info.resolution;
  const double yaw = quaternionYaw(map->info.origin.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double ox = map->info.origin.position.x;
  const double oy = map->info.origin.position.y;

  const double local_x[4] = {0.0, width_m, 0.0, width_m};
  const double local_y[4] = {0.0, 0.0, height_m, height_m};

  for (int k = 0; k < 4; ++k) {
    const double wx = ox + c * local_x[k] - s * local_y[k];
    const double wy = oy + s * local_x[k] + c * local_y[k];
    *min_x = std::min(*min_x, wx);
    *min_y = std::min(*min_y, wy);
    *max_x = std::max(*max_x, wx);
    *max_y = std::max(*max_y, wy);
  }
}

unsigned char RiskLayer::riskValueToCost(int8_t value) const
{
  if (value < min_risk_value_ || value < 0) {
    return nav2_costmap_2d::FREE_SPACE;
  }

  const double normalized =
    std::clamp(static_cast<double>(value) / 100.0, 0.0, 1.0);
  const int cost = static_cast<int>(
    std::lround(normalized * static_cast<double>(max_cost_)));
  return static_cast<unsigned char>(std::clamp(cost, 0, 252));
}

void RiskLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) { return; }

  nav_msgs::msg::OccupancyGrid::SharedPtr map;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_map_ || !risk_map_) { return; }
    map = risk_map_;
  }

  const double resolution = map->info.resolution;
  if (resolution <= 0.0) { return; }

  const double yaw = quaternionYaw(map->info.origin.orientation);
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double ox = map->info.origin.position.x;
  const double oy = map->info.origin.position.y;

  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      double wx, wy;
      master_grid.mapToWorld(
        static_cast<unsigned int>(i), static_cast<unsigned int>(j), wx, wy);

      const double dx = wx - ox;
      const double dy = wy - oy;
      const double local_x = c * dx + s * dy;
      const double local_y = -s * dx + c * dy;

      if (local_x < 0.0 || local_y < 0.0) { continue; }

      const auto gx = static_cast<unsigned int>(std::floor(local_x / resolution));
      const auto gy = static_cast<unsigned int>(std::floor(local_y / resolution));
      if (gx >= map->info.width || gy >= map->info.height) { continue; }

      const std::size_t risk_index =
        static_cast<std::size_t>(gy) * map->info.width + gx;
      const int8_t risk_value = map->data[risk_index];
      if (risk_value < 0) { continue; }

      const unsigned char risk_cost = riskValueToCost(risk_value);
      if (risk_cost == nav2_costmap_2d::FREE_SPACE) { continue; }

      const unsigned char current_cost = master_grid.getCost(i, j);
      if (current_cost == nav2_costmap_2d::NO_INFORMATION) { continue; }
      master_grid.setCost(i, j, std::max(current_cost, risk_cost));
    }
  }
  current_ = true;
}

void RiskLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  risk_map_.reset();
  has_map_ = false;
  current_ = false;
}

}  // namespace nav2_risk_layer

PLUGINLIB_EXPORT_CLASS(nav2_risk_layer::RiskLayer, nav2_costmap_2d::Layer)
