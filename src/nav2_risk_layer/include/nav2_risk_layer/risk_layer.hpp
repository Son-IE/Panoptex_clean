#ifndef NAV2_RISK_LAYER__RISK_LAYER_HPP_
#define NAV2_RISK_LAYER__RISK_LAYER_HPP_

#include <cstdint>
#include <mutex>
#include <string>

#include "nav2_costmap_2d/layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nav2_risk_layer
{

class RiskLayer : public nav2_costmap_2d::Layer
{
public:
  RiskLayer();

  void onInitialize() override;

  void updateBounds(
    double robot_x,
    double robot_y,
    double robot_yaw,
    double * min_x,
    double * min_y,
    double * max_x,
    double * max_y) override;

  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i,
    int min_j,
    int max_i,
    int max_j) override;

  void reset() override;

  bool isClearable() override { return false; }

private:
  void riskMapCallback(nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  unsigned char riskValueToCost(int8_t value) const;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscription_;
  std::mutex mutex_;
  nav_msgs::msg::OccupancyGrid::SharedPtr risk_map_;
  std::string topic_;
  bool enabled_;
  bool has_map_;
  int max_cost_;
  int min_risk_value_;
};

}  // namespace nav2_risk_layer

#endif  // NAV2_RISK_LAYER__RISK_LAYER_HPP_
