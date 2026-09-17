cmake_minimum_required(VERSION 3.8)
project(panoptex_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(dwb_core REQUIRED)
find_package(dwb_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(panoptex_msgs REQUIRED)
# WP2: MPPI critic. nav2_mppi_controller's CriticFunction/ParametersHandler are
# header-only from our side (templates + inline), but its headers pull in
# xtensor, which ships its own CMake config (libxtensor-dev, header-only).
find_package(nav2_mppi_controller REQUIRED)
find_package(xtensor REQUIRED)

set(dependencies
  rclcpp
  rclcpp_lifecycle
  nav2_util
  nav2_core
  nav2_costmap_2d
  dwb_core
  dwb_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pluginlib
  panoptex_msgs)

set(mppi_dependencies
  ${dependencies}
  nav2_mppi_controller)

# ------------------------------------------------------- shared stack lookup
# WP2: the RiskStack geometry / freshness / layer-index rules, shared verbatim
# by the DWB and the MPPI critic (include/panoptex_nav/risk_stack_lookup.hpp).
add_library(risk_stack_lookup SHARED src/risk_stack_lookup.cpp)
target_compile_features(risk_stack_lookup PUBLIC cxx_std_17)
target_include_directories(risk_stack_lookup PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(risk_stack_lookup rclcpp geometry_msgs panoptex_msgs)

# ------------------------------------------------------------ DWB critic
add_library(predicted_risk_critic SHARED src/predicted_risk_critic.cpp)
target_compile_features(predicted_risk_critic PUBLIC cxx_std_17)
target_include_directories(predicted_risk_critic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(predicted_risk_critic risk_stack_lookup)
ament_target_dependencies(predicted_risk_critic ${dependencies})

# ------------------------------------------------------------ MPPI critic
add_library(panoptex_nav_mppi_critic SHARED src/predicted_risk_mppi_critic.cpp)
target_compile_features(panoptex_nav_mppi_critic PUBLIC cxx_std_17)
target_include_directories(panoptex_nav_mppi_critic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(panoptex_nav_mppi_critic risk_stack_lookup xtensor)
ament_target_dependencies(panoptex_nav_mppi_critic ${mppi_dependencies})

# One plugins.xml, two loaders (see that file's header comment).
pluginlib_export_plugin_description_file(dwb_core plugins.xml)
pluginlib_export_plugin_description_file(nav2_mppi_controller plugins.xml)

install(TARGETS risk_stack_lookup predicted_risk_critic panoptex_nav_mppi_critic
  EXPORT export_predicted_risk_critic
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# ------------------------------------------------------- Python (later WPs)
ament_python_install_package(${PROJECT_NAME})

# WP3.2: risk_speed_governor -- thin launcher, real node lives in
# panoptex_nav/risk_speed_governor.py (installed above by
# ament_python_install_package).
install(PROGRAMS
  scripts/risk_speed_governor
  DESTINATION lib/${PROJECT_NAME})

# WP-B: mission_supervisor -- same thin-launcher pattern, real node in
# panoptex_nav/mission_supervisor.py (+ the pure-geometry
# panoptex_nav/corridor.py it wires up).
install(PROGRAMS
  scripts/mission_supervisor
  DESTINATION lib/${PROJECT_NAME})

# WP-C: gt_tracks -- same thin-launcher pattern, real node in
# panoptex_nav/gt_tracks_node.py. Oracle perception for x3_nav.launch.py's
# perception:=oracle mode (ground-truth Detection3DArray tracks off
# /gt_tf, no cameras/GDINO/SAM2/tracker in the loop).
install(PROGRAMS
  scripts/gt_tracks
  DESTINATION lib/${PROJECT_NAME})

# ------------------------------------------------- launch / config (may be empty)
foreach(share_dir launch config)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${share_dir}")
    install(DIRECTORY ${share_dir}/ DESTINATION share/${PROJECT_NAME}/${share_dir})
  endif()
endforeach()

# ------------------------------------------------------------------- testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_predicted_risk_critic test/test_predicted_risk_critic.cpp)
  target_link_libraries(test_predicted_risk_critic predicted_risk_critic)
  ament_target_dependencies(test_predicted_risk_critic ${dependencies})

  # WP2: the shared RiskStack lookup rules (layer index incl. time shift and
  # clamping, rotated-origin cell lookup, out-of-grid -> -1, batch scoring).
  ament_add_gtest(test_risk_stack_lookup test/test_risk_stack_lookup.cpp)
  target_link_libraries(test_risk_stack_lookup risk_stack_lookup)
  ament_target_dependencies(test_risk_stack_lookup rclcpp geometry_msgs panoptex_msgs)

  # WP2: the MPPI critic -- pluginlib loadability plus the scoring semantics
  # its score() delegates to (see that file's header comment for why the
  # scoring math is tested through scoreStackBatch rather than through a
  # standalone Costmap2DROS + ParametersHandler rig).
  ament_add_gtest(test_predicted_risk_mppi_critic test/test_predicted_risk_mppi_critic.cpp)
  target_link_libraries(test_predicted_risk_mppi_critic panoptex_nav_mppi_critic)
  ament_target_dependencies(test_predicted_risk_mppi_critic ${mppi_dependencies})

  # WP3.3: the baseline/panoptex nav2 params files must differ only in the
  # sanctioned risk_layer/critics/speed_limit_topic keys -- see
  # tools/check_arm_params.py's own docstring.
  find_package(ament_cmake_test REQUIRED)
  add_test(
    NAME check_arm_params
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_arm_params.py
      --baseline ${CMAKE_CURRENT_SOURCE_DIR}/config/nav2_x3_baseline.yaml
      --arm ${CMAKE_CURRENT_SOURCE_DIR}/config/nav2_x3_panoptex.yaml)

  # WP2: the same guard for the MPPI pair.
  add_test(
    NAME check_arm_params_mppi
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_arm_params.py
      --baseline ${CMAKE_CURRENT_SOURCE_DIR}/config/nav2_x3_baseline_mppi.yaml
      --arm ${CMAKE_CURRENT_SOURCE_DIR}/config/nav2_x3_panoptex_mppi.yaml)

  # WP3.2: risk_speed_governor's policy core (compute_cap) -- pure
  # functions, no rclpy spin, see that file's module docstring.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_risk_speed_governor
    test/test_risk_speed_governor.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})

  # WP-B: corridor.py's geometry / gap acceptance / refuge search -- pure
  # numpy + dicts, no rclpy, see that file's module docstring.
  ament_add_pytest_test(test_corridor
    test/test_corridor.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})

  # The observed-lane memory: the running per-track fit, the merge/expiry
  # rules, lane_line_clearance and the refuge hysteresis it feeds. Pure
  # geometry like test_corridor.py; see that file's docstring for the
  # mppi_panoptex_2 refuges it exists to stop.
  ament_add_pytest_test(test_observed_lanes
    test/test_observed_lanes.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})

  # mission_supervisor's goal bookkeeping: which NavigateToPose result
  # belongs to which goal, and therefore what may count as a missed
  # waypoint. Builds the node (rclpy.init + parameter_overrides) with a fake
  # action client -- no launch file, no spin, no nav2. See that file's
  # docstring for the mppi_panoptex_2 race it replays.
  ament_add_pytest_test(test_mission_supervisor_goals
    test/test_mission_supervisor_goals.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})

  # "Never cross a lane to reach a refuge": the crossing guards, the derived
  # lane-clearance floor and the wall-hug fallback. Pure geometry again; see
  # that file's docstring for the mppi_panoptex_4 crossing it exists to stop.
  ament_add_pytest_test(test_refuge_side
    test/test_refuge_side.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})

  # WP-C: gt_tracks_node's velocity estimator + message-field builder --
  # pure functions (displacement_velocity/ema_update/classify_motion/
  # offset_position/build_class_id/build_detection_fields), no rclpy spin.
  # See that file's module docstring.
  ament_add_pytest_test(test_gt_tracks
    test/test_gt_tracks.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_export_targets(export_predicted_risk_critic HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(risk_stack_lookup predicted_risk_critic panoptex_nav_mppi_critic)
ament_export_dependencies(${mppi_dependencies})
ament_package()
