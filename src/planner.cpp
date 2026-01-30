#include <chrono>
#include <functional>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosplane_msgs/msg/state.hpp>
#include <rosplane_msgs/msg/waypoint.hpp>

#include "planner.hpp"

using namespace std::chrono_literals;

namespace eyes_on_guys
{

Planner::Planner()
    : Node("planner")
    , eyes_state_received_(false)
{
  declare_parameters();

  // Subscribe to eyes (UAV) state from rosplane estimator
  eyes_state_sub_ = this->create_subscription<rosplane_msgs::msg::State>(
    "estimated_state", 10,
    std::bind(&Planner::eyes_state_callback, this, std::placeholders::_1));

  // Subscribe to guy poses
  guy_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "guy_poses", 10, std::bind(&Planner::guy_pose_callback, this, std::placeholders::_1));

  // Publisher for waypoints to rosplane path manager
  waypoint_pub_ = this->create_publisher<rosplane_msgs::msg::Waypoint>("waypoint_path", 10);

  // Planning timer
  double planning_rate_hz = this->get_parameter("planning_rate_hz").as_double();
  auto timer_period = std::chrono::duration<double>(1.0 / planning_rate_hz);
  planning_timer_ = this->create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
    std::bind(&Planner::planning_timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "Planner initialized at %.1f Hz", planning_rate_hz);
}

void Planner::declare_parameters()
{
  this->declare_parameter("planning_rate_hz", 1.0);
  this->declare_parameter("R_min", 50.0);
}

void Planner::eyes_state_callback(const rosplane_msgs::msg::State & msg)
{
  current_eyes_state_ = msg;
  eyes_state_received_ = true;
}

void Planner::guy_pose_callback(const geometry_msgs::msg::PoseStamped & msg)
{
  guy_poses_[msg.header.frame_id] = msg;
}

void Planner::planning_timer_callback()
{
  if (!eyes_state_received_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for eyes state...");
    return;
  }

  if (guy_poses_.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for guy poses...");
    return;
  }

  auto waypoint = compute_next_waypoint();
  waypoint_pub_->publish(waypoint);
}

rosplane_msgs::msg::Waypoint Planner::compute_next_waypoint()
{
  // TODO: Implement decision-making logic here
  // Available data:
  //   - current_eyes_state_: UAV position (p_n, p_e, p_d), velocity, attitude, etc.
  //   - guy_poses_: map of guy_id -> PoseStamped for each tracked guy

  rosplane_msgs::msg::Waypoint waypoint;
  waypoint.header.stamp = this->get_clock()->now();

  waypoint.w[0] = current_eyes_state_.p_n;
  waypoint.w[1] = current_eyes_state_.p_e;
  waypoint.w[2] = current_eyes_state_.p_d;
  waypoint.lla = false;
  waypoint.chi_d = current_eyes_state_.chi;
  waypoint.use_chi = true;
  waypoint.va_d = 15.0f;
  waypoint.set_current = false;
  waypoint.clear_wp_list = false;

  return waypoint;
}

Eigen::Matrix3f Planner::rotz(float theta)
{
  Eigen::Matrix3f R;
  R << cosf(theta), -sinf(theta), 0, sinf(theta), cosf(theta), 0, 0, 0, 1;
  return R;
}

float Planner::mo(float in)
{
  float val;
  if (in > 0)
    val = fmod(in, 2.0f * M_PI_F);
  else {
    float n = floorf(in / 2.0f / M_PI_F);
    val = in - n * 2.0f * M_PI_F;
  }
  return val;
}

double Planner::compute_dubins_path_length(float start_n, float start_e, float start_chi,
                                           float end_n, float end_e, float end_chi)
{
  float R = static_cast<float>(this->get_parameter("R_min").as_double());

  float ell = sqrtf((start_n - end_n) * (start_n - end_n)
                    + (start_e - end_e) * (start_e - end_e));

  if (ell < 2.0f * R) {
    RCLCPP_WARN(this->get_logger(),
                "Dubins path infeasible: distance between points (%.2f) < 2*R_min (%.2f)",
                ell, 2.0f * R);
    return -1.0;
  }

  // Calculate circle centers for start position
  Eigen::Vector3f crs;  // Right circle at start
  crs(0) = start_n + R * (cosf(M_PI_2_F) * cosf(start_chi) - sinf(M_PI_2_F) * sinf(start_chi));
  crs(1) = start_e + R * (sinf(M_PI_2_F) * cosf(start_chi) + cosf(M_PI_2_F) * sinf(start_chi));
  crs(2) = 0;

  Eigen::Vector3f cls;  // Left circle at start
  cls(0) = start_n + R * (cosf(-M_PI_2_F) * cosf(start_chi) - sinf(-M_PI_2_F) * sinf(start_chi));
  cls(1) = start_e + R * (sinf(-M_PI_2_F) * cosf(start_chi) + cosf(-M_PI_2_F) * sinf(start_chi));
  cls(2) = 0;

  // Calculate circle centers for end position
  Eigen::Vector3f cre;  // Right circle at end
  cre(0) = end_n + R * (cosf(M_PI_2_F) * cosf(end_chi) - sinf(M_PI_2_F) * sinf(end_chi));
  cre(1) = end_e + R * (sinf(M_PI_2_F) * cosf(end_chi) + cosf(M_PI_2_F) * sinf(end_chi));
  cre(2) = 0;

  Eigen::Vector3f cle;  // Left circle at end
  cle(0) = end_n + R * (cosf(-M_PI_2_F) * cosf(end_chi) - sinf(-M_PI_2_F) * sinf(end_chi));
  cle(1) = end_e + R * (sinf(-M_PI_2_F) * cosf(end_chi) + cosf(-M_PI_2_F) * sinf(end_chi));
  cle(2) = 0;

  float theta, theta2;

  // Compute L1 (RSR path)
  theta = atan2f(cre(1) - crs(1), cre(0) - crs(0));
  float L1 = (crs - cre).norm()
    + R * mo(2.0f * M_PI_F + mo(theta - M_PI_2_F) - mo(start_chi - M_PI_2_F))
    + R * mo(2.0f * M_PI_F + mo(end_chi - M_PI_2_F) - mo(theta - M_PI_2_F));

  // Compute L2 (RSL path)
  ell = (cle - crs).norm();
  theta = atan2f(cle(1) - crs(1), cle(0) - crs(0));
  float L2;
  if (2.0f * R > ell)
    L2 = 9999.0f;
  else {
    theta2 = theta - M_PI_2_F + asinf(2.0f * R / ell);
    L2 = sqrtf(ell * ell - 4.0f * R * R)
      + R * mo(2.0f * M_PI_F + mo(theta2) - mo(start_chi - M_PI_2_F))
      + R * mo(2.0f * M_PI_F + mo(theta2 + M_PI_F) - mo(end_chi + M_PI_2_F));
  }

  // Compute L3 (LSR path)
  ell = (cre - cls).norm();
  theta = atan2f(cre(1) - cls(1), cre(0) - cls(0));
  float L3;
  if (2.0f * R > ell)
    L3 = 9999.0f;
  else {
    theta2 = acosf(2.0f * R / ell);
    L3 = sqrtf(ell * ell - 4.0f * R * R)
      + R * mo(2.0f * M_PI_F + mo(start_chi + M_PI_2_F) - mo(theta + theta2))
      + R * mo(2.0f * M_PI_F + mo(end_chi - M_PI_2_F) - mo(theta + theta2 - M_PI_F));
  }

  // Compute L4 (LSL path)
  theta = atan2f(cle(1) - cls(1), cle(0) - cls(0));
  float L4 = (cls - cle).norm()
    + R * mo(2.0f * M_PI_F + mo(start_chi + M_PI_2_F) - mo(theta + M_PI_2_F))
    + R * mo(2.0f * M_PI_F + mo(theta + M_PI_2_F) - mo(end_chi + M_PI_2_F));

  // Return minimum path length
  float L = L1;
  if (L2 < L) L = L2;
  if (L3 < L) L = L3;
  if (L4 < L) L = L4;

  return static_cast<double>(L);
}

} // namespace eyes_on_guys
