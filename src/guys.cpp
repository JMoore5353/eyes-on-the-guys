#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "guys.hpp"

namespace eyes_on_guys
{

int sign(double x) {return (x>0) - (x<0);}

Guys::Guys()
: Node("guy_sim")
{
  declare_parameters();

  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("guy_poses", 10);
  bits_pub_ = this->create_publisher<eyes_on_the_guys::msg::Bit>("guy_bits", 10);

  names_ = {
    "Jacob",
    "Brandon",
    "Josh",
    "Euler",
    "Gauss",
    "Cantor",
    "Fermat",
    "Riemann",
    "Lebesque",
    "Turing",
    "Ramanujan",
    "Hilbert",
    "Pythagoras",
    "Kalman",
    "Bode",
    "Nyquist",
    "Laplace"
  };
  rng_ = std::mt19937(std::random_device{}());
  initialize_guys();
  log_names();

  double publish_rate_hz = this->get_parameter("publish_rate_hz").as_double();
  auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / publish_rate_hz));
  timer_ = this->create_wall_timer(period, std::bind(&Guys::update_positions, this));
}

void Guys::declare_parameters()
{
  this->declare_parameter("publish_rate_hz", 10.0);
  this->declare_parameter("number_of_guys", 6);
  this->declare_parameter("std_dev", 0.1);
  this->declare_parameter("velocity", 1.0);
  this->declare_parameter("bits_rate_max", 1.0);
  this->declare_parameter("deflection_radius", 50.0);
  this->declare_parameter("init_min_x", -400.0);
  this->declare_parameter("init_max_x", -375.0);
  this->declare_parameter("init_min_y", -500.0);
  this->declare_parameter("init_max_y", 500.0);
}

void Guys::update_positions()
{
  double std_dev = this->get_parameter("std_dev").as_double();
  double velocity = this->get_parameter("velocity").as_double();
  double publish_rate_hz = this->get_parameter("publish_rate_hz").as_double();
  double deflection_radius = this->get_parameter("deflection_radius").as_double();

  std::normal_distribution<double> noise_dist(0.0, std_dev);
  std::uniform_real_distribution<double> angle_dist(0.0, 2*M_PI);
  auto stamp = this->now();
  std::vector<geometry_msgs::msg::Point> positions;
  positions.reserve(guys_.size());
  for (auto & guy : guys_) {
    positions.push_back(guy.pose.pose.position);
  }

  double dt = 1.0 / publish_rate_hz;

  for (std::size_t i = 0; i < guys_.size(); ++i) {
    auto & guy = guys_[i];
    double current_yaw = quaternion_to_yaw(guy.pose.pose.orientation);
    double proposed_yaw = current_yaw + noise_dist(rng_);
    auto & current = positions[i];
    double yaw_deflection = 0.0;

    for (std::size_t j = 0; j < positions.size(); ++j) {
      if (i == j) {
        continue;
      }

      auto & other = positions[j];
      double dx = current.x - other.x;
      double dy = current.y - other.y;
      double dist = std::sqrt(dx * dx + dy * dy);

      if (dist < deflection_radius) {
        double away_angle = std::atan2(dy, dx);
        double angle_error = std::atan2( std::sin(away_angle - proposed_yaw), std::cos(away_angle - proposed_yaw));
        yaw_deflection += sign(angle_error)*std::min(0.03, abs(angle_error));
      }
    }

    double step_size = velocity/publish_rate_hz;

    double new_yaw = proposed_yaw + yaw_deflection;
    double dx = step_size * std::cos(new_yaw);
    double dy = step_size * std::sin(new_yaw);

    guy.pose.pose.position.x += dx;
    guy.pose.pose.position.y += dy;
    guy.pose.pose.position.z = 0;
    guy.pose.pose.orientation = yaw_to_quaternion(new_yaw);
    guy.pose.header.stamp = stamp;
    guy.bits += static_cast<float>(guy.bits_rate * dt);

    pose_pub_->publish(guy.pose);
    eyes_on_the_guys::msg::Bit bits_msg;
    bits_msg.header.stamp = stamp;
    bits_msg.header.frame_id = guy.pose.header.frame_id;
    bits_msg.bits = guy.bits;
    bits_pub_->publish(bits_msg);
  }
}

void Guys::initialize_guys() {
  int count = this->get_parameter("number_of_guys").as_int();
  double init_min_x = this->get_parameter("init_min_x").as_double();
  double init_max_x = this->get_parameter("init_max_x").as_double();
  double init_min_y = this->get_parameter("init_min_y").as_double();
  double init_max_y = this->get_parameter("init_max_y").as_double();
  double bits_rate_max = this->get_parameter("bits_rate_max").as_double();

  std::vector<std::string> names_pool = names_;
  std::shuffle(names_pool.begin(), names_pool.end(), rng_);

  guys_.clear();
  guys_.reserve(count);

  std::uniform_real_distribution<double> heading_dist(-M_PI, M_PI);
  std::uniform_real_distribution<double> init_x_dist(init_min_x, init_max_x);
  std::uniform_real_distribution<double> init_y_dist(init_min_y, init_max_y);
  std::uniform_real_distribution<float> bits_rate_dist(0.0f, static_cast<float>(bits_rate_max));

  for (int idx = 0; idx < count; ++idx) {
    GuyState guy;
    guy.pose.header.frame_id = names_pool[idx];
    guy.pose.pose.position.x = init_x_dist(rng_);
    guy.pose.pose.position.y = init_y_dist(rng_);
    guy.pose.pose.position.z = 0.0;
    guy.pose.pose.orientation = yaw_to_quaternion(heading_dist(rng_));
    guy.bits = 0.0f;
    guy.bits_rate = bits_rate_dist(rng_);
    guys_.push_back(guy);
  }
}

void Guys::log_names()
{
  std::string message = "Guys:";
  for (const auto & guy : guys_) {
    message += " " + guy.pose.header.frame_id;
  }
  RCLCPP_INFO_STREAM(this->get_logger(), message);
}

geometry_msgs::msg::Quaternion Guys::yaw_to_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion quat;
  quat.x = 0.0;
  quat.y = 0.0;
  quat.z = std::sin(yaw * 0.5);
  quat.w = std::cos(yaw * 0.5);
  return quat;
}

double Guys::quaternion_to_yaw(const geometry_msgs::msg::Quaternion & quat)
{
  double siny_cosp = 2.0 * quat.w * quat.z;
  double cosy_cosp = 1.0 - 2.0 * quat.z * quat.z;
  return std::atan2(siny_cosp, cosy_cosp);
}

} // namespace eyes_on_guys
