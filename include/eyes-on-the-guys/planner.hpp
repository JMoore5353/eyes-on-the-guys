#ifndef PLANNER_HPP
#define PLANNER_HPP

#include <Eigen/Eigen>
#include <cmath>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosplane_msgs/msg/state.hpp>
#include <rosplane_msgs/msg/waypoint.hpp>

#define M_PI_F 3.14159265358979323846f
#define M_PI_2_F 1.57079632679489661923f

namespace eyes_on_guys
{

class Planner : public rclcpp::Node
{
public:
  Planner();

private:
  // Subscribers
  rclcpp::Subscription<rosplane_msgs::msg::State>::SharedPtr eyes_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr guy_pose_sub_;

  // Publishers
  rclcpp::Publisher<rosplane_msgs::msg::Waypoint>::SharedPtr waypoint_pub_;

  // Timer for planning loop
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // Stored state data
  rosplane_msgs::msg::State current_eyes_state_;
  bool eyes_state_received_;

  std::map<std::string, geometry_msgs::msg::PoseStamped> guy_poses_;

  // Callbacks
  void eyes_state_callback(const rosplane_msgs::msg::State & msg);
  void guy_pose_callback(const geometry_msgs::msg::PoseStamped & msg);
  void planning_timer_callback();

  // Planning
  rosplane_msgs::msg::Waypoint compute_next_waypoint();

  /**
   * @brief Computes the Dubins path length from current state to target position/heading.
   *        Uses the same algorithm as rosplane's path_manager_dubins_fillets.
   *
   * @param start_n Start north position (m)
   * @param start_e Start east position (m)
   * @param start_chi Start course angle (rad)
   * @param end_n End north position (m)
   * @param end_e End east position (m)
   * @param end_chi End course angle (rad)
   * @return Path length in meters, or -1.0 if path is infeasible
   */
  double compute_dubins_path_length(float start_n, float start_e, float start_chi,
                                    float end_n, float end_e, float end_chi);

  // Dubins path helper functions
  Eigen::Matrix3f rotz(float theta);
  float mo(float in);

  void declare_parameters();
};

} // namespace eyes_on_guys

#endif
