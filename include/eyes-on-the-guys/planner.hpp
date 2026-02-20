#ifndef PLANNER_HPP
#define PLANNER_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <map>
#include <vector>
#include <matplot/matplot.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rosplane_msgs/msg/state.hpp>
#include <rosplane_msgs/msg/waypoint.hpp>

#include "eyes_on_guys_problem.hpp"
#include <eyes_on_the_guys/msg/bit.hpp>

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
  rclcpp::Subscription<eyes_on_the_guys::msg::Bit>::SharedPtr guy_bits_sub_;

  // Publishers
  rclcpp::Publisher<rosplane_msgs::msg::Waypoint>::SharedPtr waypoint_pub_;

  // Timer for planning loop
  rclcpp::TimerBase::SharedPtr planning_timer_;

  // Stored state data
  rosplane_msgs::msg::State current_eyes_state_;
  bool eyes_state_received_;

  std::map<std::string, geometry_msgs::msg::PoseStamped> guy_poses_;
  std::map<std::string, float> guy_bits_;

  // Target tracking
  std::string current_target_guy_;
  std::vector<std::string> current_target_sequence_;
  bool has_target_;
  size_t current_guy_index_;

  // Persistent information to compute reward
  EyesOnGuysProblem problem_info_;
  rclcpp::Time time_of_last_visit_to_any_agent_;

  // Variables for plotting
  std::string mcts_name_;
  std::vector<std::string> mcts_sequence_;
  std::string bnb_name_;
  std::vector<std::string> bnb_sequence_;
  std::string fs_name_;
  std::vector<std::string> fs_sequence_;
  std::string seq_name_;
  std::vector<std::string> seq_sequence_;

  std::vector<matplot::figure_handle> figure_vector_;

  // Callbacks
  void eyes_state_callback(const rosplane_msgs::msg::State & msg);
  void guy_pose_callback(const geometry_msgs::msg::PoseStamped & msg);
  void guy_bits_callback(const eyes_on_the_guys::msg::Bit & msg);
  void planning_timer_callback();

  // Planning
  rosplane_msgs::msg::Waypoint compute_waypoint_to_guy(const std::string& name);
  void select_new_target_guy();
  double compute_horizontal_distance_to_target();
  void plot_state();
  void plot_sequence(const std::vector<std::string>& sequence,
                     const std::string& line_style,
                     const std::string& display_name);

  void declare_parameters();
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  rcl_interfaces::msg::SetParametersResult parameters_callback(const std::vector<rclcpp::Parameter> & parameters);

  /**
  * Uses MCTS to compute the next guy to fly to
  *
  * @param guy_names Vector of names of the guys (sorted in alphabetical order)
  * @return nothing
  */
  void find_next_guy_with_mcts(const std::vector<std::string>& guy_names);

  /**
  * Uses Branch and Bound to compute the next guy to fly to
  *
  * @param guy_names Vector of names of the guys (sorted in alphabetical order)
  * @return nothing
  */
  void find_next_guy_with_branch_and_bound(const std::vector<std::string>& guy_names);

  /**
  * Uses forward search to compute the next guy to fly to
  *
  * @param guy_names Vector of names of the guys (sorted in alphabetical order)
  * @return nothing
  */
  void find_next_guy_with_forward_search(const std::vector<std::string>& guy_names);

  /**
  * Sequentially iterates through the guy names
  *
  * @param guy_names Vector of names of the guys (sorted in alphabetical order)
  * @return nothing
  */
  void find_next_guy_sequentially(const std::vector<std::string>& guy_names);

  void find_next_guy_using_all_methods(const std::vector<std::string>& guy_names);

  EyesOnGuysProblem compute_initial_problem_information(const std::vector<std::string>& guy_names) const;

  void update_problem_info(const std::vector<std::string>& guy_names);
};

Eigen::MatrixXd compute_distance_between_guys(const std::vector<std::string>& guy_names,
                                              const std::map<std::string, geometry_msgs::msg::PoseStamped>& guy_poses);

} // namespace eyes_on_guys

#endif
