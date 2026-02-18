#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rosplane_msgs/msg/state.hpp>
#include <rosplane_msgs/msg/waypoint.hpp>

#include "planner.hpp"
#include "forward_search_solver.hpp"
#include "monte_carlo_tree_search.hpp"
#include "eyes_on_guys_problem.hpp"

using namespace std::chrono_literals;

namespace eyes_on_guys
{

namespace
{

int find_starting_guy_index(const std::vector<std::string> & guy_names, const std::string & current_target_guy)
{
  const std::vector<std::string>::const_iterator it =
    std::find(guy_names.begin(), guy_names.end(), current_target_guy);
  if (it == guy_names.end()) {
    return 0;
  }
  return static_cast<int>(std::distance(guy_names.begin(), it));
}

ForwardSearchSolver::ForwardSearchInput make_forward_search_input(
  const std::vector<std::string> & guy_names,
  const std::map<std::string, geometry_msgs::msg::PoseStamped> & guy_poses,
  const EyesOnGuysProblem & problem_info)
{
  ForwardSearchSolver::ForwardSearchInput input;
  input.ids = guy_names;
  for (size_t i = 0; i < guy_names.size(); ++i) {
    const std::string & id = guy_names.at(i);
    ForwardSearchSolver::GuyState state;
    const auto pose_it = guy_poses.find(id);
    if (pose_it != guy_poses.end()) {
      state.pose = pose_it->second;
    }
    if (problem_info.relays_current_info.size() == static_cast<int>(guy_names.size())) {
      state.bits = static_cast<float>(problem_info.relays_current_info(static_cast<int>(i)));
    }
    input.guy_states_by_id.emplace(id, state);
  }
  return input;
}

ForwardSearchSolver::ForwardSearchConfig make_forward_search_config(
  const Planner & planner,
  int starting_guy,
  const Eigen::MatrixXd & shared_info_matrix)
{
  ForwardSearchSolver::ForwardSearchConfig config;
  config.depth = planner.get_parameter("forward_search_depth").as_int();
  config.num_rollouts = planner.get_parameter("forward_search_num_rollouts").as_int();
  config.roll_out_depth = planner.get_parameter("forward_search_roll_out_depth").as_int();
  config.discount_factor = planner.get_parameter("forward_search_discount_factor").as_double();
  config.agent_velocity = planner.get_parameter("forward_search_agent_velocity").as_double();
  config.info_shared_weight = planner.get_parameter("forward_search_info_shared_weight").as_double();
  config.path_length_weight = planner.get_parameter("forward_search_path_length_weight").as_double();
  config.time_since_visit_weight = planner.get_parameter("forward_search_time_since_visit_weight").as_double();
  config.starting_guy = starting_guy;
  config.shared_info_matrix = shared_info_matrix;
  return config;
}

} // namespace

Planner::Planner()
    : Node("planner")
    , eyes_state_received_(false)
    , current_target_guy_{""}
    , has_target_(false)
    , current_guy_index_(0)
    , problem_info_{0, 1.0, Eigen::Matrix<double, 0, 0>::Zero()}
    , time_of_last_visit_to_any_agent_{this->get_clock()->now()}
{
  declare_parameters();

  // Subscribe to eyes (UAV) state from rosplane estimator
  eyes_state_sub_ = this->create_subscription<rosplane_msgs::msg::State>(
    "estimated_state", 10,
    std::bind(&Planner::eyes_state_callback, this, std::placeholders::_1));

  // Subscribe to guy poses
  guy_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "guy_poses", 10, std::bind(&Planner::guy_pose_callback, this, std::placeholders::_1));

  // Subscribe to the bits of info
  guy_bits_sub_ = this->create_subscription<eyes_on_the_guys::msg::Bit>(
    "guy_bits", 10, std::bind(&Planner::guy_bits_callback, this, std::placeholders::_1));

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
  this->declare_parameter("communication_radius", 25.0);
  this->declare_parameter("selection_algorithm", "forward_search");
  this->declare_parameter("mcts_num_iter", 100);
  this->declare_parameter("mcts_depth", 7);
  this->declare_parameter("mcts_discount_factor", 0.9);
  this->declare_parameter("mcts_exploration_bonus", 100.0);
  this->declare_parameter("mcts_lookahead_depth", 5);
  this->declare_parameter("mcts_lookahead_iters", 30);
  this->declare_parameter("forward_search_depth", 6);
  this->declare_parameter("forward_search_num_rollouts", 20);
  this->declare_parameter("forward_search_roll_out_depth", 5);
  this->declare_parameter("forward_search_discount_factor", 0.5);
  this->declare_parameter("forward_search_agent_velocity", 15.0);
  this->declare_parameter("forward_search_info_shared_weight", 10.0);
  this->declare_parameter("forward_search_path_length_weight", 1.0);
  this->declare_parameter("forward_search_time_since_visit_weight", 10.0);
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

void Planner::guy_bits_callback(const eyes_on_the_guys::msg::Bit & msg)
{
  guy_bits_[msg.header.frame_id] = msg.bits;
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

  // Select initial target if we don't have one
  if (!has_target_) {
    select_new_target_guy();
  }

  // Check if we're within communication radius of current target
  double distance = compute_horizontal_distance_to_target();
  double comm_radius = this->get_parameter("communication_radius").as_double();
  
  if (distance >= 0 && distance < comm_radius) {
    RCLCPP_INFO(this->get_logger(), "Within %.1fm of %s, selecting new target", 
                comm_radius, current_target_guy_.c_str());
    select_new_target_guy();
  }

  // First, send a message to clear the waypoint list
  rosplane_msgs::msg::Waypoint clear_msg;
  clear_msg.header.stamp = this->get_clock()->now();
  clear_msg.clear_wp_list = true;
  waypoint_pub_->publish(clear_msg);

  // Immediately send the next waypoints
  for (std::size_t i=0; i<current_target_sequence_.size(); ++i) {
    if (i >= current_target_sequence_.size()) {
      break;
    }
    std::string guy_name = current_target_sequence_.at(i);
    auto waypoint = compute_waypoint_to_guy(guy_name);
    waypoint_pub_->publish(waypoint);
  }

  // Update the plot
  plot_state();
}

rosplane_msgs::msg::Waypoint Planner::compute_waypoint_to_guy(const std::string& name)
{
  rosplane_msgs::msg::Waypoint waypoint;
  waypoint.header.stamp = this->get_clock()->now();

  if (!has_target_ || guy_poses_.find(name) == guy_poses_.end()) {
    // No valid target, hold current position
    waypoint.w[0] = current_eyes_state_.p_n;
    waypoint.w[1] = current_eyes_state_.p_e;
    waypoint.w[2] = current_eyes_state_.p_d;
  } else {
    // Publish waypoint 50m above the target guy
    const auto& target_pose = guy_poses_[name];
    waypoint.w[0] = target_pose.pose.position.x;
    waypoint.w[1] = target_pose.pose.position.y;
    waypoint.w[2] = target_pose.pose.position.z - 50.0f;
  }

  waypoint.lla = false;
  waypoint.chi_d = current_eyes_state_.chi;
  waypoint.use_chi = true;
  waypoint.va_d = 15.0f;
  waypoint.set_current = false;
  waypoint.clear_wp_list = false;

  return waypoint;
}

void Planner::select_new_target_guy()
{
  if (guy_poses_.empty()) {
    has_target_ = false;
    return;
  }

  if (current_target_guy_ == "") {
    current_target_guy_ = guy_poses_.begin()->first;
    current_target_sequence_ = {current_target_guy_};
  }

  // Build vector of guy names
  std::vector<std::string> guy_names;
  std::string names = "";
  for (const auto& [name, pose] : guy_poses_) {
    // Iterators of std::map iterate in ascending order of keys, so
    // this is already in order (no need to sort)
    guy_names.push_back(name);
    names += name + " ";
  }

  update_problem_info(guy_names);

  std::string selection_algorithm = this->get_parameter("selection_algorithm").as_string();
  if (selection_algorithm == "mcts") {
    find_next_guy_with_mcts(guy_names);
  } else if (selection_algorithm == "branch_and_bound") {
    find_next_guy_with_branch_and_bound(guy_names);
  } else if (selection_algorithm == "forward_search") {
    find_next_guy_with_forward_search(guy_names);
  } else if (selection_algorithm == "sequential") {
    find_next_guy_sequentially(guy_names);
  } else {
    RCLCPP_WARN(this->get_logger(),
                "Unable to parse selection_algorithm type %s! Defaulting to 'sequential'!",
                selection_algorithm.c_str());
    find_next_guy_sequentially(guy_names);
  }

  has_target_ = true;
  RCLCPP_INFO(this->get_logger(), "Selected new target: %s", current_target_guy_.c_str());
}

double Planner::compute_horizontal_distance_to_target()
{
  if (!has_target_ || guy_poses_.find(current_target_guy_) == guy_poses_.end()) {
    return -1.0;
  }

  const auto& target_pose = guy_poses_[current_target_guy_];
  double delta_n = current_eyes_state_.p_n - target_pose.pose.position.x;
  double delta_e = current_eyes_state_.p_e - target_pose.pose.position.y;
  
  return std::sqrt(delta_n * delta_n + delta_e * delta_e);
}

void Planner::plot_state()
{
  using namespace matplot;
  auto fig = gcf();
  fig->quiet_mode(true);
  cla();

  // Define colors for guys (cycle through a color palette) - must be float arrays
  std::vector<std::array<float, 3>> colors = {
    {0.12f, 0.47f, 0.71f},  // blue
    {1.00f, 0.50f, 0.05f},  // orange
    {0.17f, 0.63f, 0.17f},  // green
    {0.84f, 0.15f, 0.16f},  // red
    {0.58f, 0.40f, 0.74f},  // purple
    {0.55f, 0.34f, 0.29f},  // brown
    {0.89f, 0.47f, 0.76f},  // pink
    {0.50f, 0.50f, 0.50f},  // gray
    {0.74f, 0.74f, 0.13f},  // olive
    {0.09f, 0.75f, 0.81f}   // cyan
  };

  size_t color_idx = 0;

  // Plot each guy
  for (const auto& [name, pose] : guy_poses_) {
    std::vector<double> x = {pose.pose.position.x};
    std::vector<double> y = {pose.pose.position.y};

    const auto& color = colors[color_idx % colors.size()];

    if (has_target_ && name == current_target_guy_) {
      // Selected guy: square marker
      auto s = scatter(x, y);
      s->marker(line_spec::marker_style::square);
      s->marker_size(10);
      s->marker_color(color);
      s->marker_face_color(color);
      s->display_name(name);
    } else {
      // Non-selected guys: circle marker
      auto s = scatter(x, y);
      s->marker(line_spec::marker_style::circle);
      s->marker_size(10);
      s->marker_color(color);
      s->marker_face_color(color);
      s->display_name(name);
    }

    hold(on);
    color_idx++;
  }

  // Plot UAV as black triangle
  std::vector<double> uav_x = {current_eyes_state_.p_n};
  std::vector<double> uav_y = {current_eyes_state_.p_e};

  auto uav = scatter(uav_x, uav_y);
  uav->marker(line_spec::marker_style::upward_pointing_triangle);
  uav->marker_size(12);
  uav->marker_color({0.0f, 0.0f, 0.0f});
  uav->marker_face_color({0.0f, 0.0f, 0.0f});
  uav->display_name("UAV");

  xlabel("North (m)");
  ylabel("East (m)");
  title("Eyes on the Guys - UAV Tracking");
  legend();
  grid(on);
  axis(equal);
  xlim({-600, 600});
  ylim({-600, 600});

  fig->draw();
}

void Planner::find_next_guy_with_mcts(const std::vector<std::string>& guy_names)
{
  int num_agents = static_cast<int>(guy_names.size());

  std::vector<std::string>::const_iterator it = std::find(guy_names.begin(), guy_names.end(), current_target_guy_);
  int initial_state = std::distance(guy_names.begin(), it);

  MonteCarloTreeSearch tree_searcher{num_agents};
  int optimal_action = tree_searcher.search_for_best_action(initial_state,
                                                            this->get_parameter("mcts_num_iter").as_int(),
                                                            this->get_parameter("mcts_depth").as_int(),
                                                            this->get_parameter("mcts_discount_factor").as_double(),
                                                            this->get_parameter("mcts_exploration_bonus").as_double(),
                                                            problem_info_,
                                                            this->get_parameter("mcts_lookahead_depth").as_int(),
                                                            this->get_parameter("mcts_lookahead_iters").as_int());
  std::vector<int> optimal_sequence = tree_searcher.get_greedy_sequence();

  current_target_guy_ = guy_names.at(optimal_action);
  current_target_sequence_.clear();
  for (int idx : optimal_sequence) {
    current_target_sequence_.push_back(guy_names.at(idx));
  }
}

void Planner::find_next_guy_with_branch_and_bound(const std::vector<std::string>& guy_names)
{
  find_next_guy_sequentially(guy_names);
}

void Planner::find_next_guy_with_forward_search(const std::vector<std::string>& guy_names)
{
  const int starting_guy = find_starting_guy_index(guy_names, current_target_guy_);
  const ForwardSearchSolver::ForwardSearchInput input = make_forward_search_input(
    guy_names,
    guy_poses_,
    problem_info_);
  const ForwardSearchSolver::ForwardSearchConfig config = make_forward_search_config(
    *this,
    starting_guy,
    problem_info_.shared_info_matrix);

  RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Forward search input guy_names.size(): " << guy_names.size());
  RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Forward search config depth passed to solver: " << config.depth);

  ForwardSearchSolver solver;
  const auto solve_start = std::chrono::steady_clock::now();
  const ForwardSearchSolver::ForwardSearchResult result = solver.solve(input, config);
  const auto solve_end = std::chrono::steady_clock::now();
  const double solve_elapsed_sec =
    std::chrono::duration<double>(solve_end - solve_start).count();
  RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Forward search solve() elapsed seconds: " << solve_elapsed_sec);
  if (!result.success || result.sequence_ids.empty()) {
    throw std::runtime_error("Forward search failed to produce a sequence: " + result.message);
  }

  std::ostringstream sequence_stream;
  for (size_t i = 0; i < result.sequence_ids.size(); ++i) {
    if (i > 0) {
      sequence_stream << " -> ";
    }
    sequence_stream << result.sequence_ids.at(i);
  }
  RCLCPP_INFO_STREAM(
    this->get_logger(),
    "Forward search optimal sequence (len=" << result.sequence_ids.size() << "): "
      << sequence_stream.str());

  current_target_guy_ = result.sequence_ids.at(0);
  current_target_sequence_ = result.sequence_ids;

  RCLCPP_INFO_STREAM(this->get_logger(), "FORWARD NEXT: " << current_target_guy_);
}

void Planner::find_next_guy_sequentially(const std::vector<std::string>& guy_names)
{
  current_guy_index_ = current_guy_index_ % guy_names.size();
  current_target_guy_ = guy_names[current_guy_index_];

  current_target_sequence_.clear();
  for (int i=0; i<5; ++i) {
    int guy_idx = (current_guy_index_ + i) % guy_names.size();
    current_target_sequence_.push_back(guy_names.at(guy_idx));
  }

  current_guy_index_++;
}

EyesOnGuysProblem Planner::compute_initial_problem_information(const std::vector<std::string>& guy_names) const
{
  int num_agents = static_cast<int>(guy_names.size());
  double curr_speed = std::max(std::sqrt(current_eyes_state_.v_x * current_eyes_state_.v_x +
                                         current_eyes_state_.v_y * current_eyes_state_.v_y +
                                         current_eyes_state_.v_z * current_eyes_state_.v_z), 15.0f);
  Eigen::MatrixXd distance_between_agents = compute_distance_between_guys(guy_names, guy_poses_);

  return EyesOnGuysProblem{num_agents, curr_speed, distance_between_agents};
}

  void Planner::update_problem_info(const std::vector<std::string>& guy_names)
{
  if (problem_info_.num_agents() != static_cast<int>(guy_names.size())) {
    problem_info_ = compute_initial_problem_information(guy_names);
  }

  std::vector<std::string>::const_iterator it = std::find(guy_names.begin(), guy_names.end(), current_target_guy_);
  int current_state = std::distance(guy_names.begin(), it);

  problem_info_.update_relay_information(current_state, guy_bits_[current_target_guy_]);

  rclcpp::Time now = this->get_clock()->now();
  double dt_sec = (now - time_of_last_visit_to_any_agent_).seconds();
  time_of_last_visit_to_any_agent_ = now;

  problem_info_.update_time_since_last_visit(current_state, dt_sec);
  problem_info_.update_distance_matrix(compute_distance_between_guys(guy_names, guy_poses_));
}

Eigen::MatrixXd compute_distance_between_guys(const std::vector<std::string>& guy_names,
                                              const std::map<std::string, geometry_msgs::msg::PoseStamped>& guy_poses)
{
  std::size_t num_agents = guy_poses.size();
  Eigen::MatrixXd distance_between_agents = Eigen::MatrixXd::Zero(num_agents, num_agents);

  for (std::size_t i=0; i<num_agents; ++i) {
    std::string ith_guy_name = guy_names.at(i);

    for (std::size_t j=i+1; j<num_agents; ++j) {
      std::string jth_guy_name = guy_names.at(j);

      double x_dist = guy_poses.at(ith_guy_name).pose.position.x - guy_poses.at(jth_guy_name).pose.position.x;
      double y_dist = guy_poses.at(ith_guy_name).pose.position.y - guy_poses.at(jth_guy_name).pose.position.y;
      double z_dist = guy_poses.at(ith_guy_name).pose.position.z - guy_poses.at(jth_guy_name).pose.position.z;
      double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);

      distance_between_agents(i,j) = dist;
      distance_between_agents(j,i) = dist;
    }
  }

  return distance_between_agents;
}

} // namespace eyes_on_guys
