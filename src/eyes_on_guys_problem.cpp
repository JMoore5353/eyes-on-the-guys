#include <Eigen/Core>
#include <limits>

#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

EyesOnGuysProblem::EyesOnGuysProblem(const int num_agents, const double relay_speed,
                                     const Eigen::MatrixXd & distance_between_agents)
    : relays_current_info{Eigen::VectorXd::Zero(num_agents)}
    , shared_info_matrix{Eigen::MatrixXd::Zero(num_agents, num_agents)}
    , time_since_last_relay_contact_with_agent{Eigen::VectorXd::Zero(num_agents)}
    , relay_speed{relay_speed}
    , distance_between_agents{distance_between_agents}
{}

EyesOnGuysProblem EyesOnGuysProblem::create_child_eyes_on_guys_state(const int curr_state,
                                                                     const int action)
{
  if (action >= relays_current_info.size() || action < 0) {
    return *this;
  }

  EyesOnGuysProblem new_problem = *this;

  double dt = compute_time_to_take_action(curr_state, action, relay_speed, distance_between_agents);
  for (int i = 0; i < new_problem.relays_current_info.size(); ++i) {
    new_problem.time_since_last_relay_contact_with_agent[i] += dt;
  }

  double agents_new_info =
    simulate_agent_info_gain(new_problem.time_since_last_relay_contact_with_agent[action]);

  new_problem.relays_current_info[action] += agents_new_info;
  new_problem.shared_info_matrix.row(action) = new_problem.relays_current_info;
  new_problem.time_since_last_relay_contact_with_agent[action] = 0.0;

  return new_problem;
}

double simulate_agent_info_gain(const double time_since_last_visit)
{
  // TODO: Update this with the current model
  return time_since_last_visit;
}

double compute_time_to_take_action(const int curr_state, const int action, const double relay_speed,
                                   const Eigen::MatrixXd & distance_between_agents)
{
  if (relay_speed == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return distance_between_agents(curr_state, action) / relay_speed;
}

} // namespace eyes_on_guys
