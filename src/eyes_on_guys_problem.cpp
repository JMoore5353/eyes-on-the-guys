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
  new_problem.shared_info_matrix(action, action) = 0;
  new_problem.time_since_last_relay_contact_with_agent[action] = 0.0;

  return new_problem;
}

void EyesOnGuysProblem::update_relay_information(const int curr_state, const double new_total_bits)
{
  relays_current_info[curr_state] = new_total_bits;
  shared_info_matrix.row(curr_state) = relays_current_info;
  shared_info_matrix(curr_state, curr_state) = 0;
}

void EyesOnGuysProblem::update_time_since_last_visit(const int curr_state, const double dt)
{
  for (int i = 0; i < relays_current_info.size(); ++i) {
    time_since_last_relay_contact_with_agent[i] += dt;
  }
  time_since_last_relay_contact_with_agent[curr_state] = 0.0;
}

double simulate_agent_info_gain(const double time_since_last_visit)
{
  // TODO: (PREDICTIVE_MODEL) Replace this placeholder with the information-gain model.
  // Currently, we approximate the agent's information gain as increasing linearly with
  // the time since its last visit. This is a simple proxy that assumes one unit of
  // "information" per unit of time without saturation or noise.
  //
  // Intended future behavior:
  //  - Use the current sensing/observation model to map time-since-last-visit to
  //    expected information gain (e.g., a saturating or exponential curve).
  //  - Update any tests that rely on the linear proxy once the calibrated model
  //    is available and agreed upon.
  //
  // Until the full model is implemented, we retain this linear proxy to avoid
  // changing behavior in existing simulations.
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

double compute_reward_model(const int curr_state, const int next_state,
                            const EyesOnGuysProblem & curr_state_info,
                            const EyesOnGuysProblem & next_state_info)
{
  double gamma{1.0};
  double beta{1.0};
  double zeta{1.0};

  Eigen::MatrixXd delta_shared_info =
    next_state_info.shared_info_matrix - curr_state_info.shared_info_matrix;
  double shared_info_reward = gamma * delta_shared_info.norm();

  double distance_penalty = beta * curr_state_info.distance_between_agents(curr_state, next_state);

  Eigen::VectorXd delta_time_since_last_contact =
    next_state_info.time_since_last_relay_contact_with_agent
    - curr_state_info.time_since_last_relay_contact_with_agent;
  double time_since_last_visit_penalty = delta_time_since_last_contact.sum() * zeta;

  return shared_info_reward - distance_penalty - time_since_last_visit_penalty;
}
} // namespace eyes_on_guys
