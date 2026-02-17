#ifndef EYES_ON_GUYS_PROBLEM_HPP
#define EYES_ON_GUYS_PROBLEM_HPP

#include <Eigen/Core>

namespace eyes_on_guys
{

struct EyesOnGuysProblem
{
public:
  EyesOnGuysProblem(const int num_agents, const double relay_speed,
                    const Eigen::MatrixXd & distance_between_agents);

  Eigen::VectorXd relays_current_info;                      // nx1
  Eigen::MatrixXd shared_info_matrix;                       // nxn
  Eigen::VectorXd time_since_last_relay_contact_with_agent; // nx1
  double relay_speed;
  Eigen::MatrixXd distance_between_agents; // nxn where n is num of agents

  EyesOnGuysProblem create_child_eyes_on_guys_state(const int curr_state, const int action);
};

double simulate_agent_info_gain(const double time_since_last_visit);
double compute_time_to_take_action(const int curr_state, const int action, const double relay_speed,
                                   const Eigen::MatrixXd & distance_between_agents);
double compute_reward_model(const int curr_state, const int next_state,
                            const EyesOnGuysProblem & curr_state_info,
                            const EyesOnGuysProblem & next_state_info,
                            const bool include_penalties = true);

} // namespace eyes_on_guys

#endif
