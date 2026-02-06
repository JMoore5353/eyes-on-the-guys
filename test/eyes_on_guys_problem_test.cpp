#include <gtest/gtest.h>

#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

TEST(eyes_on_guys_problem, WhenCreatingChild_ExpectCorrectValues)
{
  int num_agents{4};
  double relay_speed{5.0};
  Eigen::Matrix4d dist_between_agents;
  dist_between_agents << 0, 20, 40, 60, 20, 0, 80, 100, 40, 80, 0, 120, 60, 100, 120, 0;

  EyesOnGuysProblem problem{num_agents, relay_speed, dist_between_agents};
  problem.relays_current_info = Eigen::Vector4d{30, 10, 40, 0};
  Eigen::Matrix4d info_matrix;
  info_matrix << 0, 10, 20, 0, 0, 0, 0, 0, 30, 10, 0, 0, 0, 0, 0, 0;
  problem.shared_info_matrix = info_matrix;
  problem.time_since_last_relay_contact_with_agent = Eigen::Vector4d{10, 30, 0, 40};
  int curr_state{0};
  int action{3};

  EyesOnGuysProblem child_state = problem.create_child_eyes_on_guys_state(curr_state, action);

  double dt = compute_time_to_take_action(curr_state, action, relay_speed, dist_between_agents);
  double info_gain = simulate_agent_info_gain(40 + dt);
  Eigen::Vector4d expected_relay_current_info{30, 10, 40, info_gain};
  Eigen::Matrix4d expected_info_matrix = info_matrix;
  expected_info_matrix.row(action) = expected_relay_current_info;
  Eigen::Vector4d expected_time_since_last_relay_contact_with_agent{10 + dt, 30 + dt, dt, 0};

  EXPECT_EQ(expected_relay_current_info, child_state.relays_current_info);
  EXPECT_EQ(expected_info_matrix, child_state.shared_info_matrix);
  EXPECT_EQ(expected_time_since_last_relay_contact_with_agent,
            child_state.time_since_last_relay_contact_with_agent);
}

} // namespace eyes_on_guys
