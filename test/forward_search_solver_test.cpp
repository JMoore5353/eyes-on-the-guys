#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>

#include "eyes_on_guys_problem.hpp"
#include "forward_search_solver.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

namespace
{

std::vector<int> ids_to_actions(const std::vector<std::string> & ids)
{
  std::vector<int> actions;
  actions.reserve(ids.size());
  for (const std::string & id : ids) {
    actions.push_back(std::stoi(id));
  }
  return actions;
}

ForwardSearchSolver::ForwardSearchConfig make_default_forward_config(const int depth,
                                                                     const int starting_guy)
{
  ForwardSearchSolver::ForwardSearchConfig config;
  config.depth = depth;
  config.num_rollouts = 0;
  config.roll_out_depth = 0;
  config.discount_factor = 1.0;
  config.agent_velocity = 1.0;
  config.info_shared_weight = 1.0;
  config.path_length_weight = 1.0;
  config.time_since_visit_weight = 1.0;
  config.starting_guy = starting_guy;
  return config;
}

ForwardSearchSolver::ForwardSearchInput make_two_agent_forward_input(const Eigen::MatrixXd & shared_info_matrix)
{
  ForwardSearchSolver::ForwardSearchInput input;
  input.ids = {"0", "1"};

  ForwardSearchSolver::GuyState s0;
  s0.pose.pose.position.x = 0.0;
  s0.pose.pose.position.y = 0.0;
  s0.pose.pose.position.z = 0.0;

  ForwardSearchSolver::GuyState s1;
  s1.pose.pose.position.x = 10.0;
  s1.pose.pose.position.y = 0.0;
  s1.pose.pose.position.z = 0.0;

  input.guy_states_by_id.emplace("0", s0);
  input.guy_states_by_id.emplace("1", s1);
  input.shared_info_matrix = shared_info_matrix;
  return input;
}

ForwardSearchSolver::ForwardSearchInput make_three_agent_equilateral_input(
  const Eigen::Vector3d & bits,
  const Eigen::MatrixXd & shared_info_matrix)
{
  ForwardSearchSolver::ForwardSearchInput input;
  input.ids = {"0", "1", "2"};

  ForwardSearchSolver::GuyState s0;
  s0.pose.pose.position.x = 0.0;
  s0.pose.pose.position.y = 0.0;
  s0.pose.pose.position.z = 0.0;
  s0.bits = static_cast<float>(bits(0));

  ForwardSearchSolver::GuyState s1;
  s1.pose.pose.position.x = 10.0;
  s1.pose.pose.position.y = 0.0;
  s1.pose.pose.position.z = 0.0;
  s1.bits = static_cast<float>(bits(1));

  ForwardSearchSolver::GuyState s2;
  s2.pose.pose.position.x = 5.0;
  s2.pose.pose.position.y = 8.660254037844386;
  s2.pose.pose.position.z = 0.0;
  s2.bits = static_cast<float>(bits(2));

  input.guy_states_by_id.emplace("0", s0);
  input.guy_states_by_id.emplace("1", s1);
  input.guy_states_by_id.emplace("2", s2);
  input.shared_info_matrix = shared_info_matrix;
  return input;
}

} // namespace

class FSTwoAgentTest : public ::testing::Test
{
public:
  FSTwoAgentTest()
      : num_agents{2}
      , initial_state{0}
      , num_iter{10}
      , depth{10}
      , discount_factor{1.0}
      , exploration_bonus{1.0}
      , relay_speed{1.0}
      , problem_info{num_agents, relay_speed, dist_between_agents}
      , lookahead_depth{0}
      , lookahead_iters{1}
      , optimal_action{1}
      , forward_input{}
      , forward_config{make_default_forward_config(depth, initial_state)}
  {
    dist_between_agents << 0.0, 10.0, 10.0, 0.0;
    problem_info.shared_info_matrix = dist_between_agents;
    forward_input = make_two_agent_forward_input(problem_info.shared_info_matrix);
  }

protected:
  int num_agents;
  int initial_state;
  int num_iter;
  int depth;
  double discount_factor;
  double exploration_bonus;
  double relay_speed;
  Eigen::Matrix2d dist_between_agents;
  EyesOnGuysProblem problem_info;
  int lookahead_depth;
  int lookahead_iters;
  int optimal_action;
  ForwardSearchSolver::ForwardSearchInput forward_input;
  ForwardSearchSolver::ForwardSearchConfig forward_config;
};

TEST_F(FSTwoAgentTest, Given2AgentsAndOneIteration_ExpectActionIsOtherAgent)
{
  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.sequence_ids.empty());
  EXPECT_EQ(std::stoi(result.sequence_ids.front()), optimal_action);
}

class FSThreeAgentTest : public ::testing::Test
{
public:
  FSThreeAgentTest()
      : num_agents{3}
      , initial_state{0}
      , num_iter{20}
      , depth{15}
      , discount_factor{1.0}
      , exploration_bonus{1.0}
      , relay_speed{1.0}
      , problem_info{num_agents, relay_speed, dist_between_agents}
      , lookahead_depth{0}
      , lookahead_iters{0}
      , forward_config{make_default_forward_config(depth, initial_state)}
  {}

protected:
  int num_agents;
  int initial_state;
  int num_iter;
  int depth;
  double discount_factor;
  double exploration_bonus;
  double relay_speed;
  Eigen::Matrix3d dist_between_agents;
  EyesOnGuysProblem problem_info;
  int lookahead_depth;
  int lookahead_iters;
  int optimal_action;
  std::vector<int> optimal_greedy_seq;
  ForwardSearchSolver::ForwardSearchInput forward_input;
  ForwardSearchSolver::ForwardSearchConfig forward_config;
};

class FSThreeAgentTestOne : public FSThreeAgentTest
{
public:
  FSThreeAgentTestOne()
  {
    dist_between_agents << 0.0, 10.0, 10.0,
                           10.0, 0.0, 10.0,
                           10.0, 10.0, 0.0;
    problem_info.distance_between_agents = dist_between_agents;

    problem_info.relays_current_info = Eigen::Vector3d{0.0, 100.0, 0.0};
    problem_info.shared_info_matrix = Eigen::MatrixXd::Zero(3, 3);
    problem_info.time_since_last_relay_contact_with_agent = Eigen::Vector3d{0.0, 0.0, 0.0};

    optimal_action = 2;
    optimal_greedy_seq = {2, 0, 1};
    forward_input = make_three_agent_equilateral_input(
      problem_info.relays_current_info, problem_info.shared_info_matrix);
  }
};

class FSThreeAgentTestTwo : public FSThreeAgentTest
{
public:
  FSThreeAgentTestTwo()
  {
    dist_between_agents << 0.0, 10.0, 10.0,
                           10.0, 0.0, 10.0,
                           10.0, 10.0, 0.0;
    problem_info.distance_between_agents = dist_between_agents;

    problem_info.relays_current_info = Eigen::Vector3d{100.0, 0.0, 0.0};
    problem_info.shared_info_matrix = Eigen::MatrixXd({{0, 0, 0}, {100, 0, 0}, {0, 0, 0}});
    problem_info.time_since_last_relay_contact_with_agent = Eigen::Vector3d{0.0, 5.0, 0.0};

    optimal_action = 2;
    optimal_greedy_seq = {2, 1, 0};
    forward_input = make_three_agent_equilateral_input(
      problem_info.relays_current_info, problem_info.shared_info_matrix);
  }
};

TEST_F(FSThreeAgentTestTwo, GivenThreeIters_WhenTakingAction_ExpectOptimalAction)
{
  forward_config.depth = 3;
  forward_config.num_rollouts = 0;

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(std::stoi(result.sequence_ids.front()), optimal_action);
  EXPECT_EQ(ids_to_actions(result.sequence_ids), optimal_greedy_seq);
}

TEST_F(FSThreeAgentTestTwo, Given1000Iters_WhenTakingAction_ExpectOptimalAction)
{
  num_iter = 1000;
  depth = 11;
  optimal_greedy_seq = {2, 1, 0, 2, 1, 0, 2, 1, 0, 2};
  forward_config.depth = depth;
  forward_config.num_rollouts = 0;

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(std::stoi(result.sequence_ids.front()), optimal_action);
  auto res = result.sequence_ids;
  res.pop_back();
  EXPECT_EQ(ids_to_actions(res), optimal_greedy_seq);
}

TEST_F(FSThreeAgentTestTwo, Given1000Iters_WhenTakingAction_ExpectMatchesMcts)
{
  num_iter = 1000;
  depth = 3;
  forward_config.depth = depth;
  forward_config.num_rollouts = 0;

  MonteCarloTreeSearch tree_searcher{num_agents};
  int mcts_action = tree_searcher.search_for_best_action(initial_state, num_iter, depth, discount_factor,
                                                         exploration_bonus, problem_info,
                                                         lookahead_depth, lookahead_iters);
  std::vector<int> mcts_greedy_seq = tree_searcher.get_greedy_sequence();

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(std::stoi(result.sequence_ids.front()), mcts_action);
  EXPECT_EQ(ids_to_actions(result.sequence_ids), mcts_greedy_seq);
}

TEST_F(FSThreeAgentTestTwo, GivenManyLookaheadIters_WhenTakingAction_ExpectOptimalAction)
{
  lookahead_iters = 1000;
  lookahead_depth = 5;
  optimal_greedy_seq = {2, 1, 0};
  forward_config.num_rollouts = 0;
  forward_config.depth = 3;

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  EXPECT_EQ(std::stoi(result.sequence_ids.front()), optimal_action);
  EXPECT_EQ(ids_to_actions(result.sequence_ids), optimal_greedy_seq);
}

class FSActualProblem : public ::testing::Test
{
public:
  FSActualProblem()
      : num_agents{6}
      , initial_state{0}
      , num_iter{100}
      , depth{7}
      , discount_factor{0.9}
      , exploration_bonus{100.0}
      , relay_speed{15.0}
      , problem_info{num_agents, relay_speed, dist_between_agents}
      , lookahead_depth{0}
      , lookahead_iters{0}
      , forward_config{make_default_forward_config(depth, initial_state)}
  {
    Eigen::Matrix<double, 6, 6> distance_between_agents;
    distance_between_agents <<
            0, 536.669, 260.261, 437.994, 141.305, 115.668,
      536.669, 0, 276.562, 99.0206, 395.931, 421.336,
      260.261, 276.562, 0, 178.291, 120.54, 145.473,
      437.994, 99.0206, 178.291, 0, 297.063, 322.526,
      141.305, 395.931, 120.54, 297.063, 0, 25.6663,
      115.668, 421.336, 145.473, 322.526, 25.6663, 0;
    problem_info.distance_between_agents = distance_between_agents;
    problem_info.shared_info_matrix = Eigen::MatrixXd::Zero(num_agents, num_agents);

    forward_input.ids = {"0", "1", "2", "3", "4", "5"};
    for (int i = 0; i < num_agents; ++i) {
      ForwardSearchSolver::GuyState s;
      s.pose.pose.position.x = static_cast<double>(i) * 50.0;
      s.pose.pose.position.y = 0.0;
      s.pose.pose.position.z = 0.0;
      forward_input.guy_states_by_id.emplace(std::to_string(i), s);
    }
    forward_input.shared_info_matrix = problem_info.shared_info_matrix;
  }

protected:
  int num_agents;
  int initial_state;
  int num_iter;
  int depth;
  double discount_factor;
  double exploration_bonus;
  double relay_speed;
  Eigen::Matrix3d dist_between_agents;
  EyesOnGuysProblem problem_info;
  int lookahead_depth;
  int lookahead_iters;
  int optimal_action;
  std::vector<int> optimal_greedy_seq;
  ForwardSearchSolver::ForwardSearchInput forward_input;
  ForwardSearchSolver::ForwardSearchConfig forward_config;
};

class FSActualProblemTestOne : public FSActualProblem
{
public:
  FSActualProblemTestOne()
  {
    initial_state = 1;
    forward_config.starting_guy = initial_state;
  }
};

TEST_F(FSActualProblem, ExpectOptimalActionIsNotSelf)
{
  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.sequence_ids.empty());
  EXPECT_NE(std::stoi(result.sequence_ids.front()), initial_state);
}

TEST_F(FSActualProblem, ExpectOptimalActionIsNotSelfComparedToMcts)
{
  MonteCarloTreeSearch tree_searcher{num_agents};
  int mcts_action = tree_searcher.search_for_best_action(initial_state,
                                                         num_iter,
                                                         depth,
                                                         discount_factor,
                                                         exploration_bonus,
                                                         problem_info,
                                                         lookahead_depth,
                                                         lookahead_iters);

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.sequence_ids.empty());
  EXPECT_NE(mcts_action, initial_state);
  EXPECT_NE(std::stoi(result.sequence_ids.front()), initial_state);
}

TEST_F(FSActualProblemTestOne, ExpectOptimalActionIsNotSelf)
{
  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.sequence_ids.empty());
  EXPECT_NE(std::stoi(result.sequence_ids.front()), initial_state);
}

TEST_F(FSActualProblemTestOne, ExpectOptimalActionIsNotSelfComparedToMcts)
{
  MonteCarloTreeSearch tree_searcher{num_agents};
  int mcts_action = tree_searcher.search_for_best_action(initial_state,
                                                         num_iter,
                                                         depth,
                                                         discount_factor,
                                                         exploration_bonus,
                                                         problem_info,
                                                         lookahead_depth,
                                                         lookahead_iters);

  ForwardSearchSolver solver;
  ForwardSearchSolver::ForwardSearchResult result = solver.solve(forward_input, forward_config);

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.sequence_ids.empty());
  EXPECT_NE(mcts_action, initial_state);
  EXPECT_NE(std::stoi(result.sequence_ids.front()), initial_state);
}

} // namespace eyes_on_guys
