#include <Eigen/Core>
#include <gtest/gtest.h>
#include <memory>

#include "eyes_on_guys_problem.hpp"
#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

namespace eyes_on_guys
{

TEST(find_greedy_action, WhenFindingGreedyAction_ExpectTheNodesGreedyAction)
{
  auto node = std::make_shared<MCTSNode>(0, 10, 1.0);
  int node_greedy_action = node->explore_best_action();

  int greedy_action = find_greedy_action(node);

  EXPECT_EQ(greedy_action, node_greedy_action);
}

TEST(transition_from_state, WhenTransitioning_ExpectTransitionIsDeterministic)
{
  auto node = std::make_shared<MCTSNode>(0, 10, 1.0);
  int action{2};

  for (int i = 0; i < 1000; ++i) {
    std::shared_ptr<MCTSNode> new_node = transition_from_state(node, action);
    EXPECT_EQ(new_node->get_id(), action);
    EXPECT_NE(new_node->get_id(), node->get_id());
  }
}

TEST(get_random_action, ExpectRandomActionNeverEqualsStateAndIsWithinBounds)
{
  int state{3};
  int num_states{4};
  for (int i = 0; i < 100; ++i) {
    int action = lookahead_get_random_action(state, num_states);
    EXPECT_NE(action, state);
    EXPECT_GE(action, 0);
    EXPECT_LE(action, num_states);
  }
}

class MCTSTwoAgentTest : public ::testing::Test
{
public:
  MCTSTwoAgentTest()
      : num_agents{2}
      , initial_state{0}
      , num_iter{1}
      , depth{10}
      , discount_factor{1.0}
      , exploration_bonus{1.0}
      , relay_speed{1.0}
      , problem_info{num_agents, relay_speed, dist_between_agents}
      , lookahead_depth{0}
      , lookahead_iters{1}
      , optimal_action{1}
  {
    dist_between_agents << 0.0, 10.0, 10.0, 0.0;
    problem_info.shared_info_matrix = dist_between_agents;
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
};

TEST_F(MCTSTwoAgentTest, Given2AgentsAndOneIteration_ExpectActionIsOtherAgent)
{
  MonteCarloTreeSearch tree_searcher{num_agents};

  int action = tree_searcher.search_for_best_action(initial_state, num_iter, depth, discount_factor,
                                                    exploration_bonus, problem_info,
                                                    lookahead_depth, lookahead_iters);

  EXPECT_EQ(action, optimal_action);
}

TEST_F(MCTSTwoAgentTest, WhenLookaheadingWithZeroDepth_ExpectCurrStateUnchanged)
{
  auto curr_node =
    std::make_shared<MCTSNode>(initial_state, num_agents, exploration_bonus, problem_info);

  double lookahead_reward = lookahead_value_function_estimate(
    curr_node, num_agents, lookahead_depth, discount_factor, lookahead_iters);

  EXPECT_EQ(lookahead_reward, 0.0);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(curr_node->get_children_vector()[i], nullptr);
  }
}

TEST_F(MCTSTwoAgentTest, WhenLookaheadingWith10Depth_ExpectCurrStateUnchanged)
{
  auto curr_node =
    std::make_shared<MCTSNode>(initial_state, num_agents, exploration_bonus, problem_info);
  lookahead_depth = 10;

  double lookahead_reward = lookahead_value_function_estimate(
    curr_node, num_agents, lookahead_depth, discount_factor, lookahead_iters);

  EXPECT_NE(lookahead_reward, 0.0);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(curr_node->get_children_vector()[i], nullptr);
  }
}

TEST_F(MCTSTwoAgentTest, WhenLookaheading_ExpectNonZeroReward)
{
  auto curr_node =
    std::make_shared<MCTSNode>(initial_state, num_agents, exploration_bonus, problem_info);
  lookahead_depth = 10;

  double lookahead_reward = lookahead_value_function_estimate(
    curr_node, num_agents, lookahead_depth, discount_factor, lookahead_iters);

  EXPECT_NE(lookahead_reward, 0.0);
}

TEST_F(MCTSTwoAgentTest, ExpectRewardFromSuccessorStateIsNotZero)
{
  auto curr_node =
    std::make_shared<MCTSNode>(initial_state, num_agents, exploration_bonus, problem_info);
  std::shared_ptr<MCTSNode> next_node = transition_from_state(curr_node, 1);

  double reward = compute_reward_from_transitioning(curr_node, next_node);

  EXPECT_NE(reward, 0.0);
}

class MCTSThreeAgentTest : public ::testing::Test
{
public:
  MCTSThreeAgentTest()
      : num_agents{3}
      , initial_state{0}
      , num_iter{3}
      , depth{3}
      , discount_factor{1.0}
      , exploration_bonus{1.0}
      , relay_speed{1.0}
      , problem_info{num_agents, relay_speed, dist_between_agents}
      , lookahead_depth{0}
      , lookahead_iters{0}
      , optimal_action{1}
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
};

class MCTSThreeAgentTestOne : public MCTSThreeAgentTest
{
public:
  MCTSThreeAgentTestOne()
  {
    dist_between_agents << 0.0, 10.0, 10.0, 10.0, 0.0, 10., 10., 10., 0.0;
    problem_info.shared_info_matrix = dist_between_agents;

    problem_info.relays_current_info = Eigen::Vector3d{0.0, 100.0, 0.0};
    problem_info.shared_info_matrix = Eigen::MatrixXd::Zero(3, 3);
    problem_info.time_since_last_relay_contact_with_agent = Eigen::Vector3d{0.0, 0.0, 0.0};
  }
};

TEST_F(MCTSThreeAgentTestOne, GivenThreeIters_WhenTakingAction_ExpectOptimalActionIs1)
{
  MonteCarloTreeSearch tree_searcher{num_agents};

  double action = tree_searcher.search_for_best_action(
    initial_state, num_iter, depth, discount_factor, exploration_bonus, problem_info,
    lookahead_depth, lookahead_iters);

  EXPECT_EQ(action, optimal_action);
}

class MCTSThreeAgentTestTwo : public MCTSThreeAgentTest
{
public:
  MCTSThreeAgentTestTwo()
  {
    dist_between_agents << 0.0, 10.0, 10.0, 10.0, 0.0, 10., 10., 10., 0.0;
    problem_info.shared_info_matrix = dist_between_agents;

    problem_info.relays_current_info = Eigen::Vector3d{100.0, 0.0, 0.0};
    problem_info.shared_info_matrix = Eigen::MatrixXd({{0, 0, 0}, {100, 0, 0}, {0, 0, 0}});
    problem_info.time_since_last_relay_contact_with_agent = Eigen::Vector3d{0.0, 0.0, 0.0};

    optimal_action = 2;
  }
};

TEST_F(MCTSThreeAgentTestTwo, GivenThreeIters_WhenTakingAction_ExpectOptimalActionIs2)
{
  MonteCarloTreeSearch tree_searcher{num_agents};

  double action = tree_searcher.search_for_best_action(
    initial_state, num_iter, depth, discount_factor, exploration_bonus, problem_info,
    lookahead_depth, lookahead_iters);

  EXPECT_EQ(action, optimal_action);
}

} // namespace eyes_on_guys
