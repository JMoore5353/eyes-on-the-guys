#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <vector>

#include "branch_and_bound_solver.hpp"

namespace eyes_on_guys
{

class BranchAndBoundTest : public ::testing::Test
{
public:
  BranchAndBoundTest()
  {
    distance_between_agents << 0, 20, 40, 60, 20, 0, 80, 100, 40, 80, 0, 120, 60, 100, 120, 0;
    shared_info_matrix << 0, 10, 20, 0, 0, 0, 0, 0, 30, 10, 0, 0, 0, 0, 0, 0;
  }

protected:
  int num_agents{4};
  double relay_speed{1.0};
  Eigen::Matrix4d distance_between_agents;
  Eigen::Matrix4d shared_info_matrix;
  int curr_state{0};
  int max_depth{10};
  int max_iterations{1000};
  double discount_factor{0.9};
  bool debug_mode{true};
};

TEST_F(BranchAndBoundTest, OrderingsAreCorrect_Qmax)
{
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  std::multiset<BranchAndBoundSolver::NodePtr, BranchAndBoundSolver::QMaxComparator> nodes;

  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    5.0, std::vector<int>{1}, 0.0, 0, problem, 3U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    10.0, std::vector<int>{2}, 0.0, 0, problem, 2U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    7.0, std::vector<int>{3}, 0.0, 0, problem, 4U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    10.0, std::vector<int>{4}, 0.0, 0, problem, 1U));

  ASSERT_EQ(nodes.size(), 4U);

  auto it = nodes.begin();
  EXPECT_EQ((*it)->id, 1U);
  EXPECT_DOUBLE_EQ((*it)->q_max, 10.0);

  ++it;
  EXPECT_EQ((*it)->id, 2U);
  EXPECT_DOUBLE_EQ((*it)->q_max, 10.0);

  ++it;
  EXPECT_EQ((*it)->id, 4U);
  EXPECT_DOUBLE_EQ((*it)->q_max, 7.0);

  ++it;
  EXPECT_EQ((*it)->id, 3U);
  EXPECT_DOUBLE_EQ((*it)->q_max, 5.0);
}

TEST_F(BranchAndBoundTest, MakeNodeBuildsNodeWithExpectedValues)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> path{0, 2};
  constexpr int test_state = 2;
  constexpr int test_depth = 3;
  constexpr double test_reward = 4.5;

  const BranchAndBoundSolver::NodePtr first_node =
    solver.make_node(test_depth, path, test_reward, problem);
  const BranchAndBoundSolver::NodePtr second_node =
    solver.make_node(test_depth + 1, std::vector<int>{0, 2, 1}, test_reward + 1.0, problem);

  ASSERT_NE(first_node, nullptr);
  ASSERT_NE(second_node, nullptr);

  EXPECT_DOUBLE_EQ(first_node->q_max, solver.q_max(test_state, path, problem, test_reward, test_depth));
  EXPECT_EQ(first_node->path.back(), test_state);
  EXPECT_EQ(first_node->depth, test_depth);
  EXPECT_DOUBLE_EQ(first_node->reward, test_reward);
  EXPECT_EQ(first_node->path, path);
  EXPECT_EQ(first_node->problem.relays_current_info.size(), num_agents);
  EXPECT_EQ(first_node->id, 0U);
  EXPECT_EQ(second_node->id, 1U);
}

TEST_F(BranchAndBoundTest, AddUnexploredNodeAddsNodeToAllIndexes)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node =
    std::make_shared<BranchAndBoundSolver::Node>(
      2.0, std::vector<int>{1}, 3.0, 1, problem, 42U);

  solver.add_unexplored_node(node);

  ASSERT_EQ(solver.unexplored_nodes_.size(), 1U);

  EXPECT_EQ(*solver.unexplored_nodes_.begin(), node);
}

TEST_F(BranchAndBoundTest, EraseUnexploredNodeRemovesNodeFromAllIndexes)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node_to_keep =
    std::make_shared<BranchAndBoundSolver::Node>(
      3.0, std::vector<int>{1}, 0.0, 1, problem, 10U);
  const BranchAndBoundSolver::NodePtr node_to_erase =
    std::make_shared<BranchAndBoundSolver::Node>(
      4.0, std::vector<int>{2}, 0.0, 1, problem, 11U);

  solver.add_unexplored_node(node_to_keep);
  solver.add_unexplored_node(node_to_erase);
  ASSERT_EQ(solver.unexplored_nodes_.size(), 2U);

  solver.erase_unexplored_node(node_to_erase);

  EXPECT_EQ(solver.unexplored_nodes_.size(), 1U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_to_erase), 0U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_to_keep), 1U);
}

TEST_F(BranchAndBoundTest, PruneNodesWithQMaxBelowRemovesOnlyThresholdMatches)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node_below_1 =
    std::make_shared<BranchAndBoundSolver::Node>(
      2.0, std::vector<int>{1}, 0.0, 1, problem, 50U);
  const BranchAndBoundSolver::NodePtr node_below_2 =
    std::make_shared<BranchAndBoundSolver::Node>(
      3.5, std::vector<int>{2}, 0.0, 1, problem, 51U);
  const BranchAndBoundSolver::NodePtr node_at_threshold =
    std::make_shared<BranchAndBoundSolver::Node>(
      5.0, std::vector<int>{3}, 0.0, 1, problem, 52U);
  const BranchAndBoundSolver::NodePtr node_above =
    std::make_shared<BranchAndBoundSolver::Node>(
      7.0, std::vector<int>{4}, 0.0, 1, problem, 53U);

  solver.add_unexplored_node(node_below_1);
  solver.add_unexplored_node(node_below_2);
  solver.add_unexplored_node(node_at_threshold);
  solver.add_unexplored_node(node_above);

  ASSERT_EQ(solver.unexplored_nodes_.size(), 4U);

  // Set the best reward to 5.0 to test pruning behavior
  solver.best_reward_ = 5.0;
  const std::size_t pruned = solver.prune_nodes();

  EXPECT_EQ(pruned, 3U);
  EXPECT_EQ(solver.unexplored_nodes_.size(), 1U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_below_1), 0U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_below_2), 0U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_at_threshold), 0U);
  EXPECT_EQ(solver.unexplored_nodes_.count(node_above), 1U);
  EXPECT_EQ(*solver.unexplored_nodes_.begin(), node_above);
}

TEST_F(BranchAndBoundTest, PopNodeWithHighestQMaxReturnsAndRemovesBestNode)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr low_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      2.0, std::vector<int>{1}, 0.0, 1, problem, 30U);
  const BranchAndBoundSolver::NodePtr best_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      5.0, std::vector<int>{2}, 0.0, 1, problem, 31U);
  const BranchAndBoundSolver::NodePtr mid_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      3.0, std::vector<int>{3}, 0.0, 1, problem, 32U);

  solver.add_unexplored_node(low_node);
  solver.add_unexplored_node(best_node);
  solver.add_unexplored_node(mid_node);

  const BranchAndBoundSolver::NodePtr popped = solver.pop_node_with_highest_q_max();
  ASSERT_NE(popped, nullptr);
  EXPECT_EQ(popped, best_node);
  EXPECT_DOUBLE_EQ(popped->q_max, 5.0);
  EXPECT_EQ(solver.unexplored_nodes_.size(), 2U);
  EXPECT_EQ(solver.unexplored_nodes_.count(best_node), 0U);
  EXPECT_EQ(solver.unexplored_nodes_.count(low_node), 1U);
  EXPECT_EQ(solver.unexplored_nodes_.count(mid_node), 1U);
}

TEST_F(BranchAndBoundTest, ProblemDimensionsAreValidChecksKeyShapeMismatchCases)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};

  EyesOnGuysProblem valid_problem{num_agents, relay_speed, distance_between_agents};
  EXPECT_TRUE(solver.problem_dimensions_are_valid(valid_problem));

  EyesOnGuysProblem bad_vector_problem{num_agents, relay_speed, distance_between_agents};
  bad_vector_problem.relays_current_info = Eigen::VectorXd::Zero(num_agents - 1);
  EXPECT_FALSE(solver.problem_dimensions_are_valid(bad_vector_problem));

  EyesOnGuysProblem bad_matrix_problem{num_agents, relay_speed, distance_between_agents};
  bad_matrix_problem.shared_info_matrix = Eigen::MatrixXd::Zero(num_agents, num_agents - 1);
  EXPECT_FALSE(solver.problem_dimensions_are_valid(bad_matrix_problem));
}

TEST_F(BranchAndBoundTest, MaybeUpdateBestSolutionTracksHighestRewardNonEmptyPath)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr null_node = nullptr;
  const BranchAndBoundSolver::NodePtr empty_path_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, std::vector<int>{}, 10.0, 1, problem, 40U);
  const BranchAndBoundSolver::NodePtr low_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, std::vector<int>{1}, 5.0, 1, problem, 41U);
  const BranchAndBoundSolver::NodePtr high_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, std::vector<int>{2, 3}, 8.0, 1, problem, 42U);
  const BranchAndBoundSolver::NodePtr lower_late_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, std::vector<int>{0, 1}, 7.0, 1, problem, 43U);

  solver.maybe_update_best_solution(null_node);
  EXPECT_EQ(solver.best_path_.size(), 0U);

  solver.maybe_update_best_solution(empty_path_node);
  EXPECT_EQ(solver.best_path_.size(), 0U);

  solver.maybe_update_best_solution(low_reward_node);
  EXPECT_DOUBLE_EQ(solver.best_reward_, 5.0);
  EXPECT_EQ(solver.best_path_, std::vector<int>({1}));

  solver.maybe_update_best_solution(high_reward_node);
  EXPECT_DOUBLE_EQ(solver.best_reward_, 8.0);
  EXPECT_EQ(solver.best_path_, std::vector<int>({2, 3}));

  solver.maybe_update_best_solution(lower_late_node);
  EXPECT_DOUBLE_EQ(solver.best_reward_, 8.0);
  EXPECT_EQ(solver.best_path_, std::vector<int>({2, 3}));
}

TEST_F(BranchAndBoundTest, SolveResetsBetweenCalls)
{
  const int test_max_depth = 2;
  BranchAndBoundSolver solver{
    num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> first_result = solver.solve(0, problem);
  const double first_best = solver.best_reward_;
  const std::size_t first_explored = solver.explored_nodes_count_;

  const std::vector<int> second_result = solver.solve(0, problem);

  EXPECT_DOUBLE_EQ(solver.best_reward_, first_best);
  EXPECT_EQ(solver.explored_nodes_count_, first_explored);
  EXPECT_EQ(first_result, second_result);
  // Note: completed_paths_count may be less than total possible paths due to pruning
  EXPECT_GT(solver.completed_paths_count_, 0U);
}

TEST_F(BranchAndBoundTest, SolveReturnsEmptyForInvalidDimensions)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, false};

  EyesOnGuysProblem bad_problem{num_agents, relay_speed, distance_between_agents};
  bad_problem.relays_current_info = Eigen::VectorXd::Zero(num_agents - 1);

  const std::vector<int> result = solver.solve(0, bad_problem);
  EXPECT_TRUE(result.empty());
}

TEST_F(BranchAndBoundTest, SolveRespectsMaxIterationsLimit)
{
  const int test_max_depth = 5;
  const int limited_iterations = 10;
  BranchAndBoundSolver solver{
    num_agents, test_max_depth, limited_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  solver.solve(0, problem);

  EXPECT_LE(solver.explored_nodes_count_, static_cast<std::size_t>(limited_iterations));
}

TEST_F(BranchAndBoundTest, SolveWithDepthOne)
{
  const int test_max_depth = 1;
  BranchAndBoundSolver solver{
    num_agents, test_max_depth, max_iterations, 1.0, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const int initial_state = 0;
  const std::vector<int> best_path = solver.solve(initial_state, problem);

  ASSERT_EQ(best_path.size(), 2U);
  EXPECT_EQ(best_path[0], initial_state);
  EXPECT_GE(best_path[1], 0);
  EXPECT_LT(best_path[1], num_agents);
  // Note: completed_paths_count may be less than num_agents due to pruning
  EXPECT_GT(solver.completed_paths_count_, 0U);

  double best_single_step_reward = std::numeric_limits<double>::lowest();
  int best_action = -1;
  for (int a = 0; a < num_agents; ++a) {
    auto child = problem.create_child_eyes_on_guys_state(initial_state, a);
    double r = compute_reward_model(initial_state, a, problem, child);
    if (r > best_single_step_reward) {
      best_single_step_reward = r;
      best_action = a;
    }
  }
  EXPECT_EQ(best_path[1], best_action);
  EXPECT_DOUBLE_EQ(solver.best_reward_, best_single_step_reward);
}

TEST_F(BranchAndBoundTest, SolveAppliesDiscountFactor)
{
  const int test_max_depth = 2;
  BranchAndBoundSolver solver_no_discount{
    num_agents, test_max_depth, max_iterations, 1.0, false};
  BranchAndBoundSolver solver_with_discount{
    num_agents, test_max_depth, max_iterations, 0.5, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.time_since_last_relay_contact_with_agent =
    Eigen::VectorXd::Constant(num_agents, 100.0);

  solver_no_discount.solve(0, problem);
  solver_with_discount.solve(0, problem);

  EXPECT_NE(solver_no_discount.best_reward_, solver_with_discount.best_reward_);
}

TEST_F(BranchAndBoundTest, SolveMatchesBruteForceForSmallProblem)
{
  const int test_max_depth = 3;
  BranchAndBoundSolver solver{
    num_agents, test_max_depth, max_iterations, 1.0, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const int initial_state = 0;
  const std::vector<int> solver_path = solver.solve(initial_state, problem);

  double brute_best_reward = std::numeric_limits<double>::lowest();
  std::vector<int> brute_best_path;

  for (int a1 = 0; a1 < num_agents; ++a1) {
    auto p1 = problem.create_child_eyes_on_guys_state(initial_state, a1);
    double r1 = compute_reward_model(initial_state, a1, problem, p1);

    for (int a2 = 0; a2 < num_agents; ++a2) {
      auto p2 = p1.create_child_eyes_on_guys_state(a1, a2);
      double r2 = r1 + compute_reward_model(a1, a2, p1, p2);

      for (int a3 = 0; a3 < num_agents; ++a3) {
        auto p3 = p2.create_child_eyes_on_guys_state(a2, a3);
        double r3 = r2 + compute_reward_model(a2, a3, p2, p3);

        if (r3 > brute_best_reward) {
          brute_best_reward = r3;
          brute_best_path = {initial_state, a1, a2, a3};
        }
      }
    }
  }

  EXPECT_EQ(solver_path, brute_best_path);
  EXPECT_DOUBLE_EQ(solver.best_reward_, brute_best_reward);
}

TEST_F(BranchAndBoundTest, AntiGreedyPathReturnsEmptyWhenAtMaxDepth)
{
  const int test_max_depth = 3;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> path = solver.anti_greedy_path(0, {0}, problem, test_max_depth);
  EXPECT_TRUE(path.empty());
}

TEST_F(BranchAndBoundTest, AntiGreedyPathPicksFarthestNeighbor)
{
  const int test_max_depth = 1;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  // Start at state 0, only state 1 is visited (row 1 zeroed, row 0 intact)
  const std::vector<int> path = solver.anti_greedy_path(0, {1}, problem, 0);

  ASSERT_EQ(path.size(), 1U);
  EXPECT_EQ(path[0], 3);  // State 3 is farthest from state 0 (dist=60)
}

TEST_F(BranchAndBoundTest, AntiGreedyPathAvoidsVisitedNodes)
{
  const int test_max_depth = 2;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> path = solver.anti_greedy_path(2, {0, 2}, problem, 0);

  ASSERT_EQ(path.size(), 2U);
  for (int state : path) {
    EXPECT_GE(state, 0);
    EXPECT_LT(state, num_agents);
  }
}

TEST_F(BranchAndBoundTest, AntiGreedyPathResetsWhenAllNodesVisited)
{
  const int test_max_depth = 5;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> path = solver.anti_greedy_path(0, {0, 1, 2, 3}, problem, 0);

  ASSERT_EQ(path.size(), 5U);
  EXPECT_EQ(path[0], 3);
  for (int state : path) {
    EXPECT_GE(state, 0);
    EXPECT_LT(state, num_agents);
  }
}

TEST_F(BranchAndBoundTest, AntiGreedyPathLengthMatchesRemainingDepth)
{
  const int test_max_depth = 6;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  for (int depth = 0; depth <= test_max_depth; ++depth) {
    const std::vector<int> path = solver.anti_greedy_path(0, {0}, problem, depth);
    EXPECT_EQ(static_cast<int>(path.size()), test_max_depth - depth)
      << "Failed for depth=" << depth;
  }
}

TEST_F(BranchAndBoundTest, AntiGreedyPathWithSingleAgent)
{
  const int single_agent = 1;
  const int test_max_depth = 3;
  Eigen::MatrixXd single_dist(1, 1);
  single_dist << 0;
  BranchAndBoundSolver solver{single_agent, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{single_agent, relay_speed, single_dist};

  const std::vector<int> path = solver.anti_greedy_path(0, {0}, problem, 0);

  ASSERT_EQ(path.size(), 3U);
  for (int state : path) {
    EXPECT_EQ(state, 0);
  }
}

TEST_F(BranchAndBoundTest, QMaxIsAtLeastPathReward)
{
  const int test_max_depth = 3;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.shared_info_matrix = shared_info_matrix;
  problem.time_since_last_relay_contact_with_agent = Eigen::VectorXd::Constant(num_agents, 50.0);

  const double path_reward = 10.0;
  const double upper = solver.q_max(0, {0}, problem, path_reward, 0);

  EXPECT_GE(upper, path_reward);
}

TEST_F(BranchAndBoundTest, QMaxWithoutPenaltiesIsGreaterOrEqualToFullReward)
{
  const int test_max_depth = 2;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.shared_info_matrix = shared_info_matrix;
  problem.time_since_last_relay_contact_with_agent = Eigen::VectorXd::Constant(num_agents, 50.0);

  const double upper = solver.q_max(0, {0}, problem, 0.0, 0);

  auto ag_path = solver.anti_greedy_path(0, {0}, problem, 0);
  double reward_with_penalties = 0.0;
  EyesOnGuysProblem curr_problem = problem;
  int curr_state = 0;
  int curr_depth = 0;
  for (int next_state : ag_path) {
    auto next_problem = curr_problem.create_child_eyes_on_guys_state(curr_state, next_state);
    reward_with_penalties += std::pow(discount_factor, curr_depth) *
      compute_reward_model(curr_state, next_state, curr_problem, next_problem, true);
    curr_state = next_state;
    curr_problem = next_problem;
    curr_depth++;
  }

  EXPECT_GE(upper, reward_with_penalties);
}

TEST_F(BranchAndBoundTest, QMaxAtMaxDepthEqualsPathReward)
{
  const int test_max_depth = 3;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.shared_info_matrix = shared_info_matrix;
  problem.time_since_last_relay_contact_with_agent = Eigen::VectorXd::Constant(num_agents, 50.0);

  const double path_reward = 42.0;
  const double upper = solver.q_max(0, {0, 1, 2}, problem, path_reward, test_max_depth);

  EXPECT_DOUBLE_EQ(upper, path_reward);
}

TEST_F(BranchAndBoundTest, QMaxIsConsistentWithAntiGreedyPath)
{
  const int test_max_depth = 4;
  BranchAndBoundSolver solver{num_agents, test_max_depth, max_iterations, discount_factor, false};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.shared_info_matrix = shared_info_matrix;
  problem.time_since_last_relay_contact_with_agent = Eigen::VectorXd::Constant(num_agents, 50.0);

  const double path_reward = 5.0;
  const int start_state = 1;
  const std::vector<int> current_path = {0, 1};
  const int start_depth = 1;

  const double upper = solver.q_max(start_state, current_path, problem, path_reward, start_depth);

  auto ag_path = solver.anti_greedy_path(start_state, current_path, problem, start_depth);
  double manual_reward = 0.0;
  EyesOnGuysProblem curr_problem = problem;
  int curr_state = start_state;
  int curr_depth = start_depth;
  for (int next_state : ag_path) {
    auto next_problem = curr_problem.create_child_eyes_on_guys_state(curr_state, next_state);
    manual_reward += std::pow(discount_factor, curr_depth) *
      compute_reward_model(curr_state, next_state, curr_problem, next_problem, false);
    curr_state = next_state;
    curr_problem = next_problem;
    curr_depth++;
  }

  EXPECT_DOUBLE_EQ(upper, path_reward + manual_reward);
}

// ============================================================================
// Performance Test (keep this as the last unit test)
// ============================================================================

TEST_F(BranchAndBoundTest, PerformanceTestLargeProblem)
{
  // Test parameters - adjust these to explore different problem sizes
  const int num_agents = 6;
  const int test_depth = 10;  // Adjust between 1-10 to test different depths
  const int max_iterations = 5000000;  // Large enough to not limit the search
  
  // Create a realistic distance matrix for 6 agents
  // Distances are in some arbitrary units (e.g., meters)
  Eigen::MatrixXd distance_between_agents(num_agents, num_agents);
  distance_between_agents << 
    0,  50, 80, 120, 150, 200,
    50, 0,  60,  90, 130, 170,
    80, 60,  0,  70, 110, 140,
    120, 90, 70,  0,  80, 100,
    150, 130, 110, 80,  0,  90,
    200, 170, 140, 100, 90,  0;
  
  // Create a realistic shared information matrix
  // Represents information gain between agents
  Eigen::MatrixXd shared_info_matrix(num_agents, num_agents);
  shared_info_matrix <<
    0,  10,  5,  0,  0,  0,
    0,  0,  15,  5,  0,  0,
    0,  0,  0,  20, 10,  0,
    0,  0,  0,  0,  25, 15,
    0,  0,  0,  0,  0,  30,
    0,  0,  0,  0,  0,  0;
  
  // Relay speed (units per time step)
  const double relay_speed = 10.0;
  
  // Create the problem
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};
  problem.shared_info_matrix = shared_info_matrix;
  
  // Initialize time since last contact to create urgency
  problem.time_since_last_relay_contact_with_agent = 
    Eigen::VectorXd::LinSpaced(num_agents, 10.0, 100.0);
  
  // Create solver with debug mode to see progress
  BranchAndBoundSolver solver{
    num_agents, test_depth, max_iterations, 0.9, false};  // Enable debug output
  
  std::cout << "\n=== Performance Test Parameters ===" << std::endl;
  std::cout << "Number of agents: " << num_agents << std::endl;
  std::cout << "Search depth: " << test_depth << std::endl;
  std::cout << "Maximum iterations: " << max_iterations << std::endl;
  std::cout << "Total possible paths: " << std::pow(num_agents, test_depth) << std::endl;
  std::cout << "Discount factor: 0.95" << std::endl;
  std::cout << "Relay speed: " << relay_speed << std::endl;
  std::cout << "\nStarting solve..." << std::endl;
  
  // Time the solve operation
  auto start_time = std::chrono::high_resolution_clock::now();
  
  const int initial_state = 0;
  const std::vector<int> best_path = solver.solve(initial_state, problem);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);
  
  // Print performance metrics
  std::cout << "\n=== Performance Results ===" << std::endl;
  std::cout << "Solve time: " << duration.count() << " ms" << std::endl;
  std::cout << "Best path found: ";
  for (size_t i = 0; i < best_path.size(); ++i) {
    std::cout << best_path[i];
    if (i < best_path.size() - 1) std::cout << " -> ";
  }
  std::cout << std::endl;
  std::cout << "Best reward: " << solver.best_reward_ << std::endl;
  std::cout << "Nodes explored: " << solver.explored_nodes_count_ << std::endl;
  std::cout << "Nodes pr (pruned): " << solver.total_pruned_nodes_ << std::endl;
  std::cout << "Completed paths: " << solver.completed_paths_count_ << std::endl;
  std::cout << "Final best reward: " << solver.best_reward_ << std::endl;
  
  // Calculate efficiency metrics
  // Geometric series: total nodes = (b^(d+1) - 1) / (b - 1) for branching factor b, depth d
  const double total_possible_nodes =
    (std::pow(num_agents, test_depth + 1) - 1) / (num_agents - 1);
  const double exploration_percentage = (solver.explored_nodes_count_ / total_possible_nodes) * 100.0;
  const double pruning_percentage = (solver.total_pruned_nodes_ / total_possible_nodes) * 100.0;
  
  std::cout << "\n=== Efficiency Metrics ===" << std::endl;
  std::cout << "Total nodes in full tree: " << total_possible_nodes << std::endl;
  std::cout << "Tree explored: " << exploration_percentage << "%" << std::endl;
  std::cout << "Tree pr: " << pruning_percentage << "%" << std::endl;
  std::cout << "Nodes per millisecond: " << 
    (solver.explored_nodes_count_ / std::max(1.0, static_cast<double>(duration.count()))) << std::endl;
  
  // Note: This test doesn't assert anything - it's purely for observation
  // You can adjust the parameters above to test different scenarios
}

} // namespace eyes_on_guys
