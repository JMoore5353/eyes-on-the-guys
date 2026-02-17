#include <Eigen/Core>
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
  double discount_factor{0.95};
  bool debug_mode{true};
};

TEST_F(BranchAndBoundTest, OrderingsAreCorrect_Qmax)
{
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  std::multiset<BranchAndBoundSolver::NodePtr, BranchAndBoundSolver::QMaxComparator> nodes;

  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0, 5.0, std::vector<int>{1}, 0.0, curr_state, 0, problem, 3U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0, 10.0, std::vector<int>{2}, 0.0, curr_state, 0, problem, 2U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0, 7.0, std::vector<int>{3}, 0.0, curr_state, 0, problem, 4U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0, 10.0, std::vector<int>{4}, 0.0, curr_state, 0, problem, 1U));

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
    solver.make_node(test_state, test_depth, path, test_reward, problem);
  const BranchAndBoundSolver::NodePtr second_node =
    solver.make_node(1, test_depth + 1, std::vector<int>{0, 2, 1}, test_reward + 1.0, problem);

  ASSERT_NE(first_node, nullptr);
  ASSERT_NE(second_node, nullptr);

  EXPECT_DOUBLE_EQ(first_node->u_min, solver.u_min(test_state, problem, test_reward, test_depth));
  EXPECT_DOUBLE_EQ(first_node->q_max, solver.q_max(test_state, problem, test_reward, test_depth));
  EXPECT_EQ(first_node->state, test_state);
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
      1.0, 2.0, std::vector<int>{1}, 3.0, curr_state, 1, problem, 42U);

  solver.add_unexplored_node(node);

  ASSERT_EQ(solver.unexplored_nodes_by_q_max_.size(), 1U);

  EXPECT_EQ(*solver.unexplored_nodes_by_q_max_.begin(), node);
}

TEST_F(BranchAndBoundTest, EraseUnexploredNodeRemovesNodeFromAllIndexes)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node_to_keep =
    std::make_shared<BranchAndBoundSolver::Node>(
      1.0, 3.0, std::vector<int>{1}, 0.0, curr_state, 1, problem, 10U);
  const BranchAndBoundSolver::NodePtr node_to_erase =
    std::make_shared<BranchAndBoundSolver::Node>(
      2.0, 4.0, std::vector<int>{2}, 0.0, curr_state, 1, problem, 11U);

  solver.add_unexplored_node(node_to_keep);
  solver.add_unexplored_node(node_to_erase);
  ASSERT_EQ(solver.unexplored_nodes_by_q_max_.size(), 2U);

  solver.erase_unexplored_node(node_to_erase);

  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.size(), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_to_erase), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_to_keep), 1U);
}

TEST_F(BranchAndBoundTest, PruneNodesWithQMaxBelowRemovesOnlyThresholdMatches)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node_below_1 =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 2.0, std::vector<int>{1}, 0.0, curr_state, 1, problem, 50U);
  const BranchAndBoundSolver::NodePtr node_below_2 =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 3.5, std::vector<int>{2}, 0.0, curr_state, 1, problem, 51U);
  const BranchAndBoundSolver::NodePtr node_at_threshold =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 5.0, std::vector<int>{3}, 0.0, curr_state, 1, problem, 52U);
  const BranchAndBoundSolver::NodePtr node_above =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 7.0, std::vector<int>{4}, 0.0, curr_state, 1, problem, 53U);

  solver.add_unexplored_node(node_below_1);
  solver.add_unexplored_node(node_below_2);
  solver.add_unexplored_node(node_at_threshold);
  solver.add_unexplored_node(node_above);

  ASSERT_EQ(solver.unexplored_nodes_by_q_max_.size(), 4U);

  // Set the threshold to 5.0 to test pruning behavior
  solver.u_min_threshold_ = 5.0;
  const std::size_t pruned = solver.prune_nodes();

  EXPECT_EQ(pruned, 3U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.size(), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_below_1), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_below_2), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_at_threshold), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(node_above), 1U);
  EXPECT_EQ(*solver.unexplored_nodes_by_q_max_.begin(), node_above);
}

TEST_F(BranchAndBoundTest, PopNodeWithHighestQMaxReturnsAndRemovesBestNode)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr low_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 2.0, std::vector<int>{1}, 0.0, curr_state, 1, problem, 30U);
  const BranchAndBoundSolver::NodePtr best_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 5.0, std::vector<int>{2}, 0.0, curr_state, 1, problem, 31U);
  const BranchAndBoundSolver::NodePtr mid_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 3.0, std::vector<int>{3}, 0.0, curr_state, 1, problem, 32U);

  solver.add_unexplored_node(low_node);
  solver.add_unexplored_node(best_node);
  solver.add_unexplored_node(mid_node);

  const BranchAndBoundSolver::NodePtr popped = solver.pop_node_with_highest_q_max();
  ASSERT_NE(popped, nullptr);
  EXPECT_EQ(popped, best_node);
  EXPECT_DOUBLE_EQ(popped->q_max, 5.0);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.size(), 2U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(best_node), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(low_node), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_q_max_.count(mid_node), 1U);
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
      0.0, 0.0, std::vector<int>{}, 10.0, curr_state, 1, problem, 40U);
  const BranchAndBoundSolver::NodePtr low_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 0.0, std::vector<int>{1}, 5.0, curr_state, 1, problem, 41U);
  const BranchAndBoundSolver::NodePtr high_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 0.0, std::vector<int>{2, 3}, 8.0, curr_state, 1, problem, 42U);
  const BranchAndBoundSolver::NodePtr lower_late_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0, 0.0, std::vector<int>{0, 1}, 7.0, curr_state, 1, problem, 43U);

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
  EXPECT_EQ(solver.completed_paths_count_, std::pow(num_agents, test_max_depth));
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
  EXPECT_EQ(solver.completed_paths_count_, static_cast<std::size_t>(num_agents));

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

} // namespace eyes_on_guys
