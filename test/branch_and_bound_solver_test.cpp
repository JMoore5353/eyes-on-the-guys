#include <Eigen/Core>
#include <gtest/gtest.h>
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
  float discount_factor{0.95F};
  bool debug_mode{true};
};

TEST_F(BranchAndBoundTest, OrderingsAreCorrect_Umax)
{
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  std::multiset<BranchAndBoundSolver::NodePtr, BranchAndBoundSolver::UMaxComparator> nodes;

  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0F, 5.0F, std::vector<int>{1}, 0.0F, curr_state, 0, problem, 3U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0F, 10.0F, std::vector<int>{2}, 0.0F, curr_state, 0, problem, 2U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0F, 7.0F, std::vector<int>{3}, 0.0F, curr_state, 0, problem, 4U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    0.0F, 10.0F, std::vector<int>{4}, 0.0F, curr_state, 0, problem, 1U));

  ASSERT_EQ(nodes.size(), 4U);

  auto it = nodes.begin();
  EXPECT_EQ((*it)->id, 1U);
  EXPECT_FLOAT_EQ((*it)->u_max, 10.0F);

  ++it;
  EXPECT_EQ((*it)->id, 2U);
  EXPECT_FLOAT_EQ((*it)->u_max, 10.0F);

  ++it;
  EXPECT_EQ((*it)->id, 4U);
  EXPECT_FLOAT_EQ((*it)->u_max, 7.0F);

  ++it;
  EXPECT_EQ((*it)->id, 3U);
  EXPECT_FLOAT_EQ((*it)->u_max, 5.0F);
}

TEST_F(BranchAndBoundTest, OrderingsAreCorrect_Umin)
{
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  std::multiset<BranchAndBoundSolver::NodePtr, BranchAndBoundSolver::UMinComparator> nodes;

  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    3.0F, 0.0F, std::vector<int>{1}, 0.0F, curr_state, 0, problem, 4U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    1.0F, 0.0F, std::vector<int>{2}, 0.0F, curr_state, 0, problem, 2U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    2.0F, 0.0F, std::vector<int>{3}, 0.0F, curr_state, 0, problem, 3U));
  nodes.insert(std::make_shared<BranchAndBoundSolver::Node>(
    1.0F, 0.0F, std::vector<int>{4}, 0.0F, curr_state, 0, problem, 1U));

  ASSERT_EQ(nodes.size(), 4U);

  auto it = nodes.begin();
  EXPECT_EQ((*it)->id, 1U);
  EXPECT_FLOAT_EQ((*it)->u_min, 1.0F);

  ++it;
  EXPECT_EQ((*it)->id, 2U);
  EXPECT_FLOAT_EQ((*it)->u_min, 1.0F);

  ++it;
  EXPECT_EQ((*it)->id, 3U);
  EXPECT_FLOAT_EQ((*it)->u_min, 2.0F);

  ++it;
  EXPECT_EQ((*it)->id, 4U);
  EXPECT_FLOAT_EQ((*it)->u_min, 3.0F);
}

TEST_F(BranchAndBoundTest, MakeNodeBuildsNodeWithExpectedValues)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const std::vector<int> path{0, 2};
  constexpr int test_state = 2;
  constexpr int test_depth = 3;
  constexpr float test_reward = 4.5F;

  const BranchAndBoundSolver::NodePtr first_node =
    solver.make_node(test_state, test_depth, path, test_reward, problem);
  const BranchAndBoundSolver::NodePtr second_node =
    solver.make_node(1, test_depth + 1, std::vector<int>{0, 2, 1}, test_reward + 1.0F, problem);

  ASSERT_NE(first_node, nullptr);
  ASSERT_NE(second_node, nullptr);

  EXPECT_FLOAT_EQ(first_node->u_min, solver.u_min(test_state, problem, test_reward, test_depth));
  EXPECT_FLOAT_EQ(first_node->u_max, solver.u_max(test_state, problem, test_reward, test_depth));
  EXPECT_EQ(first_node->state, test_state);
  EXPECT_EQ(first_node->depth, test_depth);
  EXPECT_FLOAT_EQ(first_node->reward, test_reward);
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
      1.0F, 2.0F, std::vector<int>{1}, 3.0F, curr_state, 1, problem, 42U);

  solver.add_unexplored_node(node);

  ASSERT_EQ(solver.unexplored_nodes_by_umax_.size(), 1U);
  ASSERT_EQ(solver.unexplored_nodes_by_umin_.size(), 1U);
  ASSERT_EQ(solver.unexplored_node_lookup_.size(), 1U);

  EXPECT_EQ(*solver.unexplored_nodes_by_umax_.begin(), node);
  EXPECT_EQ(*solver.unexplored_nodes_by_umin_.begin(), node);

  const auto lookup_it = solver.unexplored_node_lookup_.find(node->id);
  ASSERT_NE(lookup_it, solver.unexplored_node_lookup_.end());
  EXPECT_EQ(lookup_it->second, node);
}

TEST_F(BranchAndBoundTest, EraseUnexploredNodeRemovesNodeFromAllIndexes)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr node_to_keep =
    std::make_shared<BranchAndBoundSolver::Node>(
      1.0F, 3.0F, std::vector<int>{1}, 0.0F, curr_state, 1, problem, 10U);
  const BranchAndBoundSolver::NodePtr node_to_erase =
    std::make_shared<BranchAndBoundSolver::Node>(
      2.0F, 4.0F, std::vector<int>{2}, 0.0F, curr_state, 1, problem, 11U);

  solver.add_unexplored_node(node_to_keep);
  solver.add_unexplored_node(node_to_erase);
  ASSERT_EQ(solver.unexplored_nodes_by_umax_.size(), 2U);
  ASSERT_EQ(solver.unexplored_nodes_by_umin_.size(), 2U);
  ASSERT_EQ(solver.unexplored_node_lookup_.size(), 2U);

  solver.erase_unexplored_node(node_to_erase);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.size(), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_umin_.size(), 1U);
  EXPECT_EQ(solver.unexplored_node_lookup_.size(), 1U);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.count(node_to_erase), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_umin_.count(node_to_erase), 0U);
  EXPECT_EQ(solver.unexplored_node_lookup_.count(node_to_erase->id), 0U);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.count(node_to_keep), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_umin_.count(node_to_keep), 1U);
  EXPECT_EQ(solver.unexplored_node_lookup_.count(node_to_keep->id), 1U);
}

TEST_F(BranchAndBoundTest, PopNodeWithHighestUMaxReturnsAndRemovesBestNode)
{
  BranchAndBoundSolver solver{
    num_agents, max_depth, max_iterations, discount_factor, debug_mode};
  EyesOnGuysProblem problem{num_agents, relay_speed, distance_between_agents};

  const BranchAndBoundSolver::NodePtr low_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 2.0F, std::vector<int>{1}, 0.0F, curr_state, 1, problem, 30U);
  const BranchAndBoundSolver::NodePtr best_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 5.0F, std::vector<int>{2}, 0.0F, curr_state, 1, problem, 31U);
  const BranchAndBoundSolver::NodePtr mid_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 3.0F, std::vector<int>{3}, 0.0F, curr_state, 1, problem, 32U);

  solver.add_unexplored_node(low_node);
  solver.add_unexplored_node(best_node);
  solver.add_unexplored_node(mid_node);

  const BranchAndBoundSolver::NodePtr popped = solver.pop_node_with_highest_u_max();
  ASSERT_NE(popped, nullptr);
  EXPECT_EQ(popped, best_node);
  EXPECT_FLOAT_EQ(popped->u_max, 5.0F);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.size(), 2U);
  EXPECT_EQ(solver.unexplored_nodes_by_umin_.size(), 2U);
  EXPECT_EQ(solver.unexplored_node_lookup_.size(), 2U);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.count(best_node), 0U);
  EXPECT_EQ(solver.unexplored_nodes_by_umin_.count(best_node), 0U);
  EXPECT_EQ(solver.unexplored_node_lookup_.count(best_node->id), 0U);

  EXPECT_EQ(solver.unexplored_nodes_by_umax_.count(low_node), 1U);
  EXPECT_EQ(solver.unexplored_nodes_by_umax_.count(mid_node), 1U);
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
      0.0F, 0.0F, std::vector<int>{}, 10.0F, curr_state, 1, problem, 40U);
  const BranchAndBoundSolver::NodePtr low_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 0.0F, std::vector<int>{1}, 5.0F, curr_state, 1, problem, 41U);
  const BranchAndBoundSolver::NodePtr high_reward_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 0.0F, std::vector<int>{2, 3}, 8.0F, curr_state, 1, problem, 42U);
  const BranchAndBoundSolver::NodePtr lower_late_node =
    std::make_shared<BranchAndBoundSolver::Node>(
      0.0F, 0.0F, std::vector<int>{0, 1}, 7.0F, curr_state, 1, problem, 43U);

  solver.maybe_update_best_solution(null_node);
  EXPECT_EQ(solver.best_path_.size(), 0U);

  solver.maybe_update_best_solution(empty_path_node);
  EXPECT_EQ(solver.best_path_.size(), 0U);

  solver.maybe_update_best_solution(low_reward_node);
  EXPECT_FLOAT_EQ(solver.best_reward_, 5.0F);
  EXPECT_EQ(solver.best_path_, std::vector<int>({1}));

  solver.maybe_update_best_solution(high_reward_node);
  EXPECT_FLOAT_EQ(solver.best_reward_, 8.0F);
  EXPECT_EQ(solver.best_path_, std::vector<int>({2, 3}));

  solver.maybe_update_best_solution(lower_late_node);
  EXPECT_FLOAT_EQ(solver.best_reward_, 8.0F);
  EXPECT_EQ(solver.best_path_, std::vector<int>({2, 3}));
}

} // namespace eyes_on_guys
