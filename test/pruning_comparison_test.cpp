#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "branch_and_bound_solver.hpp"
#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

// Test fixture for pruning comparison tests
class PruningComparisonTest : public ::testing::Test
{
protected:
  // Helper function to create a problem with given parameters
  EyesOnGuysProblem create_problem(int num_agents, const Eigen::MatrixXd & distances,
                                   const Eigen::MatrixXd & shared_info)
  {
    EyesOnGuysProblem problem{num_agents, 10.0, distances};
    problem.shared_info_matrix = shared_info;
    problem.time_since_last_relay_contact_with_agent = Eigen::VectorXd::Constant(num_agents, 50.0);
    return problem;
  }

  // Helper function to compare results from pruning enabled vs disabled
  void compare_pruning_results(int num_agents, int depth, const EyesOnGuysProblem & problem,
                               int initial_state)
  {
    const int max_iterations = 10000000;
    const double discount_factor = 0.9;

    // Solve with pruning enabled
    BranchAndBoundSolver solver_with_pruning{num_agents, depth, max_iterations, discount_factor,
                                             false, true};
    std::vector<int> result_with_pruning = solver_with_pruning.solve(initial_state, problem);

    // Solve with pruning disabled
    BranchAndBoundSolver solver_without_pruning{num_agents, depth, max_iterations, discount_factor,
                                                false, false};
    std::vector<int> result_without_pruning = solver_without_pruning.solve(initial_state, problem);

    // Both should find valid paths
    ASSERT_FALSE(result_with_pruning.empty()) << "Pruning enabled should find a valid path";
    ASSERT_FALSE(result_without_pruning.empty()) << "Pruning disabled should find a valid path";

    // Both should have the same length
    EXPECT_EQ(result_with_pruning.size(), result_without_pruning.size())
      << "Path lengths should match";

    // Both should start at the same initial state
    EXPECT_EQ(result_with_pruning[0], initial_state);
    EXPECT_EQ(result_without_pruning[0], initial_state);

    // Both should find reasonable rewards (pruning may find suboptimal due to heuristic bounds)
    // The reward with pruning should be at least 90% of the reward without pruning
    EXPECT_GT(solver_with_pruning.best_reward_, 
              0.9 * solver_without_pruning.best_reward_)
      << "Pruning should find a reasonably good solution (at least 90% of optimal)";
    
    // Note: We don't require identical paths because pruning uses heuristic upper bounds
    // that may cause it to prune away optimal paths in favor of exploring other paths first

    // Pruning should explore fewer nodes
    EXPECT_LE(solver_with_pruning.explored_nodes_count_,
              solver_without_pruning.explored_nodes_count_)
      << "Pruning should explore fewer or equal nodes";

    // Pruning should have some pruned nodes (unless problem is trivial)
    if (depth > 1 && num_agents > 2) {
      EXPECT_GT(solver_with_pruning.total_pruned_nodes_, 0U)
        << "Pruning should prune some nodes for non-trivial problems";
    }

    // Print detailed comparison
    std::cout << "\n=== Pruning Comparison (agents=" << num_agents << ", depth=" << depth 
              << ", start=" << initial_state << ") ===\n";
    
    // Node exploration comparison
    std::cout << "Node Exploration:\n";
    std::cout << "  With pruning:    " << solver_with_pruning.explored_nodes_count_
              << " nodes explored, " << solver_with_pruning.total_pruned_nodes_ << " pruned\n";
    std::cout << "  Without pruning: " << solver_without_pruning.explored_nodes_count_
              << " nodes explored\n";
    std::cout << "  Efficiency gain: "
              << (1.0 - static_cast<double>(solver_with_pruning.explored_nodes_count_) /
                        solver_without_pruning.explored_nodes_count_) * 100.0
              << "%\n";
    
    // Reward comparison
    std::cout << "Rewards:\n";
    std::cout << "  With pruning:    " << solver_with_pruning.best_reward_ << "\n";
    std::cout << "  Without pruning: " << solver_without_pruning.best_reward_ << "\n";
    double reward_ratio = (solver_with_pruning.best_reward_ / solver_without_pruning.best_reward_) * 100.0;
    std::cout << "  Pruning achieves: " << reward_ratio << "% of unpruned reward\n";
    
    // Path comparison
    std::cout << "Paths:\n";
    std::cout << "  With pruning:    ";
    for (size_t i = 0; i < result_with_pruning.size(); ++i) {
      std::cout << result_with_pruning[i];
      if (i < result_with_pruning.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n";
    std::cout << "  Without pruning: ";
    for (size_t i = 0; i < result_without_pruning.size(); ++i) {
      std::cout << result_without_pruning[i];
      if (i < result_without_pruning.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n";
  }
};

// Test 1: Small problem with 3 agents, depth 3
TEST_F(PruningComparisonTest, SmallProblem3Agents)
{
  const int num_agents = 3;
  const int depth = 3;

  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 10, 20,
               10, 0, 15,
               20, 15, 0;

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 5, 3,
                 0, 0, 8,
                 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 2: Medium problem with 4 agents, depth 4
TEST_F(PruningComparisonTest, MediumProblem4Agents)
{
  const int num_agents = 4;
  const int depth = 4;

  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 15, 25, 40,
               15, 0, 20, 30,
               25, 20, 0, 18,
               40, 30, 18, 0;

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 10, 5, 2,
                 0, 0, 12, 8,
                 0, 0, 0, 15,
                 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 3: Asymmetric problem with 4 agents, depth 3
TEST_F(PruningComparisonTest, AsymmetricProblem4Agents)
{
  const int num_agents = 4;
  const int depth = 3;

  // Asymmetric distance matrix
  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 10, 50, 100,
               10, 0, 20, 80,
               50, 20, 0, 30,
               100, 80, 30, 0;

  // Asymmetric information gain
  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 20, 5, 1,
                 0, 0, 15, 3,
                 0, 0, 0, 25,
                 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 4: Larger problem with 5 agents, depth 4
TEST_F(PruningComparisonTest, LargerProblem5Agents)
{
  const int num_agents = 5;
  const int depth = 4;

  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 12, 24, 36, 48,
               12, 0, 18, 30, 42,
               24, 18, 0, 22, 34,
               36, 30, 22, 0, 26,
               48, 42, 34, 26, 0;

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 8, 6, 4, 2,
                 0, 0, 10, 7, 5,
                 0, 0, 0, 12, 9,
                 0, 0, 0, 0, 14,
                 0, 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 5: Different starting state with 4 agents
TEST_F(PruningComparisonTest, DifferentStartingState)
{
  const int num_agents = 4;
  const int depth = 3;

  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 20, 35, 50,
               20, 0, 25, 40,
               35, 25, 0, 30,
               50, 40, 30, 0;

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 12, 8, 4,
                 0, 0, 15, 10,
                 0, 0, 0, 18,
                 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  
  // Test from different starting states
  compare_pruning_results(num_agents, depth, problem, 1);
  compare_pruning_results(num_agents, depth, problem, 2);
}

// Test 6: Uniform distances with 4 agents
TEST_F(PruningComparisonTest, UniformDistances)
{
  const int num_agents = 4;
  const int depth = 3;

  // All distances are equal
  Eigen::MatrixXd distances = Eigen::MatrixXd::Constant(num_agents, num_agents, 10.0);
  distances.diagonal().setZero();

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 10, 8, 6,
                 0, 0, 12, 9,
                 0, 0, 0, 15,
                 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 7: Sparse information gain with 5 agents
TEST_F(PruningComparisonTest, SparseInformationGain)
{
  const int num_agents = 5;
  const int depth = 3;

  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 15, 30, 45, 60,
               15, 0, 20, 35, 50,
               30, 20, 0, 25, 40,
               45, 35, 25, 0, 30,
               60, 50, 40, 30, 0;

  // Very sparse information gain (only a few non-zero entries)
  Eigen::MatrixXd shared_info = Eigen::MatrixXd::Zero(num_agents, num_agents);
  shared_info(0, 1) = 20.0;
  shared_info(1, 3) = 25.0;
  shared_info(3, 4) = 30.0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

// Test 8: Chain topology with 5 agents, depth 4
TEST_F(PruningComparisonTest, ChainTopology)
{
  const int num_agents = 5;
  const int depth = 4;

  // Chain-like distance matrix (neighbors are close, distant agents are far)
  Eigen::MatrixXd distances(num_agents, num_agents);
  distances << 0, 10, 100, 200, 300,
               10, 0, 10, 100, 200,
               100, 10, 0, 10, 100,
               200, 100, 10, 0, 10,
               300, 200, 100, 10, 0;

  Eigen::MatrixXd shared_info(num_agents, num_agents);
  shared_info << 0, 15, 10, 5, 2,
                 0, 0, 15, 10, 5,
                 0, 0, 0, 15, 10,
                 0, 0, 0, 0, 15,
                 0, 0, 0, 0, 0;

  EyesOnGuysProblem problem = create_problem(num_agents, distances, shared_info);
  compare_pruning_results(num_agents, depth, problem, 0);
}

} // namespace eyes_on_guys
