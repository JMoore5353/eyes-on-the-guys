#ifndef BRANCH_AND_BOUND_SOLVER_HPP
#define BRANCH_AND_BOUND_SOLVER_HPP

#include <cstddef>
#include <gtest/gtest_prod.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "eyes_on_guys_problem.hpp"

// Multiplier for encouraging deep search rather than wide search
// 
const double TUNNELING_WEIGHT_PARAM = 0.5;

namespace eyes_on_guys
{

// Branch-and-bound solver for selecting the next relay target/action.
class BranchAndBoundSolver
{
public:
  explicit BranchAndBoundSolver(int num_states, int max_depth, int max_iterations,
                                double discount_factor = 1.0, bool debug_mode = false,
                                bool enable_pruning = true);

  // Executes branch-and-bound from initial_state and returns the best path found.
  // Returns an empty vector when inputs/configuration are invalid or no valid path is found.
  std::vector<int> solve(int initial_state, const EyesOnGuysProblem & problem_info);

private:
  // Give unit tests access to private members
  FRIEND_TEST(BranchAndBoundTest, OrderingsAreCorrect_Qmax);
  FRIEND_TEST(BranchAndBoundTest, MakeNodeBuildsNodeWithExpectedValues);
  FRIEND_TEST(BranchAndBoundTest, AddUnexploredNodeAddsNodeToAllIndexes);
  FRIEND_TEST(BranchAndBoundTest, EraseUnexploredNodeRemovesNodeFromAllIndexes);
  FRIEND_TEST(BranchAndBoundTest, PruneNodesWithQMaxBelowRemovesOnlyThresholdMatches);
  FRIEND_TEST(BranchAndBoundTest, PopNodeWithHighestQMaxReturnsAndRemovesBestNode);
  FRIEND_TEST(BranchAndBoundTest, ProblemDimensionsAreValidChecksKeyShapeMismatchCases);
  FRIEND_TEST(BranchAndBoundTest, MaybeUpdateBestSolutionTracksHighestRewardNonEmptyPath);
  FRIEND_TEST(BranchAndBoundTest, SolveFindsOptimalPathInBasicScenario);
  FRIEND_TEST(BranchAndBoundTest, SolveResetsBetweenCalls);
  FRIEND_TEST(BranchAndBoundTest, SolveReturnsEmptyForInvalidDimensions);
  FRIEND_TEST(BranchAndBoundTest, SolveRespectsMaxIterationsLimit);
  FRIEND_TEST(BranchAndBoundTest, SolveWithDepthOne);
  FRIEND_TEST(BranchAndBoundTest, SolveAppliesDiscountFactor);
  FRIEND_TEST(BranchAndBoundTest, SolveMatchesBruteForceForSmallProblem);
  FRIEND_TEST(BranchAndBoundTest, PerformanceTestLargeProblem);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathReturnsEmptyWhenAtMaxDepth);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathPicksFarthestNeighbor);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathAvoidsVisitedNodes);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathResetsWhenAllNodesVisited);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathLengthMatchesRemainingDepth);
  FRIEND_TEST(BranchAndBoundTest, AntiGreedyPathWithSingleAgent);
  FRIEND_TEST(BranchAndBoundTest, QMaxIsAtLeastPathReward);
  FRIEND_TEST(BranchAndBoundTest, QMaxWithoutPenaltiesIsGreaterOrEqualToFullReward);
  FRIEND_TEST(BranchAndBoundTest, QMaxAtMaxDepthEqualsPathReward);
  FRIEND_TEST(BranchAndBoundTest, QMaxIsConsistentWithAntiGreedyPath);
  FRIEND_TEST(BranchAndBoundTest, UMinEqualsPathReward);
  FRIEND_TEST(BranchAndBoundTest, WeightingParameterAffectsNodeOrdering);
  FRIEND_TEST(BranchAndBoundTest, WeightingParameterZeroWeightPreservesQMaxOrdering);
  FRIEND_TEST(BranchAndBoundTest, WeightingParameterDeepNodesPreferredWithPositiveWeight);
  FRIEND_TEST(BranchAndBoundTest, WeightingParameterAffectsSolverPathSelection);
  friend class PruningComparisonTest;

  // Unified node structure used for both algorithm values and queue bookkeeping.
  struct Node
  {
    // q_max: optimistic upper bound on total achievable reward from this node.
    // weighted_q_max: q_max weighted by depth to encourage deeper search.
    // path: sequence of actions taken from root to reach this node (last element is current state).
    // reward: accumulated discounted reward along path.
    // depth/problem/id: search bookkeeping for expansion and indexing.
    Node(double q_max, double weighted_q_max, std::vector<int> path, double reward,
         int depth, EyesOnGuysProblem problem, std::size_t id)
        : q_max(q_max)
        , weighted_q_max(weighted_q_max)
        , path(std::move(path))
        , reward(reward)
        , depth(depth)
        , problem(std::move(problem))
        , id(id)
    {}

    double q_max;
    double weighted_q_max;
    std::vector<int> path;
    double reward;
    int depth;
    EyesOnGuysProblem problem;
    std::size_t id;
  };

  using NodePtr = std::shared_ptr<Node>;

  // Orders unexplored nodes by descending q_max so begin() is best candidate.
  struct QMaxComparator
  {
    bool operator()(const NodePtr & lhs, const NodePtr & rhs) const;
  };

  // Bound evaluator used when creating/expanding nodes.
  double q_max(int state, std::vector<int> path,
               const EyesOnGuysProblem & problem_state, double path_reward, int depth) const;
  
  // Helper functions for computing bounds
  // Determines the longest path from the current node till search depth
  std::vector<int> anti_greedy_path(int curr_state, std::vector<int> current_path,
                                     const EyesOnGuysProblem & problem_state, int depth) const;

  // Creates a fully initialized node with computed bounds and unique id.
  // Current state is inferred from path.back().
  NodePtr make_node(int depth, std::vector<int> path, double reward,
                    EyesOnGuysProblem problem_state);

  // Inserts/removes node in the unexplored-node q_max index.
  void add_unexplored_node(const NodePtr & node);
  void erase_unexplored_node(const NodePtr & node);

  // Prunes all unexplored nodes whose weighted_q_max is below the best reward found so far.
  std::size_t prune_nodes();

  // Pops the unexplored node with largest q_max.
  NodePtr pop_node_with_highest_q_max();

  // Validates that problem matrices/vectors match configured num_states_.
  bool problem_dimensions_are_valid(const EyesOnGuysProblem & problem_info) const;

  // Updates incumbent solution when node reward beats current best.
  void maybe_update_best_solution(const NodePtr & node);

  // Optional per-iteration debug printout.
  void maybe_print_debug_info() const;

  // Resets all mutable search state so the solver can be reused across calls.
  void reset();

  // Fixed search configuration.
  int num_states_;
  int max_depth_;
  int max_iterations_;
  double discount_factor_;
  bool debug_mode_;
  bool enable_pruning_;

  // Unexplored nodes indexed by descending q_max for pop/prune operations.
  std::multiset<NodePtr, QMaxComparator> unexplored_nodes_by_q_max_;

  // Runtime search counters.
  std::size_t next_node_id_;
  std::size_t explored_nodes_count_;
  std::size_t completed_paths_count_;
  std::size_t total_pruned_nodes_;

  // Incumbent best solution summary.
  double best_reward_;
  std::vector<int> best_path_;
};

} // namespace eyes_on_guys

#endif
