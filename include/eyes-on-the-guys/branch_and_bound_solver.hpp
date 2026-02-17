#ifndef BRANCH_AND_BOUND_SOLVER_HPP
#define BRANCH_AND_BOUND_SOLVER_HPP

#include <cstddef>
#include <gtest/gtest_prod.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "eyes_on_guys_problem.hpp"

namespace eyes_on_guys
{

// Branch-and-bound solver for selecting the next relay target/action.
class BranchAndBoundSolver
{
public:
  explicit BranchAndBoundSolver(int num_states, int max_depth, int max_iterations,
                                double discount_factor = 1.0, bool debug_mode = false);

  // Executes branch-and-bound from initial_state and returns the best path found.
  // Returns an empty vector when inputs/configuration are invalid or no valid path is found.
  std::vector<int> solve(int initial_state, const EyesOnGuysProblem & problem_info);

private:
  // Give unit tests access to Node struct
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

  // Unified node structure used for both algorithm values and queue bookkeeping.
  struct Node
  {
    // u_min: pessimistic lower bound on total achievable reward from this node.
    // q_max: optimistic upper bound on total achievable reward from this node.
    // path: sequence of actions taken from root to reach this node.
    // reward: accumulated discounted reward along path.
    // state/depth/problem/id: search bookkeeping for expansion and indexing.
    Node(double u_min, double q_max, std::vector<int> path, double reward, int state, int depth,
         EyesOnGuysProblem problem, std::size_t id)
        : u_min(u_min)
        , q_max(q_max)
        , path(std::move(path))
        , reward(reward)
        , state(state)
        , depth(depth)
        , problem(std::move(problem))
        , id(id)
    {}

    double u_min;
    double q_max;
    std::vector<int> path;
    double reward;
    int state;
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

  // Bound evaluators used when creating/expanding nodes.
  double u_min(int curr_state, const EyesOnGuysProblem & problem_state, double path_reward,
              int depth) const;
  double q_max(int curr_state, const EyesOnGuysProblem & problem_state, double path_reward,
              int depth) const;

  // Creates a fully initialized node with computed bounds and unique id.
  NodePtr make_node(int curr_state, int depth, std::vector<int> path, double reward,
                    EyesOnGuysProblem problem_state);

  // Inserts/removes node in the unexplored-node q_max index.
  void add_unexplored_node(const NodePtr & node);
  void erase_unexplored_node(const NodePtr & node);

  // Prunes all unexplored nodes whose q_max is below u_min_threshold.
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
  double u_min_threshold_;
};

} // namespace eyes_on_guys

#endif
