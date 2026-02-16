#ifndef BRANCH_AND_BOUND_SOLVER_HPP
#define BRANCH_AND_BOUND_SOLVER_HPP

#include <cstddef>
#include <gtest/gtest_prod.h>
#include <memory>
#include <set>
#include <unordered_map>
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
                                float discount_factor = 1.0F, bool debug_mode = false);

  // Executes branch-and-bound from initial_state and returns the best next action.
  // Returns -1 when inputs/configuration are invalid or no valid action is found.
  int solve(int initial_state, const EyesOnGuysProblem & problem_info);

private:
  // Give unit tests access to Node struc
  FRIEND_TEST(BranchAndBoundTest, OrderingsAreCorrect_Umax);
  FRIEND_TEST(BranchAndBoundTest, OrderingsAreCorrect_Umin);
  FRIEND_TEST(BranchAndBoundTest, MakeNodeBuildsNodeWithExpectedValues);
  FRIEND_TEST(BranchAndBoundTest, AddUnexploredNodeAddsNodeToAllIndexes);
  FRIEND_TEST(BranchAndBoundTest, EraseUnexploredNodeRemovesNodeFromAllIndexes);
  FRIEND_TEST(BranchAndBoundTest, PruneNodesWithUMinBelowRemovesOnlyThresholdMatches);
  FRIEND_TEST(BranchAndBoundTest, PopNodeWithHighestUMaxReturnsAndRemovesBestNode);
  FRIEND_TEST(BranchAndBoundTest, ProblemDimensionsAreValidChecksKeyShapeMismatchCases);
  FRIEND_TEST(BranchAndBoundTest, MaybeUpdateBestSolutionTracksHighestRewardNonEmptyPath);

  // Unified node structure used for both algorithm values and queue bookkeeping.
  struct Node
  {
    // u_min: pessimistic lower bound on total achievable reward from this node.
    // u_max: optimistic upper bound on total achievable reward from this node.
    // path: sequence of actions taken from root to reach this node.
    // reward: accumulated discounted reward along path.
    // state/depth/problem/id: search bookkeeping for expansion and indexing.
    Node(float u_min, float u_max, std::vector<int> path, float reward, int state, int depth,
         const EyesOnGuysProblem & problem, std::size_t id)
        : u_min(u_min)
        , u_max(u_max)
        , path(std::move(path))
        , reward(reward)
        , state(state)
        , depth(depth)
        , problem(problem)
        , id(id)
    {}

    float u_min;
    float u_max;
    std::vector<int> path;
    float reward;
    int state;
    int depth;
    EyesOnGuysProblem problem;
    std::size_t id;
  };

  using NodePtr = std::shared_ptr<Node>;

  // Orders unexplored nodes by descending u_max so begin() is best candidate.
  struct UMaxComparator
  {
    bool operator()(const NodePtr & lhs, const NodePtr & rhs) const;
  };

  // Orders unexplored nodes by ascending u_min to support threshold pruning.
  struct UMinComparator
  {
    bool operator()(const NodePtr & lhs, const NodePtr & rhs) const;
  };

  // Bound evaluators used when creating/expanding nodes.
  float u_min(int curr_state, const EyesOnGuysProblem & problem_state, float path_reward,
              int depth) const;
  float u_max(int curr_state, const EyesOnGuysProblem & problem_state, float path_reward,
              int depth) const;

  // Creates a fully initialized node with computed bounds and unique id.
  NodePtr make_node(int curr_state, int depth, const std::vector<int> & path, float reward,
                    const EyesOnGuysProblem & problem_state);

  // Inserts/removes node in all unexplored-node indices.
  void add_unexplored_node(const NodePtr & node);
  void erase_unexplored_node(const NodePtr & node);

  // Prunes all unexplored nodes whose u_min is below threshold.
  std::size_t prune_nodes_with_u_min_below(float threshold);

  // Pops the unexplored node with largest u_max.
  NodePtr pop_node_with_highest_u_max();

  // Validates that problem matrices/vectors match configured num_states_.
  bool problem_dimensions_are_valid(const EyesOnGuysProblem & problem_info) const;

  // Updates incumbent solution when node reward beats current best.
  void maybe_update_best_solution(const NodePtr & node);

  // Optional per-iteration debug printout.
  void maybe_print_debug_info(int iteration, std::size_t pruned_this_iteration) const;

  // Fixed search configuration.
  int num_states_;
  int max_depth_;
  int max_iterations_;
  float discount_factor_;
  bool debug_mode_;

  // Dual indices for efficient best-first pop and threshold pruning.
  std::multiset<NodePtr, UMaxComparator> unexplored_nodes_by_umax_;
  std::multiset<NodePtr, UMinComparator> unexplored_nodes_by_umin_;
  std::unordered_map<std::size_t, NodePtr> unexplored_node_lookup_;

  // Runtime search counters.
  std::size_t next_node_id_;
  std::size_t explored_nodes_count_;
  std::size_t completed_paths_count_;
  std::size_t total_pruned_nodes_;

  // Incumbent best solution summary.
  float best_reward_;
  std::vector<int> best_path_;
};

} // namespace eyes_on_guys

#endif
