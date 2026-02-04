#include <Eigen/Core>
#include <gtest/gtest.h>
#include <memory>

#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

TEST(find_greedy_action, WhenFindingGreedyAction_ExpectTheNodesGreedyAction)
{
  auto node = std::make_shared<eyes_on_guys::Node>(0, 10, 1.0);
  int node_greedy_action = node->explore_best_action();

  int greedy_action = find_greedy_action(node);

  EXPECT_EQ(greedy_action, node_greedy_action);
}

// TODO: Continue testing the non-member functions!
