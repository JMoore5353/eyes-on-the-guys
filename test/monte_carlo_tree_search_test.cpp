#include <Eigen/Core>
#include <gtest/gtest.h>
#include <memory>

#include "eyes_on_guys_problem.hpp"
#include "monte_carlo_node.hpp"
#include "monte_carlo_tree_search.hpp"

TEST(find_greedy_action, WhenFindingGreedyAction_ExpectTheNodesGreedyAction)
{
  auto node = std::make_shared<eyes_on_guys::MTCSNode>(0, 10, 1.0);
  int node_greedy_action = node->explore_best_action();

  int greedy_action = find_greedy_action(node);

  EXPECT_EQ(greedy_action, node_greedy_action);
}

TEST(transition_from_state, WhenTransitioning_ExpectTransitionIsDeterministic)
{
  auto node = std::make_shared<eyes_on_guys::MTCSNode>(0, 10, 1.0);
  int action{2};

  for (int i = 0; i < 1000; ++i) {
    std::shared_ptr<eyes_on_guys::MTCSNode> new_node =
      eyes_on_guys::transition_from_state(node, action);
    EXPECT_EQ(new_node->get_id(), action);
  }
}
