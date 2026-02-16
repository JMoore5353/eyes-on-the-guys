#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "planner.hpp"

namespace eyes_on_guys
{

TEST(compute_distance, ExpectDistanceBetweenGuysIsCorrect)
{
  std::string guy1 = "Brandon";
  std::string guy2 = "Ian";
  std::string guy3 = "Jacob";
  std::vector<std::string> guy_names{guy1, guy2, guy3};

  geometry_msgs::msg::PoseStamped pose1;
  geometry_msgs::msg::PoseStamped pose2;
  geometry_msgs::msg::PoseStamped pose3;

  pose1.pose.position.x = 0.0;
  pose1.pose.position.y = 0.0;
  pose1.pose.position.z = 0.0;

  pose2.pose.position.x = 10.0;
  pose2.pose.position.y = 0.0;
  pose2.pose.position.z = 0.0;

  pose3.pose.position.x = 20.0;
  pose3.pose.position.y = 0.0;
  pose3.pose.position.z = 0.0;

  std::map<std::string, geometry_msgs::msg::PoseStamped> guy_poses{{guy1, pose1}, {guy2, pose2}, {guy3, pose3}};

  Eigen::MatrixXd dists = compute_distance_between_guys(guy_names, guy_poses);

  Eigen::Matrix3d expected_dists;
  expected_dists << 0.0, 10.0, 20.0, 10.0, 0.0, 10.0, 20.0, 10.0, 0.0;

  EXPECT_EQ(dists, expected_dists);
}

} // namespace eyes_on_guys
