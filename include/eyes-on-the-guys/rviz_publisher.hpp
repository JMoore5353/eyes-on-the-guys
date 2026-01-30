#ifndef RVIZ_PUBLISHER_HPP
#define RVIZ_PUBLISHER_HPP

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace eyes_on_guys
{

class RvizPublisher : public rclcpp::Node
{
public:
  RvizPublisher();

private:
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr guy_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr guy_pose_sub_;
  rclcpp::QoS qos_transient_local_20_;

  std::map<std::string, std::size_t> guy_ids_to_idxs_;
  std::vector<visualization_msgs::msg::Marker> guy_marker_vector_;

  void declare_parameters();
  void guy_pose_callback(const geometry_msgs::msg::PoseStamped & msg);
  void add_new_marker_to_guy_vector(const std::string & name);
};

} // namespace eyes_on_guys

#endif
