#ifndef RVIZ_ROSPLANE_WAYPOINT_ONLY_PUBLISHER_HPP
#define RVIZ_ROSPLANE_WAYPOINT_ONLY_PUBLISHER_HPP

#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "rosplane_msgs/msg/waypoint.hpp"

namespace rosplane_gcs
{

class RvizWaypointOnlyPublisher : public rclcpp::Node
{
public:
  RvizWaypointOnlyPublisher();

private:
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr rviz_wp_pub_;
  rclcpp::Subscription<rosplane_msgs::msg::Waypoint>::SharedPtr waypoint_sub_;

  void declare_parameters();

  void new_wp_callback(const rosplane_msgs::msg::Waypoint & wp);
  void publish_markers_to_clear_waypoints();
  visualization_msgs::msg::Marker create_new_waypoint_marker(const rosplane_msgs::msg::Waypoint& wp);
  void update_waypoint_line_list(const rosplane_msgs::msg::Waypoint& wp);
  visualization_msgs::msg::Marker create_new_waypoint_text_marker(const rosplane_msgs::msg::Waypoint& wp);

  // Persistent rviz markers
  visualization_msgs::msg::Marker line_list_;
  std::vector<geometry_msgs::msg::Point> line_points_;

  int num_wps_;
};

} // namespace rosplane_gcs

#endif

