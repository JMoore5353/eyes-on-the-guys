#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "rviz_publisher.hpp"

namespace eyes_on_guys
{

RvizPublisher::RvizPublisher()
    : Node("rviz_guys_publisher")
    , qos_transient_local_20_(20)
{
  declare_parameters();

  qos_transient_local_20_.transient_local();
  guy_pub_ =
    this->create_publisher<visualization_msgs::msg::Marker>("rviz/guys", qos_transient_local_20_);
  guy_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "guy_poses", 10, std::bind(&RvizPublisher::guy_pose_callback, this, std::placeholders::_1));
}

void RvizPublisher::declare_parameters()
{
  this->declare_parameter("guy_model_file", "resource/maeserstatue_small.stl");
}

void RvizPublisher::guy_pose_callback(const geometry_msgs::msg::PoseStamped & msg)
{
  if (!guy_ids_to_idxs_.contains(msg.header.frame_id)) {
    guy_ids_to_idxs_.insert({msg.header.frame_id, guy_marker_vector_.size()});
    add_new_marker_to_guy_vector(msg.header.frame_id);
  }

  std::size_t idx = guy_ids_to_idxs_.at(msg.header.frame_id);
  guy_marker_vector_[idx].pose = msg.pose;
  guy_marker_vector_[idx].header.stamp = this->get_clock()->now();
  guy_pub_->publish(guy_marker_vector_[idx]);
}

void RvizPublisher::add_new_marker_to_guy_vector(const std::string & name)
{
  visualization_msgs::msg::Marker new_marker;
  new_marker.header.frame_id = "NED";
  new_marker.ns = name;
  new_marker.id = 0;
  new_marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
  new_marker.action = visualization_msgs::msg::Marker::ADD;
  new_marker.scale.x = 0.05;
  new_marker.scale.y = 0.05;
  new_marker.scale.z = 0.05;
  new_marker.color.r = 0.0f;
  new_marker.color.g = 0.46f;
  new_marker.color.b = 0.93f;
  new_marker.color.a = 1.0f;
  new_marker.lifetime.sec = 0;
  new_marker.lifetime.nanosec = 0;
  new_marker.mesh_resource =
    "package://eyes_on_the_guys/" + this->get_parameter("guy_model_file").as_string();

  guy_marker_vector_.push_back(new_marker);
}

} // namespace eyes_on_guys
