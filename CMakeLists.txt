cmake_minimum_required(VERSION 3.8)
project(eyes-on-the-guys)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosplane_msgs REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Matplot++ QUIET)
if(NOT Matplot++_FOUND)
  message(STATUS "Matplot++ not found, downloading via FetchContent")
  include(FetchContent)
  FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master)
  
  FetchContent_GetProperties(matplotplusplus)
  if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()


#####################
##   EXECUTABLES   ##
#####################
add_executable(rviz_publisher
  src/rviz_publisher_node.cpp
  src/rviz_publisher.cpp)
target_include_directories(rviz_publisher
  PRIVATE
    include/eyes-on-the-guys)
ament_target_dependencies(rviz_publisher rclcpp geometry_msgs visualization_msgs)
install(
  TARGETS rviz_publisher
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(guys
  src/guys_node.cpp
  src/guys.cpp)
target_include_directories(guys
  PRIVATE
    include/eyes-on-the-guys)
ament_target_dependencies(guys rclcpp geometry_msgs)
install(
  TARGETS guys
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(planner
  src/planner_node.cpp
  src/planner.cpp)
target_include_directories(planner
  PRIVATE
    include/eyes-on-the-guys
    ${EIGEN3_INCLUDE_DIRS})
if(Matplot++_FOUND)
  target_link_libraries(planner Matplot++::matplot)
else()
  target_link_libraries(planner matplot)
endif()
ament_target_dependencies(planner rclcpp geometry_msgs rosplane_msgs Eigen3)
install(
  TARGETS planner
  DESTINATION lib/${PROJECT_NAME}
)

########################
##   PYTHON SCRIPTS   ##
########################
install(
  PROGRAMS
    src/example.py
  DESTINATION lib/${PROJECT_NAME}
)

###################
##   RESOURCES   ##
###################
install(
  DIRECTORY resource launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
