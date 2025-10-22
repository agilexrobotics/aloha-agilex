cmake_minimum_required(VERSION 2.8.3)
project(aloha)
find_package(OpenCV REQUIRED)
## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  # interbotix_xs_msgs
  # interbotix_xsarm_control
  rosbag
  roscpp
  tf2_ros
  rviz
  sensor_msgs
  std_msgs
  std_srvs
  pcl_conversions
  tf
  geometry_msgs
  cv_bridge
  kdl_parser
)
find_package(PCL REQUIRED QUIET)
find_package(orocos_kdl REQUIRED)
###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
catkin_package(
 CATKIN_DEPENDS
 rosbag
 roscpp
 tf2_ros
 rviz
 sensor_msgs
 std_msgs
 std_srvs
 geometry_msgs
 kdl_parser
)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_LIBRARY_DIRS}
)
add_executable(${PROJECT_NAME}_endPoseForwardSolution src/endPoseForwardSolution.cpp)
add_dependencies(${PROJECT_NAME}_endPoseForwardSolution ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_endPoseForwardSolution ${catkin_LIBRARIES} ${PCL_LIBRARIES} orocos-kdl)
add_executable(${PROJECT_NAME}_testing_kdl src/testing_kdl.cpp)
add_dependencies(${PROJECT_NAME}_testing_kdl ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_testing_kdl ${catkin_LIBRARIES} ${PCL_LIBRARIES} orocos-kdl)

#add_executable(${PROJECT_NAME}_jointRadInverseSolution src/jointRadInverseSolution.cpp)
#add_dependencies(${PROJECT_NAME}_jointRadInverseSolution ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_jointRadInverseSolution ${catkin_LIBRARIES} ${PCL_LIBRARIES})
