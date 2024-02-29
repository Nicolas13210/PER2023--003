cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 11)
# For debugging, make sure this is set
# set(CMAKE_BUILD_TYPE Debug)

project(mushr_gp)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2
)

find_package(SBPL REQUIRED)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${SBPL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(planner_node src/planner.cpp)

target_link_libraries(planner_node ${catkin_LIBRARIES} ${SBPL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
