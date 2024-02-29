# Global Planner
[![build-test](https://github.com/prl-mushr/mushr_gp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/prl-mushr/mushr_gp/actions)

This ROS package hosts the global planner for the MuSHR system. It wraps a search-based planner from SBPL. Search based methods have two parts - first, the planning problem and environment must be represented by a graph, and then the graph must be searched from a starting point to a goal. This MuSHR global planner builds upon the [SBPL repo](https://github.com/sbpl/sbpl) by Search-Based Planning Lab at Carnegie Mellon University. MuSHR specifically makes use of the [ARA* planning algorithm](https://en.wikipedia.org/wiki/Anytime_A*) and plans on an implicit graph in a SE2 (x, y, θ) state space.

## Tutorial
The following [tutorial](https://mushr.io/tutorials/navigation/) goes through installing/running the car.

## Installing on the car
Make sure you install [pysbpl](https://github.com/schmittlema/pysbpl) into your `~/catkin_ws/src`. Note that pysbpl depends on [SBPL](https://github.com/sbpl/sbpl). Please make sure you follow the instructions in the pysbpl repo to install all necessary dependencies first.

Clone [this](https://github.com/prl-mushr/mushr_gp) repo into `~/catkin_ws/src`

Make sure that you have the tf dependency installed: `sudo apt install ros-melodic-tf`

You should be good to go!

## `mushr_gh` ROS API

#### Publishers
Topic | Type | Description
------|------|------------
`/path_topic`|[geometry_msgs/Path](http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/Path.html)|The trajectory computed by the planner.

#### Subscribers
Topic | Type | Description
------|------|------------
`/map`|[nav_msgs/OccupancyGrid](http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html)|Uses the provided occupancy grid as the graph for planning.
`/goal_topic`|[geometry_msgs/PoseStamped](http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html)|Goal to compute path to.
`/start_topic`|[geometry_msgs/PoseStamped](http://docs.ros.org/api/geometry_msgs/html/msg/PoseStamped.html)|Starting location of the path being computed.
