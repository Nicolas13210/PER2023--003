#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <sbpl/headers.h>
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/MapMetaData.h"
#include "nav_msgs/Path.h"
#include <tf2/impl/utils.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>

using cv::Mat;
using geometry_msgs::PoseStamped;
using geometry_msgs::TransformStamped;
using geometry_msgs::Vector3;
using nav_msgs::MapMetaData;
using ros::NodeHandle;
using ros::Publisher;
using ros::Subscriber;
using ros::shutdown;
using std::fstream;
using std::ios;
using std::string;
using std::unique_ptr;
using std::vector;
using tf2::Quaternion;
using YAML::Node;

class PlannerNode
{
    private:
    Publisher path_pub_;
    Subscriber start_sub_;
    Subscriber goal_sub_;
    Subscriber map_sub_;

    unique_ptr<PoseStamped> car_pose_;
    unique_ptr<PoseStamped> goal_pose_;
    unique_ptr<EnvironmentNAVXYTHETALAT> env_;
    unique_ptr<SBPLPlanner> planner_;
    unique_ptr<vector<int>> path_;
    unique_ptr<MapMetaData> map_meta_data_;

    int start_id_;
    int goal_id_;
    uint pose_counter_;

    vector<sbpl_2Dpt_t> perimeter_;  // rectangular shape of the car, in meters
    string mprim_path_;              // path to the .mprim file
    double car_vel_;                 // velocity of the car, meters per second
    double time_45_;                 // meaningless for ackermann vehicles, time to turn 45 degrees in place in seconds
    unsigned char obs_thresh_;       // obstacle threashold for the map
    double replan_time_;             // how many seconds to run the planner for when recieving new data. See search_mode below
    double initial_eps_;             // ratio of how close the first valid solution should be to the optimal solution
    bool search_mode_;               // if true, search until a solution is found. If false, try to find and improve a solution for replan_time_ seconds
    bool forward_search_;            // if the planner should search forwards or backwards
    bool update_costmap_;            // if true, the costmap will be updated dynamically, if false, only at the beginning
    bool regular_replan_;            // if true, the replan upon recieving regular_replan_freq_ car_pose_ updates.
    int regular_replan_freq_;        // how many car_pose_ updates to wait for before replanning, if regular_replan_ is true

    float motion_res_;               
    uint thetas_;

    public:
    PlannerNode(NodeHandle nh)
    {
        string car_pos;
        nh.getParam("global_planner/start_topic", car_pos);
        string goal_pos;
        nh.getParam("global_planner/goal_topic", goal_pos);
        string map;
        nh.getParam("global_planner/map", map);
        string path;
        nh.getParam("global_planner/path_topic", path);
        
        path_pub_ = nh.advertise<nav_msgs::Path>(path.c_str(), 1);
        start_sub_ = nh.subscribe(car_pos.c_str(), 1000, &PlannerNode::start_cb, this);
        goal_sub_ = nh.subscribe(goal_pos.c_str(), 1000, &PlannerNode::goal_cb, this);
        map_sub_ = nh.subscribe(map.c_str(), 1000, &PlannerNode::map_cb, this);

        car_pose_ = nullptr;
        goal_pose_ = nullptr;
        env_ = nullptr;
        planner_ = nullptr;
        path_ = unique_ptr<vector<int>>(new vector<int>);
        map_meta_data_ = nullptr;
        
        nh.getParam("/car/global_planner/mprim_path", mprim_path_);

        fstream mprim_file;
        mprim_file.open(mprim_path_, ios::in);
        if (mprim_file.is_open())
        {
            string f_line;
            getline(mprim_file, f_line);
            motion_res_ = stof(f_line.substr(f_line.find(" ") + 1));
            getline(mprim_file, f_line);
            thetas_ = stoul(f_line.substr(f_line.find(" ") + 1));
            mprim_file.close();
        }
        else
        {
            ROS_ERROR("UNABLE TO OPEN MOTION PRIMITIVES FILE FOR NODE INITIALIZATION");
            shutdown();
        }

        string config_path;
        nh.getParam("/car/global_planner/config_path", config_path);
        Node config_file = YAML::LoadFile(config_path);
        if (!(config_file["vel"] && config_file["time_45_deg"] && config_file["obs_thresh"] && config_file["perimeter"]))
        {
            ROS_ERROR("config.yaml file needs to specif parameters 'vel' (velocity), 'time_45_deg'" \
                      "(time for 45 degree turn), 'obs_thresh' (map occupied threshhold), and 'perimeter' (robot shape)");
            shutdown();
        }   
        car_vel_ = config_file["vel"].as<double>();
        time_45_ = config_file["time_45_deg"].as<double>();
        obs_thresh_ = config_file["obs_thresh"].as<unsigned char>();
        replan_time_ = config_file["replan_time"].as<double>();
        initial_eps_ = config_file["initial_eps"].as<double>();
        forward_search_ = config_file["forward_search"].as<bool>();
        update_costmap_ = config_file["update_costmap"].as<bool>();
        search_mode_ = config_file["search_mode"].as<bool>();
        regular_replan_ = config_file["regular_replan"].as<bool>();
        regular_replan_freq_ = config_file["regular_replan_freq"].as<int>();
        perimeter_.reserve(config_file["perimeter"].size());
        for (size_t i = 0; i < config_file["perimeter"].size(); i++) {
            sbpl_2Dpt_t pt;
            pt.x = config_file["perimeter"][i][0].as<double>();
            pt.y = config_file["perimeter"][i][1].as<double>();
            perimeter_.push_back(pt);
        }

        start_id_ = -1;
        goal_id_ = -1;
        pose_counter_ = 0;

        ROS_INFO("Planner ROS node constructed. Need to initialize state");
    }

    void start_cb(const geometry_msgs::PoseStamped& msg)
    {
        car_pose_.reset(new PoseStamped(msg));
        if (car_pose_ == nullptr || (regular_replan_ && pose_counter_ % regular_replan_freq_ == 0))
        {
            processState(true);
        }
        pose_counter_++;
    }

    void goal_cb(const geometry_msgs::PoseStamped& msg)
    {
        // receive a goal and plan to it, publish path
        ROS_INFO("Received goal message");
        goal_pose_.reset(new PoseStamped(msg));
        processState(false);
    }

    // convert map into sbpl format and save it
    // pass into planner when recieved, refuse to plan until map received
    void map_cb(const nav_msgs::OccupancyGrid& msg)
    {
        if (env_ != nullptr && !update_costmap_)
            return;
        map_meta_data_.reset(new MapMetaData(msg.info));
        Mat resized;
        resizeOccupancyGrid(msg, resized);
        if (env_ == nullptr)
            initializeEnv(msg, resized);
        // Iterate over indices for traversing sbpl environment
        for (int env_iy = 0; env_iy < resized.rows; ++env_iy)
        {
            for (int env_ix = 0; env_ix < resized.cols; ++env_ix)
            {
                uchar val = resized.data[env_iy * resized.cols + env_ix];
                val = val != 0 ? 255 : 0;
                env_->UpdateCost(env_ix, env_iy, val);
            }
        }
        ROS_INFO("Updated costmap");
    }

    // tries to generate a state_id if the environment is defined.
    // will also check if the environment, planner, and both the car_pose
    // and goal are defined, and do any of the necessary initialization.
    void processState(const bool is_start)
    {
        if (env_ == nullptr)
        {
            if (is_start)
                ROS_INFO("Recieved car_pose, but environment isn't loaded");
            else
                ROS_INFO("Recieved goal_pose, but environment isn't loaded");
            return;
        }
        TransformStamped translation;
        TransformStamped rotation;
        Vector3 vec;
        vec.x = -map_meta_data_->origin.position.x;
        vec.y = -map_meta_data_->origin.position.y;
        vec.z = -map_meta_data_->origin.position.z;
        rotation.transform.rotation = map_meta_data_->origin.orientation;
        rotation.transform.rotation.w *= -1;
        translation.transform.translation = vec;
        translation.transform.rotation.w = 1;
        PoseStamped actual;
        Quaternion quatty;
        if (is_start)
        {
            tf2::doTransform(*car_pose_, actual, translation);
            tf2::doTransform(actual, actual, rotation);
            tf2::convert(actual.pose.orientation, quatty);
            start_id_ = env_->SetStart(actual.pose.position.x, actual.pose.position.y, tf2::impl::getYaw(quatty));
            if (goal_pose_ != nullptr)
            {
                tf2::doTransform(*goal_pose_, actual, translation);
                tf2::doTransform(actual, actual, rotation);
                tf2::convert(actual.pose.orientation, quatty);
                goal_id_ = env_->SetGoal(actual.pose.position.x, actual.pose.position.y, tf2::impl::getYaw(quatty));
            }
        }
        else
        {
            tf2::doTransform(*goal_pose_, actual, translation);
            tf2::doTransform(actual, actual, rotation);
            tf2::convert(actual.pose.orientation, quatty);
            goal_id_ = env_->SetGoal(actual.pose.position.x, actual.pose.position.y, tf2::impl::getYaw(quatty));
            if (car_pose_ != nullptr)
            {
                tf2::doTransform(*car_pose_, actual, translation);
                tf2::doTransform(actual, actual, rotation);
                tf2::convert(actual.pose.orientation, quatty);
                start_id_ = env_->SetStart(actual.pose.position.x, actual.pose.position.y, tf2::impl::getYaw(quatty));
            }
        }
        if (planner_ != nullptr)
            runPlanner();
        else if (car_pose_ != nullptr && goal_pose_ != nullptr)        
            initializePlanner();
    }

    void initializePlanner()
    {
        if (planner_ != nullptr)
        {
            ROS_ERROR("should not be reinitializing the planner, should instead be replaning");
            return;
        }
        planner_.reset(new ARAPlanner(env_.get(), forward_search_));
        planner_->set_search_mode(search_mode_);
        planner_->set_initialsolution_eps(initial_eps_);
        ROS_INFO("Planner initialized");
        runPlanner();
    }

    void runPlanner()
    {
        if (start_id_ < 0 || planner_->set_start(start_id_) == 0)
        {
            ROS_ERROR("Invalid start state");
            return;
        }
        if (goal_id_ < 0 || planner_->set_goal(goal_id_) == 0)
        {
            ROS_ERROR("Invalid goal state");
            return;
        }
        if (!planner_->replan(replan_time_, path_.get()))
        {
            ROS_INFO("Solution does not exist!");
            path_->clear();
        } else
        {
            vector<sbpl_xy_theta_pt_t> converted_path;
            env_->ConvertStateIDPathintoXYThetaPath(path_.get(), &converted_path);
            nav_msgs::Path pubPath = nav_msgs::Path();
            pubPath.header.frame_id = "map";
            pubPath.poses.reserve(path_->size());
            for (sbpl_xy_theta_pt_t pt : converted_path)
            {
                // pose initially represents the output from SBPL directly
                PoseStamped pose;
                pose.pose.position.x = pt.x;
                pose.pose.position.y = pt.y;
                Quaternion quat;
                quat.setEuler(pt.theta, 0, 0);
                pose.pose.orientation.w = quat.getW();
                pose.pose.orientation.x = quat.getX();
                pose.pose.orientation.y = quat.getY();
                pose.pose.orientation.z = quat.getZ();

                // create the transforms we need
                TransformStamped rotation;
                rotation.transform.rotation = map_meta_data_->origin.orientation;

                TransformStamped translation;
                Vector3 vec;
                vec.x = map_meta_data_->origin.position.x;
                vec.y = map_meta_data_->origin.position.y;
                vec.z = map_meta_data_->origin.position.z;
                translation.transform.translation = vec;
                translation.transform.rotation.w = 1;

                // finally, do the sequence of transforms and append them to the path
                tf2::doTransform(pose, pose, rotation);
                tf2::doTransform(pose, pose, translation);
                pose.header.frame_id = "map";
                pubPath.poses.push_back(pose);
            }
            ROS_INFO("Solution found! Publishing...");
            path_pub_.publish(pubPath);
        }
    }

    // Initializes the environment
    void initializeEnv(const nav_msgs::OccupancyGrid& msg, Mat& resized)
    {
        if (env_ != nullptr)
        {
            ROS_ERROR("Attempting to initialize the enviroment multiple times.");
            return;
        }
        EnvNAVXYTHETALAT_InitParms params = {thetas_};
        env_.reset(new EnvironmentNAVXYTHETALAT);
        env_->InitializeEnv(resized.cols, resized.rows, perimeter_, motion_res_, car_vel_, time_45_, obs_thresh_,
                            mprim_path_.c_str(), params);
        ROS_INFO("Environment initialized");
    }

    void resizeOccupancyGrid(const nav_msgs::OccupancyGrid& grid, Mat &resized)
    {
        ROS_INFO("Resizing the map");
        float ratio = grid.info.resolution / motion_res_;
        Mat src = Mat(grid.data).reshape(0, grid.info.height);
        src.convertTo(src, CV_8UC1);
        if (ratio != 1.0)
        {
            
            resize(src, resized, cv::Size(), ratio, ratio, cv::INTER_AREA);
        }
        else
        {
            resized = src;
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planner");
    NodeHandle n;

    PlannerNode node = PlannerNode(n);
    ros::spin();
}
