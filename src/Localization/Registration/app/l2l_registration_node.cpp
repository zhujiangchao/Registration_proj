#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <string>

#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>

#include "Registration/registration_utils.hpp"

using namespace std;
using namespace RegistrationAlgorithm;

typedef pcl::PointXYZ PointType;

// global flag
bool g_is_other_map_received = false;
bool g_is_my_map_received = false;

// other map point_clouds
bool g_is_use_pcd = true;
string g_other_map_fn;
std::vector<float> g_map_size_vec;
pcl::PointCloud<PointType>::Ptr g_other_map_ptr;
float g_resolution, g_radiusfilter_radius;
int g_radiusfilter_neibors;

// my map point clouds
pcl::PointCloud<PointType>::Ptr g_my_map_ptr;

// registration params
string g_reg_pattern;
int g_reg_iter, g_ransac_iter, g_neibor_k, g_drone_id;
float g_trans_eps, g_fitness_eps, g_max_cor_dis, g_ndt_resolution, g_ndt_stepsize;
float g_trigger_time;
std::vector<double> g_params_vec;
std::vector<float> g_init_guess_vec;

// ros msgs
nav_msgs::Odometry modify_odom, local_odom, old_odom;
sensor_msgs::PointCloud2 modify_pc, old_pc;
sensor_msgs::PointCloud2 other_map_msg;

// Subscriber and Publisher
ros::Subscriber other_map_sub, my_map_sub, lidar_odom_sub, lidar_map_sub;
ros::Publisher other_map_pub, modify_odom_pub, modify_my_map_pub,\
              local_odom_pub,old_odom_pub,old_my_map_pub, reg_trigger_pub;

// tranformation
Eigen::Matrix4f g_HT_init_guess = Eigen::Matrix4f::Identity();
Eigen::Matrix4f g_HT = Eigen::Matrix4f::Identity();              

void other_map_cb(const sensor_msgs::PointCloud2Ptr &msg)
{
  ROS_WARN("[l2l_registration_node] other_map_cb");
  msg->header.frame_id = "world";
  pcl::fromROSMsg(*msg, *g_other_map_ptr);
  other_map_msg = *msg;

  if (g_my_map_ptr->size() > 0 && g_other_map_ptr->size() > 0) {
    std::vector<double> ret;        
    pcl::Correspondences cor;  
	g_HT = RegFunMap<PointType, PointType>[g_reg_pattern](\
					g_my_map_ptr, g_other_map_ptr, g_HT_init_guess.cast<float>(), g_params_vec, ret, cor);            

    local_odom = Matrix4d2odom(g_HT.cast<double>());
    local_odom.header.frame_id = "world";    
    std::cout << "rel_p = " << g_HT.block<3,1>(0,3).transpose() << std::endl;
  }
}


void my_map_cb(const sensor_msgs::PointCloud2Ptr &msg)
{
  ROS_WARN("[l2l_registration_node] my_map_cb");

  msg->header.frame_id = "world";

  pcl::fromROSMsg(*msg, *g_my_map_ptr);

  if (g_my_map_ptr->size() > 0 && g_other_map_ptr->size() > 0) {
    std::vector<double> ret; 
    pcl::Correspondences cor;  
	  g_HT = RegFunMap<PointType, PointType>[g_reg_pattern](\
					g_my_map_ptr, g_other_map_ptr, g_HT_init_guess.cast<float>(), g_params_vec, ret, cor);            

    local_odom = Matrix4d2odom(g_HT.cast<double>());
    local_odom.header.frame_id = "world";

    std::cout << "rel_p = " << g_HT.block<3,1>(0,3).transpose() << std::endl;
  }
}

// void lidar_odom_cb(const nav_msgs::OdometryPtr &msg) {
//   old_odom = *msg;
//   old_odom.header.frame_id = "world";

//   Eigen::Matrix4d HT_body_wrt_local = odom2Matrix4d(*msg);
//   Eigen::Matrix4d HT_body_wrt_global = g_HT.cast<double>() * HT_body_wrt_local;

//   modify_odom = Matrix4d2odom(HT_body_wrt_global);
//   modify_odom.header.frame_id = "world";

//   modify_odom_pub.publish(modify_odom);

// }

// void lidar_pc_cb(const sensor_msgs::PointCloud2Ptr &msg) {
//   old_pc = *msg;
//   old_pc.header.frame_id = "world";

//   pcl::PointCloud<PointType>::Ptr lidar_pc_local(new pcl::PointCloud<PointType>);
//   pcl::PointCloud<PointType>::Ptr lidar_pc_global(new pcl::PointCloud<PointType>);  
//   pcl::fromROSMsg(*msg, *lidar_pc_local);
//   pcl::transformPointCloud(*lidar_pc_local, *lidar_pc_global, g_HT);
//   pcl::toROSMsg(*lidar_pc_global, modify_pc);
//   modify_pc.header.frame_id = "world";

//   modify_my_map_pub.publish(modify_pc);
// }



int main(int argc, char** argv)
{
    ros::init(argc, argv, "l2l_registration_node");
    ros::NodeHandle nh("~");
    g_other_map_ptr.reset(new pcl::PointCloud<PointType>);
    g_my_map_ptr.reset(new pcl::PointCloud<PointType>);

    // step 1: load parameters 
    nh.getParam("drone_id", g_drone_id);    
    nh.getParam("init_guess", g_init_guess_vec);
    nh.getParam("map_size", g_map_size_vec);  
    nh.getParam("is_use_pcd", g_is_use_pcd);
    nh.getParam("other_map_fn", g_other_map_fn);    
    nh.getParam("resolution", g_resolution);    
    nh.getParam("radiusfilter_neibors", g_radiusfilter_neibors);
    nh.getParam("radiusfilter_radius", g_radiusfilter_radius); 

    nh.getParam("trigger_time", g_trigger_time);

    nh.getParam("reg_pattern", g_reg_pattern);  
    nh.getParam("reg_iter", g_reg_iter);
    nh.getParam("trans_eps", g_trans_eps);
    nh.getParam("fitness_eps", g_fitness_eps);
    nh.getParam("max_cor_dis", g_max_cor_dis);
    nh.getParam("ransac_iter", g_ransac_iter);
    nh.getParam("neibor_k", g_neibor_k);    
    nh.getParam("ndt_resolution", g_ndt_resolution);
    nh.getParam("ndt_stepsize", g_ndt_stepsize);

    g_init_guess_vec[3] = g_init_guess_vec[3] * M_PI / 180;
    g_init_guess_vec[4] = g_init_guess_vec[4] * M_PI / 180;
    g_init_guess_vec[5] = g_init_guess_vec[5] * M_PI / 180;

    std::cout << "is_use_pcd = " << g_is_use_pcd << std::endl;
    std::cout << "drone_id = " << g_drone_id << std::endl;

    Eigen::Vector3d eular_init_guess;
    eular_init_guess = Eigen::Vector3d(g_init_guess_vec[3], g_init_guess_vec[4], g_init_guess_vec[5]);
    Eigen::Quaternionf q_init_guess = euler2quaternion(eular_init_guess).cast<float>();

    g_HT_init_guess.block<3,1>(0,3) = Eigen::Vector3f(g_init_guess_vec[0], g_init_guess_vec[1], g_init_guess_vec[2]);
    g_HT_init_guess.block<3,3>(0,0) = q_init_guess.toRotationMatrix();

    g_HT = g_HT_init_guess;

	g_params_vec.push_back(g_reg_iter);
	g_params_vec.push_back(g_trans_eps);
	g_params_vec.push_back(g_fitness_eps);
	g_params_vec.push_back(g_max_cor_dis);
	g_params_vec.push_back(g_ransac_iter);

	g_params_vec.push_back(g_neibor_k);	

	g_params_vec.push_back(g_ndt_resolution);
	g_params_vec.push_back(g_ndt_stepsize);	

    // step 2: register subscriber and publisher
    my_map_sub = nh.subscribe("my_map", 1, my_map_cb, ros::TransportHints().tcpNoDelay());
    if (!g_is_use_pcd) {
      other_map_sub = nh.subscribe("/other_map", 1, other_map_cb, ros::TransportHints().tcpNoDelay());      
    } else {
      if (pcl::io::loadPCDFile<PointType>(g_other_map_fn, *g_other_map_ptr) == -1)
      {
          ROS_ERROR("other map pcd load fail!!!");
          return 0;
      }
      pcl_downSampling<PointType>(g_other_map_ptr, g_resolution);

      cut_mapxyz<PointType>(g_other_map_ptr, g_map_size_vec);
            
      pcl::toROSMsg(*g_other_map_ptr, other_map_msg);
      other_map_msg.header.frame_id = "world";
      g_is_other_map_received = true;
    }
    reg_trigger_pub = nh.advertise<std_msgs::Empty>("/trigger_to_drones", 1);


    // lidar_odom_sub = nh.subscribe("Odometry_old", 10, lidar_odom_cb, ros::TransportHints().tcpNoDelay());
    // lidar_map_sub = nh.subscribe("cloud_old", 10, lidar_pc_cb, ros::TransportHints().tcpNoDelay());
    // modify_odom_pub = nh.advertise<nav_msgs::Odometry>("Odometry_new",1);
    // modify_my_map_pub = nh.advertise<sensor_msgs::PointCloud2>("cloud_new",1);

    local_odom_pub = nh.advertise<nav_msgs::Odometry>("/local_ref",1);
    ros::Rate rate(10);

    ros::Time start_time = ros::Time::now();
    bool trigger_flag = false;
    
    while(ros::ok()) {

      if ((ros::Time::now() - start_time).toSec() > g_trigger_time && !trigger_flag) {
        std_msgs::Empty msg;
        reg_trigger_pub.publish(msg);
        ROS_WARN("[l2l_registration_node]trigger!!!");
        trigger_flag = true;
      }
      rate.sleep();
      ros::spinOnce();
    }
    return 0;
}