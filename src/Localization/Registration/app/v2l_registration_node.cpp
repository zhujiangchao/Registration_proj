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
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Eigenvalues>

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>

#include "Registration/registration_utils.hpp"

using namespace pcl;
using namespace std;
using namespace RegistrationAlgorithm;

#define NUM_NEIGHBOR    5

typedef pcl::PointNormal PointNormalType;
typedef pcl::PointXYZ PointType;

// global flag
bool g_is_fullmap_received = false;
bool g_is_localmap_received = false;

int g_drone_id;

// full map point clouds
std::vector<float> g_map_size_vec;
std::vector<float> g_cut_fullmap_size_vec;
bool g_is_use_pcd = true;
string g_fullmap_fn;
pcl::PointCloud<PointType>::Ptr g_fullmap_ptr(new pcl::PointCloud<PointType>);
float g_resolution, g_radiusfilter_radius;
int g_radiusfilter_neibors;

// visual map point clouds
pcl::PointCloud<PointType>::Ptr g_localmap_ptr(new pcl::PointCloud<PointType>);

// registration params
string g_reg_pattern;
int g_reg_iter, g_ransac_iter, g_neibor_k;
float g_trans_eps, g_fitness_eps, g_max_cor_dis, g_ndt_resolution, g_ndt_stepsize;
float g_trigger_time;
std::vector<double> g_params_vec;
std::vector<float> g_init_guess_vec;

// evaluate params
float g_ground_height, g_socre_thre;
int g_total_num = 0, g_good_num = 0;

// ros msgs
nav_msgs::Odometry modify_odom, local_odom, old_odom;
sensor_msgs::PointCloud2 modify_pc, old_pc, full_map_msg;

// tranformation
Eigen::Matrix4d g_HT_init_guess = Eigen::Matrix4d::Identity();
Eigen::Matrix4f g_HT = Eigen::Matrix4f::Identity();


bool plane_fit(Eigen::Matrix<float, 3, Eigen::Dynamic> A, Eigen::Vector4f &plane_coeff)
{
    Eigen::Vector3f centroid = A.rowwise().mean();
    Eigen::MatrixXf centered = A.colwise() - centroid;
    Eigen::Matrix3f cov = centered * centered.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
    Eigen::Vector3f normal = eig.eigenvectors().col(0);
    float d = -normal.dot(centroid);
    Eigen::Vector4f plane_eq;
    plane_coeff << normal, d; // std::cout << "Equation of plane1: " << plane_coeff(0) << "x + " << plane_coeff(1) << "y + " << plane_coeff(2) << "z + " << plane_coeff(3) << " = 0" << std::endl;

    // Compute residuals
    Eigen::VectorXf residuals = centered.transpose() * normal;

    // Check for large residuals
    for (int i = 0; i < residuals.size(); i++)
    {
        if (fabs(residuals(i)) > 0.1f)
            return false;
    }
    return true;
}

void evaluate_transformation(PointCloud<PointType>::Ptr src, PointCloud<PointType>::Ptr tar1, \
                             Eigen::Matrix4f transformation, std::vector<double>& ret)
{   
    pcl::PointCloud<PointType>::Ptr tar(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*tar1, *tar);

    pcl::PointCloud<PointType>::Ptr pc_src_world(new pcl::PointCloud<PointType>());
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(tar);
    pcl::transformPointCloud(*src, *pc_src_world, transformation);

    Eigen::Matrix4f transform = transformation;
    Eigen::Vector3f transition = transformation.block<3,1>(0,3);
    Eigen::Quaternionf _q(transformation.block<3,3>(0,0));
    _q = _q.normalized();

    int effct_feat_num = 0;
    std::vector<Eigen::Vector4f> plane_seq;       // norm(a,b,c) + d => ax + by + cz + d =0
    std::vector<Eigen::Vector3f> world_point_seq; // norm(a,b,c) + d => ax + by + cz + d =0
    std::vector<Eigen::Vector3f> plane_point_seq; // norm(a,b,c) + d => ax + by + cz + d =0
    plane_seq.clear();
    world_point_seq.clear();
    plane_point_seq.clear();

    // Matching features
    sensor_msgs::PointCloud2 scan_pc_msg_val;
    pcl::PointCloud<PointType> valid_point;
    valid_point.clear();

    for (int i = 0; i < pc_src_world->size(); i++)
    {
        PointType point_world = pc_src_world->points[i];
        pcl::PointCloud<PointType> pc_vis;
        pc_vis.clear();
        
        std::vector<int> pointIdxNKNSearch(NUM_NEIGHBOR);
        std::vector<float> pointNKNSquaredDistance(NUM_NEIGHBOR);            
        kdtree.nearestKSearch (point_world, NUM_NEIGHBOR, pointIdxNKNSearch, pointNKNSquaredDistance);
        float maxValue = *max_element(pointNKNSquaredDistance.begin(),pointNKNSquaredDistance.end()); 
        
        if (maxValue > g_max_cor_dis)
        {
            continue; // 距离太远，退出
        }

        Eigen::Matrix<float, 3, NUM_NEIGHBOR> A;

        for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j)
        {
            PointType point = tar->at(pointIdxNKNSearch[j]);
            A(0, j) = point.x;
            A(1, j) = point.y;
            A(2, j) = point.z;
            pc_vis.push_back(point);
        }

        Eigen::Vector4f plane_coeff;
        if (!plane_fit(A, plane_coeff))
        {
            continue;
        }
        valid_point.push_back(point_world);
        Eigen::Vector3f centroid = A.rowwise().mean();
        plane_point_seq.push_back(centroid);
        plane_seq.push_back(plane_coeff);
        world_point_seq.push_back(Eigen::Vector3f(point_world.x, point_world.y, point_world.z));
    }

    if (world_point_seq.size() != plane_seq.size())
        cout << "[NEVER reach] seq error continue!" << endl;

    effct_feat_num = world_point_seq.size();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> h_x = Eigen::MatrixXf::Zero(effct_feat_num, 1);    // b
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> J_x = Eigen::MatrixXf::Zero(effct_feat_num, 3);    // A
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> H_x = Eigen::MatrixXf::Zero(6, 6);                 // H
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x_opt1 = Eigen::MatrixXf::Zero(3, effct_feat_num); //x_opt^1
    double total_dist = 0;
    int num_above_ground = 0;
    for (int i = 0; i < effct_feat_num; i++)
    {
        Eigen::Vector3f norm = plane_seq[i].block<3, 1>(0, 0);
        Eigen::Vector3f w_point = world_point_seq[i] - transition;
        Eigen::Vector3f cross_vc = w_point.cross(norm);
        Eigen::Matrix3f H_tmp = Eigen::MatrixXf::Zero(3, 3);

        Eigen::Vector3f L_point = _q.inverse()*(world_point_seq[i] - transition);
        Eigen::Matrix3f L_point_hat;
        L_point_hat << 0,-L_point(2),L_point(1),L_point(2),0,-L_point(0),-L_point(1),L_point(0),0;
        Eigen::Matrix3f R_q(_q.matrix());
        Eigen::Vector3f J_tmp_last3term = L_point_hat*R_q.transpose()*norm;
        // J_x.row(i) << norm(0), norm(1), norm(2), J_tmp_last3term(0), J_tmp_last3term(1), J_tmp_last3term(2);
        J_x.row(i) << norm(0), norm(1), norm(2);
        double dist = norm.transpose()*(world_point_seq[i]-plane_point_seq[i]);
        h_x(i) = dist*dist;
        if (L_point(2) > g_ground_height) {
            total_dist += sqrt(dist*dist);
            num_above_ground ++;
        }
        
    }
    // assessment
    // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> J = J_x;
    Eigen::MatrixXf JTJ = J_x.transpose() * J_x;
    JTJ = JTJ.array() / effct_feat_num;


    Eigen::MatrixXf cov = JTJ.inverse();
    // cout << "Jx\n" << J_x.transpose() << endl;
    cout << "JTJ\n" << JTJ << endl;
    cout << "cov\n" << cov << endl;

    // 进行特征值和特征向量的求解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> esolver_JTJ(JTJ);
    Eigen::Matrix<float, 1, 3> matE;
    Eigen::Matrix<float, 3, 3> matV;
    matE = esolver_JTJ.eigenvalues().real();
    matV = esolver_JTJ.eigenvectors().real();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> esolver_cov(cov);
    Eigen::Matrix<float, 1, 3> matE_cov;
    Eigen::Matrix<float, 3, 3> matV_cov;
    matE_cov = esolver_cov.eigenvalues().real();
    matV_cov = esolver_cov.eigenvectors().real();

    if (num_above_ground > 0) {
      ret.push_back(total_dist/num_above_ground);
    } else {
      ret.push_back(1);
    }
    ret.push_back(matE(0,0));
    ret.push_back(matE(0,1));
    ret.push_back(matE(0,2));
    ret.push_back(num_above_ground);

    // regres_msg.cov.clear();
    // local_odom.pose.covariance.resize(36);
    local_odom.twist.twist.linear.x = ret[0];
    for (int i = 0; i < cov.size(); i++) {
      local_odom.pose.covariance[6*(i/3)+(i%3)] = cov.data()[i]; 
      // regres_msg.cov.push_back(cov.data()[i]);
    }
    std::cout << "Eigenvectors:\n" << matV << std::endl;
    std::cout << "Eigenvalues:\n" << matE << std::endl;      
    
     
}

void full_map_cb(const sensor_msgs::PointCloud2Ptr &msg)
{
  // ROS_WARN("global_map_cb");
  pcl::fromROSMsg(*msg, *g_fullmap_ptr);
  g_is_fullmap_received = true;
}

void local_map_cb(const sensor_msgs::PointCloud2Ptr &msg)
{
  ROS_WARN("local_map_cb");
  ros::Time map_ts = msg->header.stamp;
  pcl::fromROSMsg(*msg, *g_localmap_ptr);
  old_pc = *msg;
  old_pc.header.frame_id = "world";

  radiusFilter<PointType>(g_localmap_ptr, g_radiusfilter_radius, g_radiusfilter_neibors);

  if (g_localmap_ptr->size() > 0 && g_fullmap_ptr->size() > 0) {
    g_total_num++;

    ros::Time t1 = ros::Time::now();

    // cut global map based on current odom
    std::vector<float> cut_range(6, 0.0);
    cut_range[0] = old_odom.pose.pose.position.x + g_cut_fullmap_size_vec[0];
    cut_range[1] = old_odom.pose.pose.position.x + g_cut_fullmap_size_vec[1];
    cut_range[2] = old_odom.pose.pose.position.y + g_cut_fullmap_size_vec[2];
    cut_range[3] = old_odom.pose.pose.position.y + g_cut_fullmap_size_vec[3];
    cut_range[4] = /*old_odom.pose.pose.position.z +*/g_cut_fullmap_size_vec[4];
    cut_range[5] = /*old_odom.pose.pose.position.z +*/g_cut_fullmap_size_vec[5];

    pcl::PointCloud<PointType>::Ptr fullmap_cut_ptr(new pcl::PointCloud<PointType>);
    cut_mapxyz<PointType>(g_fullmap_ptr, fullmap_cut_ptr, cut_range);

    pcl::toROSMsg(*fullmap_cut_ptr, full_map_msg);
    full_map_msg.header.frame_id = "world";

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();  
    std::vector<double> ret; 
    pcl::Correspondences cor;

    transformation = RegFunMap<PointType, PointType>[g_reg_pattern](\
    					g_localmap_ptr, fullmap_cut_ptr, g_HT_init_guess.cast<float>(), g_params_vec, ret, cor);

    ret.clear();
    evaluate_transformation(g_localmap_ptr, fullmap_cut_ptr, transformation, ret);        
    
    pcl::PointCloud<PointType>::Ptr localmap_trans_ptr(new pcl::PointCloud<PointType>);

    pcl::transformPointCloud(*g_localmap_ptr, *localmap_trans_ptr, transformation);
    pcl::toROSMsg(*localmap_trans_ptr, modify_pc);
    modify_pc.header.frame_id = "world";

    local_odom = Matrix4d2odom(transformation.cast<double>());
    local_odom.header.frame_id = "world";
    local_odom.header.stamp = map_ts;

    if (ret[0] < g_socre_thre) {
      g_good_num++;
      g_HT = transformation;
    }

    ros::Time t2 = ros::Time::now();  
    cout << "T(ms): " << (t2-t1).toSec() * 1000 << endl; 
    cout << "Rate: " << 1.0*g_good_num/g_total_num << endl;
  }
}

void visual_odom_cb(const nav_msgs::OdometryPtr &msg) {
  ROS_WARN("visual_odom_cb");
  old_odom = *msg;
  old_odom.header.frame_id = "world";

  Eigen::Matrix4d HT_body_wrt_local = odom2Matrix4d(*msg);
  Eigen::Matrix4d HT_body_wrt_global = g_HT.cast<double>() * HT_body_wrt_local;

  modify_odom = Matrix4d2odom(HT_body_wrt_global);
  modify_odom.header.frame_id = "world";
  modify_odom.header.stamp = old_odom.header.stamp;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "v2l_registration_node");
    ros::NodeHandle nh("~");
    g_fullmap_ptr.reset(new pcl::PointCloud<PointType>);
    g_localmap_ptr.reset(new pcl::PointCloud<PointType>);

    // step 1: load parameters
    nh.getParam("drone_id", g_drone_id);   

	// full map point clouds params
    nh.getParam("map_size", g_map_size_vec);  
    nh.getParam("cut_globalmap_size", g_cut_fullmap_size_vec); 
    nh.getParam("is_use_pcd", g_is_use_pcd);
    nh.getParam("fullmap_fn", g_fullmap_fn);       
    nh.getParam("resolution", g_resolution);    
    nh.getParam("radiusfilter_radius", g_radiusfilter_radius);     
    nh.getParam("radiusfilter_neibors", g_radiusfilter_neibors);

    // registration params
    nh.getParam("reg_pattern", g_reg_pattern);  
    nh.getParam("init_guess", g_init_guess_vec);
    nh.getParam("reg_iter", g_reg_iter);
    nh.getParam("trans_eps", g_trans_eps);
    nh.getParam("fitness_eps", g_fitness_eps);
    nh.getParam("max_cor_dis", g_max_cor_dis);
    nh.getParam("ransac_iter", g_ransac_iter);
    nh.getParam("neibor_k", g_neibor_k);    
    nh.getParam("ndt_resolution", g_ndt_resolution);
    nh.getParam("ndt_stepsize", g_ndt_stepsize);
	
    // evaluate params
    nh.getParam("ground_height", g_ground_height);
    nh.getParam("socre_thre", g_socre_thre);	

    std::cout << "is_use_pcd = " << g_is_use_pcd << std::endl;

    g_init_guess_vec[3] = g_init_guess_vec[3] * M_PI / 180;
    g_init_guess_vec[4] = g_init_guess_vec[4] * M_PI / 180;
    g_init_guess_vec[5] = g_init_guess_vec[5] * M_PI / 180;

    Eigen::Vector3d eular_init_guess;
    eular_init_guess = Eigen::Vector3d(g_init_guess_vec[3], g_init_guess_vec[4], g_init_guess_vec[5]);
    Eigen::Quaterniond q_init_guess = euler2quaternion(eular_init_guess);

    g_HT_init_guess.block<3,1>(0,3) = Eigen::Vector3d(g_init_guess_vec[0], g_init_guess_vec[1], g_init_guess_vec[2]);
    g_HT_init_guess.block<3,3>(0,0) = q_init_guess.toRotationMatrix();

    g_params_vec.push_back(g_reg_iter);
    g_params_vec.push_back(g_trans_eps);
    g_params_vec.push_back(g_fitness_eps);
    g_params_vec.push_back(g_max_cor_dis);
    g_params_vec.push_back(g_ransac_iter);

    g_params_vec.push_back(g_neibor_k); 

    g_params_vec.push_back(g_ndt_resolution);
    g_params_vec.push_back(g_ndt_stepsize);     

    // step 2: register subscriber and publisher
    ros::Subscriber full_map_sub, visual_odom_sub, visual_map_sub;
    ros::Publisher full_map_pub, modify_odom_pub, modify_localmap_pub, local_ref_pub, old_odom_pub, old_local_map_pub;

    full_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/test_full_map", 1);
    local_ref_pub = nh.advertise<nav_msgs::Odometry>("/local_ref",1);

    modify_odom_pub = nh.advertise<nav_msgs::Odometry>("Odometry_modify",1);
    modify_localmap_pub = nh.advertise<sensor_msgs::PointCloud2>("Map_modify",1);
    
    visual_map_sub = nh.subscribe("Map_src", 1, local_map_cb, ros::TransportHints().tcpNoDelay());
    visual_odom_sub = nh.subscribe("Odometry_src", 1, visual_odom_cb, ros::TransportHints().tcpNoDelay());

    if (!g_is_use_pcd) {
      full_map_sub = nh.subscribe("full_map", 1, full_map_cb, ros::TransportHints().tcpNoDelay());      
    } else {
      std::cout << "load pcd file\n";
      if (pcl::io::loadPCDFile<PointType>(g_fullmap_fn, *g_fullmap_ptr) == -1)
      {
          ROS_ERROR("full map pcd load fail!!!");
          return 0;
      }

      pcl::toROSMsg(*g_fullmap_ptr, full_map_msg);
      full_map_msg.header.frame_id = "world";

      //down sampling
      pcl_downSampling<PointType>(g_fullmap_ptr, g_resolution);

      // radius filter
      radiusFilter<PointType>(g_fullmap_ptr, g_radiusfilter_radius, g_radiusfilter_neibors);
      // calculate norm before
      // estimateNorm<PointType>(g_fullmap_ptr, g_full_map_norm_ptr, g_neibor_k);
      
      std::cout << "size = " << g_fullmap_ptr->size() << std::endl;

      g_is_fullmap_received = true;
    }

    ros::Rate rate(10);

    while(ros::ok()) {
      full_map_pub.publish(full_map_msg);

      local_ref_pub.publish(local_odom);

      modify_odom_pub.publish(modify_odom);
      modify_localmap_pub.publish(modify_pc);

      rate.sleep();
      ros::spinOnce();
    }
    return 0;
}