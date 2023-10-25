#pragma once
#include <Eigen/src/Core/Matrix.h>
#include <nav_msgs/Odometry.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

Eigen::Quaterniond euler2quaternion(Eigen::Vector3d euler)
{
  double cr = cos(euler(0)/2);
  double sr = sin(euler(0)/2);
  double cp = cos(euler(1)/2);
  double sp = sin(euler(1)/2);
  double cy = cos(euler(2)/2);
  double sy = sin(euler(2)/2);
  Eigen::Quaterniond q;
  q.w() = cr*cp*cy + sr*sp*sy;
  q.x() = sr*cp*cy - cr*sp*sy;
  q.y() = cr*sp*cy + sr*cp*sy;
  q.z() = cr*cp*sy - sr*sp*cy;
  return q; 
}

Eigen::Matrix4d odom2Matrix4d(const nav_msgs::Odometry &odom) {
	Eigen::Matrix4d HT = Eigen::Matrix4d::Identity();
	Eigen::Vector3d p(odom.pose.pose.position.x,\
	                odom.pose.pose.position.y,\
	                odom.pose.pose.position.z);

	Eigen::Quaterniond q(odom.pose.pose.orientation.w,\
	                   odom.pose.pose.orientation.x,\
	                   odom.pose.pose.orientation.y,\
	                   odom.pose.pose.orientation.z);
	HT.block<3, 1>(0, 3) = p;
	HT.block<3, 3>(0, 0) = q.toRotationMatrix();
	return HT;
}

nav_msgs::Odometry Matrix4d2odom(const Eigen::Matrix4d &HT) {
	nav_msgs::Odometry odom;
	Eigen::Vector3d p = HT.block<3, 1>(0, 3);
	Eigen::Quaterniond q(HT.block<3, 3>(0, 0));
	odom.pose.pose.position.x = p[0];
	odom.pose.pose.position.y = p[1];
	odom.pose.pose.position.z = p[2];
	odom.pose.pose.orientation.w = q.w();
	odom.pose.pose.orientation.x = q.x();
	odom.pose.pose.orientation.y = q.y();
	odom.pose.pose.orientation.z = q.z();

	return odom;
}

template<typename PointT>
void pcl_downSampling(typename pcl::PointCloud<PointT>::Ptr &cloud_ds, float resolution) 
{
  typename pcl::PointCloud<PointT>::Ptr cloud_tmp(new pcl::PointCloud<PointT>);

  std::cout << "before downsampling: " << cloud_ds->size() << std::endl;

  // 创建VoxelGrid滤波器对象
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(cloud_ds);  // 设置输入点云
  sor.setLeafSize(resolution, resolution, resolution);  // 设置体素的大小（降采样的分辨率）

  // 执行降采样滤波
  sor.filter(*cloud_tmp);

  *cloud_ds = *cloud_tmp;
  //
  std::cout << "after downsampling: " << cloud_ds->size() << std::endl;
}

template<typename PointT, typename PointNormalT>
void estimateNorm(typename pcl::PointCloud<PointT>::Ptr pc,
                  typename pcl::PointCloud<PointNormalT>::Ptr pc_norm,
                  int num_neigh=5)
{
  pcl::copyPointCloud(*pc, *pc_norm);
  pcl::NormalEstimation<PointNormalT, PointNormalT> norm_est;
  norm_est.setSearchMethod(pcl::search::KdTree<PointNormalT>::Ptr(new pcl::search::KdTree<PointNormalT>));
  norm_est.setKSearch(num_neigh);
  norm_est.setInputCloud(pc_norm);
  norm_est.compute(*pc_norm);
}

template<typename PointT>
void radiusFilter(typename pcl::PointCloud<PointT>::Ptr pc, float radius=0.15, int num_neigh=5)
{
  typename pcl::PointCloud<PointT>::Ptr cloud_filtered_inliers(new pcl::PointCloud<PointT>);  
  pcl::RadiusOutlierRemoval<PointT> sor;
  sor.setInputCloud(pc); 
  sor.setRadiusSearch(radius); 
  sor.setMinNeighborsInRadius(num_neigh);
  sor.filter(*cloud_filtered_inliers);      
  *pc = *cloud_filtered_inliers;
}

template<typename PointT>
void cut_mapxyz(typename pcl::PointCloud<PointT>::Ptr &cloud, std::vector<float> size) 
{
  typename pcl::ConditionAnd<PointT>::Ptr range_cond(new pcl::ConditionAnd<PointT>);//实例化条件指针

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::GT,size[0])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::LT,size[1])));

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::GT,size[2])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::LT,size[3])));

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::GT,size[4])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::LT,size[5])));

  //build the filter
  pcl::ConditionalRemoval<PointT> condrem;
  condrem.setCondition(range_cond);
  condrem.setInputCloud(cloud);
  condrem.setKeepOrganized(false);//保存原有点云结结构就是点的数目没有减少，采用nan代替了
  condrem.filter(*cloud);
}

template<typename PointT>
void cut_mapxyz(typename pcl::PointCloud<PointT>::Ptr &in_pc, \
                typename pcl::PointCloud<PointT>::Ptr &out_pc,
                std::vector<float> size) 
{
  typename pcl::ConditionAnd<PointT>::Ptr range_cond(new pcl::ConditionAnd<PointT>);//实例化条件指针

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::GT,size[0])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::LT,size[1])));

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::GT,size[2])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::LT,size[3])));

  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::GT,size[4])));
  range_cond->addComparison(typename pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::LT,size[5])));

  //build the filter
  pcl::ConditionalRemoval<PointT> condrem;
  condrem.setCondition(range_cond);
  condrem.setInputCloud(in_pc);
  condrem.setKeepOrganized(false);//保存原有点云结结构就是点的数目没有减少，采用nan代替了
  condrem.filter(*out_pc);
}