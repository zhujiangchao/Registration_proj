#pragma once
#include <iostream>

#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/warp_point_rigid_3d.h>

// #include <fast_gicp/gicp/fast_gicp.hpp>
// #include <fast_gicp/gicp/fast_gicp_st.hpp>
// #include <fast_gicp/gicp/fast_vgicp.hpp>

#include <Eigen/src/Core/Matrix.h>

#include "utils.hpp"

typedef pcl::PointXYZ Point_T;
typedef pcl::PointNormal PointWithNoram_T;

template<typename PointSource, typename PointTarget>
using RegFunPtr = Eigen::Matrix4f (*)(	typename pcl::PointCloud<PointSource>::Ptr, \
										typename pcl::PointCloud<PointTarget>::Ptr, \
										Eigen::Matrix4f, std::vector<double>, \
										std::vector<double>&, pcl::Correspondences&);

namespace RegistrationAlgorithm {

template<typename PointSource, typename PointNormal>
void addNormal(typename pcl::PointCloud<PointSource>::Ptr pc,
			   typename pcl::PointCloud<PointNormal>::Ptr pc_with_normals,
			   int num)
{
	typename pcl::PointCloud<PointNormal>::Ptr normals(new pcl::PointCloud<PointNormal>);

	typename pcl::search::KdTree<PointSource>::Ptr searchTree(new pcl::search::KdTree<PointSource>);
	searchTree->setInputCloud(pc);

	pcl::NormalEstimation<PointSource, PointNormal> normalEstimator;
	normalEstimator.setInputCloud(pc);
	normalEstimator.setSearchMethod(searchTree);
	normalEstimator.setKSearch(num);
	normalEstimator.compute(*normals);

	pcl::concatenateFields(*pc, *normals, *pc_with_normals);
}

/**
 * @brief 	point-to-point icp algorithm.
 * 			The XYZ coordinates are used for registration.
 * 
 * @param src_pc: source point clouds
 * @oaram tar_pc: target point clouds
 * 
 * @param guess: init guess of icp
 * @param params: icp parameters
 * 			params[0]: 	max iteration numbers
 * 			params[1]: 	transformation eps
 * 			params[2]: 	fitness eps
 * 			params[3]: 	max correspondence distance 
 * 			params[4]:	ransac iteration numbers 
 * 
 * @param cor: correspondences between source and target
 * @param ret: result
 * 			ret[0]: convergence, 0 or 1
 * 			ret[1]: score
 * 
 * @return final transformation
 */
template<typename PointSource, typename PointTarget>
Eigen::Matrix4f PointToPointICP(typename pcl::PointCloud<PointSource>::Ptr src_pc,
								typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
								Eigen::Matrix4f guess,
								std::vector<double> params,
								std::vector<double> &ret,
								pcl::Correspondences &cor)
{
	// std::cout << "begin PointToPointICP\n";

	// set input
	pcl::IterativeClosestPoint<PointSource, PointTarget> icp;

	typename pcl::search::KdTree<PointSource>::Ptr tree1(new pcl::search::KdTree<PointSource>);
	tree1->setInputCloud(src_pc);
	typename pcl::search::KdTree<PointTarget>::Ptr tree2(new pcl::search::KdTree<PointTarget>);
	tree2->setInputCloud(tar_pc);

	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);

	icp.setInputSource(src_pc);
	icp.setInputTarget(tar_pc);

	// set icp parameters
	icp.setMaximumIterations(params[0]);
	icp.setTransformationEpsilon(params[1]);
	icp.setEuclideanFitnessEpsilon(params[2]);
	icp.setMaxCorrespondenceDistance(params[3]);
	icp.setRANSACIterations(params[4]);  

	// align
	typename pcl::PointCloud<PointTarget>::Ptr output(new pcl::PointCloud<PointTarget>());
	icp.align(*output,guess);

	// get icp result
	ret.clear();
	ret.push_back(icp.hasConverged());
	ret.push_back(icp.getFitnessScore());  

	cor = *(icp.correspondences_);  
	
	// std::cout << "end PointToPointICP\n";	
	return icp.getFinalTransformation();
}								


/**
 * @brief 	point-to-point icp algorithm. The XYZ coordinates and normals of points are used for registration, 
 * 			where the normals of the point cloud are estimated online.
 * 			
 * 
 * @param src_pc: source point clouds
 * @param tar_pc: target point clouds
 * 
 * @param guess: init guess of icp
 * @param params: icp parameters
 * 			params[0]: number of neighbour points used for normal estimation 
 * 			params[0]: 	max iteration numbers
 * 			params[1]: 	transformation eps
 * 			params[2]: 	fitness eps
 * 			params[3]: 	max correspondence distance 
 * 			params[4]:	ransac iteration numbers 
 * 			params[5]:	number of neighbour points used for normal estimation
 * 
 * @param cor: correspondences between source and target
 * @param ret: result
 * 			ret[0]: convergence, 0 or 1
 * 			ret[1]: score
 * 
 * @return final transformation
 * 
 * 
 */
template<typename PointSource, typename PointTarget>
Eigen::Matrix4f PointToPointICPWithNormal(	typename pcl::PointCloud<PointSource>::Ptr src_pc,
											typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
											Eigen::Matrix4f guess,
											std::vector<double> params,
											std::vector<double> &ret,
											pcl::Correspondences &cor)
{
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_source_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_target_normals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_source_trans_normals(new pcl::PointCloud<pcl::PointNormal>());

	addNormal<pcl::PointXYZ, pcl::PointNormal>(tar_pc, cloud_target_normals, int(params[5]));
	addNormal<pcl::PointXYZ, pcl::PointNormal>(src_pc, cloud_source_normals, int(params[5]));

	pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
	
	icp.setMaximumIterations(params[0]);
	icp.setTransformationEpsilon(params[1]);
	icp.setEuclideanFitnessEpsilon(params[2]);
	icp.setMaxCorrespondenceDistance(params[3]);
	icp.setRANSACIterations(params[4]);  	

	icp.setInputSource(cloud_source_normals);  //
	icp.setInputTarget(cloud_target_normals);
	icp.align(*cloud_source_trans_normals, guess);  //

	cor = *(icp.correspondences_);

	ret.clear();
	ret.push_back(icp.hasConverged());
	ret.push_back( icp.getFitnessScore());  

	// std::cout << icp.getConvergeCriteria()->getConvergenceState() << std::endl;
	// return icp.getFinalTransformation();

	// // set input
	// pcl::IterativeClosestPointWithNormals<PointWithNoram_T, PointWithNoram_T> icp;

	// pcl::PointCloud<PointWithNoram_T>::Ptr src_pc_normals(new pcl::PointCloud<PointWithNoram_T>());
	// pcl::PointCloud<PointWithNoram_T>::Ptr tar_pc_normals(new pcl::PointCloud<PointWithNoram_T>());

	// addNormal<PointSource, PointWithNoram_T>(src_pc, src_pc_normals, int(params[5]));
	// addNormal<PointSource, PointWithNoram_T>(tar_pc, tar_pc_normals, int(params[5]));

	// std::cout << "finish add noraml\n";

	// icp.setInputSource(src_pc_normals); 
	// icp.setInputTarget(tar_pc_normals);	

	// // set icp parameters
	// icp.setMaximumIterations(params[0]);
	// icp.setTransformationEpsilon(params[1]);
	// icp.setEuclideanFitnessEpsilon(params[2]);
	// icp.setMaxCorrespondenceDistance(params[3]);
	// icp.setRANSACIterations(params[4]);  

	// std::cout << "begin align\n";
	// // align
	// pcl::PointCloud<PointWithNoram_T>::Ptr output(new pcl::PointCloud<PointWithNoram_T>());
	// icp.align(*output, guess);
	// std::cout << "finsish align\n";


	// // get icp result
	// ret.clear();
	// ret.push_back(icp.hasConverged());
	// ret.push_back(icp.getFitnessScore());  
	// cor = *(icp.correspondences_);

	std::cout << "end PointToPointICPWithNormal\n";
	return icp.getFinalTransformation();
}											

template<typename PointSource, typename PointTarget>
Eigen::Matrix4f PointToPlaneICPV1(	typename pcl::PointCloud<PointSource>::Ptr src_pc,
									typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
									Eigen::Matrix4f guess,
									std::vector<double> params,
									std::vector<double> &ret,
									pcl::Correspondences &cor)
{
	// std::cout << "begin PointToPlaneICPV1\n";

	pcl::IterativeClosestPoint<PointWithNoram_T, PointWithNoram_T> icp;
	typedef pcl::registration::TransformationEstimationPointToPlaneLLS<PointWithNoram_T, PointWithNoram_T> PointToPlaneLLS;
	boost::shared_ptr<PointToPlaneLLS> point_to_plane_lls(new PointToPlaneLLS);
	icp.setTransformationEstimation(point_to_plane_lls);

	pcl::PointCloud<PointWithNoram_T>::Ptr src(new pcl::PointCloud<PointWithNoram_T>);
	pcl::PointCloud<PointWithNoram_T>::Ptr tgt(new pcl::PointCloud<PointWithNoram_T>);

	pcl::copyPointCloud(*src_pc, *src);
	pcl::copyPointCloud(*tar_pc, *tgt);

	pcl::NormalEstimation<PointWithNoram_T, PointWithNoram_T> norm_est;
	norm_est.setSearchMethod(pcl::search::KdTree<PointWithNoram_T>::Ptr(new pcl::search::KdTree<PointWithNoram_T>));

	norm_est.setKSearch(int(params[5]));
	norm_est.setInputCloud(tgt);
	norm_est.compute(*tgt);

	icp.setInputSource(src);
	icp.setInputTarget(tgt);	

	// set icp parameters
	icp.setMaximumIterations(params[0]);
	icp.setTransformationEpsilon(params[1]);
	icp.setEuclideanFitnessEpsilon(params[2]);
	icp.setMaxCorrespondenceDistance(params[3]);
	icp.setRANSACIterations(params[4]);  

	pcl::PointCloud<PointWithNoram_T>::Ptr output(new pcl::PointCloud<PointWithNoram_T>());
	icp.align(*output,guess); 

	ret.clear();
	ret.push_back(icp.hasConverged());
	ret.push_back(icp.getFitnessScore());

	cor = *(icp.correspondences_);

	// std::cout << "end PointToPlaneICPV1\n";

	return icp.getFinalTransformation();
}

template<typename PointSource, typename PointTarget>
Eigen::Matrix4f PointToPlaneICPV2( 	typename pcl::PointCloud<PointSource>::Ptr src_pc,
									typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
									Eigen::Matrix4f guess,
									std::vector<double> params,
									std::vector<double> &ret,
									pcl::Correspondences &cor)
{
	// std::cout << "begin PointToPlaneICPV2\n";

	pcl::IterativeClosestPoint<PointWithNoram_T, PointWithNoram_T> icp;
	typedef pcl::registration::TransformationEstimationPointToPlaneLLS<PointWithNoram_T, PointWithNoram_T> PointToPlaneLLS;
	boost::shared_ptr<PointToPlaneLLS> point_to_plane_lls(new PointToPlaneLLS);
	icp.setTransformationEstimation(point_to_plane_lls);

	pcl::PointCloud<PointWithNoram_T>::Ptr src(new pcl::PointCloud<PointWithNoram_T>);
	pcl::copyPointCloud(*src_pc, *src);
	pcl::PointCloud<PointWithNoram_T>::Ptr tgt(new pcl::PointCloud<PointWithNoram_T>);
	pcl::copyPointCloud(*tar_pc, *tgt);

	icp.setInputSource(src);
	icp.setInputTarget(tgt);

	// set icp parameters
	icp.setMaximumIterations(params[0]);
	icp.setTransformationEpsilon(params[1]);
	icp.setEuclideanFitnessEpsilon(params[2]);
	icp.setMaxCorrespondenceDistance(params[3]);
	icp.setRANSACIterations(params[4]);  

	pcl::PointCloud<PointWithNoram_T>::Ptr output(new pcl::PointCloud<PointWithNoram_T>());
	icp.align(*output,guess); 

	ret.clear();
	ret.push_back(icp.hasConverged());
	ret.push_back(icp.getFitnessScore());

	cor = *(icp.correspondences_);

	// std::cout << "end PointToPlaneICPV2\n";

	return icp.getFinalTransformation();
}
/**
 * @brief   pcl General-ICP algorithm. It is also often named as plane-to-plane ICP.
 * 
 * @param src_pc: source point clouds
 * @oaram tar_pc: target point clouds
 * 
 * @param guess: init guess of icp
 * @param params: icp parameters
 * 			params[0]: 	max iteration numbers
 * 			params[1]: 	transformation eps
 * 			params[2]: 	fitness eps
 * 			params[3]: 	max correspondence distance 
 * 			params[4]:	ransac iteration numbers 
 * 
 * @param cor: correspondences between source and target
 * @param ret: result
 * 			ret[0]: convergence, 0 or 1
 * 			ret[1]: score
 * 
 * @return final transformation
 */
template<typename PointSource, typename PointTarget>
Eigen::Matrix4f pclGICP(typename pcl::PointCloud<PointSource>::Ptr src_pc,
						typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
						Eigen::Matrix4f guess,
						std::vector<double> params,
						std::vector<double> &ret,
						pcl::Correspondences &cor)
{
	// std::cout << "begin pclGICP\n";

	pcl::GeneralizedIterativeClosestPoint<PointSource, PointSource> icp;  // GICP 泛化的ICP，或者叫Plane to Plane
	icp.setInputSource(src_pc);
	icp.setInputTarget(tar_pc);
	
	// set icp parameters
	icp.setMaximumIterations(params[0]);
	icp.setTransformationEpsilon(params[1]);
	icp.setEuclideanFitnessEpsilon(params[2]);
	icp.setMaxCorrespondenceDistance(params[3]);
	icp.setRANSACIterations(params[4]);  

	typename pcl::PointCloud<PointSource>::Ptr output(new pcl::PointCloud<PointSource>());
	icp.align(*output,guess);  

	ret.clear();
	ret.push_back(icp.hasConverged());
	ret.push_back(icp.getFitnessScore());  

	cor = *(icp.correspondences_);

	// std::cout << "end pclGICP\n";

	return icp.getFinalTransformation();
}



/**
 * @brief   pcl NDT.
 * 
 * @param src_pc: source point clouds
 * @oaram tar_pc: target point clouds
 * 
 * @param guess: init guess of icp
 * @param params: icp parameters
 * 			params[0]: 	max iteration numbers
 * 			params[1]: 	transformation eps
 * 			params[2]: 	fitness eps
 * 			params[3-5]:	not used 	
 * 			params[6]:	g_ndt_resolution	
 * 			params[7]:	g_ndt_stepsize
 * 
 * @param cor: correspondences between source and target, none in ndt
 * @param ret: result
 * 			ret[0]: convergence, 0 or 1
 * 			ret[1]: score
 * 
 * @return final transformation
 */
template<typename PointSource, typename PointTarget>
Eigen::Matrix4f pclNDT(	typename pcl::PointCloud<PointSource>::Ptr src_pc,
						typename pcl::PointCloud<PointTarget>::Ptr tar_pc,
						Eigen::Matrix4f guess,
						std::vector<double> params,
						std::vector<double> &ret,
						pcl::Correspondences &cor)
{

	// std::cout << "begin pclNDT\n";

	pcl::NormalDistributionsTransform<Point_T, Point_T> ndt;

	ndt.setInputSource(src_pc);
	ndt.setInputTarget(tar_pc);

	ndt.setMaximumIterations (params[0]);
	ndt.setTransformationEpsilon (params[1]);
	ndt.setEuclideanFitnessEpsilon(params[2]);	

	ndt.setResolution(params[6]);    
	ndt.setStepSize(params[7]);              

	pcl::PointCloud<Point_T>::Ptr output(new pcl::PointCloud<Point_T>());
	ndt.align(*output,guess);  

	ret.clear();
	ret.push_back(ndt.hasConverged());
	ret.push_back(ndt.getFitnessScore());  

	// std::cout << "end pclNDT\n";
	return ndt.getFinalTransformation();
}

// Eigen::Matrix4f fastGICPST(pcl::PointCloud<Point_T>::Ptr cloud1,
//						pcl::PointCloud<Point_T>::Ptr cloud2,
//						Eigen::Matrix4f& guess,
//						std::vector<double> &ret,
//						pcl::Correspondences& cor,
//						std::vector<double> params);

// Eigen::Matrix4f fastGICPMT(pcl::PointCloud<Point_T>::Ptr cloud1,
//						pcl::PointCloud<Point_T>::Ptr cloud2,
//						Eigen::Matrix4f& guess,
//						std::vector<double> &ret,
//						pcl::Correspondences& cor,
//						std::vector<double> params);

template<typename PointSource, typename PointTarget>
std::map<std::string, RegFunPtr<PointSource, PointTarget>> RegFunMap = {	\
	{"PointToPointICP", &PointToPointICP<PointSource, PointTarget>},
	{"PointToPointICPWithNormal", &PointToPointICPWithNormal<PointSource, PointTarget>},
	{"PointToPlaneICPV1", &PointToPlaneICPV1<PointSource, PointTarget>},
	{"pclGICP", &pclGICP<PointSource, PointTarget>},
	{"pclNDT", &pclNDT<PointSource, PointTarget>}
};

} // namespace RegistrationAlgorithm