//
// Created by yxy on 2022/1/9.
//

#ifndef ORB_SLAM3_UTILS_H
#define ORB_SLAM3_UTILS_H

#include <Eigen/Core>
#include <vector>
#include "opencv2/core.hpp"

void write_vector3_txt(const char *file_name, const double data[][3], int seq_len);
void read_vector3_txt(const char *file_name, double data[][3], int seq_len);
void convert_to_unity_motion(const char *trajectory_file, int seq_len, const char *output_dir);
void save_unity_motion(const std::vector<Eigen::Vector3f> &pose, const std::vector<Eigen::Vector3f> &tran, const char *output_dir);
cv::Mat_<float> FindRigidTransform(const cv::Mat_<cv::Vec3f> &points1, const cv::Mat_<cv::Vec3f> &points2);
void cv_visualize_matrix(const float *data, int rows, int cols, float lower=0, float upper=1);

#endif //ORB_SLAM3_UTILS_H
