//
// Created by yxy on 2022/1/9.
//

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace std;


static int read_trajectory_txt(const char *file_name, double data[][8]) {
    ifstream raw_file(file_name);
    for (int i = 0; ; ++i) {
        for (int j = 0; j < 8; ++j) {
            raw_file >> data[i][j];
            if (raw_file.eof()) return i;
        }
    }
}

void write_vector3_txt(const char *file_name, const double data[][3], int seq_len) {
    ofstream of(file_name);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 3; ++j) {
            of << data[i][j] << ',';
        }
        of << '\n';
    }
}

void read_vector3_txt(const char *file_name, double data[][3], int seq_len) {
    ifstream f(file_name);
    char c;
    for (int i = 0; i < seq_len; ++i) {
        f >> data[i][0] >> c >> data[i][1] >> c >> data[i][2];
    }
}

void convert_to_unity_motion(const char *trajectory_file, int seq_len, const char *output_dir) {
    // save offline_tran.txt and offline_pose.txt for camera translations and axis angles.
    double data[seq_len][8], tran[seq_len][3], poseq[seq_len][4], pose[seq_len][3];
    memset(tran, 0, sizeof(tran));
    memset(poseq, 0, sizeof(poseq));

    int num_lines = read_trajectory_txt(trajectory_file, data), cur = 0;
    for (int i = 0; i < num_lines; ++i) {
        int f = (int)(data[i][0] / 1e9 * 30);
        while (++cur < f) {    // lost frames
            memcpy(tran[cur], tran[cur - 1], sizeof(double) * 3);
            memcpy(poseq[cur], poseq[cur - 1], sizeof(double) * 4);
        }
        memcpy(tran[f], data[i] + 1, sizeof(double) * 3);
        memcpy(poseq[f], data[i] + 4, sizeof(double) * 4);
    }
    while (++cur < seq_len) {    // lost frames
        memcpy(tran[cur], tran[cur - 1], sizeof(double) * 3);
        memcpy(poseq[cur], poseq[cur - 1], sizeof(double) * 4);
    }

    // xyzw quaternion (poseq) to axis angle (pose)
    for (int i = 0; i < seq_len; ++i) {
        const double theta_half = acos(poseq[i][3]);
        const double sin_theta_half = sin(theta_half);
        for (int j = 0; j < 3; ++j) {
            pose[i][j] = sin_theta_half == 0 ? 0 : poseq[i][j] / sin_theta_half * 2 * theta_half;
        }
    }

    write_vector3_txt((output_dir + "offline_tran.txt"s).c_str(), tran, seq_len);
    write_vector3_txt((output_dir + "offline_pose.txt"s).c_str(), pose, seq_len);
}


void save_unity_motion(const vector<Eigen::Vector3f> &pose, const vector<Eigen::Vector3f> &tran, const char *output_dir) {
    // save online_tran.txt and online_pose.txt for camera translations and axis angles.
    int seq_len = pose.size();
    double pose_arr[seq_len][3], tran_arr[seq_len][3];
    for (int i = 0; i < pose.size(); ++i) {
        pose_arr[i][0] = pose[i].x();
        pose_arr[i][1] = pose[i].y();
        pose_arr[i][2] = pose[i].z();
        tran_arr[i][0] = tran[i].x();
        tran_arr[i][1] = tran[i].y();
        tran_arr[i][2] = tran[i].z();
    }
    write_vector3_txt((output_dir + "online_tran.txt"s).c_str(), tran_arr, seq_len);
    write_vector3_txt((output_dir + "online_pose.txt"s).c_str(), pose_arr, seq_len);
}


static cv::Vec3f CalculateMean(const cv::Mat_<cv::Vec3f> &points)
{
    cv::Mat_<cv::Vec3f> result;
    cv::reduce(points, result, 0, cv::REDUCE_AVG);
    return result(0, 0);
}

// https://stackoverflow.com/questions/21206870/opencv-rigid-transformation-between-two-3d-point-clouds
cv::Mat_<float> FindRigidTransform(const cv::Mat_<cv::Vec3f> &points1, const cv::Mat_<cv::Vec3f> &points2)
{
    /* Calculate centroids. */
    cv::Vec3f t1 = -CalculateMean(points1);
    cv::Vec3f t2 = -CalculateMean(points2);

    cv::Mat_<float> T1 = cv::Mat_<float>::eye(4, 4);
    T1(0, 3) = t1[0];
    T1(1, 3) = t1[1];
    T1(2, 3) = t1[2];

    cv::Mat_<float> T2 = cv::Mat_<float>::eye(4, 4);
    T2(0, 3) = -t2[0];
    T2(1, 3) = -t2[1];
    T2(2, 3) = -t2[2];

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
     * which is used for scale calculation.
     */
    cv::Mat_<float> C(3, 3, 0.0);
    float p1Rms = 0, p2Rms = 0;
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++) {
        cv::Vec3f p1 = points1(ptIdx, 0) + t1;
        cv::Vec3f p2 = points2(ptIdx, 0) + t2;
        p1Rms += p1.dot(p1);
        p2Rms += p2.dot(p2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C(i, j) += p2[i] * p1[j];
            }
        }
    }

    cv::Mat_<float> u, s, vh;
    cv::SVD::compute(C, s, u, vh);

    cv::Mat_<float> R = u * vh;

    if (cv::determinant(R) < 0) {
        R -= u.col(2) * (vh.row(2) * 2.0);
    }

    float scale = sqrt(p2Rms / p1Rms);
    R *= scale;

    cv::Mat_<float> M = cv::Mat_<float>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<float> result = T2 * M * T1;
    result /= result(3, 3);

    return result;
}

void cv_visualize_matrix(const float *data, int rows, int cols, float lower, float upper) {
    constexpr int SQUIRE_SIZE = 20;
    cv::Mat im = cv::Mat::zeros(rows * SQUIRE_SIZE, cols * SQUIRE_SIZE, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; ++j) {
            float value = data[i * cols + j];
            float v = (value - lower) / (upper - lower);
            if (v > 1) v = 1;
            else if (v < 0) v = 0;
            int c = 255 - v * 255.;
            cv::rectangle(im, cv::Point(j * SQUIRE_SIZE, i * SQUIRE_SIZE),
                          cv::Point(j * SQUIRE_SIZE + SQUIRE_SIZE, i * SQUIRE_SIZE + SQUIRE_SIZE),
                          c, -1);
            cv::putText(im, to_string(value).substr(0, 5),
                        cv::Point(j * SQUIRE_SIZE + 2, i * SQUIRE_SIZE + SQUIRE_SIZE / 2),
                        0, SQUIRE_SIZE / 100., c > 128 ? 0 : 255);
        }
    }
    cv::imshow("matrix", im);
    // cv::waitKey(1);
    // cv::destroyWindow("matrix");
}
