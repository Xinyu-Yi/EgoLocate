#include <iostream>
#include <opencv2/core/core.hpp>
#include "include/System.h"
#include <Eigen/Core>

#define DLL_EXPORT extern "C" __attribute ((visibility("default")))



/***************************************** CHANGE THIS *****************************************/
bool use_viewer = true;     // whether to use ORB-SLAM3 viewer
string vocab_file = "/home/yxy/release/EgoLocate/slam/Vocabulary/ORBvoc.txt";
string cam_settings_dir = "/home/yxy/release/EgoLocate/slam/CameraSettings/";     // end with '/'
string map_points_save_dir = "/home/yxy/release/EgoLocate/results/map_points/";   // end with '/'
/***********************************************************************************************/



ORB_SLAM3::System *pSLAM = NULL;


// Init the slam system by setting the camera setting file and head-camera rotation offset (RHC).
// If we do not perform curve initialization, RHC will be used to calculate slam-mocap global frame transformation.
DLL_EXPORT void init(int setting_file_id, double RHC_x, double RHC_y, double RHC_z) {
    string setting_file = cam_settings_dir + to_string(setting_file_id) + ".yaml";
    pSLAM = new ORB_SLAM3::System(vocab_file, setting_file, ORB_SLAM3::System::MONOCULAR, use_viewer);
    pSLAM->SetRHC((float)RHC_x, (float)RHC_y, (float)RHC_z);
}

// Used when visualizing map points in real time for the live demo. We directly send the points from SLAM to Unity3D.
DLL_EXPORT void accept_socket() {
    pSLAM->AcceptSocket();
}

// Used for the evaluation of mapping accuracy
DLL_EXPORT void save_map_points(const char *name) {
    ofstream of(map_points_save_dir + name + "map.txt");
    of << pSLAM->GetMapPointString();
    of.close();
}

// main function of tracking with mocap awareness
// input: image and timestamp (Row 1); mocap-estimated head pose (Row 2) and tran (Row 3);
// output: optimized head pose (not used, Row 4) and tran (Row 5); tracking state; number of matched points (confidence);
//         curve trajectory alignment error (when using curve initialization); did SLAM send map points to Unity3D;
DLL_EXPORT void track_with_tp(unsigned char *im_data, int rows, int cols, double tframe,
                              double tp_pose_x, double tp_pose_y, double tp_pose_z,
                              double tp_tran_x, double tp_tran_y, double tp_tran_z,
                              double *pose_x, double *pose_y, double *pose_z,
                              double *tran_x, double *tran_y, double *tran_z,
                              int *state, int *inliers, double *initMSEerr, bool *isSend) {
    cv::Mat im(rows, cols, CV_8UC3, im_data);
    Eigen::Vector3d tp_axis_angle(tp_pose_x, tp_pose_y, tp_pose_z);
    Eigen::Vector3d tp_tran(tp_tran_x, tp_tran_y, tp_tran_z);
    Sophus::SE3d tp_pose(Sophus::SO3d::exp(tp_axis_angle), tp_tran);

    Sophus::SE3f T = pSLAM->TrackMonocular6IMU(im, tframe, tp_pose.cast<float>(), isSend);

    auto pose = T.so3().log();
    *pose_x = pose(0, 0);
    *pose_y = pose(1, 0);
    *pose_z = pose(2, 0);

    auto tran = T.translation();
    *tran_x = tran(0, 0);
    *tran_y = tran(1, 0);
    *tran_z = tran(2, 0);

    *state = pSLAM->GetTrackingState();
    *inliers = pSLAM->mnMatchesInliers;
    *initMSEerr = pSLAM->initMSEerr;
}
