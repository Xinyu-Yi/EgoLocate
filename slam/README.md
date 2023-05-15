# SLAM library

The SLAM library is designed based on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) released under [GPLv3 license](https://github.com/UZ-SLAMLab/ORB_SLAM3/LICENSE). 

- [**[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)**] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. **[PDF](https://arxiv.org/abs/2007.11898)**.

## Prerequisites

We use Ubuntu 22.04. It would be easy to build this library on other Linux systems. If there is any problem, users may first try to build [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) on the same system to get some experience.

- C++14 Compiler

- [Pangolin](https://github.com/stevenlovegrove/Pangolin)

- [OpenCV >= 4.4.0](http://opencv.org) 
- [Eigen3 >= 3.1.0](eigen.tuxfamily.org) 

## Configuration

Open `src/SLAM.cc`, modify the following lines  (L11~14):

```c++
bool use_viewer = false;     // whether to use ORB-SLAM3 viewer
string vocab_file = "<absolute path>/slam/Vocabulary/ORBvoc.txt";
string cam_settings_dir = "<absolute path>/slam/CameraSettings/";     // end with '/'
string map_points_save_dir = "<absolute output path>/";    // end with '/', output path
```

*Note: you should use absolute paths. Set `use_viewer=true` to open the ORB-SLAM3 built-in visualization.*

*The camera parameters and some other SLAM options are set in the camera setting `.yaml` file.*

## Build

Make sure all dependencies have been installed and the paths have been correctly set. Then, execute:
```
cd slam
chmod +x build.sh
./build.sh
```

This will automatically build the *Thirdparty* libraries and finally create  `libslam.so` in the `lib/` folder. If you move the generated library file, edit `wrapper.py` to set the correct library file path `slam_library_file`.

