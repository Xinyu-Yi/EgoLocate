# Motion Capture 

The motion capture (mocap) module of the project is designed based on our previous works [PIP](https://xinyu-yi.github.io/PIP/) and [TransPose](https://xinyu-yi.github.io/TransPose/). 

- [**[PIP](https://github.com/Xinyu-Yi/PIP)**] Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shimada, Vladislav Golyanik, Christian Theobalt, and Feng Xu. 2022. **Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors**. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. **[PDF](https://arxiv.org/abs/2203.08528)**.
- [**[TransPose](https://github.com/Xinyu-Yi/TransPose)**] Xinyu Yi, Yuxiao Zhou, and Feng Xu. 2021. **TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors**. *ACM Transactions on Graphics 40 (08 2021)*. **[PDF](https://arxiv.org/abs/2105.04605)**.

## Prerequisites

### Install dependencies

We use python 3.8 on Ubuntu 22.04. Please install the dependences in `requirements.txt`.

You also need to compile and install [rbdl 2.6.0](https://github.com/rbdl/rbdl)  with python bindings. Also install the urdf reader addon. If you have configured [PIP](https://xinyu-yi.github.io/PIP/) on Linux with python 3.8 before, you can directly use the compiled RBDL library.

*Our codes do not need GPUs. Whether installing `pytorch` with CUDA or not does not matter.*

### Prepare SMPL body model

1. Download SMPL model from [here](https://smpl.is.tue.mpg.de/). You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. Then unzip it.
2. In `config.py`, set `paths.smpl_file` to the model path.

*If you have configured [TransPose](https://github.com/Xinyu-Yi/TransPose/)/[PIP](https://xinyu-yi.github.io/PIP/), just copy its settings here.*

### Prepare physics body model

1. Download the physics body model from [here](https://xinyu-yi.github.io/PIP/files/urdfmodels.zip) and unzip it.
2. In `config.py`, set `paths.physics_model_file` to the body model path.
3. In `config.py`, set `paths.plane_file`  to `plane.urdf`. Please put `plane.obj` next to it.

*The physics model and the ground plane are modified from [physcap](https://github.com/soshishimada/PhysCap_demo_release).*

*If you have configured [PIP](https://xinyu-yi.github.io/PIP/), just copy its settings here.*

### Prepare pre-trained network weights

1. Download weights from [here](https://xinyu-yi.github.io/PIP/files/weights.pt).
2. In `config.py`, set `paths.weights_file` to the weights path.

*If you have configured [PIP](https://xinyu-yi.github.io/PIP/), just copy its settings here (same weights).*

