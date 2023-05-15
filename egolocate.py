import torch
import numpy as np
from pygame.time import Clock
from slam.wrapper import SLAM
from mocap.utils import KalmanFilter, Body
from mocap.net import FastPIP


class EgoLocate:
    flat_floor = False
    use_clock = False

    def __init__(self, cam_joint_rbdl=Body.HEAD, cam_joint_smpl_index=15, slam_setting_file_id=-1,
                 cam_local_pos=torch.tensor([0, 0, 0.]), Rjkck_aa=(0, 0, np.pi), visualize=False):
        SLAM.init(slam_setting_file_id, Rjkck_aa)
        self.accept_socket = SLAM.accept_socket
        self.save_map_points = SLAM.save_map_points
        self.cam_joint_smpl_index = cam_joint_smpl_index
        self.net = FastPIP(cam_joint_rbdl, cam_local_pos)
        self.clock = Clock()
        self.reset()

        if visualize:
            from functools import partial
            from mocap.articulate.utils.bullet import MotionViewer
            MotionViewer.colors = [[198 / 255, 238 / 255, 0., 1.], [0., 0., 0., 1.], [1., 1., 1., 1.]]
            self.motion_viewer = MotionViewer(3)
            self.motion_viewer.connect()
            self.set_slam_pose = partial(self.motion_viewer.update, index=0)
            self.set_gt_pose = partial(self.motion_viewer.update, index=1)
            self.set_pip_pose = partial(self.motion_viewer.update, index=2)

    def reset(self):
        self.net.reset()
        self.initialized = False
        dt = 1.0 / 60
        var_a = dt * dt * 1
        F = np.array([[1., 0., 0., dt, 0., 0.],
                      [0., 1., 0., 0., dt, 0.],
                      [0., 0., 1., 0., 0., dt],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]])
        B = np.array([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [dt, 0., 0.],
                      [0., dt, 0.],
                      [0., 0., dt]])
        H = np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.]])
        Q = np.array([[0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., var_a, 0., 0.],
                      [0., 0., 0., 0., var_a, 0.],
                      [0., 0., 0., 0., 0., var_a]])
        P = np.array([[0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.]])
        self.kf = KalmanFilter(F=F, B=B, H=H, Q=Q, P=P)

    @torch.no_grad()
    def forward_frame(self, glb_acc, glb_rot, im=None, tframe=None):
        root_acc, pose, tran, cam_ori, cam_pos = self.net(glb_acc, glb_rot, self.flat_floor or not self.initialized)
        x = self.kf.predict(u=root_acc)
        tran_kf = torch.from_numpy(x[:3, 0]).float()
        self.is_send = False    # does SLAM send point clouds to unity?

        if im is not None and cam_ori is not None:
            if self.use_clock:
                self.clock.tick(31)
            cam_ori_opt, cam_pos_opt, state, inliers, _, self.is_send = SLAM.track_with_tp(im, tframe, cam_ori, cam_pos)
            if state < 2:
                self.kf.update(z=(cam_pos_opt - cam_pos + tran).reshape(3, 1), r=1)
            elif state == 2:  # OK
                self.initialized = True
                r = 1000 / (inliers + 0.001)
                if (cam_pos_opt - cam_pos + tran - tran_kf).norm() > 1:  # relocalization
                    print('reloc')
                    self.kf.x[:3] = (cam_pos_opt - cam_pos + tran).view(3, 1).numpy()
                else:
                    self.kf.update(z=(cam_pos_opt - cam_pos + tran).reshape(3, 1), r=r)

        return pose, tran_kf
