__all__ = ['set_pose', 'smpl_to_rbdl', 'rbdl_to_smpl', 'normalize_and_concat', 'print_title', 'Body',
           'smpl_to_rbdl_data', 'KalmanFilter']


import enum
import torch
import numpy as np
import pybullet as p
from articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, euler_convert_np, \
    normalize_angle


_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl


def set_pose(id_robot, q):
    r"""
    Set the robot configuration.
    """
    p.resetJointStatesMultiDof(id_robot, list(range(1, p.getNumJoints(id_robot))), q[6:][_rbdl_to_bullet].reshape(-1, 1))
    glb_rot = p.getQuaternionFromEuler(euler_convert_np(q[3:6], 'zyx', 'xyz')[[2, 1, 0]])
    p.resetBasePositionAndOrientation(id_robot, q[:3], glb_rot)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans


def normalize_and_concat(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data


def print_title(s):
    print('============ %s ============' % s)


class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65


class KalmanFilter:
    r"""
    https://zhuanlan.zhihu.com/p/113685503
    """
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.n = F.shape[0]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z, r=None):
        R = np.eye(self.m) * r if r is not None else self.R
        y = z - np.dot(self.H, self.x)
        S = R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
