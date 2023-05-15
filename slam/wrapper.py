r"""
    Wrapper for libslam.so
"""

import ctypes as C
from math import pi
import torch
import os


slam_library_file = os.path.dirname(__file__) + '/lib/libslam.so'


class SLAM:
    core = C.cdll.LoadLibrary(slam_library_file)

    @staticmethod
    def init(setting_file_id, Rjkck_aa=(0, 0, pi)):
        SLAM.core.init(C.c_int(setting_file_id), C.c_double(Rjkck_aa[0]), C.c_double(Rjkck_aa[1]), C.c_double(Rjkck_aa[2]))

    @staticmethod
    def accept_socket():
        SLAM.core.accept_socket()

    @staticmethod
    def save_map_points(path: str):
        SLAM.core.save_map_points(C.c_char_p(path.encode('ascii')))

    @staticmethod
    def track_with_tp(im, tframe, tp_ori_aa, tp_pos):
        rows, cols = im.shape[0], im.shape[1]
        pose_x, pose_y, pose_z = C.c_double(0), C.c_double(0), C.c_double(0)
        tran_x, tran_y, tran_z = C.c_double(0), C.c_double(0), C.c_double(0)
        state, inliers, mse = C.c_int(0), C.c_int(0), C.c_double(0)
        is_send = C.c_bool(False)
        SLAM.core.track_with_tp(im.ctypes.data_as(C.POINTER(C.c_ubyte)), rows, cols, C.c_double(tframe),
                                C.c_double(tp_ori_aa[0]), C.c_double(tp_ori_aa[1]), C.c_double(tp_ori_aa[2]),
                                C.c_double(tp_pos[0]), C.c_double(tp_pos[1]), C.c_double(tp_pos[2]),
                                C.byref(pose_x), C.byref(pose_y), C.byref(pose_z),
                                C.byref(tran_x), C.byref(tran_y), C.byref(tran_z),
                                C.byref(state), C.byref(inliers), C.byref(mse), C.byref(is_send))
        return torch.tensor([pose_x.value, pose_y.value, pose_z.value]),\
               torch.tensor([tran_x.value, tran_y.value, tran_z.value]),\
               state.value, inliers.value, mse.value, is_send.value
