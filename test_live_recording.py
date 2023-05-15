import torch
import os
import cv2
import mocap.articulate as art
from mocap.articulate.utils.bullet import MotionViewer
from pygame.time import Clock
import sys
sys.path.insert(0, os.path.abspath('mocap'))
from egolocate import EgoLocate


use_unity = False
imu_data = torch.load('example_data/imu.pt')
video = cv2.VideoCapture('example_data/video.mp4')
clock = Clock()
net = EgoLocate(slam_setting_file_id=3)

if use_unity:
    import socket
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('0.0.0.0', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect at 8888.')
    conn, addr = server_for_unity.accept()
    net.accept_socket()  # connect SLAM and unity
else:
    motion_viewer = MotionViewer(1)
    motion_viewer.connect()

for i, frame in enumerate(imu_data):
    clock.tick(64)
    im = video.read()[1] if frame[2] is True else None
    pose, tran = net.forward_frame(frame[0], frame[1], im=im, tframe=frame[3])
    if use_unity:
        s = ','.join(['%g' % v for v in art.math.rotation_matrix_to_axis_angle(pose).view(-1)]) + '#' + \
            ','.join(['%g' % v for v in tran.view(-1)]) + '#' + ('1' if net.is_send else '0') + '$'
        conn.send(s.encode('utf8'))
    else:
        motion_viewer.update(pose, tran, 0)
