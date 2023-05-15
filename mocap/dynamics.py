import torch
import numpy as np
import articulate as art
from articulate.utils.bullet import *
from articulate.utils.rbdl import *
from utils import *
from qpsolvers import solve_qp
from config import paths


class PhysicsOptimizer:
    def __init__(self, cam_local_pos, cam_joint_rbdl=Body.HEAD):
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True, gravity=np.array([0, 0, 0.]))
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)
        self.cam_local_pos = cam_local_pos
        self.cam_joint_rbdl = cam_joint_rbdl
        self.reset_states()

    def reset_states(self):
        self.last_x = None
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.qddot = np.zeros(self.model.qdot_size)

    def optimize_frame(self, pose, jvel, contact, acc, flat_floor=False):
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        v_ref = jvel.numpy()
        c_ref = contact.sigmoid().numpy()
        a_ref = acc.numpy()
        q = self.q
        qdot = self.qdot

        if q is None:
            self.q = q_ref
            return self.qddot[:3, None], pose, torch.zeros(3), None, None

        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))

        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        As1, bs1 = [], []
        Gs1, hs1 = [], []

        # joint angle PD controller
        A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
        b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
        As1.append(A)  # 72 * 75
        bs1.append(b)  # 72

        # contacting foot velocity
        is_float = flat_floor
        for joint_name, stable in zip(['LFOOT', 'RFOOT'], c_ref):
            joint_id = vars(Body)[joint_name]
            J = self.model.calc_point_Jacobian(q, joint_id)
            v = self.model.calc_point_velocity(q, qdot, joint_id)

            th = 0.001 if stable > 0.5 else 1000000
            if flat_floor:
                pos = self.model.calc_body_position(q, joint_id)
                th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
                is_float = is_float and pos[1] - self.params['floor_y'] > 0.02
            else:
                th_y = -1000
            Gs1.append(-self.params['delta_t'] * J)
            hs1.append(v - [-th, th_y, -th])
            Gs1.append(self.params['delta_t'] * J)
            hs1.append(-v + [th, 1000, th])  # max(th, th_y) + 1e-6

        # if is float, add gravity velocity
        if is_float and c_ref.max() > 0.5:
            v_ref = v_ref + [0, -0.3, 0]

        # joint position PD controller
        for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                                  'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                                  'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
            joint_id = vars(Body)[joint_name]
            if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
            cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
            a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
            A = self.model.calc_point_Jacobian(q, joint_id)
            b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
            As1.append(A * self.params['coeff_jvel'])
            bs1.append(b * self.params['coeff_jvel'])

        As1, bs1 = torch.from_numpy(np.vstack(As1)), torch.from_numpy(np.concatenate(bs1))
        G_, h_ = np.vstack(Gs1), np.concatenate(hs1)
        P_ = As1.t().mm(As1).numpy()
        q_ = -As1.t().mm(bs1.view(-1, 1)).view(-1).numpy()

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x
        A_ = b_ = None
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='quadprog', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='scs', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)

        if x is None:
            x = solve_qp(P_, q_, None, None, A_, b_, solver='quadprog', initvals=init)

        qddot = x
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        self.q = q
        self.qdot = qdot
        self.qddot = qddot
        self.last_x = x

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]

        # todo: this is slow
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        cam_pos = torch.from_numpy(self.model.calc_body_position(q, self.cam_joint_rbdl)).float()
        cam_ori = torch.from_numpy(self.model.calc_body_orientation(q, self.cam_joint_rbdl)).float()   # T-pose not smpl
        cam_ori = cam_ori.mm(torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0.]]))
        cam_pos = cam_pos + cam_ori.mm(self.cam_local_pos.view(3, 1)).view(3)
        cam_ori = art.math.rotation_matrix_to_axis_angle(cam_ori)[0]
        return self.qddot[:3, None], pose_opt, tran_opt, cam_ori, cam_pos
