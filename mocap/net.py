from torch.nn.utils.rnn import *
from torch.nn.functional import relu
import articulate as art
from articulate.utils.torch.rnn import *
from config import *
from utils import *
from dynamics import PhysicsOptimizer


class FastPIP(torch.nn.Module):
    name = 'FastPIP'
    n_hidden = 256

    def __init__(self, cam_joint_rbdl, cam_local_pos=torch.tensor([0, 0, 0.])):
        super(FastPIP, self).__init__()
        self.rnn1 = RNNWithInit(input_size=72,
                                output_size=joint_set.n_leaf * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                dropout=0.4)
        self.rnn2 = RNN(input_size=72 + joint_set.n_leaf * 3,
                        output_size=joint_set.n_full * 3,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn3 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=joint_set.n_reduced * 6,
                        hidden_size=self.n_hidden,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn4 = RNNWithInit(input_size=72 + joint_set.n_full * 3,
                                output_size=24 * 3,
                                hidden_size=self.n_hidden,
                                num_rnn_layer=2,
                                dropout=0.4)
        self.rnn5 = RNN(input_size=72 + joint_set.n_full * 3,
                        output_size=2,
                        hidden_size=64,
                        num_rnn_layer=2,
                        dropout=0.4)

        body_model = art.ParametricModel(paths.smpl_file)
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        self.forward_kinematics = body_model.forward_kinematics
        self.dynamics_optimizer = PhysicsOptimizer(cam_local_pos=cam_local_pos, cam_joint_rbdl=cam_joint_rbdl)
        self.load_state_dict(torch.load(paths.weights_file, map_location=torch.device('cpu')))
        self.eval()
        self.reset()

    def _reduced_glb_6d_to_full_local_rot(self, rootrot, glb6dpose):
        reduced_pose = art.math.r6d_to_rotation_matrix(glb6dpose).view(-1, len(joint_set.reduced), 3, 3)
        full_pose = torch.eye(3, device=glb6dpose.device).repeat(reduced_pose.shape[0], 24, 1, 1)
        full_pose[:, joint_set.reduced] = reduced_pose
        pose = self.inverse_kinematics_R(full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = rootrot.view(-1, 3, 3)
        return pose

    def reset(self, num_subject=1):
        self.dynamics_optimizer.reset_states()
        self.rnn_states = [None for _ in range(5)]
        self.n_frames = 0

        device = next(self.rnn1.init_net[0].parameters()).device
        lj_init = self.forward_kinematics(torch.eye(3).expand(1, 24, 3, 3))[1][0, joint_set.leaf].view(-1)
        nd, nh = self.rnn1.rnn.num_layers * (2 if self.rnn1.rnn.bidirectional else 1), self.rnn1.rnn.hidden_size
        self.rnn_states[0] = [_.repeat(1, num_subject, 1) for _ in self.rnn1.init_net(lj_init.view(1, -1).to(device)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)]

        if hasattr(self.rnn4, 'init_net'):
            jvel_init = torch.zeros(24 * 3)
            nd, nh = self.rnn4.rnn.num_layers * (2 if self.rnn4.rnn.bidirectional else 1), self.rnn4.rnn.hidden_size
            self.rnn_states[3] = [_.repeat(1, num_subject, 1) for _ in self.rnn4.init_net(jvel_init.view(1, -1).to(device)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)]

    @torch.no_grad()
    def forward(self, glb_acc, glb_rot, flat_floor=True):
        self.n_frames += 1
        imu = normalize_and_concat(glb_acc, glb_rot)

        x, self.rnn_states[0] = self.rnn1.rnn(relu(self.rnn1.linear1(imu), inplace=True).unsqueeze(0), self.rnn_states[0])
        x = self.rnn1.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x, self.rnn_states[1] = self.rnn2.rnn(relu(self.rnn2.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[1])
        x = self.rnn2.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x1, self.rnn_states[2] = self.rnn3.rnn(relu(self.rnn3.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[2])
        global_6d_pose = self.rnn3.linear2(x1[0])

        x1, self.rnn_states[3] = self.rnn4.rnn(relu(self.rnn4.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[3])
        joint_velocity = self.rnn4.linear2(x1[0])

        x1, self.rnn_states[4] = self.rnn5.rnn(relu(self.rnn5.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[4])
        contact = self.rnn5.linear2(x1[0])

        pose = self._reduced_glb_6d_to_full_local_rot(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(24, 3).mm(glb_rot[-1].t()) * vel_scale
        root_acc, pose, tran, cam_ori, cam_pos = self.dynamics_optimizer.optimize_frame(
            pose[0].cpu(), joint_velocity.cpu(), contact[0].cpu(), glb_acc.cpu(), flat_floor=flat_floor)

        return root_acc, pose, tran, cam_ori, cam_pos
