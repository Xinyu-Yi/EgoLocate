import os


_base = os.path.dirname(__file__)


class paths:
    smpl_file = _base + '/assets/SMPL_male.pkl'
    physics_model_file = _base + '/assets/physics.urdf'
    plane_file = _base + '/assets/plane.urdf'
    weights_file = _base + '/assets/weights.pt'
    physics_parameter_file = _base + '/physics_parameters.json'


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


vel_scale = 3
