r"""
    Temporal filters.
"""


__all__ = ['ESFilter', 'ESFilterRotation']


from . import math as M
import quaternion
import torch


class ESFilter:
    r"""
    Exponential smoothing.
    """
    def __init__(self, a=0.8):
        r"""
        Current = Lerp(Last, Current, a)

        :math:`s_t = ax_t + (1 - a)x_{t-1}, a \in [0, 1]`
        """
        self.a = a
        self.x = None

    def __call__(self, x):
        r"""
        Smooth the current value x.
        """
        if self.x is None:
            self.x = x
        else:
            self.x = M.lerp(self.x, x, self.a)
        return self.x

    def reset(self):
        r"""
        Reset the filter states.
        """
        self.x = None


class ESFilterRotation(ESFilter):
    r"""
    Exponential smoothing for rotation matrices.
    """
    def __init__(self, a=0.8):
        r"""
        Current = Lerp(Last, Current, a)
        """
        super().__init__(a)

    def __call__(self, x):
        r"""
        Smooth the current rotations x.

        :param x: Tensor that can reshape to [n, 3, 3] for rotation matrices.
        """
        qs = quaternion.from_rotation_matrix(x.detach().cpu().numpy(), nonorthogonal=True).ravel()
        if self.x is None:
            self.x = qs
        else:
            for i in range(len(qs)):
                self.x[i] = quaternion.np.slerp_vectorized(self.x[i], qs[i], self.a)
        x = torch.from_numpy(quaternion.as_rotation_matrix(self.x)).float().view_as(x)
        return x
