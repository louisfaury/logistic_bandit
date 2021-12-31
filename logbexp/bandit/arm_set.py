import numpy as np
from enum import Enum

"""
Arm-set class. 

Attributes
----------
arm_set_type : str
    the type of arm-set; options are ['fixed_discrete', 'tv_discrete', 'ball']
dim : int
    dimension
length : int
    number of arms - does not matter for ball arm-set
arm_norm_ub : float
    upper-bound on the ell-two norm of the arms
"""


class AdmissibleArmSet(Enum):
    fxd = 'fixed_discrete'
    tvd = 'tv_discrete'
    ball = 'ball'


class ArmSet(object):
    def __init__(self, arm_set_type, dim, length, arm_norm_ub):
        self.type = arm_set_type
        self.dim = dim
        self.length = length
        self.arm_norm_ub = arm_norm_ub
        self.arm_list = None

    def generate_arm_list(self):
        """
        Compute and stores the arm list.
        """
        if not self.type == AdmissibleArmSet.ball:
            u = np.random.normal(0, 1, (self.length, self.dim))
            norm = np.linalg.norm(u, axis=1)[:, None]
            r = np.random.uniform(0, 1, (self.length,1)) ** (1.0 / self.dim)
            self.arm_list = r * u / norm

    def argmax(self, fun):
        """
        Find the arm which maximizes fun (only valid for finite arm sets).
        :param fun: function to maximize
        """
        if self.type == AdmissibleArmSet.ball:
            raise ValueError('argmax function is only compatible with finite arm sets')
        arm_and_values = list(zip(self.arm_list, [fun(a) for a in self.arm_list]))
        return max(arm_and_values, key=lambda x: x[1])[0]

    def random(self):
        """
        Draw and returns a random arm from arm_set.
        """
        if self.type == AdmissibleArmSet.ball:
            u = np.random.normal(0, 1, self.dim)
            norm = np.sqrt(np.sum(u ** 2))
            r = np.random.uniform(0, self.arm_norm_ub) ** (1.0 / self.dim)
            res = r * u / norm
        else:
            idx = np.random.randint(0, self.length-1, 1)
            res = self.arm_list[np.asscalar(idx)]
        return res
