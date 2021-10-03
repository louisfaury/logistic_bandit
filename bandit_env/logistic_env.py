import numpy as np

from bandit_env.logistic_oracle import LogisticOracle
from bandit_env.arm_set import ArmSet

"""
Logistic Bandit Env. class

...

Attributes
---------
oracle : LogisticOracle
    
arm_set : ArmSet
"""


class LogisticBanditEnv(object):
    def __init__(self, theta_star, arm_set_type, arm_set_size, arm_norm_ub):
        self.oracle = LogisticOracle(theta_star)
        self.arm_set = ArmSet(arm_set_type, len(theta_star), arm_set_size, arm_norm_ub)
        self.arm_set.generate_arm_list()

    def interact(self, arm):
        """
        Returns the reward associated with arm, and the instantaneous regret
        :param arm: np.array(dim)
        :return: np.float
        """
        reward = self.oracle.pull(arm)
        regret = self.get_best_arm_exp_reward() - self.oracle.expected_reward(arm)
        if self.arm_set.type == 'tv_fixed':
            self.arm_set.generate_arm_list()
        return reward, regret

    def get_best_arm_exp_reward(self):
        """
        Returns the expected reward of the best arm
        :return: float
        """
        if self.arm_set.type == 'ball':
            best_arm = self.arm_set.arm_norm_ub * self.oracle.theta_star / np.linalg.norm(self.oracle.theta_star)
        else:
            perf_fun = lambda x: np.sum(x*self.oracle.theta_star)
            best_arm = self.arm_set.argmax(perf_fun)
        return self.oracle.expected_reward(best_arm)


def create_env(config):
    theta_star = config["theta_star"]
    arm_set_type = config["arm_set_type"]
    arm_set_size = config["arm_set_size"]
    arm_norm_ub = config["arm_norm_ub"]
    return LogisticBanditEnv(theta_star, arm_set_type, arm_set_size, arm_norm_ub)
