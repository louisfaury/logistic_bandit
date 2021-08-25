import numpy as np

from utils.utils import sigmoid

"""
Class for the oracle of a Logistic Bandit environment

...

Attributes
----------
theta_star: np.array(dim)
    unknown shared parameter
"""


class LogisticOracle(object):
    def __init__(self, theta_star):
        self.theta_star = theta_star

    def expected_reward(self, arm):
        """

        :param arm: np.array(dim)
        :return: the expected reward associated to arm
        """
        return sigmoid(np.sum(arm * self.theta_star))

    def pull(self, arm):
        """
        Draw reward according to the distribution associated with arm
        :param arm: np.array(dim)
        :return: bool, binary reward
        """
        return int(np.random.uniform(0, 1) < self.expected_reward(arm))
