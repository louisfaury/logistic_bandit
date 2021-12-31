import numpy as np

from logbexp.utils.utils import sigmoid

"""
Class for the oracle of a Logistic Bandit environment

Attributes
----------
theta_star: np.array()
    unknown shared parameter
"""


class LogisticOracle(object):
    def __init__(self, theta_star):
        self.theta_star = theta_star

    def expected_reward(self, arm):
        """
        Returns the expected reward associated to arm.
        """
        return sigmoid(np.sum(arm * self.theta_star))

    def pull(self, arm) -> (0, 1):
        """
        Draw reward according to the Bernoulli distribution associated with arm
        """
        return int(np.random.uniform(0, 1) < self.expected_reward(arm))
