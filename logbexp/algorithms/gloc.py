
import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from logbexp.utils.utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid

"""
Class for the GLOC algorithm of [Jun et al. 2017].

Additional Attributes
---------------------
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
kappa: float
    minimal variance
v_matrix_inv: np.array(dim x dim)
    inverse design matrix for Ol2m
zt : np.array(dim)
    for computing theta_hat
tetha_hat : np.array(dim)
    center of confidence set
theta: np.array(dim)
    ONS parameter
oco_regret_bound: float
    data-dependent bound on ONS's OCO regret
conf_width : float
    radius of confidence set 
"""


class Gloc(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level):
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'GLOC'
        self.l2reg = dim
        self.kappa = dsigmoid(param_norm_ub * arm_norm_ub) # 1/kappa in the paper, but here to match the def. of [Jun et al. 2017]
        self.v_matrix = self.l2reg * np.eye(self.dim)
        self.v_matrix_inv = (1/self.l2reg)*np.eye(self.dim)
        self.zt = np.zeros((dim,))
        self.theta_hat = np.zeros((self.dim,))
        self.theta = np.zeros((self.dim,))
        self.oco_regret_bound = 2 * self.kappa * self.param_norm_ub**2 * self.l2reg
        self.conf_radius = 0

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.v_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.zt = np.zeros((self.dim,))
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta = np.random.normal(0, 1, (self.dim,))
        self.oco_regret_bound = 2 * self.kappa * self.param_norm_ub ** 2 * self.l2reg
        self.conf_radius = 0

    def learn(self, arm, reward):
        """
        Updates estimators.
        """
        # update OCO regret bound (Thm. 3 of [Jun et al. 2017]
        current_grad = (sigmoid(np.dot(arm, self.theta)) - reward) * arm
        self.oco_regret_bound += (0.5/self.kappa) * weighted_norm(current_grad, self.v_matrix_inv)**2

        # compute new confidence set center
        self.zt += np.dot(self.theta, arm) * arm
        self.theta_hat = np.dot(self.v_matrix_inv, self.zt)

        # compute new ONS parameter
        unprojected_estimate = self.theta - np.dot(self.v_matrix_inv, current_grad) / self.kappa
        self.theta = self.param_norm_ub * unprojected_estimate / np.linalg.norm(unprojected_estimate)

    def pull(self, arm_set):
        self.update_ucb_bonus()
        if not arm_set.type == 'ball':
            # find optimistic arm
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        else:  # TS, only valid for unit ball arm-set
            param = gaussian_sample_ellipsoid(self.theta, self.v_matrix, self.conf_radius)
            arm = self.arm_norm_ub * param / np.linalg.norm(param)
        # update design matrix and inverse
        self.v_matrix_inv += - np.dot(self.v_matrix_inv,
                                      np.dot(np.outer(arm, arm), self.v_matrix_inv)) / (
                                     1 + np.dot(arm, np.dot(self.v_matrix_inv, arm)))
        self.v_matrix += np.outer(arm, arm)
        return arm

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (cf. Thm 1 of [Jun et al. 2017])
        """
        res_square = 1 + (4/self.kappa)*self.oco_regret_bound
        res_square += 8 * (self.param_norm_ub / self.kappa) ** 2 * np.log(2 * np.sqrt(
            1 + 2 * self.oco_regret_bound / self.kappa + 4 * (
                        self.param_norm_ub / self.kappa) ** 4 / self.failure_level ** 2) / self.failure_level)
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm):
        """
        Returns the UCB for arm.
        """
        norm = weighted_norm(arm, self.v_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta*arm))
        bonus = self.conf_radius * norm
        return pred_reward+bonus
