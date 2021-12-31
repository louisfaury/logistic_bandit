import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from logbexp.utils.optimization import fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from logbexp.utils.utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid

"""
Class for the ECOLog algorithm.
Additional Attributes
---------------------
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
v_tilde_matrix: np.array(dim x dim)
    matrix tilde{V}_t from the paper
v_tilde_inv_matrix: np.array(dim x dim)
    inverse of matrix tilde{V}_t from the paper
theta : np.array(dim)
    online estimator
conf_radius : float
    confidence set radius
cum_loss : float
    cumulative loss between theta and theta_bar
ctr : int
    counter
"""


class EcoLog(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level):
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'adaECOLog'
        self.l2reg = 1
        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

    def reset(self):
        self.vtilde_matrix = self.l2reg * np.eye(self.dim)
        self.vtilde_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta = np.zeros((self.dim,))
        self.conf_radius = 0
        self.cum_loss = 0
        self.ctr = 1

    def learn(self, arm, reward):
        # compute new estimate theta
        self.theta = np.real_if_close(fit_online_logistic_estimate(arm=arm,
                                                                   reward=reward,
                                                                   current_estimate=self.theta,
                                                                   vtilde_matrix=self.vtilde_matrix,
                                                                   vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                   constraint_set_radius=self.param_norm_ub,
                                                                   diameter=self.param_norm_ub,
                                                                   precision=1/self.ctr))
        # compute theta_bar (needed for data-dependent conf. width)
        theta_bar = np.real_if_close(fit_online_logistic_estimate_bar(arm=arm,
                                                                      current_estimate=self.theta,
                                                                      vtilde_matrix=self.vtilde_matrix,
                                                                      vtilde_inv_matrix=self.vtilde_matrix_inv,
                                                                      constraint_set_radius=self.param_norm_ub,
                                                                      diameter=self.param_norm_ub,
                                                                      precision=1/self.ctr))
        negative_norm = np.clip(weighted_norm(self.theta-theta_bar, self.vtilde_matrix),0, np.inf)

        # update matrices
        sensitivity = dsigmoid(np.dot(self.theta, arm))
        self.vtilde_matrix += sensitivity * np.outer(arm, arm)
        self.vtilde_matrix_inv += - sensitivity * np.dot(self.vtilde_matrix_inv,
                                                         np.dot(np.outer(arm, arm), self.vtilde_matrix_inv)) / (
                                          1 + sensitivity * np.dot(arm, np.dot(self.vtilde_matrix_inv, arm)))

        # sensitivity check
        sensitivity_bar = dsigmoid(np.dot(theta_bar, arm))
        if sensitivity_bar / sensitivity > 2:
            msg = f"\033[95m Oops. ECOLog has a problem: the data-dependent condition was not met. This is rare; try increasing the regularization (self.l2reg) \033[95m"
            raise ValueError(msg)

        # update sum of losses
        coeff_theta = sigmoid(np.dot(self.theta, arm))
        loss_theta = -reward * np.log(coeff_theta) - (1-reward) * np.log(1-coeff_theta)
        coeff_bar = sigmoid(np.dot(theta_bar, arm))
        loss_theta_bar = -reward * np.log(coeff_bar) - (1-reward) * np.log(1-coeff_bar)
        self.cum_loss += 2*(1+self.param_norm_ub)*(loss_theta_bar - loss_theta) - 0.5*negative_norm

    def pull(self, arm_set):
        # bonus-based version (strictly equivalent to param-based for this algo) of OL2M
        self.update_ucb_bonus()
        if not arm_set.type == 'ball':
            # find optimistic arm
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        else:  # TS version, here only valid for unit ball arm-set
            param = gaussian_sample_ellipsoid(self.theta, self.vtilde_matrix, self.conf_radius)
            arm = self.arm_norm_ub * param / np.linalg.norm(param)
        # update ctr
        self.ctr += 1
        return arm

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (a more precise version of Thm3 in ECOLog paper, refined for the no-warm up alg)
        """
        gamma = np.sqrt(self.l2reg) / 2 + 2 * np.log(
            2 * np.sqrt(1 + self.ctr / (4 * self.l2reg)) / self.failure_level) / np.sqrt(self.l2reg)
        res_square = 2*self.l2reg*self.param_norm_ub**2 + (1+self.param_norm_ub)**2*gamma + self.cum_loss
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm):
        """
        Returns prediction + exploration_bonus for arm.
        """
        norm = weighted_norm(arm, self.vtilde_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta * arm))
        bonus = self.conf_radius * norm
        return pred_reward + bonus
