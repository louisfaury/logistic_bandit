import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import solve, slogdet

from logbexp.utils.utils import sigmoid, dsigmoid, weighted_norm
from scipy.stats import chi2

"""
Class for the LogisticUCB1 algorithm of [Faury et al. 2020]. Inherits from the LogisticBandit class.

Additional Attributes
---------------------
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
design_matrix: np.array(dim x dim)
    sum of arms outer product (V_t)
design_matrix_inv: np.array(dim x dim)
    inverse of design_matrix
hessian_matrix: np.array(dim x dim)
    hessian of the log-loss at current estimation (H_t)   
theta_hat : np.array(dim)
    maximum-likelihood estimator
ctr : int
    counter for lazy updates
ucb_bonus : float
    upper-confidence bound bonus
kappa : float
    inverse of minimum worst-case reward-sensitivity
"""


class LogisticUCB1(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, lazy_update_fr=5):
        """

        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'LogisticUCB1'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.l2reg = self.dim
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.ucb_bonus = 0
        self.kappa = 1 / dsigmoid(self.param_norm_ub * self.arm_norm_ub)
        # containers
        self.arms = []
        self.rewards = []

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """
        Updates estimators.
        """
        self.arms.append(arm)
        self.rewards.append(reward)

        # learn the m.l.e by iterative approach (a few steps of Newton descent)
        self.l2reg = self.dim * np.log(2 + len(self.rewards))
        if self.ctr % self.lazy_update_fr == 0 or len(self.rewards) < 200:
            # if lazy we learn with a reduced frequency
            theta_hat = self.theta_hat
            for _ in range(5):
                coeffs = sigmoid(np.dot(self.arms, theta_hat)[:, None])
                y = coeffs - np.array(self.rewards)[:, None]
                grad = self.l2reg * theta_hat + np.sum(y * self.arms, axis=0)
                hessian = np.dot(np.array(self.arms).T,
                                 coeffs * (1 - coeffs) * np.array(self.arms)) + self.l2reg * np.eye(self.dim)
                theta_hat -= np.linalg.solve(hessian, grad)
            self.theta_hat = theta_hat
            self.hessian_matrix = hessian
        # update counter
        self.ctr += 1

    def pull(self, arm_set):
        # update bonus bonus
        self.update_ucb_bonus()
        # select arm
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        # update design matrix and inverse
        self.design_matrix += np.outer(arm, arm)
        self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
                                  / (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))
        return arm

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (slight refinment from the concentration result of Faury et al. 2020)
        """
        _, logdet = slogdet(self.hessian_matrix)
        gamma_1 = np.sqrt(self.l2reg) / 2 + (2 / np.sqrt(self.l2reg)) \
                  * (np.log(1 / self.failure_level) + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg) +
                     np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)))
        gamma_2 = 1 + np.log(1 / self.failure_level) \
                  + np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)) \
                  + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg)
        gamma = np.min([gamma_1, gamma_2])
        res = 0.25 * np.sqrt(self.kappa) * np.min(
            [np.sqrt(1 + 2 * self.param_norm_ub) * gamma, gamma + gamma ** 2 / np.sqrt(self.l2reg)])
        res += np.sqrt(self.l2reg) * self.param_norm_ub
        self.ucb_bonus = res

    def compute_optimistic_reward(self, arm):
        """
        Computes UCB for arm.
        """
        norm = weighted_norm(arm, self.design_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta_hat * arm))
        bonus = self.ucb_bonus * norm
        return pred_reward + bonus
