import numpy as np

from algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import solve, slogdet
from utils.utils import sigmoid
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import chi2

"""
Class for the OFULog-r algorithm of [Abeille et al. 2021]. Inherits from the LogisticBandit class.

...

Additional Attributes
---------------------
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
hessian_matrix: np.array(dim x dim)
    hessian of the log-loss at current estimation (H_t)   
theta_hat : np.array(dim)
    maximum-likelihood estimator
log_loss_hat : float
    log-loss at current estimate theta_hat
ctr : int
    counter for lazy updates
"""


class OFULogr(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, lazy_update_fr=10):
        """

        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'OFULog-r'
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.l2reg = self.dim
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        # containers
        self.arms = []
        self.rewards = []

    def reset(self):
        """

        Resets the underlying learning algorithm
        :return: None
        """
        self.hessian_matrix = self.l2reg * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 1
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """

        :param arm: np.array(dim)
        :param reward: bool
        :return: None
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
        """

        :param arm_set: list of available arms
        :return:
        """
        self.update_ucb_bonus()
        self.log_loss_hat = self.logistic_loss(self.theta_hat)
        arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        return arm

    def update_ucb_bonus(self):
        """
        Update the ucb bonus function (refined concentration result from Faury et al. 2020)
        :return:
        """
        _, logdet = slogdet(self.hessian_matrix)
        gamma_1 = np.sqrt(self.l2reg) / 2 + (2 / np.sqrt(self.l2reg)) \
                  * (np.log(1 / self.failure_level) + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg) +
                     np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)))
        gamma_2 = 1 + np.log(1 / self.failure_level) \
                  + np.log(chi2(self.dim).cdf(2 * self.l2reg) / chi2(self.dim).cdf(self.l2reg)) \
                  + 0.5 * logdet - 0.5 * self.dim * np.log(self.l2reg)
        gamma = np.min([gamma_1, gamma_2])
        res = (gamma + gamma ** 2 / self.l2reg) ** 2
        self.ucb_bonus = res

    def compute_optimistic_reward(self, arm):
        """
        Planning according to Algo. 2 of Abeille et al. 2021
        :param arm: np.array(dim)
        :return:
        """
        if self.ctr == 1:
            res = np.random.normal(0, 1)
        else:
            obj = lambda theta: -np.sum(arm * theta)
            cstrf = lambda theta: self.logistic_loss(theta) - self.log_loss_hat
            cstrf_norm = lambda theta: np.linalg.norm(theta)
            constraint = NonlinearConstraint(cstrf, 0, self.ucb_bonus)
            constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub ** 2)
            opt = minimize(obj, x0=self.theta_hat, method='COBYLA', constraints=[constraint, constraint_norm],
                           options={'maxiter': 20})
            res = np.sum(arm * opt.x)
        return res

    def logistic_loss(self, theta):
        """
        Compute the full log-loss at theta
        :param theta: np.array(dim)
        :return: float
        """
        res = self.l2reg / 2 * np.linalg.norm(theta)**2
        if len(self.rewards) > 0:
            coeffs = np.clip(sigmoid(np.dot(self.arms, theta)[:, None]), 1e-12, 1-1e-12)
            res += -np.sum(np.array(self.rewards)[:, None] * np.log(coeffs / (1 - coeffs)) + np.log(1 - coeffs))
        return res
