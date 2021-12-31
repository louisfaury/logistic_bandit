import numpy as np

from logbexp.algorithms.logistic_bandit_algo import LogisticBandit
from numpy.linalg import solve, slogdet
from scipy.optimize import minimize, NonlinearConstraint
from logbexp.utils.utils import sigmoid, dsigmoid, weighted_norm, gaussian_sample_ellipsoid

"""
Class for the GLM-UCB algorithm of [Filippi et al. 2010].

Additional Attributes
---------------------
do_proj : bool
    whether to perform the projection step required by theory
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
design_matrix: np.array(dim x dim)
    sum of arms outer product (V_t)
design_matrix_inv: np.array(dim x dim)
    inverse of design_matrix
theta_hat : np.array(dim)
    maximum-likelihood estimator
theta_tilde : np.array(dim)
    projected version of theta_hat
ctr : int
    counter for lazy updates
ucb_bonus : float
    upper-confidence bound bonus
kappa : float
    inverse of minimum worst-case reward-sensitivity
"""


class GlmUCB(LogisticBandit):
    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level, do_proj=False, lazy_update_fr=5):
        """
        :param do_proj: whether to perform the projection step required by theory (default: False)
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        super().__init__(param_norm_ub, arm_norm_ub, dim, failure_level)
        self.name = 'GLM-UCB'
        self.do_proj = do_proj
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.l2reg = self.dim
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
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
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.arms = []
        self.rewards = []

    def learn(self, arm, reward):
        """
        Update the MLE, project if required/needed.
        """
        self.arms.append(arm)
        self.rewards.append(reward)

        # learn the m.l.e by iterative approach (a few steps of Newton descent)
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

        # update counter
        self.ctr += 1

        # perform projection (if required)
        if self.do_proj and len(self.rewards) > 2:
            if np.linalg.norm(self.theta_hat) > self.param_norm_ub:
                self.theta_tilde = self.theta_hat
            else:
                self.theta_tilde = self.projection(self.arms)
        else:
            self.theta_tilde = self.theta_hat

    def pull(self, arm_set):
        # update bonus bonus
        self.update_ucb_bonus()
        if not arm_set.type == 'ball':
            # find optimistic arm
            arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        else:  # TS, only valid for unit ball arm-set
            param = gaussian_sample_ellipsoid(self.theta_tilde, self.design_matrix, self.ucb_bonus)
            arm = self.arm_norm_ub * param / np.linalg.norm(param)
        # update design matrix and inverse
        self.design_matrix += np.outer(arm, arm)
        self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
                                  / (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))
        return arm

    def update_ucb_bonus(self):
        """
        Updates the UCB bonus.
        """
        logdet = slogdet(self.design_matrix)[1]
        res = np.sqrt(2 * np.log(1 / self.failure_level) + logdet - self.dim * np.log(self.l2reg))
        res *= 0.25 * self.kappa
        res += np.sqrt(self.l2reg)*self.param_norm_ub
        self.ucb_bonus = res

    def compute_optimistic_reward(self, arm):
        """
        Computes the UCB.
        """
        norm = weighted_norm(arm, self.design_matrix_inv)
        pred_reward = sigmoid(np.sum(self.theta_tilde * arm))
        bonus = self.ucb_bonus * norm
        return pred_reward + bonus

    def proj_fun(self, theta, arms):
        """
        Filippi et al. projection function
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        fun = np.dot(diff_gt, np.dot(self.design_matrix_inv, diff_gt))
        return fun

    def proj_grad(self, theta, arms):
        """
        Filippi et al. projection function gradient
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        grads = 2 * np.dot(self.design_matrix_inv, np.dot(self.hessian(theta, arms), diff_gt))
        return grads

    def gt(self, theta, arms):
        coeffs = sigmoid(np.dot(arms, theta))[:, None]
        res = np.sum(arms * coeffs, axis=0) + self.l2reg / self.kappa * theta
        return res

    def hessian(self, theta, arms):
        coeffs = dsigmoid(np.dot(arms, theta))[:, None]
        res = np.dot(np.array(arms).T, coeffs * arms) + self.l2reg / self.kappa * np.eye(self.dim)
        return res

    def projection(self, arms):
        fun = lambda t: self.proj_fun(t, arms)
        grads = lambda t: self.proj_grad(t, arms)
        norm = lambda t: np.linalg.norm(t)
        constraint = NonlinearConstraint(norm, 0, self.param_norm_ub)
        opt = minimize(fun, x0=np.zeros(self.dim), method='SLSQP', jac=grads, constraints=constraint)
        return opt.x
