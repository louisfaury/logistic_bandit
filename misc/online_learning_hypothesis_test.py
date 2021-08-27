import numpy as np

from bandit_env.logistic_env import LogisticBanditEnv
from utils.optimization import fit_batch_logistic_mle, fit_online_logistic_estimate, fit_online_logistic_estimate_bar
from utils.utils import dsigmoid

# TODO sometimes fail because complex number appear .. I'm probably taking square root of matrix which is not really PSD
# TODO investigate ..

"""
Script designed to try out how often the data-dependent condition for logistic online learning
is met ---- evaluate the learning procedure. 
"""

# some configs
dim = 10
param_norm_ub = 4
theta_star = param_norm_ub * np.ones(dim) / np.sqrt(dim)
horizon = 50000
l2reg = param_norm_ub * dim
init_burn_in = param_norm_ub * dim
bandit_env = LogisticBanditEnv(theta_star, 'fixed_discrete', 50, 1)

# initialization
theta = np.zeros((dim,))
theta_bar = np.zeros((dim,))
theta_hat = np.zeros((dim,))
vtilde_matrix = l2reg * np.eye(dim)
vtilde_inv_matrix = (1 / l2reg) * np.eye(dim)

# small burn_in
bi_arms = np.empty((init_burn_in, dim))
bi_rewards = np.empty((init_burn_in,))
for t in range(init_burn_in):
    arm = bandit_env.arm_set.random()
    reward, _ = bandit_env.interact(arm)
    bi_arms[t, :] = arm
    bi_rewards[t] = reward
# fit initial MLE
theta_hat = fit_batch_logistic_mle(bi_arms, bi_rewards)
theta = theta_hat

# start online learning procedure
print("If nothing appears, the condition was never broken ..")
for _ in range(horizon):
    arm = bandit_env.arm_set.random()

    theta_one = fit_online_logistic_estimate(arm=arm,
                                             reward=1,
                                             current_estimate=theta,
                                             vtilde_matrix=vtilde_matrix,
                                             vtilde_inv_matrix=vtilde_inv_matrix,
                                             constraint_set_radius=param_norm_ub)

    theta_zero = fit_online_logistic_estimate(arm=arm,
                                             reward=0,
                                             current_estimate=theta,
                                             vtilde_matrix=vtilde_matrix,
                                             vtilde_inv_matrix=vtilde_inv_matrix,
                                             constraint_set_radius=param_norm_ub)

    theta_bar = fit_online_logistic_estimate_bar(arm=arm,
                                                 current_estimate=theta,
                                                 vtilde_matrix=vtilde_matrix,
                                                 vtilde_inv_matrix=vtilde_inv_matrix,
                                                 constraint_set_radius=param_norm_ub)

    # checking the different values for \dot\mu
    dsigmoid_theta_one = dsigmoid(np.dot(arm, theta_one))
    dsigmoid_theta_zero = dsigmoid(np.dot(arm, theta_zero))
    dsigmoid_theta_bar = dsigmoid(np.dot(arm, theta_bar))

    if (dsigmoid_theta_bar/dsigmoid_theta_one>2) or (dsigmoid_theta_bar/dsigmoid_theta_zero>2):
        print('condition not satisfied!')

    reward, _ = bandit_env.interact(arm)
    if reward == 1:
        theta = theta_one
    else:
        theta = theta_zero

    dsigmoid_theta = dsigmoid(np.sum(theta*arm))
    rank_one_update = dsigmoid_theta * np.outer(arm, arm)
    vtilde_matrix += rank_one_update
    vtilde_inv_matrix -= np.dot(np.dot(vtilde_inv_matrix, rank_one_update), vtilde_inv_matrix) / (
               1 + dsigmoid_theta * np.dot(arm, np.dot(vtilde_inv_matrix, arm)))
