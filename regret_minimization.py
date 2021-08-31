
# logistic_regrets.py
# run logistic bandit experiments and plots regret curves

import json
import matplotlib.pyplot as plt
import numpy as np

from algorithms.algo_factory import create_algo
from bandit_env.logistic_env import create_env
from joblib import Parallel, delayed
from utils.utils import dsigmoid


def one_bandit_exp(config):
    env = create_env(config)
    algo = create_algo(config)
    horizon = config["horizon"]
    regret_array = np.empty(horizon)
    # lets go
    for t in range(horizon):
        arm = algo.pull(env.arm_set)
        reward, regret = env.interact(arm)
        regret_array[t] = regret
        algo.learn(arm, reward)
    return regret_array


def many_bandit_exps(config):
    run_bandit_exp = lambda x: one_bandit_exp(config)
    regret = Parallel(n_jobs=10)(delayed(run_bandit_exp)(i) for i in range(config["repeat"]))
    cum_regret = np.cumsum(regret, axis=1)
    return np.mean(cum_regret, axis=0)


if __name__ == '__main__':
    # load config
    config = json.load(open('configs/example_config.json', 'r'))

    # add some entries to config
    theta_star = (config["norm_theta_star"] / np.sqrt(config["dim"])) * np.array([1, 1])
    config["theta_star"] = theta_star
    config["param_norm_ub"] = config["norm_theta_star"] + 1

    # find kappa
    kappa = 1/dsigmoid(config["param_norm_ub"] * config["arm_norm_ub"])
    print('kappa is: ', kappa)

    # start
    mean_regret = many_bandit_exps(config)
    plt.plot(mean_regret)
    plt.show()
