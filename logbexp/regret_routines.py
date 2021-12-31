
"""
Helper functions for regret minimization
"""

import numpy as np

from logbexp.algorithms.algo_factory import create_algo
from logbexp.bandit.logistic_env import create_env
from joblib import Parallel, delayed


def one_bandit_exp(config):
    env = create_env(config)
    algo = create_algo(config)
    horizon = config["horizon"]
    regret_array = np.empty(horizon)
    # let's go
    for t in range(horizon):
        arm = algo.pull(env.arm_set)
        reward, regret = env.interact(arm)
        regret_array[t] = regret
        algo.learn(arm, reward)
    return regret_array


def many_bandit_exps(config):
    def run_bandit_exp(*args):
        return one_bandit_exp(config)
    regret = Parallel(n_jobs=10)(delayed(run_bandit_exp)(i) for i in range(config["repeat"]))
    cum_regret = np.cumsum(regret, axis=1)
    return np.mean(cum_regret, axis=0)
