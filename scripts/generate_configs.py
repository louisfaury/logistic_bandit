"""
Creating configs, stored in configs/generated_configs/
"""

import json
import numpy as np
import os

# hyper-params
repeat = 30 # independent runs
horizon = 1000 # normalized (later multiplied by sqrt(dim))
arm_set_type = 'fixed_discrete' # from ['fixed_discrete', 'tv_discrete', 'ball']
arm_set_size = 10 # normalized (later multiplied by dim)
failure_level = 0.05 # delta
dims = np.array([2,])
param_norm = np.array([3, 4 ,5]) # ||theta_star||
algos = ['GLM-UCB', 'OL2M', 'GLOC', 'adaECOLog', 'OFULog-r'] # from ['GLM-UCB', 'LogUCB1', 'OFULog-r', 'OL2M', 'GLOC', 'adaECOLog']

# create right config directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(cur_dir, 'configs', 'generated_configs')
if not os.path.exists(config_dir):
    print('not exist')
    os.mkdir(config_dir)
# clear existing configs
for file in os.listdir(config_dir):
    os.remove(os.path.join(config_dir, file))

# create configs
for d in dims:
    for pn in param_norm:
        for algo in algos:
            theta_star = pn / np.sqrt(d) * np.ones([d])
            pn_ub = pn + 1 # parameter upper-bound (S = ||theta_star|| + 1)
            config_path = os.path.join(config_dir, 'h{}d{}a{}n{}.json'.format(horizon, d, algo, pn))
            config_dict = {"repeat": int(repeat), "horizon": int(np.ceil(np.sqrt(d))*horizon), "dim": int(d),
                           "algo_name": algo, "theta_star": theta_star.tolist(), "param_norm_ub": int(pn_ub),
                           "failure_level": float(failure_level), "arm_set_type": arm_set_type,
                           "arm_set_size": int(d*arm_set_size), "arm_norm_ub": 1, "norm_theta_star": float(pn)}

            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
