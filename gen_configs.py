import json
import numpy as np
import os

# create right config directory
config_dir = os.path.join('configs', 'generated_configs')
if not os.path.exists(config_dir):
    os.mkdir(config_dir)
# clear existing configs
for file in os.listdir(config_dir):
    os.remove(os.path.join(config_dir, file))

# hyper-params
repeat = 30
horizon = 1000
arm_set_type = 'fixed_discrete' # from ['fixed_discrete', 'tv_discrete', 'ball']
arm_set_size = 10
failure_level = 0.05
dims = np.array([2, 50, 10])
param_norm = np.array([2, 3, 4 ,5])
algos = ['GLM-UCB', 'OL2M', 'GLOC', 'ECOLog']
# algos = ['OFULog-r']

# create configs
for d in dims:
    for pn in param_norm:
        for algo in algos:
            theta_star = pn / np.sqrt(d) * np.ones([d])
            pn_ub = pn + 1
            config_path = os.path.join(config_dir, 'h{}d{}a{}n{}.json'.format(horizon, d, algo, pn))
            config_dict = {"repeat": int(repeat), "horizon": int(np.ceil(np.sqrt(d))*horizon), "dim": int(d),
                           "algo_name": algo, "theta_star": theta_star.tolist(), "param_norm_ub": int(pn_ub),
                           "failure_level": float(failure_level), "arm_set_type": arm_set_type,
                           "arm_set_size": int(d*arm_set_size), "arm_norm_ub": 1, "norm_theta_star": float(pn)}

            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
