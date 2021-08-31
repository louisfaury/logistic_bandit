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
repeat = 100
horizon = 1000
dims = np.array([2, 5, 10, 20])
param_norm = np.array([2, 3, 4, 5])
algos = ['GLM-UCB', 'LogUCB1', 'OFULog-r', 'OL2M', 'GLOC', 'ECOLog']

# create configs
for d in dims:
    for pn in param_norm:
        for algo in algos:
            theta_star = pn / np.sqrt(d) * np.ones([d])
            pn_ub = pn + 1
            config_path = os.path.join(config_dir, 'h{}d{}a{}n{}.json'.format(horizon, d, algo, pn))
            config_dict = {"repeat": int(repeat), "horizon": int(np.ceil(d)*horizon), "dimension": int(d), "algo": algo,
                           "theta_star": theta_star.tolist(), "param_norm_ub": int(pn_ub)}

            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
