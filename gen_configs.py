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
algos = ['GLM-UCB', 'LogUCB1', 'OFULog-r', 'OL2M', 'GLOC', 'ECOLog']

# create configs
for d in dims:
    for algo in algos:
        config_path = os.path.join(config_dir, 'h{}d{}a{}.json'.format(horizon, d, algo))
        config_dict = {"repeat": int(repeat), "horizon": int(np.ceil(d)*horizon), "dimension": int(d), "algo": algo}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)
