"""
Automatically creates configs, stored in configs/generated_configs/
"""

import argparse
import json
import numpy as np
import os


# arg parser
parser = argparse.ArgumentParser(description='Automatically creates configs, stored in configs/generated_configs/',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dims', nargs='+', type=int, help='Dimension')
parser.add_argument('-pn', nargs='+', type=float, help='Parameter norm (||theta_star||)')
parser.add_argument('-algos', type=str, nargs='+', help='Algorithms. Possibilities include GLM-UCB, LogUCB1, OFULog-r, OL2M, GLOC or adaECOLog')
parser.add_argument('-r', type=int, nargs='?', default=20, help='# of independent runs')
parser.add_argument('-hz', type=int, nargs='?', default=1000, help='Horizon, normalized (later multiplied by sqrt(dim))')
parser.add_argument('-ast', type=str, nargs='?', default='fixed_discrete', help='Arm set type. Must be either fixed_discrete, tv_discrete or ball')
parser.add_argument('-ass', type=int, nargs='?', default='10', help='Arm set size, normalized (later multiplied by dim)')
parser.add_argument('-fl', type=float, nargs='?', default=0.05, help='Failure level, must be in (0,1)')
args = parser.parse_args()

# set args (no sanity check, please read the doc)
repeat = args.r
horizon = args.hz
arm_set_type = args.ast
arm_set_size = args.ass
failure_level = args.fl
dims = np.array(args.dims)
param_norm = np.array(args.pn)
algos = args.algos

# create right config directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(cur_dir, 'configs', 'generated_configs')
if not os.path.exists(config_dir):
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
