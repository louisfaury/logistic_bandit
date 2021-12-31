"""
Extracting to csv format for plotting in LaTeX
"""

import json
import numpy as np
import os
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(__file__))


logs_dir = os.path.join(root_dir, 'logs')
dimension = 2
algos = ['GLM-UCB', 'OFULog-r', 'GLOC', 'OL2M', 'adaECOLog', 'OFULog-r']
param_norm = 5
res_dict = dict()
horizon = np.arange(1, 3000)

data = pd.DataFrame({'t': horizon[0::5]})

for log_path in os.listdir(logs_dir):
    log_dict = json.load(open(os.path.join(logs_dir, log_path), 'r'))
    log_cum_regret = np.array(log_dict["cum_regret"])

    # eliminate logs with undesired dimension
    dim = int(log_dict["dim"])
    if not dim == dimension:
        continue

    # eliminate logs with undesired algo
    algo = log_dict["algo_name"]
    if algo not in algos:
        continue

    # eliminate logs with undesired param_norm
    norm_theta_star = log_dict["norm_theta_star"]
    if not norm_theta_star == param_norm:
        continue

    # store
    data[algo] =  log_cum_regret[0::5]

data.to_csv('regretd{}n{}.dat'.format(dimension, param_norm))
