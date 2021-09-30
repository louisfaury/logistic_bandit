import json
import matplotlib.pyplot as plt
import numpy as np
import os

root_dir = os.path.dirname(os.path.dirname(__file__))


logs_dir = os.path.join(root_dir, 'logs')
dimension = 2
res_dict = dict()

for log_path in os.listdir(logs_dir):
    log_dict = json.load(open(os.path.join(logs_dir, log_path), 'r'))
    log_cum_regret = np.array(log_dict["cum_regret"])
    # find the right line to store
    dim = int(log_dict["dim"])
    if not dim == dimension:
        continue
    algo = log_dict["algo_name"]
    norm_theta_star = log_dict["norm_theta_star"]
    horizon = len(log_cum_regret)
    if norm_theta_star not in res_dict.keys():
        res_dict[norm_theta_star] = dict()
    res_dict[norm_theta_star][algo] = log_cum_regret

num_figs = len(res_dict.keys())
print(num_figs)
fig, ax = plt.subplots(1, num_figs)
print(ax)

for idx, norm in enumerate(np.sort(list(res_dict.keys()))):
    print(idx)
    for algo in res_dict[norm].keys():
        ax[idx].plot(res_dict[norm][algo], label=algo)
    ax[idx].legend()
    ax[idx].set_title(norm)
plt.show()
