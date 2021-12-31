"""
usage: plot_regret.py [-h] [-d [D]] [-pn [PN]]

Plot regret curves (by default for dimension=2 and parameter norm=3)

optional arguments:
  -h, --help  show this help message and exit
  -d [D]      Dimension (default: 2)
  -pn [PN]    Parameter norm (default: 4.0)
"""


import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from logbexp.utils.utils import dsigmoid

# parser
parser = argparse.ArgumentParser(description='Plot regret curves, by default for dimension=2 and parameter norm=3',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', type=int, nargs='?', default=2, help='Dimension')
parser.add_argument('-pn', type=float, nargs='?', default=4.0, help='Parameter norm')
args = parser.parse_args()
dimension = args.d
param_norm = args.pn
print("Plotting for dimension={} and parameter norm={}".format(dimension, param_norm))

# path to logs/
root_dir = os.path.dirname(os.path.dirname(__file__))
logs_dir = os.path.join(root_dir, 'logs')

# accumulate results
res_dict = dict()
for log_path in os.listdir(logs_dir):
    log_dict = json.load(open(os.path.join(logs_dir, log_path), 'r'))
    log_cum_regret = np.array(log_dict["cum_regret"])

    # eliminate logs with undesired dimension
    dim = int(log_dict["dim"])
    if not dim == dimension:
        continue
    # eliminate logs with undesired param norm
    pn = int(log_dict["norm_theta_star"])
    if not pn == param_norm:
        continue

    algo = log_dict["algo_name"]
    res_dict[algo] = log_cum_regret

# plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
for algorithm in res_dict.keys():
    plt.plot(res_dict[algorithm], label=algorithm)
plt.legend()
plt.title(r"$\kappa = {}$".format(int(1/dsigmoid(args.pn+1))))
plt.show()
