import json
import matplotlib.pyplot as plt
import numpy as np
import os

from lin_ts import many_bandit_exps

config_dict = 'configs'
config_file = 'config.json'
config_path = os.path.join(config_dict, config_file)

config = json.load(open(config_path, 'r'))
horizon = config["horizon"]
dimension = config["dimension"]
repeat = config["repeat"]
alpha = config["alpha"]

mean_cum_regret, _ = many_bandit_exps(repeat, horizon, dimension, alpha)
plt.plot(np.arange(1, horizon+1), mean_cum_regret)
plt.show()
