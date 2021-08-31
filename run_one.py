import json
import matplotlib.pyplot as plt
import numpy as np
import os

from regret_minimization import many_bandit_exps

config_file = 'h1000d2aECOLogn2.json'
config_path = os.path.join('configs', 'generated_configs', config_file)

config = json.load(open(config_path, 'r'))
mean_cum_regret = many_bandit_exps(config)
plt.plot(np.arange(1, len(mean_cum_regret)+1), mean_cum_regret)
plt.show()
