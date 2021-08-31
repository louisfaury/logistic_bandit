import json
import os

from regret_minimization import many_bandit_exps

configs_path = os.path.join('configs', 'generated_configs')
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

for config_path in os.listdir(configs_path):
    config = json.load(open(os.path.join(configs_path, config_path), 'r'))
    mean_cum_regret = many_bandit_exps(config)
    log_path = os.path.join(logs_dir, 'h{}d{}a{}'.format(config["horizon"], config["dim"], config["algo_name"]))
    log_dict = config
    log_dict["cum_regret"] = mean_cum_regret.tolist()
    with open(log_path, 'w') as f:
        json.dump(log_dict, f)
