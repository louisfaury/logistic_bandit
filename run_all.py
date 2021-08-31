import json
import os

from lin_ts import many_bandit_exps

configs_path = os.path.join('configs', 'gen_configs_regret')
logs_path = 'logs/regret'

for config_path in os.listdir(configs_path):
    print(config_path)
    config = json.load(open(os.path.join(configs_path, config_path), 'r'))
    mean_cum_regret, _ = many_bandit_exps(config["repeat"], config["horizon"], config["dimension"], config["alpha"])
    log_path = os.path.join(logs_path, 'h{}d{}a{}'.format(config["horizon"], config["dimension"], int(100*config["alpha"])))
    log_dict = config
    log_dict["cum_regret"] = mean_cum_regret.tolist()
    with open(log_path, 'w') as f:
        json.dump(log_dict, f)
