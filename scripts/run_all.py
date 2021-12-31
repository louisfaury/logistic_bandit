"""
Running regret minimization for all configs in configs/generated_configs/
"""

import json
import os

from logbexp.regret_routines import many_bandit_exps


def run(config_path):
    msg = "Currently treating: {}".format(config_path)
    if 'OFULog-r' in config_path:
        msg += f"\033[95m (warning, this algorithm is slooooow.)\033[95m"
    print(msg)
    config = json.load(open(os.path.join(configs_path, config_path), 'r'))
    mean_cum_regret = many_bandit_exps(config)
    log_path = os.path.join(logs_dir, 'h{}d{}a{}n{}'.format(config["horizon"], config["dim"], config["algo_name"],
                                                            config['norm_theta_star']))
    log_dict = config
    log_dict["cum_regret"] = mean_cum_regret.tolist()
    with open(log_path, 'w') as f:
        json.dump(log_dict, f)


if __name__ == '__main__':
    configs_path = os.path.join('configs', 'generated_configs')
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    for cf_path in os.listdir(configs_path):
        run(cf_path)
