from algorithms.ecolog import EcoLog
from algorithms.glm_ucb import GlmUCB
from algorithms.gloc import Gloc
from algorithms.logistic_ucb_1 import LogisticUCB1
from algorithms.ol2m import Ol2m
from algorithms.ofulogr import OFULogr


def create_algo(config):
    algo = None
    if config["algo_name"] == 'GLM-UCB':
        algo = GlmUCB(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"])
    elif config["algo_name"] == 'LogUCB1':
        algo = LogisticUCB1(param_norm_ub=config["param_norm_ub"],
                            arm_norm_ub=config["arm_norm_ub"],
                            dim=config["dim"],
                            failure_level=config["failure_level"])
    elif config["algo_name"] == 'OFULog-r':
        algo = OFULogr(param_norm_ub=config["param_norm_ub"],
                       arm_norm_ub=config["arm_norm_ub"],
                       dim=config["dim"],
                       failure_level=config["failure_level"])
    elif config["algo_name"] == 'OL2M':
        algo = Ol2m(param_norm_ub=config["param_norm_ub"],
                    arm_norm_ub=config["arm_norm_ub"],
                    dim=config["dim"],
                    failure_level=config["failure_level"])
    elif config["algo_name"] == 'GLOC':
        algo = Gloc(param_norm_ub=config["param_norm_ub"],
                    arm_norm_ub=config["arm_norm_ub"],
                    dim=config["dim"],
                    failure_level=config["failure_level"])
    elif config["algo_name"] == 'ECOLog':
        algo = EcoLog(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"])
    return algo
