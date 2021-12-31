from logbexp.algorithms.ecolog import EcoLog
from logbexp.algorithms.glm_ucb import GlmUCB
from logbexp.algorithms.gloc import Gloc
from logbexp.algorithms.logistic_ucb_1 import LogisticUCB1
from logbexp.algorithms.ol2m import Ol2m
from logbexp.algorithms.ofulogr import OFULogr


ALGOS = ['GLM-UCB', 'LogUCB1', 'OFULog-r', 'OL2M', 'GLOC', 'adaECOLog']


def create_algo(config):
    """
    Creates algorithm from config.
    """
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
    elif config["algo_name"] == 'adaECOLog':
        algo = EcoLog(param_norm_ub=config["param_norm_ub"],
                      arm_norm_ub=config["arm_norm_ub"],
                      dim=config["dim"],
                      failure_level=config["failure_level"])
    if algo is None:
        raise ValueError("Oops. The algorithm {} is not implemented. You must choose within ({})".format(
            config["algo_name"], ''.join(['\''+elem+'\''+', ' for elem in ALGOS])))
    return algo
