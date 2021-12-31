

class LogisticBandit(object):
    """
    Class representing a base logistic bandit algorithm

    Attributes
    ----------
    param_norm_ub : float
        upper bound on the ell-two norm of theta_star (S)
    arm_norm_ub : float
        upper bound on the ell-two norm of any arm in the arm-set (L)
    dim : int
        problem dimension (d)
    failure_level: float
        failure level of the algorithm (delta)
    name : str
        algo name

    Methods
    -------
    pull(arm_set)
        play an arm within the given arm set

    learn(dataset)
        update internal parameters

    reset()
        reset attributes to initial value
    """

    def __init__(self, param_norm_ub, arm_norm_ub, dim, failure_level):
        self.param_norm_ub = param_norm_ub
        self.arm_norm_ub = arm_norm_ub
        self.dim = dim
        self.failure_level = failure_level
        self.name = None

    def pull(self, arm_set):
        raise NotImplementedError

    def learn(self, arm, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
