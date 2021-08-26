
import numpy as np

from utils.utils import sigmoid


def fit_batch_logistic_mle(arms, rewards, l2reg=0.1, starting_point=None):
    """

    :param arms: np.array(len,dim)
    :param rewards: np.array(len)
    :param l2reg: float (positive)
    :param starting_point: np.array(dim,)
    :return: np.array(dim,); mle-estimate
    """
    dim = len(arms[0, :])
    if starting_point is None:
        theta_hat = np.zeros((dim,))
    else:
        theta_hat = starting_point
    # few steps of Newton descent
    for _ in range(20):
        predict_probas = sigmoid(np.dot(arms, theta_hat)[:, None])
        y = predict_probas - np.array(rewards)[:, None]
        grad = l2reg * theta_hat + np.sum(y * arms, axis=0)
        hessian = np.dot(np.array(arms).T, predict_probas * (1 - predict_probas) * np.array(arms)) + l2reg * np.eye(dim)
        theta_hat -= np.linalg.solve(hessian, grad)
    return theta_hat


def fit_online_logistic_estimate(arm, reward, current_estimate, vtilde_matrix):
    """

    :param arm:
    :param reward:
    :param vtilde_matrix:
    :return:
    """
    #TODO set right amount of iterations
    #TODO make sure correct normalization (should involve param_norm_ub
    # few steps of projected gradient descent

    return 0


def project_ellipsoid(x_to_proj, ell_center, ell_design):
    """

    :param x_to_proj:
    :param ell_center:
    :param ell_design:
    :return:
    """

    return 0