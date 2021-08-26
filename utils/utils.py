
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def weighted_norm(x, A):
    return np.sqrt(np.dot(x, np.dot(A, x)))



