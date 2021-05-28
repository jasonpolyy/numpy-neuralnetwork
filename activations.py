import numpy as np

"""
Definitions for loss and activation functions that fit with the MLP class.
"""


# activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def softmax(x):
    """
    Calculate the softmax values of a vector `x`

    Softmax is defined as the exponential of a vector divided by the sum of the exponentials of a vector.
    :param x: vector to transform using softmax
    :return: vector transformed using the softmax function
    """
    e = np.exp(x)
    return e / np.sum(e)


# loss functions
def cross_entropy(x):
    return -np.log(x)


def mse(y_pred, y_actuals):
    s = (np.square(y_pred - y_actuals))
    s = np.sum(s) / len(y_actuals)
    return (s)

