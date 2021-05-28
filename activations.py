import numpy as np

"""
Definitions for loss and activation functions that fit with the MLP class.
"""


# activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def softmax(x_i, x):
    return np.exp(x_i) / np.sum(np.exp(x))


# loss functions
def cross_entropy(x):
    return -np.log(x)


def cross_entropy_mult(x_i, x):
    return -np.log(softmax(x_i, x))


def mse(y_pred, y_actuals):
    s = (np.square(y_pred - y_actuals))
    s = np.sum(s) / len(y_actuals)
    return (s)

