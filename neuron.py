import numpy as np


class Neuron():
    """
    Class to construct a neuron.
    Neuron is defined by weights, data `x` and biases `b` passed through an activation function.
    """

    def __init__(self, w, x, b, activation):
        # initialisation of a neuron
        self.w = w
        self.x = x
        self.b = b
        self.activation = activation

    def calc(self):
        # calculate output for the neuron
        neuron = self.x.dot(self.w) + self.b
        return self.activation(neuron)