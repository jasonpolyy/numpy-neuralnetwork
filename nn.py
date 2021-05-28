import numpy as np
import statistics
from activations import *


class MLP:
    """
    Class for a simple multi-layer perceptron neural network.
    Implements feed forward, loss and back propagation methods.

    loss_function method must have two arguments for both predicted and actual values.

    Some code referenced from:
        https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/
    """

    def __init__(self, layer_sizes, loss_function, activation, tol=0.001, epochs=100):

        # Initialisation of a simple neural network.
        if type(layer_sizes) not in [tuple, list, np.ndarray]:
            raise ValueError("layer_sizes must be one of tuple, list, np.ndarray")

        self.layer_sizes = layer_sizes
        self.loss_function = loss_function
        self.tol = tol
        self.epochs = 100
        self.weights = None
        self.input_layer_size = None
        self.num_hidden_layers = self.__get_number_of_hidden_layers()
        self.losses = list()
        self.activation = activation

    def fit(self, X, y):
        """
        Fit the neural network using some data X and y.
        """
        self.input_layer_size = X.shape[1]
        self.__initialise_weights()

        # iterate through epochs to train
        for epoch in range(self.epochs):

            epoch_loss = []
            for x_i, y_i in zip(X, y):
                output = self.__feed_forward(x_i, sigmoid)

                # calculate the instance loss
                loss = self.__loss(y_i, output, f=self.loss_function)
                epoch_loss.append(loss)

                # self.weights = self.__back_propagation(y_i, output, learning_rate = 0.1)

            # append the average loss across the epoch
            self.losses.append(statistics.mean(epoch_loss))

        # print(self.weights)

    def predict(self, X):
        """
        Create predictions using the trained neural network.
        """

        # check if network is fitted or not. Raise error if not.
        self.__is_network_fitted()

        outputs = np.array([])
        for x_i in X:
            outputs = np.append(outputs, self.__feed_forward(x_i, self.activation))

        return outputs

    def score(self, X, y):
        """
        Use some score metric to score the accuracy of the neural network.
        """

        # check if network is fitted or not. Raise error if not.
        self.__is_network_fitted()

    def __feed_forward(self, x_i, activation):
        """
        Feed input forward through the neural network.
        """
        output = x_i
        for layer_idx in range(self.num_hidden_layers):
            output = activation(self.weights['w' + str(layer_idx)].T.dot(output) + self.weights['b' + str(layer_idx)])

        return output

    def __initialise_weights(self, seed=42):
        """
        Randomly initialise the weights and biases for all layers in the network.
        """
        np.random.seed(seed)
        input_layer_size = self.input_layer_size

        # set input layer sizes
        param_values = {'w0': np.random.rand(self.input_layer_size, self.layer_sizes[0]) * 0.1,
                        'b0': np.random.rand(self.layer_sizes[0]) * 0.1}

        for layer_i_minus1, layer_i in zip(np.array(range(0, self.num_hidden_layers - 1)),
                                           np.array(range(1, self.num_hidden_layers))):
            param_values['w' + str(layer_i)] = np.random.randn(self.layer_sizes[layer_i_minus1],
                                                               self.layer_sizes[layer_i]) * 0.1
            param_values['b' + str(layer_i)] = np.random.randn(self.layer_sizes[layer_i]) * 0.1

        self.weights = param_values

    def __loss(self, y_actuals, y_pred, f):
        return f(y_actuals, y_pred)

    def __get_number_of_hidden_layers(self):
        """
        Get the number of hidden layers in the network from self.layer_sizes
        """
        # initialise all neurons
        if type(self.layer_sizes) in [list, tuple]:
            return len(self.layer_sizes)

        elif type(self.layer_sizes) is np.ndarray:
            return self.layer_sizes.size

        else:
            return 1

    def __back_propagation(self, y_i, output, learning_rate=0.1):
        """
        Back propagation algorithm for tuning weights in the neural network.
        """
        return None

    def __is_network_fitted(self):
        """
        Check if network is fitted or not.
        """
        if self.weights is None:
            raise Exception("Network has not been fitted. Fit the network using `.fit()`")
