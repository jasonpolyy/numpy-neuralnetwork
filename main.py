from sklearn.datasets import make_classification, make_regression

from nn import *
from activations import *


# main function
def main():
    # make a dummy classification dataset
    X, y = make_classification(n_samples=500, n_features=10, n_informative=3, n_classes=3)

    test_mlp = MLP([2, 3, 2], loss_function=mse, activation=relu)

    test_mlp.fit(X, y)

    print(test_mlp.predict(X).shape)

    return None


# initialise main
if __name__ == "__main__":
    main()
