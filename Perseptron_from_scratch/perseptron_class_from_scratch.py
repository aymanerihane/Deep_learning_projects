import numpy as np


def activation_function(v):
    """Heaviside Step function. It returns 1 if v >= 0, otherwise it returns 0."""
    return 1 if v >= 0 else 0

class Perceptron:
    def __init__(self, lr=0.01,itr=1000):
        self.lr = lr
        self.itr = itr

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.itr):
            for i ,x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = activation_function(linear_output)

                # update weights
                update = self.lr * (y[i] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.array([activation_function(i) for i in linear_output])
        return y_pred