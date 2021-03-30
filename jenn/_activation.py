"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""
from importlib.util import find_spec

if find_spec("matplotlib"):
    from matplotlib import pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

import numpy as np

tensor = np.ndarray


class Singleton:

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance


class Activation(Singleton):

    def evaluate(self, z):
        """
        Evaluate activation function

        :param z: a scalar or numpy array of any size
        :return: activation value at z
        """
        raise NotImplementedError

    def first_derivative(self, z):
        """
        Evaluate gradient of activation function

        :param z: a scalar or numpy array of any size
        :return: gradient at z
        """
        raise NotImplementedError

    def second_derivative(self, z):
        """
        Evaluate second derivative of activation function

        :param z: a scalar or numpy array of any size
        :return: second derivative at z
        """
        raise NotImplementedError


class Sigmoid(Activation):

    def evaluate(self, z):
        a = 1. / (1. + np.exp(-z))
        return a

    def first_derivative(self, z):
        a = self.evaluate(z)
        da = a * (1. - a)
        return da

    def second_derivative(self, z):
        a = self.evaluate(z)
        da = self.first_derivative(z)
        dda = da * (1 - 2 * a)
        return dda


class Tanh(Activation):

    def evaluate(self, z):
        numerator = np.exp(z) - np.exp(-z)
        denominator = np.exp(z) + np.exp(-z)
        a = np.divide(numerator, denominator)
        return a

    def first_derivative(self, z):
        a = self.evaluate(z)
        da = 1 - np.square(a)
        return da

    def second_derivative(self, z):
        a = self.evaluate(z)
        da = self.first_derivative(z)
        dda = -2 * a * da
        return dda


class Relu(Activation):

    def evaluate(self, z):
        a = (z > 0) * z
        return a

    def first_derivative(self, z):
        da = 1.0 * (z > 0)
        return da

    def second_derivative(self, z):
        dda = 0.0
        return dda


class Linear(Activation):

    def evaluate(self, z):
        return z

    def first_derivative(self, z):
        return np.ones(z.shape)

    def second_derivative(self, z):
        return np.zeros(z.shape)


ACTIVATIONS = {'identity': Linear(),
               'tanh': Tanh(),
               'logistic': Sigmoid(),
               'relu': Relu()}


if __name__ == "__main__":
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    x = np.linspace(-10, 10, 100)
    activations = {'tanh': Tanh(), 'sigmoid': Sigmoid(), 'relu': Relu()}
    for name, activation in activations.items():
        plt.plot(x, activation.evaluate(x))
        plt.title(name)
        plt.show()


