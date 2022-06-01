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


class Activation:

    @staticmethod
    def evaluate(z):
        """
        Evaluate activation function

        :param z: a scalar or numpy array of any size
        :return: activation value at z
        """
        raise NotImplementedError

    @staticmethod
    def first_derivative(z, a):
        """
        Evaluate gradient of activation function

        :param z: a scalar or numpy array of any size
        :return: gradient at z
        """
        raise NotImplementedError

    @staticmethod
    def second_derivative(z, a, da):
        """
        Evaluate second derivative of activation function

        :param z: a scalar or numpy array of any size
        :return: second derivative at z
        """
        raise NotImplementedError


class Sigmoid(Activation):

    @staticmethod
    def evaluate(z):
        a = 1. / (1. + np.exp(-z))
        return a

    @staticmethod
    def first_derivative(z, a):
        da = a * (1. - a)
        return da

    @staticmethod
    def second_derivative(z, a, da):
        dda = da * (1 - 2 * a)
        return dda


# class Tanh(Activation):
#
#     def evaluate(self, z):
#         exp_z = np.exp(z)
#         neg_exp_z = np.exp(-z)
#         numerator = exp_z - neg_exp_z
#         denominator = exp_z + neg_exp_z
#         a = np.divide(numerator, denominator)
#         return a
#
#     def first_derivative(self, z):
#         a = self.evaluate(z)
#         da = 1 - np.square(a)
#         return da
#
#     def second_derivative(self, z):
#         a = self.evaluate(z)
#         da = self.first_derivative(z)
#         dda = -2 * a * da
#         return dda

class Tanh(Activation):

    @staticmethod
    def evaluate(z):
        return np.tanh(z)

    @staticmethod
    def first_derivative(z, a):
        da = 1 - np.square(a)
        return da

    @staticmethod
    def second_derivative(z, a, da):
        dda = -2 * a * da
        return dda


class Relu(Activation):

    @staticmethod
    def evaluate(z):
        a = (z > 0) * z
        return a

    @staticmethod
    def first_derivative(z, a):
        da = 1.0 * (z > 0)
        return da

    @staticmethod
    def second_derivative(z, a, da):
        dda = 0.0
        return dda


class Linear(Activation):

    @staticmethod
    def evaluate(z):
        return z

    @staticmethod
    def first_derivative(z, a):
        return np.ones(z.shape)

    @staticmethod
    def second_derivative(z, a, da):
        return np.zeros(z.shape)


ACTIVATIONS = {'identity': Linear,
               'tanh': Tanh,
               'logistic': Sigmoid,
               'relu': Relu}


if __name__ == "__main__":
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    x = np.linspace(-10, 10, 100)
    activations = {'tanh': Tanh, 'sigmoid': Sigmoid, 'relu': Relu}
    for name, activation in activations.items():
        plt.plot(x, activation.evaluate(x))
        plt.title(name)
        plt.show()


