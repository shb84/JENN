"""Activation functions."""
import abc
import numpy as np


class Activation:
    """ Evaluate activation function dynamically or inplace """

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, x: np.ndarray, y: np.ndarray = None):
        """ Evaluate activation function in place: y = g(x)

        Parameters
        ----------
        x: np.ndarray
            Point at which to evaluate function

        y: np.ndarray, optional
            If provided, the function writes the result into it and returns a
            reference to y. Otherwise, new array is created. Default is None.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def first_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
    ):
        """ Evaluate gradient of activation function in place: dy = g'(x)

        Parameters
        ----------
        x: np.ndarray
            Point at which to evaluate function

        y: np.ndarray, optional
            If provided, the function uses the values; otherwise, it computes
            it. Providing avoids having to recompute y. Default is None.

        dy: np.ndarray, optional
            If provided, the function writes the 1st derivative into it and
            returns a reference to dy. Otherwise, new array is created. Default
            is None.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def second_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
            ddy: np.ndarray = None,
    ):
        """ Evaluate second derivative of activation function: ddy = g''(x)

        Parameters
        ----------
        x: np.ndarray
            Point at which to evaluate function

        y: np.ndarray, optional
            If provided, the function uses the values; otherwise, it computes
            it. Providing avoids having to recompute y. Default is None.

        dy: np.ndarray, optional
            If provided, the function uses the values; otherwise, it computes
            it. Providing avoids having to recompute dy. Default is None.

        ddy: np.ndarray, optional
            If provided, the function writes the 2nd derivative into it and
            returns a reference to ddy. Otherwise, new array is created.
            Default is None.
        """
        raise NotImplementedError


class Tanh(Activation):

    @classmethod
    def evaluate(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
    ):
        return np.tanh(x, out=y)  # evaluated in place if y is not None

    @classmethod
    def first_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
    ):
        if y is None:
            y = cls.evaluate(x)
        if dy is None:
            return 1 - np.square(y, out=dy)
        dy[:] = 1 - np.square(y, out=dy)
        return dy

    @classmethod
    def second_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
            ddy: np.ndarray = None,
    ):
        if y is None:
            y = cls.evaluate(x)
        if dy is None:
            dy = cls.first_derivative(x, y)
        if ddy is None:
            return -2 * y * dy
        ddy[:] = -2 * y * dy
        return ddy


class Relu(Activation):

    @classmethod
    def evaluate(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
    ):
        if y is None:
            y = x.copy()
        else:
            y[:] = x
        y[x <= 0] = 0
        return y

    @classmethod
    def first_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
    ):
        if dy is None:
            dy = np.ones(x.shape)
        else:
            dy[:] = 1.0
        dy[x <= 0] = 0
        return dy

    @classmethod
    def second_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
            ddy: np.ndarray = None,
    ):
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0.0
        return ddy


class Linear(Activation):

    @classmethod
    def evaluate(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
    ):
        if y is None:
            y = x.copy()
        else:
            y[:] = x
        return y

    @classmethod
    def first_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
            **kwargs,
    ):
        if dy is None:
            dy = np.ones(x.shape)
        else:
            dy[:] = 1
        return dy

    @classmethod
    def second_derivative(
            cls,
            x: np.ndarray,
            y: np.ndarray = None,
            dy: np.ndarray = None,
            ddy: np.ndarray = None,
            **kwargs,
    ):
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0
        return ddy

