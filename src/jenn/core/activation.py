"""Activation.
==============

This module implements activation functions used by the neural network."""

import abc
from typing import Union

import numpy as np


class Activation:
    """Activation function base class."""

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate activation function.

        :param x: input array at which to evaluate the function
        :param y: output array in which to write the results (optional)
        :return: activation function evaluated at `x` (as new array if `y` not provided as input)
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate 1st derivative of activation function.

        :param x: input array at which to evaluate the function
        :param y: response already evaluated at x (optional)
        :param dy: output array in which to write the 1st derivative (optional)
        :return: 1st derivative (as new array if `dy` not provided as input)
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def second_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
        ddy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate 2nd derivative of activation function.

        :param x: input array at which to evaluate the function
        :param y: response already evaluated at x (optional)
        :param y: 1st derivative already evaluated at x (optional)
        :param ddy: output array in which to write the 2nd derivative (optional)
        :return: 2nd derivative (as new array if `ddy` not provided as input)
        """
        raise NotImplementedError


class Tanh(Activation):
    r"""Hyperbolic tangent.

    .. math::
        y = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    """

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        return np.tanh(x, out=y)  # evaluated in place if y is not None

    @classmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if y is None:
            y = cls.evaluate(x)
        if dy is None:
            return 1 - np.square(y)
        dy[:] = 1 - np.square(y, out=dy)
        return dy

    @classmethod
    def second_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
        ddy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if y is None:
            y = cls.evaluate(x)
        if dy is None:
            dy = cls.first_derivative(x, y)
        if ddy is None:
            return -2 * y * dy
        ddy[:] = -2 * y * dy
        return ddy


class Relu(Activation):
    r"""Rectified linear unit activation.

    .. math::
        y = \begin{cases}
            x & \text{if}~ x \ge 0 \\
            0 & \text{otherwise}
        \end{cases}
    """

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if y is None:
            y = (x > 0) * x
        else:
            y[:] = (x > 0) * x
        return y

    @classmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if dy is None:
            dy = np.asarray(x > 0, dtype=x.dtype)
        else:
            dy[:] = np.asarray(x > 0, dtype=x.dtype)
        return dy

    @classmethod
    def second_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
        ddy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0.0
        return ddy


class Linear(Activation):
    r"""Linear activation function.

    .. math::
        y = x
    """

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if y is None:
            y = x.copy()
        else:
            y[:] = x
        return y

    @classmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if dy is None:
            dy = np.ones(x.shape)
        else:
            dy[:] = 1
        return dy

    @classmethod
    def second_derivative(
        cls,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        dy: Union[np.ndarray, None] = None,
        ddy: Union[np.ndarray, None] = None,
    ) -> np.ndarray:  # noqa: D102
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0
        return ddy


ACTIVATIONS = dict(
    relu=Relu,
    tanh=Tanh,
    linear=Linear,
)
