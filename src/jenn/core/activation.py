"""Activation functions."""

import abc

import numpy as np


class Activation:
    """Evaluate activation function."""

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate activation function.

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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate 1st derivative of activation function.

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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
        ddy: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        """Evaluate 2nd derivative of activation function.

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
    """Hyperbolic tangent activation."""

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        return np.tanh(x, out=y)  # evaluated in place if y is not None

    @classmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
        ddy: np.ndarray | None = None,
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
    """Rectified linear unit activation."""

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: np.ndarray = None,
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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
        ddy: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0.0
        return ddy


class Linear(Activation):
    """Linear activation function."""

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        y: np.ndarray | None = None,
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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
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
        y: np.ndarray | None = None,
        dy: np.ndarray | None = None,
        ddy: np.ndarray | None = None,
    ) -> np.ndarray:  # noqa: D102
        if ddy is None:
            return np.zeros(x.shape)
        ddy[:] = 0
        return ddy
