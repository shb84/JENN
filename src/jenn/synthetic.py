"""Generate synthetic data from cannonical test function."""
import numpy as np

from typing import Tuple


class Sinusoid:
    """y =x * np.sin(x)"""

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:
        """Evaluate function

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (n_x, m) where m is the number of examples

        Returns
        -------
        y: np.ndarray
            Output array of shape (n_y, m)
        """
        return x * np.sin(x)

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:
        """Evaluate partial derivative

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (n_x, m) where m is the number of examples

        Returns
        -------
        dydx: np.ndarray
            Output array of shape (n_y, n_x, m)
        """
        return np.sin(x) + x * np.cos(x)

    @classmethod
    def sample(
            cls, m: int, lb: float = -np.pi, ub: float = np.pi,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data"""
        n_x = n_y = 1
        x = np.linspace(lb, ub, m).reshape((n_x, m))
        y = cls.evaluate(x).reshape((n_y, m))
        dydx = cls.first_derivative(x).reshape((n_y, n_x, m))
        return x, y, dydx
