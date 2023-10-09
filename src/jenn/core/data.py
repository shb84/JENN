import numpy as np
from dataclasses import dataclass
from functools import cached_property

from .mini_batch import mini_batches
from .normalization import normalize, normalize_partials


def avg(array):
    return np.mean(array, axis=1).reshape((-1, 1))


def std(array):
    return np.std(array, axis=1).reshape((-1, 1))


@dataclass
class Dataset:
    """Stores training data for easy access.

    Parameters
    ----------
    X: np.ndarray
        Training data outputs. An array of shape (n_x, m)

    Y: np.ndarray
        Training data outputs. An array of shape (n_y, m)

    J: np.ndarray, optional
        Training data gradients. An array of shape (n_y, n_x, m)
        Y' = d(Y)/dX where n_y = number outputs
                           n_x = number inputs
                           m = number examples
    """
    X: np.ndarray
    Y: np.ndarray
    J: np.ndarray = None

    def __post_init__(self):

        if self.X.shape[1] != self.Y.shape[1]:
            msg = f'X and Y must have the same number of examples'
            raise ValueError(msg)

        if self.J is not None:
            if self.J.shape != (self.n_y, self.n_x, self.m):
                msg = f'J must be of shape (n_y, n_x, m)'
                raise ValueError(msg)

    @property
    def m(self):
        return self.X.shape[1]

    @property
    def n_x(self):
        return self.X.shape[0]

    @property
    def n_y(self):
        return self.Y.shape[0]

    @cached_property
    def avg_x(self):
        return avg(self.X)

    @cached_property
    def avg_y(self):
        return avg(self.Y)

    @cached_property
    def std_x(self):
        return std(self.X)

    @cached_property
    def std_y(self):
        return std(self.Y)

    def mini_batches(self, batch_size: int, shuffle=True, random_state=None):
        """Breakup data into multiple batches."""
        X = self.X
        Y = self.Y
        J = self.J
        batches = mini_batches(X, batch_size, shuffle, random_state)
        if self.J is None:
            return [Dataset(X[:, b], Y[:, b]) for b in batches]
        return [Dataset(X[:, b], Y[:, b], J[:, :, b]) for b in batches]

    def normalize(self):
        """Return normalized Dataset."""
        X_norm = normalize(self.X, self.avg_x, self.std_x)
        Y_norm = normalize(self.Y, self.avg_y, self.std_y)
        J_norm = normalize_partials(self.J, self.std_x, self.std_y)
        return Dataset(X_norm, Y_norm, J_norm)

