from dataclasses import dataclass
import numpy as np


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

