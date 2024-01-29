"""Training data dataclass."""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np


def mini_batches(
    X: np.ndarray,
    batch_size: int | None,
    shuffle: bool = True,
    random_state: int = None,
) -> list[tuple[int, ...]]:
    """Create randomized mini-batches.

    Parameters
    ----------
    X: np.ndarray
        input features of the training data shape (n_x, m)
        where n_x = number of inputs
                m = number of examples

    batch_size: int | None
        mini batch size (if None, then batch_size = m)

    shuffle: bool
        Shuffle data points
        Default = True

    random_state: int
        Random seed (set to make runs repeatable)
        Default = None

    Returns
    -------
    batches: list[tuple[int, ...]]
        A list of tuples of integers, where each tuple contains
        the indices of the training data for that batch, where the
        index is in the interval [1, m]
    """
    rng = np.random.default_rng(random_state)

    batches = []
    m = X.shape[1]
    if not batch_size:
        batch_size = m
    batch_size = min(batch_size, m)

    # Step 1: Shuffle the indices
    if shuffle:
        indices = list(rng.permutation(m))
    else:
        indices = np.arange(m)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / batch_size))
    k = 0
    for _ in range(num_complete_minibatches):
        mini_batch = indices[k * batch_size : (k + 1) * batch_size]
        if mini_batch:
            batches.append(tuple(mini_batch))
        k += 1

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch = indices[(k + 1) * batch_size :]
        if mini_batch:
            batches.append(tuple(mini_batch))

    return batches


def avg(array: np.ndarray) -> np.ndarray:
    """Compute training data mean.

    Parameters
    ----------
    array: np.ndarray
        Array of shape (n, m) where m is the number of examples.

    Returns
    -------
    mean: np.ndarray
        Array of shape (n, 1) representing mean along each dimension.
    """
    return np.mean(array, axis=1).reshape((-1, 1))


def std(array: np.ndarray) -> np.ndarray:
    """Compute training data standard deviation.

    Parameters
    ----------
    array: np.ndarray
        Array of shape (n, m) where m is the number of examples.

    Returns
    -------
    std: np.ndarray
        Array of shape (n, 1) representing std along each dimension.
    """
    return np.std(array, axis=1).reshape((-1, 1))


def _safe_divide(value: np.ndarray, eps: float = np.finfo(float).eps) -> np.ndarray:
    """Add small number to avoid dividing by zero."""
    mask = value == 0.0  # noqa: PLR2004
    value[mask] += eps
    return value


def normalize(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Center data about mean and normalize by standard deviation.

    Parameters
    ----------
    data: np.ndarray
        The data to be normalized. An array of shape (n, m)
        where m is the number of examples.

    mu: np.ndarray
        The mean of the data. An array of shape (n, 1).

    sigma: np.ndarray
        The standard deviation of the data. An array of shape (n, 1).
    """
    return (data - mu) / _safe_divide(sigma)


def denormalize(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Undo normalization.

    Parameters
    ----------
    data: np.ndarray
        Normalized data. An array of shape (n, m)
        where m is the number of examples.

    mu: np.ndarray
        The mean of the data. An array of shape (n, 1).

    sigma: np.ndarray
        The standard deviation of the data. An array of shape (n, 1).
    """
    return sigma * data + mu


def normalize_partials(
    partials: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray
) -> np.ndarray:
    """Normalize partials.

    Parameters
    ----------
    partials: np.ndarray
        The partials to be normalized. An array of shape (n_y, n_x, m)
        where n_x = number of inputs
              n_y = number of outputs
                m = number of examples

    sigma_x: np.ndarray
        The standard deviation of the inputs. An array of shape (n_x, 1).

    sigma_y: np.ndarray
        The standard deviation of the outputs. An array of shape (n_y, 1).
    """
    if partials is None:
        return partials
    n_y, n_x, _ = partials.shape
    sigma_x = sigma_x.T.reshape((1, n_x, 1))
    sigma_y = sigma_y.reshape((n_y, 1, 1))
    return partials * sigma_x / _safe_divide(sigma_y)


def denormalize_partials(
    partials: np.ndarray, sigma_x: np.ndarray, sigma_y: np.ndarray
) -> np.ndarray:
    """Undo normalization of partials.

    Parameters
    ----------
    partials: np.ndarray
        Normalized partials. An array of shape (n, m)
        where m is the number of examples.

    mu: np.ndarray
        The mean of the data. An array of shape (n, 1).

    sigma: np.ndarray
        The standard deviation of the data. An array of shape (n, 1).
    """
    n_y, n_x, _ = partials.shape
    sigma_x = sigma_x.T.reshape((1, n_x, 1))
    sigma_y = sigma_y.reshape((n_y, 1, 1))
    return partials * sigma_y / _safe_divide(sigma_x)


@dataclass
class Dataset:
    """Store training data and associated metadata for easy access.

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
    J: np.ndarray | None = None

    def __post_init__(self):  # noqa: D105
        if self.X.shape[1] != self.Y.shape[1]:
            msg = "X and Y must have the same number of examples"
            raise ValueError(msg)

        n_y, n_x, m = self.n_y, self.n_x, self.m

        if self.J is not None:
            if self.J.shape != (n_y, n_x, m):
                msg = f"J must be of shape ({n_y}, {n_x}, {m})"
                raise ValueError(msg)

    @property
    def m(self) -> int:
        """Return number of training examples."""
        return self.X.shape[1]

    @property
    def n_x(self) -> int:
        """Return number of inputs."""
        return self.X.shape[0]

    @property
    def n_y(self) -> int:
        """Return number of outputs."""
        return self.Y.shape[0]

    @cached_property
    def avg_x(self) -> np.ndarray:
        """Return mean of input data as array of shape (n_x, 1)."""
        return avg(self.X)

    @cached_property
    def avg_y(self) -> np.ndarray:
        """Return mean of output data as array of shape (n_y, 1)."""
        return avg(self.Y)

    @cached_property
    def std_x(self) -> np.ndarray:
        """Return standard dev of input data, array of shape (n_x, 1)."""
        return std(self.X)

    @cached_property
    def std_y(self) -> np.ndarray:
        """Return standard dev of output data, array of shape (n_y, 1)."""
        return std(self.Y)

    def mini_batches(
        self,
        batch_size: int | None,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> list[Self]:
        """Breakup data into multiple batches and return list of Datasets.

        Parameters
        ----------
        batch_size: int | None
            Number of examples to include in each batch. This
            number ultimate determines how many batches will
            be created.

        shuffle: bool, optional
            Shuffle the data before putting it into batches.
            Default is True

        random_state: int, optional
            Random seed to use for shuffling. Default is None.
        """
        X = self.X
        Y = self.Y
        J = self.J
        batches = mini_batches(X, batch_size, shuffle, random_state)
        if self.J is None:
            return [Dataset(X[:, b], Y[:, b]) for b in batches]
        return [Dataset(X[:, b], Y[:, b], J[:, :, b]) for b in batches]

    def normalize(self) -> Self:
        """Return normalized Dataset."""
        X_norm = normalize(self.X, self.avg_x, self.std_x)
        Y_norm = normalize(self.Y, self.avg_y, self.std_y)
        J_norm = normalize_partials(self.J, self.std_x, self.std_y)
        return Dataset(X_norm, Y_norm, J_norm)
