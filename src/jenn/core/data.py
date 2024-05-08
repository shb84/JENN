"""Data.
========

This module contains convenience utilities to 
manage and handle training data. 
"""  # noqa: W291

import math
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple, Union

import numpy as np


def mini_batches(
    X: np.ndarray,
    batch_size: Union[int, None],
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> List[Tuple[int, ...]]:
    r"""Create randomized mini-batches.

    :param X: training data input :math:`X\in\mathbb{R}^{n_x\times m}`
    :param batch_size: mini batch size (if None, single batch with all
        data)
    :param shuffle: swhether to huffle data points or not
    :param random_state: random seed (useful to make runs repeatable)
    :return: list of tuples containing training data indices allocated
        to each batch
    """
    rng = np.random.default_rng(random_state)

    batches = []
    m = X.shape[1]
    if not batch_size:
        batch_size = m
    batch_size = min(batch_size, m)

    # Step 1: Shuffle the indices
    indices: list[int] = list(rng.permutation(m)) if shuffle else np.arange(m).tolist()

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
    """Compute mean and reshape as column array.

    :param array: array of shape (-1, m)
    :return: column array corresponding to mean of each row
    """
    return np.mean(array, axis=1).reshape((-1, 1))


def std(array: np.ndarray) -> np.ndarray:
    """Compute standard deviation and reshape as column array.

    :param array: array of shape (-1, m)
    :return: column array corresponding to std dev of each row
    """
    return np.std(array, axis=1).reshape((-1, 1))


def _safe_divide(
    value: np.ndarray, eps: float = float(np.finfo(float).eps)
) -> np.ndarray:
    """Add small number to avoid dividing by zero."""
    mask = value == 0.0  # noqa: PLR2004
    value[mask] += eps
    return value


def normalize(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Center data about mean and normalize by standard deviation.

    :param data: data to be normalized, array of shape (-1, m)
    :param mu: mean of the data, array of shape (-1, 1)
    :param sigma: std deviation of the data, array of shape (-1, 1)
    :return: normalized data, array of shape (-1, m)
    """
    return (data - mu) / _safe_divide(sigma)


def denormalize(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Undo normalization.

    :param data: normalized data, array of shape (-1, m)
    :param mu: mean of the data, array of shape (-1, 1)
    :param sigma: std deviation of the data, array of shape (-1, 1)
    :return: denormalized data, array of shape (-1, m)
    """
    return sigma * data + mu


def normalize_partials(
    partials: Union[np.ndarray, None], sigma_x: np.ndarray, sigma_y: np.ndarray
) -> Union[np.ndarray, None]:
    r"""Normalize partials.

    :param partials: training data partials to be normalized
        :math:`J\in\mathbb{R}^{n_y\times n_x \times m}`
    :param sigma_x: std dev of training data factors :math:`\sigma_x`,
        array of shape (-1, 1)
    :param sigma_y: std dev of training data responses :math:`\sigma_y`,
        array of shape (-1, 1)
    :return: normalized partials, array of shape (n_y, n_x, m)
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
    r"""Undo normalization of partials.

    :param partials: normalized training data partials
        :math:`\bar{J}\in\mathbb{R}^{n_y\times n_x \times m}`
    :param sigma_x: std dev of training data factors :math:`\sigma_x`,
        array of shape (-1, 1)
    :param sigma_y: std dev of training data responses :math:`\sigma_y`,
        array of shape (-1, 1)
    :return: denormalized partials, array of shape (n_y, n_x, m)
    """
    n_y, n_x, _ = partials.shape
    sigma_x = sigma_x.T.reshape((1, n_x, 1))
    sigma_y = sigma_y.reshape((n_y, 1, 1))
    return partials * sigma_y / _safe_divide(sigma_x)


@dataclass
class Dataset:
    """Store training data and associated metadata for easy access.

    :param X: training data outputs, array of shape (n_x, m)
    :param Y: training data outputs, array of shape (n_y, m)
    :param J: training data Jacobians, array of shape (n_y, n_x, m)
    """

    X: np.ndarray
    Y: np.ndarray
    J: Union[np.ndarray, None] = None

    Y_weights: Union[np.ndarray, float] = 1.0
    J_weights: Union[np.ndarray, float] = 1.0

    def __post_init__(self) -> None:  # noqa: D105
        if self.X.shape[1] != self.Y.shape[1]:
            msg = "X and Y must have the same number of examples"
            raise ValueError(msg)

        n_y, n_x, m = self.n_y, self.n_x, self.m

        self.Y_weights = self.Y_weights * np.ones((n_y, m))
        self.J_weights = self.J_weights * np.ones((n_y, n_x, m))

        if self.J is not None:
            if self.J.shape != (n_y, n_x, m):
                msg = f"J must be of shape ({n_y}, {n_x}, {m})"
                raise ValueError(msg)

    def set_weights(
        self,
        beta: Union[np.ndarray, float] = 1.0,
        gamma: Union[np.ndarray, float] = 1.0,
    ) -> None:
        """Prioritize certain points more than others.

        Rational: this can be used to reward the optimizer more in certain regions.

        :param beta: multiplier(s) on Y
        :param beta: multiplier(s) on J
        """
        self.Y_weights = beta * np.ones((self.n_y, self.m))
        self.J_weights = gamma * np.ones((self.n_y, self.n_x, self.m))

    @property
    def m(self) -> int:
        """Return number of training examples."""
        return int(self.X.shape[1])

    @property
    def n_x(self) -> int:
        """Return number of inputs."""
        return int(self.X.shape[0])

    @property
    def n_y(self) -> int:
        """Return number of outputs."""
        return int(self.Y.shape[0])

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
        batch_size: Union[int, None],
        shuffle: bool = True,
        random_state: Union[int, None] = None,
    ) -> List["Dataset"]:
        """Breakup data into multiple batches and return list of Datasets.

        :param batch_size: mini batch size (if None, single batch with
            all data)
        :param shuffle: swhether to huffle data points or not
        :param random_state: random seed (useful to make runs
            repeatable)
        :return: list of Dataset representing data broken up in batches
        """
        X = self.X
        Y = self.Y
        J = self.J
        Y_weights = np.ones(Y.shape) * self.Y_weights
        batches = mini_batches(X, batch_size, shuffle, random_state)
        if J is None:
            return [
                Dataset(X[:, b], Y[:, b], Y_weights=Y_weights[:, b]) for b in batches
            ]
        J_weights = np.ones(J.shape) * self.J_weights
        return [
            Dataset(X[:, b], Y[:, b], J[:, :, b], Y_weights[:, b], J_weights[:, :, b])
            for b in batches
        ]

    def normalize(self) -> "Dataset":
        """Return normalized Dataset."""
        X_norm = normalize(self.X, self.avg_x, self.std_x)
        Y_norm = normalize(self.Y, self.avg_y, self.std_y)
        J_norm = normalize_partials(self.J, self.std_x, self.std_y)
        return Dataset(X_norm, Y_norm, J_norm)
