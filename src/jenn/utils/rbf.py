"""Radial Basis Function."""
from typing import Union

import numpy as np


def rbf(
    r: np.ndarray,
    epsilon: float = 0.0,
    out: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Compute Gaussian Radial Basis Function (RBF).

    :param r: radius from center of RBF
    :param epsilon: hyperparameter
    """
    return np.exp(-((max(0.0, epsilon) * r) ** 2), out=out)
