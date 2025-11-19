"""Radial Basis Function."""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def rbf(
    r: np.ndarray,
    epsilon: float = 0.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Gaussian Radial Basis Function (RBF).

    :param r: radius from center of RBF
    :param epsilon: hyperparameter
    """
    return np.exp(-((max(0.0, epsilon) * r) ** 2), out=out)
