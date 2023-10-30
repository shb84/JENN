"""Goodness of fit metrics."""
import numpy as np 


def r_square(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute R-square value for each output."""
    axis = y_true.ndim - 1
    y_bar = np.mean(y_true, axis=axis)
    SSE = np.sum(np.square(y_pred - y_true), axis=axis)
    SSTO = np.sum(np.square(y_true - y_bar) + 1e-12, axis=axis)
    return 1 - SSE / SSTO