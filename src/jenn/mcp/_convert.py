"""Array conversion.
===================

Translate between the row-per-sample JSON payloads exchanged over MCP
and the feature-first arrays used internally by :mod:`jenn`.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import numpy as np


def to_feature_first(values: list[list[float]], name: str) -> np.ndarray:
    r"""Convert a row-per-sample matrix to a feature-first array.

    :param values: shape ``(m, n)`` -- ``m`` samples of ``n`` features.
        A flat list of length ``m`` is treated as ``m`` single-feature
        samples.
    :param name: label used in error messages (e.g. ``"x"`` or ``"y"``)
    :return: array of shape ``(n, m)``
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    if array.ndim != 2:
        msg = f"`{name}` must be a 2-D (n_samples, n_features) array."
        raise ValueError(msg)
    if array.size == 0:
        msg = f"`{name}` is empty; provide at least one sample."
        raise ValueError(msg)
    return array.T


def to_feature_first_partials(
    values: list[list[list[float]]],
    n_y: int,
    n_x: int,
    name: str = "dydx",
) -> np.ndarray:
    r"""Convert row-per-sample Jacobians to a feature-first array.

    :param values: shape ``(m, n_y, n_x)`` -- one Jacobian per sample (rows =
        outputs, cols = inputs). Broadcast when unambiguous: ``(m,)`` -> read
        as ``(m, 1, 1)``; ``(m, n_x)`` -> read as ``(m, 1, n_x)``.
    :param n_y: expected number of outputs (from ``y``)
    :param n_x: expected number of inputs (from ``x``)
    :param name: label used in error messages
    :return: array of shape ``(n_y, n_x, m)``
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:  # (m,) -> one output, one input per sample
        array = array.reshape((-1, 1, 1))
    elif array.ndim == 2:  # (m, n_x) -> single output per sample
        array = array.reshape((array.shape[0], 1, array.shape[1]))
    if array.ndim != 3:
        msg = (
            f"`{name}` must have shape (n_samples, n_outputs, n_inputs); "
            f"got an array with {array.ndim} dimensions."
        )
        raise ValueError(msg)
    m, got_y, got_x = array.shape
    if (got_y, got_x) != (n_y, n_x):
        msg = (
            f"`{name}` Jacobian shape {(got_y, got_x)} does not match the data: "
            f"expected (n_outputs, n_inputs) = {(n_y, n_x)}."
        )
        raise ValueError(msg)
    return np.transpose(array, (1, 2, 0))


def prepare_training_data(
    x: list[list[float]],
    y: list[list[float]],
    dydx: list[list[list[float]]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Validate and convert a full training payload to feature-first arrays.

    :return: tuple ``(X, Y, J)`` with shapes ``(n_x, m)``, ``(n_y, m)``
        and ``(n_y, n_x, m)`` (or ``J`` is ``None`` when ``dydx`` is
        ``None``).
    """
    inputs = to_feature_first(x, "x")  # (n_x, m)
    outputs = to_feature_first(y, "y")  # (n_y, m)
    if inputs.shape[1] != outputs.shape[1]:
        msg = (
            f"`x` and `y` disagree on sample count: "
            f"x has {inputs.shape[1]}, y has {outputs.shape[1]}."
        )
        raise ValueError(msg)
    partials = None
    if dydx is not None:
        n_x, m = inputs.shape
        n_y = outputs.shape[0]
        partials = to_feature_first_partials(dydx, n_y, n_x)
        if partials.shape[2] != m:
            msg = f"`dydx` has {partials.shape[2]} samples but `x` has {m}."
            raise ValueError(msg)
    return inputs, outputs, partials
