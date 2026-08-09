"""Test goodness-of-fit metrics."""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import numpy as np

from jenn.post_processing.metrics import rsquare


def test_rsquare_perfect_fit_multioutput():
    """R² is 1.0 per output when prediction equals truth (n_y > 1)."""
    y = np.random.rand(3, 50)
    rr = rsquare(y, y)
    assert np.allclose(rr, 1.0)
    assert rr.shape == (3,)


def test_rsquare_handles_3d_partials():
    """R² reduces over the last axis for 3-D Jacobians -> shape (n_y, n_x)."""
    n_y = 2
    n_x = 4
    jac = np.random.rand(n_y, n_x, 10)
    rr = rsquare(jac, jac)
    assert rr.shape == (n_y, n_x)
    assert np.allclose(rr, 1.0)
