"""Test the file-based training-data loaders in jenn.utilities."""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jenn.utilities import load_csv, load_npz

DATA = Path(__file__).parent / "data"


def test_load_csv_builds_partial_mask():
    """A CSV with one of two partials present -> feature-first arrays + mask."""
    x, y, dydx, mask = load_csv(
        DATA / "rastrigin.csv",
        inputs=["x1", "x2"],
        outputs=["y"],
        derivatives=[{"output": "y", "input": "x1", "column": "slope_wrt_x1"}],
    )
    assert x.shape == (2, 100)  # feature-first (n_x, m)
    assert y.shape == (1, 100)  # feature-first (n_y, m)
    assert dydx.shape == (1, 2, 100)  # (n_y, n_x, m)
    assert mask.tolist() == [[1.0, 0.0]]  # only d(y)/d(x1) is present
    # The absent partial is a finite 0.0 placeholder, never NaN.
    assert np.isfinite(dydx).all()
    assert np.allclose(dydx[0, 1, :], 0.0)


def test_load_csv_without_derivatives():
    """No derivatives mapping -> values only, dydx and mask are None."""
    x, y, dydx, mask = load_csv(
        DATA / "rastrigin.csv",
        inputs=["x1", "x2"],
        outputs=["y"],
    )
    assert x.shape == (2, 100)
    assert y.shape == (1, 100)
    assert dydx is None
    assert mask is None


def test_load_csv_error_paths():
    """Unknown columns and malformed derivative specs raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        load_csv(DATA / "rastrigin.csv", inputs=["nope"], outputs=["y"])

    with pytest.raises(ValueError, match="not one of inputs"):
        load_csv(
            DATA / "rastrigin.csv",
            inputs=["x1", "x2"],
            outputs=["y"],
            derivatives=[{"output": "y", "input": "z", "column": "slope_wrt_x1"}],
        )


def test_load_npz_roundtrip_with_missing_layer(tmp_path):
    """An all-NaN partial layer becomes mask 0 with a finite placeholder."""
    rng = np.random.default_rng(0)
    x = rng.random((2, 5))
    y = rng.random((1, 5))
    dydx = rng.random((1, 2, 5))
    dydx[0, 1, :] = np.nan  # d(y)/d(x2) fully absent
    path = tmp_path / "data.npz"
    np.savez(path, x=x, y=y, dydx=dydx)

    lx, ly, ldydx, mask = load_npz(path)
    assert np.allclose(lx, x)
    assert np.allclose(ly, y)
    assert mask.tolist() == [[1.0, 0.0]]
    assert np.isfinite(ldydx).all()  # NaN placeholder replaced with 0.0
    assert np.allclose(ldydx[0, 0, :], dydx[0, 0, :])  # present layer untouched


def test_load_npz_without_dydx(tmp_path):
    """Archive with only x and y -> dydx and mask are None."""
    path = tmp_path / "vals.npz"
    np.savez(path, x=np.zeros((2, 3)), y=np.zeros((1, 3)))
    x, y, dydx, mask = load_npz(path)
    assert x.shape == (2, 3)
    assert dydx is None
    assert mask is None


def test_load_npz_rejects_partial_nan(tmp_path):
    """A partial that is NaN in only some samples is unsupported -> ValueError."""
    dydx = np.ones((1, 2, 4))
    dydx[0, 1, 0] = np.nan  # only one sample NaN -> ambiguous
    path = tmp_path / "mixed.npz"
    np.savez(path, x=np.zeros((2, 4)), y=np.zeros((1, 4)), dydx=dydx)
    with pytest.raises(ValueError, match="only some samples"):
        load_npz(path)
