"""Test the JENN MCP server tools."""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import numpy as np
import pytest

import jenn
from jenn.synthetic_data import rastrigin

pytest.importorskip("mcp")  # skip this whole module if the mcp extra isn't installed

from jenn.mcp import server  # import after the importorskip guard above


def _rastrigin_rows(m_levels: int = 10):
    """Rastrigin data as row-per-sample lists (what an agent sends) + native arrays."""
    x, y, dydx = jenn.utilities.sample(
        f=rastrigin.compute,
        f_prime=rastrigin.compute_partials,
        m_random=0,
        m_levels=m_levels,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    rows = {
        "x": x.T.tolist(),
        "y": y.T.tolist(),
        "dydx": np.transpose(dydx, (2, 0, 1)).tolist(),
    }
    return rows, x  # native x is handy for a predict() comparison


def test_train_evaluate_export_roundtrip(
    tmp_path,
):  # tmp_path is a pytest built-in fixture
    """The worked-example loop: train -> evaluate holdout -> export -> reload."""
    # Training
    rows, x = _rastrigin_rows()
    out = server.train(**rows, hidden_layers=[12, 12], max_iter=400, random_state=0)
    assert out["training_metrics"]["response"]["r2_per_output"][0] > 0.9
    assert out["training_seconds"] > 0

    # Evaluation
    holdout, _ = _rastrigin_rows(m_levels=13)
    e = server.evaluate(out["model_id"], **holdout)
    assert e["dataset"] == "holdout"
    assert e["metrics"]["response"]["r2_per_output"][0] > 0.9, e["metrics"]

    # Saving and reloading
    path = tmp_path / "surrogate.json"
    res = server.export(out["model_id"], path=str(path))
    assert path.exists()
    reloaded = jenn.NeuralNet.load(res["path"])
    original = server._REGISTRY.get(out["model_id"]).model  # ruff:ignore[private-member-access]
    assert np.allclose(reloaded.predict(x), original.predict(x))


def test_evaluate_data_source_and_guards():
    """Training-data default, the XOR error, and the dim-mismatch error."""
    rows, _ = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[12], max_iter=100, random_state=0)[
        "model_id"
    ]

    # No x/y -> scores the stored training data (by reference, no re-send).
    assert server.evaluate(mid)["dataset"] == "training"

    # x without y is ambiguous -> error.
    with pytest.raises(ValueError, match="Provide both x and y"):
        server.evaluate(mid, x=rows["x"])

    # Holdout with the wrong number of inputs (model expects n_x=2) -> dim mismatch.
    with pytest.raises(ValueError, match="Data shape mismatch"):
        server.evaluate(mid, x=[[0.1]], y=[[0.2]])


def test_convert_error_paths():
    """prepare_training_data rejects inconsistent shapes before any training."""
    # x has 1 sample, y has 2 -> sample-count mismatch.
    with pytest.raises(ValueError, match="disagree on sample count"):
        server.prepare_training_data([[1.0, 2.0]], [[1.0], [2.0]], None)

    # Jacobian says n_x=3 but x has n_x=2 -> shape mismatch.
    with pytest.raises(ValueError, match="does not match"):
        server.prepare_training_data([[0.0, 0.0]], [[0.0]], [[[0.0, 0.0, 0.0]]])
