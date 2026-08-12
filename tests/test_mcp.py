"""Test the JENN MCP server tools."""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jenn
from jenn.synthetic_data import rastrigin

pytest.importorskip("mcp")  # skip this whole module if the mcp extra isn't installed

from jenn.mcp import server  # import after the importorskip guard above

DATA = Path(__file__).parent / "data"


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
    original = server._MODELS.get(out["model_id"]).model  # ruff:ignore[private-member-access]
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


def _ingest_rastrigin():
    """Ingest the CSV fixture with only d(y)/d(x1) present (x2 partial absent)."""
    return server.ingest(
        str(DATA / "rastrigin.csv"),
        inputs=["x1", "x2"],
        outputs=["y"],
        derivatives=[{"output": "y", "input": "x1", "column": "slope_wrt_x1"}],
    )


def test_ingest_partial_mask_and_train():
    """Ingest reports the missing partial; train masks it without NaNs."""
    ing = _ingest_rastrigin()
    assert ing["partials_available"] == [["y", "x1"]]
    assert ing["partials_missing"] == [["y", "x2"]]
    assert ing["n_samples"] == 100

    out = server.train(
        dataset_id=ing["dataset_id"],
        hidden_layers=[12],
        max_iter=100,
        random_state=0,
    )
    assert out["dataset_id"] == ing["dataset_id"]
    partials = out["training_metrics"]["partials"]
    # The absent partial is reported as ignored, never scored.
    assert partials["ignored"] == [{"output": "y", "input": "x2"}]
    assert len(partials["available"]) == 1
    # The mask nullifies the placeholder without poisoning the loss (no NaN).
    assert np.isfinite(partials["available"][0]["r2"])
    assert partials["r2_min"] is not None
    assert np.isfinite(partials["r2_min"])


def test_dataset_id_round_trip_and_mutual_exclusivity():
    """train/evaluate by dataset_id; passing both id and inline arrays errors."""
    ing = _ingest_rastrigin()
    out = server.train(dataset_id=ing["dataset_id"], hidden_layers=[12], max_iter=50)

    ev = server.evaluate(out["model_id"], dataset_id=ing["dataset_id"])
    assert ev["dataset"] == f"dataset:{ing['dataset_id']}"
    assert ev["metrics"]["partials"]["ignored"] == [{"output": "y", "input": "x2"}]

    # dataset_id + inline arrays together is ambiguous -> error.
    with pytest.raises(ValueError, match="not both"):
        server.train(x=[[0.0, 0.0]], y=[[0.0]], dataset_id=ing["dataset_id"])

    # list_datasets exposes the ingested dataset's metadata.
    listed = server.list_datasets()
    ids = [d["dataset_id"] for d in listed["datasets"]]
    assert ing["dataset_id"] in ids
    entry = next(d for d in listed["datasets"] if d["dataset_id"] == ing["dataset_id"])
    assert entry["has_partials"] is True
    assert entry["partials_missing"] == [["y", "x2"]]


def test_named_gamma_override():
    """Per-partial gamma overrides need a named dataset; inline data is rejected."""
    ing = _ingest_rastrigin()
    # Valid: boost the one available partial on a named dataset.
    out = server.train(
        dataset_id=ing["dataset_id"],
        hidden_layers=[12],
        max_iter=50,
        random_state=0,
        gamma=[{"output": "y", "input": "x1", "weight": 3.0}],
    )
    assert out["hyperparameters"]["gamma"] == [
        {"output": "y", "input": "x1", "weight": 3.0},
    ]

    # Same override on inline (nameless) data -> error.
    with pytest.raises(ValueError, match="require a dataset ingested with"):
        server.train(
            x=[[0.0, 0.0]],
            y=[[0.0]],
            dydx=[[[0.1, 0.2]]],
            gamma=[{"output": "y", "input": "x1", "weight": 3.0}],
        )


def test_ingest_npz_roundtrip(tmp_path):
    """NPZ ingest with an all-NaN partial layer -> masked, finite, trainable."""
    rng = np.random.default_rng(0)
    x = rng.random((2, 20))
    y = rng.random((1, 20))
    dydx = rng.random((1, 2, 20))
    dydx[0, 1, :] = np.nan  # d(y)/d(x2) absent
    path = tmp_path / "data.npz"
    np.savez(path, x=x, y=y, dydx=dydx)

    ing = server.ingest(str(path))
    assert ing["source"].startswith("npz:")
    # No column names for NPZ -> partials are labelled positionally.
    assert ing["partials_available"] == [[0, 0]]
    assert ing["partials_missing"] == [[0, 1]]

    out = server.train(dataset_id=ing["dataset_id"], max_iter=20, random_state=0)
    assert np.isfinite(out["training_metrics"]["partials"]["r2_min"])
