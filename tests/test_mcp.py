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


def test_jenn_root_env(tmp_path, monkeypatch):
    """`_jenn_root` honors $JENN_DIR and falls back to the CWD."""
    monkeypatch.setenv("JENN_DIR", str(tmp_path))
    assert server._jenn_root() == tmp_path.resolve()  # ruff:ignore[private-member-access]
    monkeypatch.delenv("JENN_DIR", raising=False)
    assert server._jenn_root() == Path.cwd().resolve()  # ruff:ignore[private-member-access]


def test_scan_files_discovers_data_and_models(tmp_path):
    """_scan_files lists CSV/NPZ data + exported models, skipping other files."""
    # comma CSV at the root
    (tmp_path / "a.csv").write_text("u,v,w\n1,2,3\n4,5,6\n")
    # ;-delimited CSV nested in a subdirectory (recursion + delimiter sniff)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.csv").write_text("p;q\n1;2\n3;4\n")
    # NPZ with named arrays
    np.savez(
        tmp_path / "c.npz",
        x=np.zeros((2, 3)),
        y=np.zeros((1, 3)),
        dydx=np.zeros((1, 2, 3)),
    )
    # a real exported JENN model, plus files that must be ignored
    rows, _ = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[8], max_iter=5, random_state=0)[
        "model_id"
    ]
    server.export(mid, path=str(tmp_path / "model.json"))
    (tmp_path / "other.json").write_text('{"hello": 1}')  # not a JENN model
    (tmp_path / "notes.txt").write_text("ignore me")

    result = server._scan_files(tmp_path)  # ruff:ignore[private-member-access]
    by_name = {entry["name"]: entry for entry in result["files"]}

    # The non-JENN JSON and the .txt are excluded.
    assert result["count"] == 4
    assert "other.json" not in by_name
    assert "notes.txt" not in by_name

    # CSV: columns + correctly sniffed delimiter; nested file found via recursion.
    assert by_name["a.csv"]["columns"] == ["u", "v", "w"]
    assert by_name["a.csv"]["delimiter"] == ","
    nested_name = str(Path("sub") / "b.csv")
    assert by_name[nested_name]["delimiter"] == ";"

    # NPZ lists its array names; model reports its architecture.
    assert by_name["c.npz"]["arrays"] == ["x", "y", "dydx"]
    assert by_name["model.json"]["kind"] == "model"
    assert by_name["model.json"]["layer_sizes"]


def test_scan_files_reads_fixture_columns():
    """The committed CSV fixture surfaces its column names for discovery."""
    result = server._scan_files(DATA)  # ruff:ignore[private-member-access]
    fixture = next(f for f in result["files"] if f["name"] == "rastrigin.csv")
    assert fixture["columns"] == ["x1", "x2", "y", "slope_wrt_x1"]


def _export_and_load(tmp_path, **train_kwargs):
    """Train -> export -> load_model; return (loaded_dict, original_model, x)."""
    rows, x = _rastrigin_rows()
    trained = server.train(**rows, **train_kwargs)
    path = tmp_path / "surrogate.json"
    server.export(trained["model_id"], path=str(path))
    loaded = server.load_model(str(path))
    original = server._MODELS.get(trained["model_id"]).model  # ruff:ignore[private-member-access]
    return loaded, original, x


def test_load_model_and_predict_roundtrip(tmp_path):
    """A reloaded model predicts identically to the original in-registry one."""
    loaded, original, x = _export_and_load(
        tmp_path,
        hidden_layers=[12, 12],
        max_iter=200,
        random_state=0,
    )
    assert loaded["n_inputs"] == 2
    assert loaded["n_outputs"] == 1
    assert loaded["source"].endswith("surrogate.json")

    # Inline row-per-sample x -> row-per-sample y that matches the original model.
    pred = server.predict(loaded["model_id"], x=x.T.tolist())
    assert np.allclose(np.array(pred["y"]), original.predict(x).T)


def test_predict_inline_with_partials():
    """Inline predict returns row-per-sample y, and dydx when requested."""
    rows, _ = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[12], max_iter=100, random_state=0)[
        "model_id"
    ]
    m = len(rows["x"])

    res = server.predict(mid, x=rows["x"])
    assert np.array(res["y"]).shape == (m, 1)  # (m, n_y)
    assert "dydx" not in res

    res = server.predict(mid, x=rows["x"], with_partials=True)
    assert np.array(res["y"]).shape == (m, 1)
    assert np.array(res["dydx"]).shape == (m, 1, 2)  # (m, n_y, n_x)


def test_predict_from_csv_and_npz_paths(tmp_path):
    """A CSV and an NPZ inputs file give the same result as inline arrays."""
    rows, x = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[12], max_iter=100, random_state=0)[
        "model_id"
    ]

    # CSV path: compare against inline on the identical fixture inputs.
    x_csv = jenn.utilities.load_csv_inputs(DATA / "rastrigin.csv", inputs=["x1", "x2"])
    from_csv = server.predict(
        mid, path=str(DATA / "rastrigin.csv"), inputs=["x1", "x2"]
    )
    inline_csv = server.predict(mid, x=x_csv.T.tolist())
    assert np.allclose(np.array(from_csv["y"]), np.array(inline_csv["y"]))

    # NPZ path: feature-first x on disk -> same as inline row-per-sample.
    npz = tmp_path / "inputs.npz"
    np.savez(npz, x=x)  # feature-first (n_x, m)
    from_npz = server.predict(mid, path=str(npz))
    inline_x = server.predict(mid, x=x.T.tolist())
    assert np.allclose(np.array(from_npz["y"]), np.array(inline_x["y"]))


def test_predict_output_path_writes_file(tmp_path):
    """output_path writes a file (feature-first NPZ / table CSV) and omits inline arrays."""
    rows, _ = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[12], max_iter=50, random_state=0)[
        "model_id"
    ]
    m = len(rows["x"])

    # CSV: response-only table; inline arrays omitted.
    csv_out = tmp_path / "pred.csv"
    res = server.predict(mid, x=rows["x"], output_path=str(csv_out))
    assert csv_out.exists()
    assert "y" not in res
    assert "dydx" not in res
    assert res["n_samples"] == m

    # NPZ with partials: feature-first x/y/dydx on disk.
    npz_out = tmp_path / "pred.npz"
    res = server.predict(mid, x=rows["x"], with_partials=True, output_path=str(npz_out))
    assert "y" not in res
    with np.load(npz_out) as arch:
        assert set(arch.files) == {"x", "y", "dydx"}
        assert arch["x"].shape == (2, m)  # feature-first
        assert arch["dydx"].shape == (1, 2, m)

    # CSV cannot hold 3-D partials -> steer to NPZ.
    with pytest.raises(ValueError, match=r"use \.npz to write partials"):
        server.predict(
            mid,
            x=rows["x"],
            with_partials=True,
            output_path=str(tmp_path / "bad.csv"),
        )


def test_predict_width_mismatch():
    """Inputs whose width != model n_inputs raise a clear error."""
    rows, _ = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[12], max_iter=20, random_state=0)[
        "model_id"
    ]
    with pytest.raises(ValueError, match="does not match model n_inputs"):
        server.predict(mid, x=[[0.1]])  # 1 feature, model expects 2


def test_load_model_error_paths(tmp_path):
    """Missing files and non-JENN JSON raise clear errors."""
    with pytest.raises(ValueError, match="No such file"):
        server.load_model(str(tmp_path / "nope.json"))

    bad = tmp_path / "bad.json"
    bad.write_text('{"hello": 1}')  # valid JSON, not a JENN model
    with pytest.raises(ValueError, match="not a valid JENN model file"):
        server.load_model(str(bad))


def test_evaluate_on_loaded_model(tmp_path):
    """A data-less loaded model errors on the training-data path, scores on holdout."""
    loaded, _, _ = _export_and_load(
        tmp_path,
        hidden_layers=[12],
        max_iter=100,
        random_state=0,
    )
    mid = loaded["model_id"]

    # No x/y: nothing to score against -> the new guard.
    with pytest.raises(ValueError, match="no training data"):
        server.evaluate(mid)

    # Holdout data -> scores normally (dimension check now uses layer_sizes).
    holdout, _ = _rastrigin_rows(m_levels=11)
    ev = server.evaluate(mid, **holdout)
    assert ev["dataset"] == "holdout"
    assert ev["metrics"]["response"]["r2_per_output"][0] > 0.5


def test_list_models_surfaces_loaded_model(tmp_path):
    """list_models tolerates data-less records (n_samples None) and shows source."""
    loaded, _, _ = _export_and_load(
        tmp_path,
        hidden_layers=[12],
        max_iter=20,
        random_state=0,
    )
    entry = next(
        m for m in server.list_models()["models"] if m["model_id"] == loaded["model_id"]
    )
    assert entry["n_samples"] is None
    assert entry["source"] == loaded["source"]


def test_resolve_path_anchors_relative_to_jenn_dir(tmp_path, monkeypatch):
    """A bare/relative path resolves under $JENN_DIR; absolute paths pass through."""
    monkeypatch.setenv("JENN_DIR", str(tmp_path))
    # relative -> under JENN_DIR
    assert (
        server._resolve_path("model.json")  # ruff:ignore[private-member-access]
        == (tmp_path / "model.json").resolve()
    )
    # absolute -> unchanged
    absolute = tmp_path / "sub" / "other.json"
    assert (
        server._resolve_path(str(absolute))  # ruff:ignore[private-member-access]
        == absolute.resolve()
    )
    # unset -> falls back to the CWD (backward compatible)
    monkeypatch.delenv("JENN_DIR", raising=False)
    assert (
        server._resolve_path("model.json")  # ruff:ignore[private-member-access]
        == (Path.cwd() / "model.json").resolve()
    )


def test_export_and_load_bare_name_under_jenn_dir(tmp_path, monkeypatch):
    """A bare filename is written and reloaded under $JENN_DIR (export/load_model)."""
    monkeypatch.setenv("JENN_DIR", str(tmp_path))
    rows, x = _rastrigin_rows()
    mid = server.train(**rows, hidden_layers=[8], max_iter=50, random_state=0)[
        "model_id"
    ]

    # a bare name lands in JENN_DIR (not the process CWD)
    res = server.export(mid, path="mymodel.json")
    assert res["path"] == str((tmp_path / "mymodel.json").resolve())
    assert (tmp_path / "mymodel.json").is_file()

    # and load_model finds it there by the same bare name
    loaded = server.load_model("mymodel.json")
    reloaded = server._MODELS.get(loaded["model_id"]).model  # ruff:ignore[private-member-access]
    original = server._MODELS.get(mid).model  # ruff:ignore[private-member-access]
    assert np.allclose(reloaded.predict(x), original.predict(x))


def test_ingest_bare_name_under_jenn_dir(tmp_path, monkeypatch):
    """A bare filename discovered under $JENN_DIR can be ingested."""
    monkeypatch.setenv("JENN_DIR", str(tmp_path))
    (tmp_path / "rastrigin.csv").write_text((DATA / "rastrigin.csv").read_text())
    out = server.ingest(
        "rastrigin.csv",
        inputs=["x1", "x2"],
        outputs=["y"],
        derivatives=[{"output": "y", "input": "x1", "column": "slope_wrt_x1"}],
    )
    assert out["n_inputs"] == 2
    assert out["n_outputs"] == 1
    assert out["n_samples"] == 100
