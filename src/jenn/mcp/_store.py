"""Registries.
==============

In-memory stores mapping opaque handles to trained models and to
ingested datasets, plus the metadata needed to evaluate, export, and
train from them. The stdio server is a single long-lived process, so
module-level registries persist across tool calls.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import numpy as np

    from jenn.core.model import NeuralNet


@dataclass(kw_only=True)
class Record:
    """Base for anything a :class:`Registry` holds: carries the opaque handle."""

    handle: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(kw_only=True)
class ModelRecord(Record):
    """A trained model plus the context needed to evaluate/export it."""

    model: NeuralNet
    layer_sizes: list[int]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_state: int | None = None
    x: np.ndarray | None = None  # feature-first training inputs  (n_x, m)
    y: np.ndarray | None = None  # feature-first training outputs (n_y, m)
    dydx: np.ndarray | None = None  # feature-first Jacobians (n_y, n_x, m) or None
    training_seconds: float  # wall-clock time spent in NeuralNet.fit
    partial_mask: np.ndarray | None = None  # (n_y, n_x) availability mask or None
    input_names: list[str] | None = None  # column names for inputs, if known
    output_names: list[str] | None = None  # column names for outputs, if known
    source: str | None = (
        None  # provenance, e.g. the ingested file path (if model reloaded)
    )


@dataclass(kw_only=True)
class DatasetRecord(Record):
    """Ingested training data plus the metadata needed to train from it."""

    x: np.ndarray  # feature-first inputs  (n_x, m)
    y: np.ndarray  # feature-first outputs (n_y, m)
    dydx: (
        np.ndarray | None
    )  # feature-first Jacobians (n_y, n_x, m), 0-filled where absent
    partial_mask: np.ndarray | None  # (n_y, n_x) of {0, 1}; None iff dydx is None
    input_names: list[str] | None  # column names for inputs, or None (positional)
    output_names: list[str] | None  # column names for outputs, or None (positional)
    source: str  # provenance, e.g. the ingested file path


R = TypeVar("R", bound=Record)


class Registry(Generic[R]):
    """Dict-backed registry of :class:`Record` values keyed by handle."""

    def __init__(self) -> None:
        self._records: dict[str, R] = {}

    def add(self, record: R) -> str:
        idx = record.handle
        self._records[idx] = record
        return idx

    def get(self, handle: str) -> R:
        if handle in self._records:
            return self._records[handle]
        raise KeyError(f"{handle} not found in records.")

    def items(self) -> list[tuple[str, R]]:
        return list(self._records.items())
