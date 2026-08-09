"""Model registry.
==================

In-memory store mapping opaque handles to trained models plus the metadata
needed to evaluate and export them. The stdio server is a single long-lived
process, so a module-level registry persists across tool calls.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from jenn.core.model import NeuralNet


@dataclass
class ModelRecord:
    """A trained model plus the context needed to evaluate/export it."""

    model: NeuralNet
    layer_sizes: list[int]
    hyperparameters: dict[str, Any]
    random_state: int | None
    x: np.ndarray  # feature-first training inputs  (n_x, m)
    y: np.ndarray  # feature-first training outputs (n_y, m)
    dydx: np.ndarray | None  # feature-first Jacobians (n_y, n_x, m) or None
    training_seconds: float  # wall-clock time spent in NeuralNet.fit
    handle: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class ModelRegistry:
    """Dict-backed registry of :class:`ModelRecord` keyed by handle."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}

    def add(self, record: ModelRecord) -> str:
        idx = record.handle
        self._records[idx] = record
        return idx

    def get(self, handle: str) -> ModelRecord:
        if handle in self._records:
            return self._records[handle]
        raise KeyError(f"{handle} not found in records.")

    def items(self) -> list[tuple[str, ModelRecord]]:
        return list(self._records.items())
