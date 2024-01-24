"""jenn module entry point."""
# Copyright (c) 2018 Steven H. Berguin
# Distributed under the terms of the MIT License.
import tomllib
from pathlib import Path

from . import core, model, synthetic, utils

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "core",
    "model",
    "utils",
    "synthetic",
]
