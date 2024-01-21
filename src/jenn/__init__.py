"""jenn module entry point."""
# Copyright (c) 2018 Steven H. Berguin
# Distributed under the terms of the MIT License.
import tomllib
from pathlib import Path

from . import core, model, synthetic, utils

JENN = Path(__file__).parent
SRC = JENN.parent
ROOT = SRC.parent
PPT = ROOT / "pyproject.toml"
PPT_DATA = tomllib.loads(PPT.read_text(encoding="utf-8"))

__version__ = PPT_DATA["project"]["version"]

__all__ = [
    "__version__",
    "core",
    "model",
    "utils",
    "synthetic",
]
