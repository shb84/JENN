"""Utils module entry point."""

# Copyright (c) 2018 Steven H. Berguin
# Distributed under the terms of the MIT License.

from importlib.util import find_spec

from . import metrics

__all__ = ["metrics"]

if find_spec("matplotlib"):
    __all__.append("plot")
