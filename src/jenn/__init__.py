"""Jenn module entry point."""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from . import core, synthetic, utils  # TODO: remove
from .core.model import NeuralNet
from .post_processing import (
    plot_actual_by_predicted,
    plot_contours,
    plot_convergence,
    plot_goodness_of_fit,
    plot_histogram,
    plot_residual_by_predicted,
    plot_sensitivity_profiles,
)

__version__ = "1.0.9.dev0"

__all__ = [
    "NeuralNet",
    "__version__",
    "core",
    "plot_actual_by_predicted",
    "plot_contours",
    "plot_convergence",
    "plot_goodness_of_fit",
    "plot_histogram",
    "plot_residual_by_predicted",
    "plot_sensitivity_profiles",
    "synthetic",
    "utils",
]
