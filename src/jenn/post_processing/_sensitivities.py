# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ._styling import LINE_STYLES


def _sensitivity_profile(
    ax: plt.Axes,
    x0: np.ndarray,
    y0: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray | list[np.ndarray],
    x_true: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    alpha: float = 1.0,
    xlabel: str = "x",
    ylabel: str = "y",
    legend_fontsize: list[str] | None = None,
    legend: list[str] | None = None,
    figsize: tuple[float, float] = (6.5, 3),
    fontsize: int = 9,
    show_cursor: bool = True,
) -> Figure:
    """Plot sensitivity profile for a single input, single output."""
    fig = plt.figure(figsize=figsize, layout="tight")
    if not ax:
        spec = fig.add_gridspec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0, 0])
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if legend is None:
        legend = []
    x0 = x0.ravel()
    y0 = y0.ravel()
    x_pred = x_pred.ravel()
    linestyles = iter(LINE_STYLES.values())
    for array in y_pred:
        linestyle = next(linestyles)
        ax.plot(x_pred, array.ravel(), color="k", linestyle=linestyle, linewidth=2)
    if x_true is not None and y_true is not None:
        x_true = x_true.ravel()
        y_true = y_true.ravel()
        ax.scatter(x_true, y_true, color="k", alpha=alpha)
        legend.append("data")
    if legend:
        ax.legend(legend, fontsize=legend_fontsize)
    if show_cursor:
        for n in range(y0.size):
            ax.scatter(x0, y0[n], color="r")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True)
    plt.close(fig)
    return fig


def plot_sensitivity_profiles(
    func: Callable | list[Callable],
    x_min: np.ndarray,
    x_max: np.ndarray,
    x0: np.ndarray | None = None,
    x_true: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
    legend_fontsize: int = 7,
    legend_label: str | list[str] | None = None,
    resolution: int = 100,
    show_cursor: bool = True,
) -> Figure:
    """Plot grid of all outputs vs. all inputs evaluated at x0.

    :param func: callable function(s) for evaluating y = func(x)
    :param x_min: lower bound, array of shape (n_x, 1)
    :param x_max: upper bound, array of shape (n_x, 1)
    :param x0: point of evaluation, array of shape (n_x, 1)
    :param x_true: true data inputs, array of shape (n_x, m)
    :param y_true: true data outputs, array of shape (n_y, m)
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param xlabels: x-axis labels
    :param ylabels: y-axis labels
    :param resolution: line resolution
    :param legend_fontsize: legend text size
    :param legend_label: legend labels for each model in func list
    :param show_cursor: show x0 as a red dot (or not)
    """
    funcs = [func] if not isinstance(func, list) else func
    legend = [legend_label] if isinstance(legend_label, str) else legend_label
    x_min = x_min.ravel()
    x_max = x_max.ravel()
    if x0 is None:
        x0 = 0.5 * (x_min + x_max)
    x0 = x0.reshape((-1, 1))
    y0 = np.concatenate([func(x0) for func in funcs], axis=1)
    n_x = x0.shape[0]
    n_y = y0.shape[0]
    x_indices = range(n_x)
    y_indices = range(n_y)
    xlabels = xlabels or [f"x_{i}" for i in x_indices]
    ylabels = ylabels or [f"y_{i}" for i in y_indices]
    width, height = figsize
    fig = plt.figure(figsize=(n_x * width, height), layout="tight")
    fig.suptitle(title)
    spec = fig.add_gridspec(ncols=n_x, nrows=n_y)
    for i in x_indices:
        x_pred = np.tile(x0, (1, resolution))
        x_pred[i] = np.linspace(x_min[i], x_max[i], resolution)
        y_preds = []
        for f in funcs:
            y_pred = f(x_pred)
            y_preds.append(y_pred)
        for j in y_indices:
            _sensitivity_profile(
                ax=fig.add_subplot(spec[j, i]),
                x0=x0[i],
                y0=y0[j],
                x_pred=x_pred[i],
                y_pred=[y_pred[j] for y_pred in y_preds],
                x_true=x_true[i] if x_true is not None else None,
                y_true=y_true[j] if y_true is not None else None,
                fontsize=fontsize,
                alpha=alpha,
                xlabel=xlabels[i],
                ylabel=ylabels[j],
                legend=legend,
                legend_fontsize=legend_fontsize,
                show_cursor=show_cursor,
            )
        plt.close(fig)
    return fig
