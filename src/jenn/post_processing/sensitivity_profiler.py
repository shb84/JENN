# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Callable, List, Tuple

from ._styling import LINE_STYLES


def sensitivity_profile(
    ax: plt.Axes,
    x0: np.ndarray,
    y0: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray | List[np.ndarray],
    x_true: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    alpha: float = 1.0,
    xlabel: str = "x",
    ylabel: str = "y",
    legend: List[str] | None = None,
    figsize: Tuple[float, float] = (6.5, 3),
    fontsize: int = 9,
    show_cursor: bool = True,
) -> Figure:
    """Plot sensitivity profile for a single input, single output.

    :param ax: the matplotlib axes on which to plot the data
    :param x0: point at which the profile is centered, array of shape
        (1,)
    :param y0: model evaluated as x0, list of arrays of shape (1,)
    :param x_pred: input values for prediction, an array of shape (m,)
    :param y_pred: predicted output values for each model, list of
        arrays of shape (m,)
    :param x_true: inputs value of actual data, array of shape (m, n_x)
    :param y_true: output values of actual data. An array of shape (m,)
    :param alpha: transparency of dots (between 0 and 1)
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param legend: legend name of each model
    :param figsize: figure size
    :param fontsize: text size
    :param show_cursor: show x0 as a red dot (or not)
    :return: matplotlib figure instance
    """
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
    ax.legend(legend, fontsize=fontsize)
    if show_cursor:
        for n in range(y0.size):
            ax.scatter(x0, y0[n], color="r")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True)
    plt.close(fig)
    return fig


def sensitivity_profiles(
    f: Callable | List[Callable],
    x_min: np.ndarray,
    x_max: np.ndarray,
    x0: np.ndarray | None = None,
    x_true: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    figsize: Tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    alpha: float = 1.0,
    title: str = "",
    xlabels: List[str] | None = None,
    ylabels: List[str] | None = None,
    legend: List[str] | None = None,
    resolution: int = 100,
    show_cursor: bool = True,
) -> Figure:
    """Plot grid of all outputs vs. all inputs evaluated at x0.

    :param f: callable function(s) for evaluating y_pred = f_pred(x)
    :param x0: point at which to evaluate profiles, array of shape (n_x,
        1)
    :param x_true: inputs at which y_true is evaluated, array of shape
        (n_x, m)
    :param y_true: true values, array of shape (n_y, m)
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param title: title of figure
    :param xlabels: x-axis labels
    :param ylabels: y-axis labels
    :param resolution: line resolution
    :param legend: legend labels for each model
    :param show_cursor: show x0 as a red dot (or not)
    """
    funcs = f
    if not isinstance(funcs, list):
        funcs = [funcs]
    x_min = x_min.ravel()
    x_max = x_max.ravel()
    if x0 is None:
        x0 = 0.5 * (x_min + x_max).reshape((-1, 1))
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
        for func in funcs:
            y_pred = func(x_pred)
            y_preds.append(y_pred)
        for j in y_indices:
            sensitivity_profile(
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
                show_cursor=show_cursor,
            )
        plt.close(fig)
    return fig