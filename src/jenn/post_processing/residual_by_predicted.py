import matplotlib.pyplot as plt
import numpy as np
from typing import List
from matplotlib.figure import Figure

from ._styling import MARKERS


def plot_residual_by_predicted(
    y_pred: np.ndarray | List[np.ndarray],
    y_true: np.ndarray | List[np.ndarray],
    response: str = "Response(s)",
    datasets: List[str] | None = None,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = None,
    alpha: float = 0.75,
    percent: bool = False,
    colorful: bool = True, 
) -> Figure:
    """Create residual by predicted plot for a single response.

    .. note::
        If shape of input NumPy arrays is two dimensional or more, it will 
        be called with ravel(). For example, y.shape = (n_y, m) will become 
        y.shape = (n_y * m,) which implies all data will be melted together. 
        Individual responses will not be treated separately. This is helpful 
        to get an aggregate understand of the goodness of fit across the board. 
        If goodness of fit is desired for each response, then this method must 
        be called separately for each one, e.g. plot_residual_by_predicted(y_pred[i], y_true[i])

    :param y_pred: predicted values for each dataset, list of arrays of shape (m,)
    :param y_true: true values for each dataset, list of arrays of shape (m,)
    :param response: name of response shown
    :param datasets: names of datasets (e.g. ["train", "val", "test"])
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param colorful: distinguish datasets by different color or different marker
    :param percent: show residuals as percentages 
    :return: matplotlib Figure instance
    """
    if not legend_fontsize: 
        legend_fontsize = fontsize 

    # Sanity check inputs 
    if isinstance(y_pred, np.ndarray): 
        y_pred = [y_pred]
    if isinstance(y_true, np.ndarray): 
        y_true = [y_true]
    if not isinstance(y_pred, list): 
        raise ValueError("ypred must be a list of arrays, e.g. [y_train_pred, y_test_pred]")
    if not isinstance(y_true, list): 
        raise ValueError("ytrue must be a list of arrays, e.g. [y_train_true, y_test_true]")
    if len(y_true) != len(y_pred): 
        raise ValueError("y_true and y_pred must have same length")
    if datasets is None: 
        datasets = ["data"] * len(y_true)
    if len(y_true) != len(datasets): 
        raise ValueError("y_true and y_pred must have same length as datasets")

    # Loop over datasets to overlay them in one plot (e.g. train, test)
    fig, ax = plt.subplots(figsize=figsize)
    legend = [] 
    markers = iter(MARKERS)
    for i, dataset in enumerate(datasets): 
        pred = y_pred[i].ravel()
        true = y_true[i].ravel()
        diff = (pred - true) / true * 100 if percent else pred - true
        if colorful: 
            ax.scatter(pred, diff, alpha=alpha)
        else: 
            ax.scatter(pred, diff, alpha=alpha, color="k", marker=next(markers))
        legend.append(dataset)

    # Add statistics 
    true = np.concatenate([y.ravel() for y in y_true]).ravel()
    pred = np.concatenate([y.ravel() for y in y_pred]).ravel()
    diff = (pred - true) / true * 100 if percent else pred - true
    avg = diff.mean()
    std = diff.std() 
    ax.axhline(y=avg, color="k", linestyle="-", linewidth=2)
    ax.axhline(y=avg + std, color="k", linestyle=":", linewidth=2)
    ax.axhline(y=avg - std, color="k", linestyle=":", linewidth=2)
    legend.extend([f"avg = {avg:.3f}", f"std = {std:.3f}"])

    # Finish annotating axes 
    ax.set_xlabel(f"Predicted {response}", fontsize=fontsize)
    ax.set_ylabel("Residuals (%)" if percent else "Residuals", fontsize=fontsize)
    ax.grid(True)
    ax.legend(legend, fontsize=legend_fontsize)

    plt.close(fig)
    return fig 
