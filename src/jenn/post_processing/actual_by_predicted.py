import matplotlib.pyplot as plt
import numpy as np
from typing import List
from matplotlib.figure import Figure

from ._styling import MARKERS
from jenn.post_processing.metrics import r_square
        

def plot_actual_by_predicted(
    y_pred: np.ndarray | List[np.ndarray],
    y_true: np.ndarray | List[np.ndarray],
    response: str = "Response(s)",
    datasets: List[str] | None = None,
    figsize: tuple[float, float] = (3.25, 3),
    fontsize: int = 9,
    legend_fontsize: int = None,
    alpha: float = 0.75,
    colorful: bool = True, 
) -> Figure:
    """Create actual by predicted plot for a single response.

    .. note::
        If shape of input NumPy arrays is two dimensional or more, it will 
        be called with ravel(). For example, y.shape = (n_y, m) will become 
        y.shape = (n_y * m,) which implies all data will be melted together. 
        Individual responses will not be treated separately. This is helpful 
        to get an aggregate understand of the goodness of fit across the board. 
        If goodness of fit is desired for each response, then this method must 
        be called separately for each one, e.g. plot_actual_by_predicted(y_pred[i], y_true[i])

    :param y_pred: predicted values for each dataset, list of arrays of shape (m,)
    :param y_true: true values for each dataset, list of arrays of shape (m,)
    :param response: name of response shown
    :param datasets: names of datasets (e.g. ["train", "val", "test"])
    :param figsize: figure size
    :param fontsize: text size
    :param alpha: transparency of dots (between 0 and 1)
    :param colorful: distinguish datasets by different color or different marker
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
    
    # Instantiate lists 
    legend = [] 
    lower = []
    upper = []

    # Loop over datasets to overlay them in one plot (e.g. train, test)
    fig, ax = plt.subplots(figsize=figsize)
    markers = iter(MARKERS)
    for i, dataset in enumerate(datasets): 
        pred = y_pred[i].ravel()
        true = y_true[i].ravel()
        if colorful: 
            ax.scatter(true, pred, alpha=alpha)
        else: 
            ax.scatter(true, pred, alpha=alpha, color="k", marker=next(markers))
        r2 = r_square(pred, true).squeeze() 
        legend.append(f"{dataset} (" + r'$R^2$' + f"={r2:.2f})")
        lower.append(true.min())
        upper.append(true.max())
    
    # Add a perfect fit line to show deviations 
    legend.append("perfect fit line")
    line = [min(lower), max(upper)]
    ax.plot(line, line, color="k", linestyle=":")

    # Finish annotating axes 
    ax.set_xlabel(f"Actual {response}", fontsize=fontsize)
    ax.set_ylabel(f"Predicted {response}", fontsize=fontsize)
    ax.grid(True)
    ax.legend(legend, fontsize=legend_fontsize)

    plt.axis("equal")
    plt.close(fig)
    return fig 
