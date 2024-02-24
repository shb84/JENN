"""Training.
============

This class implements the core algorithm responsible for training the neural networks."""

import functools
from collections import defaultdict
from typing import Union

import numpy as np

from .cache import Cache
from .cost import Cost
from .data import Dataset
from .optimization import ADAM, Backtracking, Optimizer
from .parameters import Parameters
from .propagation import model_backward, model_partials_forward


def objective_function(
    X: np.ndarray,
    cost: Cost,
    parameters: Parameters,
    cache: Cache,
    stacked_params: np.ndarray,
) -> np.float64:
    """Evaluate cost function for training.

    :param X: training data inputs, array of shape (n_x, m)
    cost: cost function to be evaluated
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    stacked_params: neural network parameters returned by the optimizer,
        represented as single array of stacked parameters for all layers.
    """
    parameters.unstack(stacked_params)
    Y_pred, J_pred = model_partials_forward(X, parameters, cache)
    return cost.evaluate(Y_pred, J_pred)


def objective_gradient(
    data: Dataset,
    parameters: Parameters,
    cache: Cache,
    lambd: float,
    gamma: float,
    stacked_params: np.ndarray,
) -> np.ndarray:  # noqa: PLR0913
    """Evaluate cost function gradient for backprop.

    :param data: object containing training and associated metadata
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    :param lambd: coefficient that multiplies regularization term in
        cost function
    :param gamma: coefficient that multiplies jacobian-enhancement term
        in cost function
    :param stacked_params: neural network parameters returned by the
        optimizer, represented as single array of stacked parameters for
        all layers.
    """
    parameters.unstack(stacked_params)
    model_backward(data, parameters, cache, lambd, gamma)
    return parameters.stack_partials()


def train_model(
    data: Dataset,
    parameters: Parameters,
    alpha: float = 0.050,
    lambd: float = 0.000,
    gamma: float = 0.000,
    beta1: float = 0.900,
    beta2: float = 0.999,
    epochs: int = 1,
    max_iter: int = 200,
    batch_size: Union[int, None] = None,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
    is_backtracking: bool = False,
    is_verbose: bool = False,
) -> dict:  # noqa: PLR0913
    r"""Train neural net.

    :param data: object containing training and associated metadata
    :param parameters: object that stores neural net parameters for each
        layer
    :param alpha: learning rate :math:`\alpha`
    :param lambd: regularization term coefficient in cost function
    :param gamma: jacobian-enhancement term coefficient in cost function
    :param beta_1: exponential decay rate of 1st moment vector
        :math:`\beta_1\in[0, 1)`
    :param beta_2: exponential decay rate of 2nd moment vector
        :math:`\beta_2\in[0, 1)`
    :param epochs: number of passes through data
    :param batch_size: mini batch size (if None, single batch with all
        data)
    :param max_iter: maximum number of optimizer iterations allowed
    :param shuffle: swhether to huffle data points or not
    :param random_state: random seed (useful to make runs repeatable)
    :param is_backtracking: whether or not to use backtracking during
        line search
    :param is_verbose: print out progress for each iteration, each
        batch, each epoch
    :return: cost function training history accessed as `cost =
        history[epoch][batch][iter]`
    """
    history: dict[str, dict[str, Union[list[float], None]]] = defaultdict(dict)

    update = ADAM(beta1, beta2)
    line_search = Backtracking(update, max_count=is_backtracking * 1_000)
    optimizer = Optimizer(line_search)

    for e in range(epochs):
        batches = data.mini_batches(batch_size, shuffle, random_state)
        for b, batch in enumerate(batches):
            cache = Cache(parameters.layer_sizes, batch.m)
            cost = Cost(batch, parameters, lambd, gamma)
            func = functools.partial(
                objective_function,
                batch.X,
                cost,
                parameters,
                cache,
            )
            grad = functools.partial(
                objective_gradient,
                batch,
                parameters,
                cache,
                lambd,
                gamma,
            )
            optimizer.minimize(
                x=parameters.stack(),
                f=func,
                dfdx=grad,
                alpha=alpha,
                max_iter=max_iter,
                verbose=is_verbose,
                epoch=e,
                batch=b,
            )
            history[f"epoch_{e}"][f"batch_{b}"] = optimizer.cost_history
    return history
