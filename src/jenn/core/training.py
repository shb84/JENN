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
from .optimization import ADAMOptimizer
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
    model_backward(data, parameters, cache, lambd)
    return parameters.stack_partials()


def train_model(
    data: Dataset,
    parameters: Parameters,
    alpha: float = 0.05,
    beta: Union[np.ndarray, float] = 1.0,
    gamma: Union[np.ndarray, float] = 1.0,
    lambd: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 0.5,
    tol: float = 1e-12,
    max_count: int = 1000,
    epsilon_absolute: float = 1e-12,
    epsilon_relative: float = 1e-12,
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
    :param beta: LSE coefficients [defaulted to one] (optional)
    :param gamma: jacobian-enhancement regularization coefficient [defaulted to zero] (optional)
    :param lambd: regularization coefficient to avoid overfitting [defaulted to zero] (optional)
    :param beta_1: exponential decay rate of 1st moment vector
        :math:`\beta_1\in[0, 1)`
    :param beta_2: exponential decay rate of 2nd moment vector
        :math:`\beta_2\in[0, 1)`
    :param tau: amount by which to reduce :math:`\alpha := \tau \times
        \alpha` on each iteration
    :param tol: stop when cost function doesn't improve more than
        specified tolerance
    :param max_count: stop when line search iterations exceed maximum
        count specified
    :param epsilon_absolute: absolute error stopping criterion
    :param epsilon_relative: relative error stopping criterion
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
    history: dict[str, dict[str, Union[list[np.ndarray], None]]] = defaultdict(dict)
    optimizer = ADAMOptimizer(
        beta_1=beta1,
        beta_2=beta2,
        tau=tau,
        tol=tol,
        max_count=is_backtracking * max_count,
    )
    data.set_weights(beta, gamma)
    for e in range(epochs):
        batches = data.mini_batches(batch_size, shuffle, random_state)
        for b, batch in enumerate(batches):
            cache = Cache(parameters.layer_sizes, batch.m)
            cost = Cost(batch, parameters, lambd)
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
            )
            optimizer.minimize(
                x=parameters.stack(),
                f=func,
                dfdx=grad,
                alpha=alpha,
                max_iter=max_iter,
                epsilon_absolute=epsilon_absolute,
                epsilon_relative=epsilon_relative,
                verbose=is_verbose,
                epoch=e,
                batch=b,
            )
            history[f"epoch_{e}"][f"batch_{b}"] = optimizer.cost_history
    return history
