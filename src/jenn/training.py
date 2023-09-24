from __future__ import annotations
import functools
from collections import defaultdict
from .parameters import Parameters
from .data import Dataset
from .cost import Cost
from .propagation import model_backward, model_forward
from .cache import Cache
from .optimization import ADAM, Backtracking, Optimizer


def objective_function(X, cost, parameters, cache, stacked_params):
    parameters.unstack(stacked_params)
    Y_pred, J_pred = model_forward(X, parameters, cache)
    return cost.evaluate(Y_pred)


def objective_gradient(data, parameters, cache, lambd, stacked_params):
    parameters.unstack(stacked_params)
    model_backward(data, parameters, cache, lambd)
    return parameters.stack_partials()


def train_model(
        data: Dataset,
        parameters: Parameters,
        alpha: float = 0.100,
        lambd: float = 0.000,
        gamma: float = 0.000,
        beta1: float = 0.900,
        beta2: float = 0.999,
        epochs: int = 1,
        max_iter: int = 100,
        batch_size: int = None,
        shuffle: bool = True,
        random_state: int = None,
        is_backtracking=False,
        is_verbose=False,
) -> dict:
    """Train neural net."""
    history = defaultdict(dict)

    update = ADAM(beta1, beta2)
    line_search = Backtracking(update, max_count=is_backtracking * 1_000)
    optimizer = Optimizer(line_search)

    stacked_params = parameters.stack()

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
            )
            optimizer.minimize(
                x=stacked_params,
                f=func,
                dfdx=grad,
                alpha=alpha,
                max_iter=max_iter,
                verbose=is_verbose,
                epoch=e,
                batch=b,
            )
            history[f'epoch_{e}'][f'batch_{b}'] = optimizer.cost_history
    return history
