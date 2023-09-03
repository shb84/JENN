from __future__ import annotations
import functools
from collections import defaultdict
from .parameters import Parameters
from .data import Dataset
from .mini_batch import mini_batches
from .cost import Cost
from .propagation import model_backward, model_forward
from .cache import Cache
from .finite_difference import finite_diff
from .optimization import ADAM, Backtracking, Optimizer


def train_model(
        data: Dataset,
        parameters: Parameters,
        alpha: float = 0.100,
        lambd: float = 0.000,
        gamma: float = 0.000,
        beta1: float = 0.900,
        beta2: float = 0.999,
        num_epochs: int = 1,
        max_iter: int = 100,
        batch_size: int = None,
        shuffle: bool = True,
        random_state: int = None,
        is_backtracking=False,
        is_verbose=False,
) -> dict:
    """Train neural net."""
    history = defaultdict(dict)
    cache = Cache(parameters.layer_sizes, data.m)
    cost = Cost(data, parameters, lambd, gamma)

    def objective_function(params, batch):
        parameters.unstack(params)
        Y_pred = model_forward(data.X, parameters, cache, batch)
        return cost.evaluate(Y_pred, batch=batch)

    def objective_gradient(params, batch):
        parameters.unstack(params)
        model_backward(data, parameters, cache, lambd, batch)
        return parameters.stack_partials()

    update = ADAM(beta1, beta2)
    line_search = Backtracking(update, max_count=is_backtracking * 1_000)
    optimizer = Optimizer(line_search)

    for e in range(num_epochs):
        batches = mini_batches(data.X, batch_size, shuffle, random_state)
        for b, batch in enumerate(batches):
            params = parameters.stack()
            func = functools.partial(objective_function, batch=batch)
            grad = functools.partial(objective_gradient, batch=batch)
            optimizer.minimize(
                x=params,
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



