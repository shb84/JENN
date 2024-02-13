"""Decorators."""

from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from time import time
from typing import Any

if find_spec("matplotlib"):
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False


def timeit(func: Callable) -> Callable:
    """Return elapsed time to run a function."""

    @wraps(func)
    def wrapper(*args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        tic = time()
        results = func(*args, **kwargs)
        toc = time()
        print(f"elapsed time: {toc-tic:.3f} s")
        return results

    return wrapper


def requires_matplotlib(func: Callable) -> Callable:
    """Return error if matplotlib not installed."""

    @wraps(func)
    def wrapper(*args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        if MATPLOTLIB_INSTALLED:
            return func(*args, **kwargs)
        raise ValueError("Matplotlib is not installed.")

    return wrapper
