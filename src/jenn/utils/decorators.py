"""Decorators."""
from functools import wraps
from importlib.util import find_spec
from time import time

if find_spec("matplotlib"):
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False


def timeit(func):  # noqa: ANN001, ANN201
    """Return elapsed time to run a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        tic = time()
        results = func(*args, **kwargs)
        toc = time()
        print(f"elapsed time: {toc-tic:.3f} s")
        return results

    return wrapper


def requires_matplotlib(func):  # noqa: ANN001, ANN201
    """Return elapsed time to run a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if MATPLOTLIB_INSTALLED:
            return func(*args, **kwargs)
        raise ValueError("Matplotlib is not installed.")

    return wrapper
