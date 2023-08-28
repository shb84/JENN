"""Decorators."""
from time import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time()
        results = func(*args, **kwargs)
        toc = time()
        print(f'elapsed time: {toc-tic:.3f} s')
        return results
    return wrapper
