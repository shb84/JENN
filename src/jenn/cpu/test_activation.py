# TODO: convert to proper unit tests
#  1) Check values are correct
#  2) Check partials
#  3) Check in place vs. dynamic

from time import time
from jenn import cpu
from importlib.util import find_spec
import numpy as np

if find_spec("matplotlib"):
    from matplotlib import pyplot as plt

    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

if not MATPLOTLIB_INSTALLED:
    raise ImportError("Matplotlib must be installed.")

in_place = True
is_plot = False

x = np.linspace(-10, 10, 1_000_000)
y = np.zeros(x.shape)
dy = np.zeros(x.shape)
ddy = np.zeros(x.shape)

y_id = id(y)
dy_id = id(dy)
ddy_id = id(ddy)

for name, activation in cpu.ACTIVATIONS.items():
    tic = time()
    if in_place:
        y = activation.evaluate(x, y)
        assert id(y) == y_id
    else:
        y = activation.evaluate(x)
        assert id(y) != y_id
    toc = time()
    elapsed_time = toc - tic
    if is_plot:
        plt.plot(x, y)
        plt.title(name + f" (elapsed time: {elapsed_time:.3f} s)")
        plt.show()

for name, activation in cpu.ACTIVATIONS.items():
    tic = time()
    if in_place:
        y = activation.evaluate(x, y)
        dy = activation.first_derivative(x, y, dy)
        assert id(dy) == dy_id
    else:
        y = activation.evaluate(x)
        dy = activation.first_derivative(x)
        assert id(dy) != dy_id
    toc = time()
    elapsed_time = toc - tic
    if is_plot:
        plt.plot(x, dy)
        plt.title(name + f" (1st derivative) (elapsed time: {elapsed_time:.3f} s)")
        plt.show()

for name, activation in cpu.ACTIVATIONS.items():
    tic = time()
    if in_place:
        y = activation.evaluate(x, y)
        dy = activation.first_derivative(x, y, dy)
        ddy = activation.second_derivative(x, y, dy, ddy)
        assert id(ddy) == ddy_id
    else:
        y = activation.evaluate(x)
        dy = activation.first_derivative(x)
        ddy = activation.second_derivative(x)
        assert id(ddy) != ddy_id
    toc = time()
    elapsed_time = toc - tic
    if is_plot:
        plt.plot(x, ddy)
        plt.title(name + f" (2nd derivative) (elapsed time: {elapsed_time:.3f} s)")
        plt.show()
