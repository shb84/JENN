"""Test that activation functions are correct."""
import numpy as np
import jenn
from time import time

from importlib.util import find_spec
from ._utils import finite_difference

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False


ACTIVATIONS = dict(
    relu=jenn.core.activation.Relu,
    tanh=jenn.core.activation.Tanh,
    linear=jenn.core.activation.Linear,
)


def _check_activation_partials(
        x: np.ndarray, activation: jenn.core.activation.Activation):
    """Test 1st and 2nd derivative of activation (default = tanh)"""
    def dy(z):
        """finite difference 1st derivative"""
        return finite_difference(activation.evaluate, z)

    def ddy(z):
        """Finite difference 2nd derivative"""
        return finite_difference(activation.first_derivative, z, dx=1e-6)

    assert np.allclose(activation.first_derivative(x), dy(x), atol=1e-6)
    assert np.allclose(activation.second_derivative(x), ddy(x), atol=1e-6)


def test_inplace():
    """Check that activations update in place."""
    x = np.linspace(-10, 10, 1_000_000)  # some large array to test speed

    y = np.zeros(x.shape)
    dy = np.zeros(x.shape)
    ddy = np.zeros(x.shape)

    id_y = id(y)
    id_dy = id(dy)
    id_ddy = id(ddy)

    for name, activation in ACTIVATIONS.items():

        tic = time()

        y = activation.evaluate(x, y)
        dy = activation.first_derivative(x, y, dy)
        ddy = activation.second_derivative(x, y, dy, ddy)

        assert id(y) == id_y
        assert id(dy) == id_dy
        assert id(ddy) == id_ddy

        toc = time()
        elapsed_time_in_place = toc - tic
        tic = time()

        y_copy = activation.evaluate(x)
        dy_copy = activation.first_derivative(x)
        ddy_copy = activation.second_derivative(x)

        toc = time()
        elapsed_time_copy = toc - tic

        assert id(y_copy) != id_y
        assert id(dy_copy) != id_dy
        assert id(ddy_copy) != id_ddy

        # The other activations are so simple that it's actually faster to
        # not evaluate them in place. Hence, this check only makes sense for
        # TanH, which is the recommended default for JENN applications anyway.
        if name == 'tanh':
            assert elapsed_time_in_place < elapsed_time_copy
        # Speed about "2 x" for TanH: 0.055s < 0.128s for x.shape = (1, 1e6),
        # which translates into significant savings given that TanH gets called
        # for every hidden node in a neural net.


class TestActivation: 
    """Test all activation functions."""

    def test_tanh(self):
        """Test tanh activation"""
        x = np.linspace(-10, 10, 51).reshape((1, -1))
        assert np.allclose(ACTIVATIONS['tanh'].evaluate(x), np.tanh(x), atol=1e-6)
        _check_activation_partials(x, activation=jenn.core.activation.Tanh)

    def test_linear(self):
        """Test linear activation"""
        x = np.linspace(-10, 10, 51).reshape((1, -1))
        assert np.allclose(ACTIVATIONS['linear'].evaluate(x), x, atol=1e-6)
        _check_activation_partials(x, activation=jenn.core.activation.Linear)

    def test_relu(self):
        """Test relu activation"""
        x = np.linspace(-10, 10, 51).reshape((1, -1))
        negative = x <= 0
        positive = x > 0
        array = ACTIVATIONS['relu'].evaluate(x)
        assert np.allclose(array[positive], x[positive], atol=1e-6)
        assert np.allclose(array[negative], 0.0, atol=1e-6)
        _check_activation_partials(
            x[negative].reshape((1, -1)), activation=jenn.core.activation.Relu)
        _check_activation_partials(
            x[positive].reshape((1, -1)), activation=jenn.core.activation.Relu)

    @classmethod
    def plot_activation(cls, name: str):
        """Plot specified activation."""
        if not MATPLOTLIB_INSTALLED:
            raise ValueError(f'Matplotlib is not installed.')

        x = np.linspace(-10, 10, 1_000)
        g = ACTIVATIONS[name]
        y = np.zeros(x.shape)
        dy = np.zeros(x.shape)
        ddy = np.zeros(x.shape)

        y = g.evaluate(x, y)
        dy = g.first_derivative(x, y, dy)
        ddy = g.second_derivative(x, y, dy, ddy)

        plt.plot(x, y)
        plt.title(name + f" (0th derivative)")
        plt.show()

        plt.plot(x, dy)
        plt.title(name + f" (1st derivative)")
        plt.show()

        plt.plot(x, ddy)
        plt.title(name + f" (2nd derivative)")
        plt.show()


if __name__ == "__main__":
    for name in ACTIVATIONS:
        TestActivation.plot_activation(name)
