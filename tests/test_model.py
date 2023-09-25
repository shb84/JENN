"""Test that model learns correctly."""
import jenn
import numpy as np

from ._utils import r_square, finite_difference


def test_sinuisoid(sinusoidal_data_1D):
    """Train a neural net against 1D sinuidal data."""
    train, test = sinusoidal_data_1D
    nn = jenn.NeuralNet([1, 12, 1], 'tanh')
    nn.fit(train.X, train.Y, alpha=.005, max_iter=2000, is_normalize=True)
    XYJ = [(train.X, train.Y, train.J), (test.X, test.Y, test.J)]
    f = lambda x: nn.predict(x)[0]
    for X, Y_true, J_true in XYJ:
        Y_pred, J_pred = nn.predict(X)
        assert np.all(r_square(Y_pred, Y_true) > 0.95)
        assert np.all(r_square(J_pred, J_true) > 0.95)
        assert np.allclose(finite_difference(f, X), J_pred)
