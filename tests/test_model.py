"""Test that model learns correctly."""
import jenn
import numpy as np

from ._utils import r_square, finite_difference


def test_sinuisoid(m_train: int = 100, m_test: int = 30):
    """Train a neural net against 1D sinuidal data."""
    #########
    # Train #
    #########

    x_train, y_train, dydx_train = jenn.synthetic.Sinusoid.sample(m_train)
    nn = jenn.NeuralNet([1, 12, 1], 'tanh')
    nn.fit(x_train, y_train, alpha=.005, max_iter=2000, is_normalize=True)

    #############################
    # Goodness of Fit: Training #
    #############################

    expected = y_train
    computed = nn.predict(x_train)
    assert np.all(r_square(expected, computed) > 0.95)

    expected = dydx_train
    computed = nn.predict_partials(x_train)
    assert np.all(r_square(expected, computed) > 0.95)

    expected = finite_difference(nn.predict, x_train)
    computed = nn.predict_partials(x_train)
    assert np.allclose(expected, computed)

    ############################
    # Goodness of Fit: Testing #
    ############################

    x_test, y_test, dydx_test = jenn.synthetic.Sinusoid.sample(m_test)

    expected = y_test
    computed = nn.predict(x_test)
    assert np.all(r_square(expected, computed) > 0.95)

    expected = dydx_test
    computed = nn.predict_partials(x_test)
    assert np.all(r_square(expected, computed) > 0.95)

    expected = finite_difference(nn.predict, x_test)
    computed = nn.predict_partials(x_test)
    assert np.allclose(expected, computed)
