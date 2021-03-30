from jenn._model import JENN
from jenn._utils import rsquare
import numpy as np

from importlib.util import find_spec

from jenn.tests.test_problems import rastrigin

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

if find_spec("pyDOE2"):
    from pyDOE2 import lhs, fullfact
    PYDOE2_INSTALLED = True
else:
    PYDOE2_INSTALLED = False


def test_forward_prop():
    """
    Use a very simple network to check to check that
    forward propagation recovers expected results that
    can be computed by hand. Concretely, the following
    network is equivalent to Y = 4 * X
    """
    model = JENN(hidden_layer_sizes=(2, 2), activation='identity')
    hidden_activation = [model.activation] * len(model.hidden_layer_sizes)
    output_activation = ['identity']
    model._a = hidden_activation + output_activation
    model._W = [np.ones((2, 1)), np.ones((2, 2)), np.ones((1, 2))]
    model._b = [np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((1, 1))]
    model._n_x = 1
    model._n_y = 1
    X = np.array([1, 2, 3, 4]).reshape((-1, 1))

    # States
    y_pred = model.predict(X).ravel()
    y_true = 4 * X.ravel()
    assert np.allclose(y_pred, y_true)

    # Partials
    dydx_pred = model.jacobian(X).ravel()
    dydx_true = np.array([4, 4, 4, 4])
    assert np.allclose(dydx_pred, dydx_true)


def test_parameter_shape():
    """
    Make sure that parameter initialization
    produces the correct parameter shapes
    """
    X = np.array([1, 2, 3, 4]).reshape((1, -1))
    y = np.array([1, 2, 3, 4]).reshape((1, -1))
    model = JENN(hidden_layer_sizes=(2, 2))
    model._n_x = X.shape[0]
    model._n_y = y.shape[0]
    model._initialize()

    assert model._W[0].shape == (2, 1)
    assert model._W[1].shape == (2, 2)
    assert model._W[2].shape == (1, 2)
    assert model._b[0].shape == (2, 1)
    assert model._b[1].shape == (2, 1)
    assert model._b[2].shape == (1, 1)


def test_model_parabola(verbose=False, show_plot=False):
    """
    Very simple test: fit a parabola. This test ensures that
    the model mechanics are working.
    """
    # Training data
    X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    Y = X ** 2
    J = (2*X).reshape((-1, 1, 1))

    # Basic neural net (no gradient-enhancement)
    model = JENN(hidden_layer_sizes=(3, 3), activation='tanh',
                 num_epochs=1, max_iter=1000,
                 learning_rate_init=0.05, alpha=0.01, gamma=0, verbose=verbose)
    model.fit(X, Y, J)
    if show_plot:
        model.goodness_fit(X, Y, title='NN')
    assert rsquare(Y, model.predict(X)) > 0.99

    # Gradient-enhanced neural net
    model = JENN(hidden_layer_sizes=(3, 3), activation='tanh',
                 num_epochs=1, max_iter=1000,
                 learning_rate_init=0.05, alpha=0.01, gamma=1, verbose=verbose)
    model.fit(X, Y, J)
    if show_plot:
        model.goodness_fit(X, Y, title='GENN')
    assert rsquare(Y, model.predict(X)) > 0.99


def test_model_rastrigin(verbose=False, show_plot=False):
    """ Test GENN on the rastrigin function (egg-crate looking function) """

    if not PYDOE2_INSTALLED:
        return None
    else:
        # Domain
        lb = -1.
        ub = 1.5

        # Training Data
        X_train = lb + (ub - lb) * lhs(2, samples=100, criterion='maximin',
                                       iterations=100, random_state=0)
        Y_train, J_train = rastrigin(X_train)

        # Test Data
        levels = 15
        X_test = fullfact([levels] * 2) / (levels - 1) * (ub - lb) + lb
        Y_test, J_test = rastrigin(X_test)

        # Train
        model = JENN(hidden_layer_sizes=[12] * 2, activation='tanh',
                     num_epochs=1, max_iter=1000,
                     is_finite_difference=False, solver='adam',
                     learning_rate='constant', random_state=0, tol=1e-6,
                     learning_rate_init=0.01, alpha=0, gamma=1,
                     verbose=verbose)

        model.fit(X_train, Y_train, J_train, is_normalize=True)

        if show_plot:
            model.goodness_fit(X_train, Y_train)
            model.goodness_fit(X_test, Y_test)

        assert rsquare(Y_train, model.predict(X_train)) > 0.95
        assert rsquare(Y_test, model.predict(X_test)) > 0.95


def test_sinusoid(verbose=False, show_plot=False, is_genn: bool = True):
    """
    Test GENN on simple sinusoid with very few points. Whereas regular NN
    will inevitably fail to provide a good fit, only GENN will succeed.
    """
    if show_plot and not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    # Test function
    f = lambda x: x * np.sin(x)

    # Test function derivative
    df_dx = lambda x: np.sin(x) + x * np.cos(x)

    # Domain
    lb = -np.pi
    ub = np.pi

    # Shapes
    n_x = 1  # number of inputs
    n_y = 1  # number of outputs

    # Training data
    m = 4  # number of training examples
    X_train = np.linspace(lb, ub, m).reshape((m, n_x))
    Y_train = f(X_train).reshape((m, n_y))
    J_train = df_dx(X_train).reshape((m, n_x, n_y))

    # Test data
    m = 30  # number of test examples
    X_test = lb + np.random.rand(m, 1).reshape((m, n_x)) * (ub - lb)
    Y_test = f(X_test).reshape((m, n_y))

    # Initialize model
    model = JENN(hidden_layer_sizes=(12,), activation='tanh',
                 num_epochs=1, max_iter=1000,
                 is_finite_difference=False,
                 learning_rate='backtracking', random_state=None, tol=1e-6,
                 learning_rate_init=0.05, alpha=0.1, gamma=int(is_genn),
                 verbose=verbose)

    # Train model
    model.fit(X_train, Y_train, J_train)

    # Predictions
    X = np.linspace(lb, ub, 100).reshape((100, n_x))
    Y_true = f(X)
    Y_pred = model.predict(X)

    if show_plot:

        if is_genn:
            title = 'GENN'
        else:
            title = 'NN'

        # Plot goodness of fit
        model.goodness_fit(X_test, Y_test)

        # Plot actual function
        fig, ax = plt.subplots()
        ax.plot(X, Y_pred, 'b-')
        ax.plot(X, Y_true, 'k--')
        ax.plot(X_test, Y_test, 'ro')
        ax.plot(X_train, Y_train, 'k+', mew=3, ms=10)
        ax.set(xlabel='x', ylabel='y', title=title)
        ax.legend(['Predicted', 'True', 'Test', 'Train'])
        plt.show()

    assert rsquare(Y_true, Y_pred) > 0.95


def run_tests():
    # test_forward_prop()
    # test_sinusoid(verbose=False, show_plot=False, is_genn=True)
    # test_parameter_shape()
    # test_model_parabola(verbose=False, show_plot=False)
    test_model_rastrigin(verbose=True, show_plot=False)


if __name__ == "__main__":
    run_tests()
