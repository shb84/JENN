from sklearn.neural_network.gradient_enhanced._model import GENN
import numpy as np


def test_L_model_backward():
    """
    Fit a simple parabola twice: once using backprop and once again using
    finite difference. Make sure that the resulting coefficients are close.

    Note: it's important to fix the random seed for comparison
    """
    X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    Y = X ** 2
    J = (2 * X).reshape((-1, 1, 1))

    model = GENN(hidden_layer_sizes=(3,), is_finite_difference=False,
                 num_epochs=1, max_iter=10, alpha=0, gamma=0, seed=0)

    model.fit(X.copy(), Y.copy(), J.copy())

    model_FD = GENN(hidden_layer_sizes=(3,), is_finite_difference=True,
                 num_epochs=1, max_iter=10, alpha=0, gamma=0, seed=0)

    model_FD.fit(X.copy(), Y.copy(), J.copy())

    for i in range(len(model.coefficients)):
        assert np.allclose(model.coefficients[i], model_FD.coefficients[i],
                           rtol=0.001)
        assert np.allclose(model.intercepts[i], model_FD.intercepts[i],
                           rtol=0.001)


if __name__ == "__main__":
    test_L_model_backward()
