from jenn._loss import squared_loss, gradient_enhancement, regularization

import numpy as np


def test_squared_loss():

    # Check that perfect prediction returns 0.0 loss
    y_pred = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    y_true = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    assert np.allclose(squared_loss(y_true, y_pred), 0.)

    # Check that a prediction offset of exactly one
    # returns 1/2 times the number of examples times the number of responses
    y_pred = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]).reshape((-1, 1))
    y_true = np.array([[2, 3, 4],
                       [5, 6, 7],
                       [8, 9, 10]]).reshape((-1, 1))
    assert np.allclose(squared_loss(y_true, y_pred), 0.5 * 3 * 3)


def test_regularization():

    # Set coefficients equal to one and check
    # that, in that case, the answer is simply
    # the total number of coefficient times the
    # scale factor alpha / 2m
    alpha = 0.1
    m = 100
    n = 5
    w = [np.ones((n,))]
    actual = regularization(w, m, alpha)
    expected = len(w) * n * 0.5 * alpha / m
    assert actual == expected


def test_gradient_enhancement():

    # Check that perfect prediction of partials returns 0.0 loss
    dy_pred = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]).reshape((3, 3, 1))
    dy_true = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]).reshape((3, 3, 1))
    assert np.allclose(gradient_enhancement(dy_true, dy_pred), 0.)

    # Check that imperfect prediction of partials, with a constant
    # error of exactly 1 for each partial, returns the number of
    # examples (m=1) times the partials times the number of responses
    # scaled by gamma / 2m
    dy_pred = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]).reshape((3, 3, 1))
    dy_true = np.array([[2, 3, 4],
                        [5, 6, 7],
                        [8, 9, 10]]).reshape((3, 3, 1))
    assert np.allclose(gradient_enhancement(dy_true, dy_pred, gamma=1), 3 * 3 * 0.5)


if __name__ == "__main__":
    test_squared_loss()
    test_regularization()
    test_gradient_enhancement()
