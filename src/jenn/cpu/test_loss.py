import unittest
import numpy as np
from ._loss import SquaredLoss, Regularization, GradientEnhancement


class TestSquaredLoss(unittest.TestCase):

    def setUp(self):
        pass

    def test_evaluate(self):
        """ Test the squared loss function """

        # Check that perfect prediction returns 0.0 loss
        y_pred = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
        y_true = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
        loss = SquaredLoss(y_true)
        self.assertTrue(np.allclose(loss.evaluate(y_pred), 0.))

        # Check that a prediction offset of exactly one
        # returns the number of examples times number of responses
        y_pred = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]).reshape((-1, 1))
        y_true = np.array([[2, 3, 4],
                           [5, 6, 7],
                           [8, 9, 10]]).reshape((-1, 1))
        n_y, n_x = y_true.shape
        loss = SquaredLoss(y_true)
        self.assertTrue(np.allclose(loss.evaluate(y_pred), n_y * n_x))

    def test_regularization(self):
        """ Test the penalty function """
        # Set coefficients equal to one and check
        # that, in that case, the answer is simply
        # the total number of coefficient
        n = 5
        weights = [np.ones((n,))]
        regularization = Regularization(weights)
        actual = regularization.evaluate(weights)
        expected = len(weights) * n
        self.assertTrue(actual == expected)

    def test_gradient_enhancement(self):
        # Check that perfect prediction of partials returns 0.0 loss
        dy_pred = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]).reshape((3, 3, 1))
        dy_true = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]).reshape((3, 3, 1))
        grad_loss = GradientEnhancement(dy_true)
        self.assertTrue(np.allclose(grad_loss.evaluate(dy_pred), 0.))

        # Check that imperfect prediction of partials, with a constant
        # error of exactly 1 for each partial, returns the number of
        # examples (m=1) times the partials times the number of responses
        dy_pred = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]).reshape((3, 3, 1))
        dy_true = np.array([[2, 3, 4],
                            [5, 6, 7],
                            [8, 9, 10]]).reshape((3, 3, 1))
        grad_loss = GradientEnhancement(dy_true)
        self.assertTrue(np.allclose(grad_loss.evaluate(dy_pred), 3 * 3))


if __name__ == "__main__":
    unittest.main()
