import numpy as np


class SquaredLoss:
    """ Least Squares Estimator """

    def __init__(self, y_true):
        """
        Parameters
        ----------
        y_true: np.ndarray
            Training data outputs. An array of shape (n_y, m)
        """
        self.y_true = y_true
        self.y_error = np.empty(y_true.shape)  # preallocate to save resources

    def __call__(self, y_pred, indices=None):
        """
        Compute least squares estimator of the states in place

        Parameters
        ----------
        y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).
        """
        if indices is None:
            indices = range(self.y_error.size)
        self.y_error[:, indices] = y_pred[:, indices] - self.y_true[:, indices]
        n_y = self.y_error.shape[0]
        cost = 0
        for j in range(0, n_y):
            cost += np.dot(
                self.y_error[j, indices], self.y_error[j, indices].T)
        return np.float64(cost)


class GradientEnhancement:
    """ Least Squares Estimator for partials """

    def __init__(self, dy_true):
        """
        Parameters
        ----------
        dy_true: np.ndarray
            Training data gradients. An array of shape (n_y, n_x, m)
            Y' = d(Y)/dX where n_y = number outputs
                               n_x = number inputs
                               m = number examples
        """
        self.dy_true = dy_true
        self.dy_error = np.empty(dy_true.shape)

    def __call__(self, dy_pred, indices=None):
        """
        Compute least squares estimator for the partials

        Parameters
        ----------
        dy_pred: np ndarray of shape (n_y, n_x, m)
            Predicted partials: AL' = d(AL)/dX where n_y = number outputs
                                                     n_x = number inputs
                                                     m = number examples

        indices: List[int], optional
            Subset of indices over which to __call__ function. Default is None,
            which implies all indices (useful for minibatch for example).
        """
        if indices is None:
            indices = range(self.dy_error.size)
        n_y, n_x, m = self.dy_true.shape
        cost = 0.0
        for k in range(0, n_y):
            for j in range(0, n_x):
                self.dy_error[k, j, indices] = \
                    dy_pred[k, j, indices] - self.dy_true[k, j, indices]
                inner_product = np.dot(
                    self.dy_error[k, j, indices],
                    self.dy_error[k, j, indices].T)
                cost += np.squeeze(inner_product)
        return np.float64(cost)


class Regularization:
    """ Compute regularization penalty """

    def __init__(self, weights):
        """
        Parameters
        ----------
        weights: List[np.ndarray]
            Parameters w associated with each layer of neural network
            i.e. a = g(z) where z = w * a_prev + b
        """
        # Preallocate for speed
        self.weights = weights
        self._squared_weights = [np.empty(w.shape) for w in weights]

    def __call__(self, alpha: float):
        """Compute L2 norm penalty.

        Parameters
        ----------
        alpha: float
            Regularization coefficient
        """
        penalty = 0.0
        if alpha > 0:
            for i, weight in enumerate(self.weights):
                squared_weights = np.square(
                    weight, out=self._squared_weights[i])
                penalty += np.squeeze(np.sum(squared_weights))
        return alpha * np.float64(penalty)
