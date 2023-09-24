"""Parameters and hyperparameters."""
import numpy as np

from .activation import Relu, Tanh, Linear


ACTIVATIONS = dict(
    relu=Relu,
    tanh=Tanh,
    linear=Linear,
)


class Parameters:

    def __init__(
            self,
            layer_sizes: list,
            hidden_activation='relu',
            output_activation='linear',
    ):
        n_x = layer_sizes[0]
        n_y = layer_sizes[-1]
        self.layers = range(len(layer_sizes))
        self.L = len(layer_sizes)
        self.W = []
        self.b = []
        self.a = []
        self.dW = []
        self.db = []
        self.mu_x = np.zeros((n_x, 1))
        self.mu_y = np.zeros((n_y, 1))
        self.sigma_x = np.eye(n_x, 1)
        self.sigma_y = np.eye(n_x, 1)
        previous_layer_size = None
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:  # input layer
                W = np.eye(layer_size)
                b = np.zeros((layer_size, 1))
                a = ACTIVATIONS['linear']
            elif i == self.L - 1:  # output layer
                W = np.random.randn(layer_size, previous_layer_size) \
                    * np.sqrt(1. / previous_layer_size)
                b = np.zeros((layer_size, 1))
                a = ACTIVATIONS[output_activation]
            else:  # hidden layer
                W = np.random.randn(layer_size, previous_layer_size) \
                    * np.sqrt(1. / previous_layer_size)
                b = np.zeros((layer_size, 1))
                a = ACTIVATIONS[hidden_activation]
            dW = np.zeros(W.shape)
            db = np.zeros(b.shape)
            self.dW.append(dW)
            self.db.append(db)
            self.W.append(W)
            self.b.append(b)
            self.a.append(a)
            previous_layer_size = layer_size
        self.layer_sizes = layer_sizes

    def stack(self):
        """Stack W, b into a single array for each layer"""
        stacks = []
        for i in range(self.L):
            W = self.W[i].ravel()
            b = self.b[i].ravel()
            stack = np.concatenate([W, b]).reshape((-1, 1))
            stacks.append(stack)
        return stacks

    def unstack(self, params: np.ndarray):
        """Unstack W, b from single array stacks back into original arrays"""
        for i, array in enumerate(params):
            n, p = self.W[i].shape
            self.W[i][:] = array[:n * p].reshape(n, p)
            self.b[i][:] = array[n * p:].reshape(n, 1)

    def stack_partials(self):
        stacks = []
        for i in range(self.L):
            dW = self.dW[i].ravel()
            db = self.db[i].ravel()
            stack = np.concatenate([dW, db]).reshape((-1, 1))
            stacks.append(stack)
        return stacks

    def unstack_partials(self, partials):
        """Unstack dW, db from single array stacks back into original arrays"""
        for i, array in enumerate(partials):
            n, p = self.dW[i].shape
            self.dW[i][:] = array[:n * p].reshape(n, p)
            self.db[i][:] = array[n * p:].reshape(n, 1)
