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
        self.n_x = layer_sizes[0]
        self.n_y = layer_sizes[-1]
        self.layers = range(len(layer_sizes))
        self.partials = range(self.n_x)
        self.L = len(layer_sizes)
        self.W = []
        self.b = []
        self.a = []
        self.dW = []
        self.db = []
        self.mu_x = np.zeros((self.n_x, 1))
        self.mu_y = np.zeros((self.n_y, 1))
        self.sigma_x = np.eye(self.n_x, 1)
        self.sigma_y = np.eye(self.n_y, 1)
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

    def stack(self, per_layer: bool = False) -> np.ndarray or list:
        """Stack W, b into a single array for each layer"""
        stacks = []
        for i in range(self.L):
            stack = np.concatenate([
                self.W[i].ravel(),
                self.b[i].ravel()
            ]).reshape((-1, 1))
            stacks.append(stack)
        if per_layer:
            return stacks
        return np.concatenate(stacks).reshape((-1, 1))

    def _column_to_stacks(self, params: np.ndarray) -> list:
        """Convert params represented as single column of stacked layers to
        list of stacks, where a stack is a one column representation of W, b
        for that layer only."""
        stacks = []
        k = 0
        for i in range(self.L):  # single stack to many stacks (for each layer)
            n_w, p = self.W[i].shape
            n_b, _ = self.b[i].shape
            n = n_w * p + n_b
            stack = params[k:k + n]
            stacks.append(stack)
            k += n
        return stacks

    def unstack(self, params: np.ndarray or list):
        """Unstack W, b back into list of arrays"""
        if isinstance(params, np.ndarray):  # single column
            params = self._column_to_stacks(params)
        for i, array in enumerate(params):  # stacks to params for each layer
            n, p = self.W[i].shape
            self.W[i][:] = array[:n * p].reshape(n, p)
            self.b[i][:] = array[n * p:].reshape(n, 1)

    def stack_partials(self, per_layer: bool = False) -> np.ndarray or list:
        stacks = []
        for i in range(self.L):
            stack = np.concatenate([
                self.dW[i].ravel(),
                self.db[i].ravel(),
            ]).reshape((-1, 1))
            stacks.append(stack)
        if per_layer:
            return stacks
        return np.concatenate(stacks).reshape((-1, 1))

    def unstack_partials(self, partials):
        """Unstack dW, db back into list of arrays"""
        if isinstance(partials, np.ndarray):  # single column
            partials = self._column_to_stacks(partials)
        for i, array in enumerate(partials):
            n, p = self.dW[i].shape
            self.dW[i][:] = array[:n * p].reshape(n, p)
            self.db[i][:] = array[n * p:].reshape(n, 1)

    def serialize(self):
        """Save parameters to json."""
        pass  # TODO

    def deserialize(self):
        """Load parameters from json"""
        pass  # TODO
