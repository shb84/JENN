"""Parameters and hyperparameters."""
import orjson 
import numpy as np

from dataclasses import dataclass 


@dataclass 
class Parameters:

    layer_sizes: list
    hidden_activation: str = 'relu'
    output_activation: str = 'linear'

    @property 
    def layers(self): 
        """Return iterator of index for each layer."""
        return range(self.L)

    @property 
    def partials(self): 
        """Return iterator of index for each partial."""
        return range(self.n_x)
    
    @property 
    def n_x(self): 
        """Return number of inputs."""
        return self.layer_sizes[0]
    
    @property 
    def n_y(self): 
        """Return number of outputs."""
        return self.layer_sizes[-1]
    
    @property 
    def L(self): 
        """Return number of layers."""
        return len(self.layer_sizes)

    def __post_init__(self):
        self.initialize() 

    def initialize(self):
        """Use 'He initialization' to initialize parameters."""
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
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0:  # input layer
                W = np.eye(layer_size)
                b = np.zeros((layer_size, 1))
                a = 'linear'
            elif i == self.L - 1:  # output layer
                W = np.random.randn(layer_size, previous_layer_size) \
                    * np.sqrt(1. / previous_layer_size)
                b = np.zeros((layer_size, 1))
                a = self.output_activation
            else:  # hidden layer
                W = np.random.randn(layer_size, previous_layer_size) \
                    * np.sqrt(1. / previous_layer_size)
                b = np.zeros((layer_size, 1))
                a = self.hidden_activation
            dW = np.zeros(W.shape)
            db = np.zeros(b.shape)
            self.dW.append(dW)
            self.db.append(db)
            self.W.append(W)
            self.b.append(b)
            self.a.append(a)
            previous_layer_size = layer_size

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

    def serialize(self) -> bytes:
        """Serialized parameters to bytes."""
        return orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY)

    def deserialize(self, saved_parameters: bytes):
        """Deserialze parameters stored as bytes."""
        params = orjson.loads(saved_parameters)
        self.W = [np.array(value) for value in params['W']]
        self.b = [np.array(value) for value in params['b']]
        self.a = params['a']
        self.dW = [np.array(value) for value in params['dW']]
        self.db = [np.array(value) for value in params['db']]
        self.mu_x = np.array(params['mu_x'])
        self.mu_y = np.array(params['mu_y'])
        self.sigma_x = np.array(params['sigma_x'])
        self.sigma_y = np.array(params['sigma_y'])
    
    def save(self, binary_file: str = 'parameters.json'): 
        """Save parameters to json file."""
        with open("params.json", "wb") as binary_file:
            binary_file.write(self.serialize())

    def load(self, binary_file: str = 'paramemeters.json'): 
        """Load parameters from json file."""
        with open("params.json", "rb") as binary_file:
            byte_stream = binary_file.read()
        self.deserialize(byte_stream)
