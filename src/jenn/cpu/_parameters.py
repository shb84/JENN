import numpy as np


class Parameters:
    """
    This class stores the parameters associated with a specific neural network
    architecture. The parameters in question are the weights (w) and biases (b)
    associated with each neuron: a = g(z) where z = a_prev * W[i] + b[i].

    Note that there are no parameters associated with first layer, such that
    the length of the lists containing the parameters is L - 1:

        W[0], b[0] = layer 1
        W[1], b[1] = layer 2
        ...
        W[L-1], b[L-1] = layer L (output layer)
    """

    def __init__(
            self,
            layer_sizes: list,
            hidden_activation='relu',
            output_activation='linear',
            input_activation='linear',
    ):
        """
        Constructor: initialize neural net params using "He" initialization

        Parameters
        ----------
        layer_sizes: List[int]
            The number of neurons in each layer, including input and out layer.
        """
        activations = [input_activation]
        activations.extend([hidden_activation] * (len(layer_sizes) - 2))
        activations.append(output_activation)
        self.W = []
        self.b = []
        self.a = activations
        self.L = len(layer_sizes)
        previous_layer_size = None
        for i, layer_size in enumerate(layer_sizes):
            if i > 0:
                W = np.random.randn(layer_size, previous_layer_size) \
                    * np.sqrt(1. / previous_layer_size)
                b = np.zeros((layer_size, 1))
            else:
                W = np.eye(layer_size)
                b = np.zeros((layer_size, 1))
            self.W.append(W)
            self.b.append(b)
            previous_layer_size = layer_size


