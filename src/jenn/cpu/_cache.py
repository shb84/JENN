import numpy as np


class Cache:
    """
    Don't recompute quantities during training. This class stores previously
    computed quantities during forward prop in pre-allocated arrays that can
    be accessed during backprop. This allows the algorithm to be faster.

    Note that the attributes of this class are not protected. It's possible
    to overwrite them instead of updating them in place. To
    ensure that an array is updated in place, please use numpy syntax for it:

        e.g. cache = Cache(shapes)
             first_layer_activations = cache.A[1]
             first_layer_activations[:] = new_array_values  # note [:]
    """

    def __init__(self, layer_sizes, number_examples=1):
        """
        Constructor

        Parameters
        ----------
        shapes: List[Tuple[int]
            The shape associated with each layer of the neural network:
                input layer  0: shapes[0] = (n_x, m)
                hidden layer 1: shapes[1] = (n_1, m)
                hidden layer 2: shapes[2] = (n_2, m)
                ...
                output layer L: shapes[L] = (n_y, m)
            where:
                m is the number of training examples
        """
        shapes = [(layer_size, number_examples) for layer_size in layer_sizes]
        n_x, m = len(layer_sizes), number_examples
        self.Z = []  # store z = w a_prev + b
        self.Z_prime = []  # store z' = dz/dx[j] for all j = 1, .., n_x
        self.Z_prime_prime = []  # store z'' = d/dx[j] ( dz/dx[j] )
        self.A = []  # store a = g(z)
        self.A_prime = []  # store a' = da/dx[j] for all j = 1, .., n_x
        self.A_prime_prime = []  # store a'' = d/dx[j] ( da/dx[j] )
        self.G_prime = []  # store g' = da/dz
        self.G_prime_prime = []  # store g'' = d/dz( da/dz )
        self.J = []  # Jacobian w.r.t. x for each layer
        for i, shape in enumerate(shapes):
            n = shape[0]
            if shape[1] != m:
                raise ValueError(
                    f'layer {i} shape should be (*, {m}) not (*, {shape[1]})')
            self.Z.append(np.zeros(shape))
            self.Z_prime.append(np.zeros((n, n_x, m)))
            self.Z_prime_prime.append(np.zeros((n, n_x, m)))
            self.G_prime.append(np.zeros(shape))
            self.G_prime_prime.append(np.zeros(shape))
            self.A.append(np.zeros(shape))
            self.A_prime.append(np.zeros((n, n_x, m)))
            self.A_prime_prime.append(np.zeros((n, n_x, m)))
            self.J.append(np.zeros((n, n_x, m)))
