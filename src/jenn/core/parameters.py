"""Parameters.
==============

This module defines a utility class to store and manage neural net parameters and metadata."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

import jsonpointer
import jsonschema
import numpy as np
import orjson

from .activation import ACTIVATIONS

_here = Path(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = json.loads((_here / "schema.json").read_text())


@dataclass
class Parameters:
    r"""Neural network parameters.

    .. warning::
        The attributes of this class are not protected. It's possible
        to overwrite them instead of updating them in place. To ensure
        that an array is updated in place, use the numpy `[:]` syntax:

        .. code-block:: python

            parameters = Parameters(**kwargs)
            layer_1_weights = parameters.W[1]
            layer_1_weights[:] = new_array_values  # note [:]

    .. note::

        The variables and their symbols refer to the theory in the companion
        `paper`_ for this library.

    :param layer_sizes: number of nodes in each layer (including
        input/output layers)
    :param hidden_activation: activation function used in hidden layers
    :param output_activation: activation function used in output layer

    :ivar W: weights :math:`\boldsymbol{W} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}` for each layer
    :vartype W: List[np.ndarray]

    :ivar b: biases :math:`\boldsymbol{b} \in \mathbb{R}^{n^{[l]} \times 1}` for each layer
    :vartype b: List[np.ndarray]

    :ivar a: activation names for each layer
    :vartype a: List[str]

    :ivar dW: partials w.r.t. weight :math:`dL/dW^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}`
    :vartype dW: List[np.ndarray]

    :ivar db: partials w.r.t. bias :math:`dL/db^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}`
    :vartype db: List[np.ndarray]

    :ivar mu_x: mean of training data inputs used for normalization :math:`\mu_x \in \mathbb{R}^{n_x \times 1}`
    :vartype mu_x: List[np.ndarray]

    :ivar mu_y: mean of training data outputs used for normalization :math:`\mu_y \in \mathbb{R}^{n_y \times 1}`
    :vartype mu_x: List[np.ndarray]

    :ivar sigma_x: standard deviation of training data inputs used for normalization :math:`\sigma_x \in \mathbb{R}^{n_x \times 1}`
    :vartype sigma_x: List[np.ndarray]

    :ivar sigma_y: standard deviation of training data outputs used for normalization :math:`\sigma_y \in \mathbb{R}^{n_y \times 1}`
    :vartype sigma_y: List[np.ndarray]
    """

    layer_sizes: List[int]
    hidden_activation: str = "relu"
    output_activation: str = "linear"

    @property
    def layers(self) -> Iterable[int]:
        """Return iterator of index for each layer."""
        return range(self.L)

    @property
    def partials(self) -> Iterable[int]:
        """Return iterator of index for each partial."""
        return range(self.n_x)

    @property
    def n_x(self) -> int:
        """Return number of inputs."""
        return self.layer_sizes[0]

    @property
    def n_y(self) -> int:
        """Return number of outputs."""
        return self.layer_sizes[-1]

    @property
    def L(self) -> int:
        """Return number of layers."""
        return len(self.layer_sizes)

    def initialize(self, random_state: Union[int, None] = None) -> None:
        """Use `He initialization <https://arxiv.org/pdf/1502.01852.pdf>`_ to
        initialize parameters.

        :param random_state: optional random seed (for repeatability)
        """
        rng = np.random.default_rng(random_state)
        self.W = []
        self.b = []
        self.a = []
        self.dW = []
        self.db = []
        self.mu_x = np.zeros((self.n_x, 1))
        self.mu_y = np.zeros((self.n_y, 1))
        self.sigma_x = np.eye(self.n_x, 1)
        self.sigma_y = np.eye(self.n_y, 1)
        previous_layer_size = -1  # Not used on first loop.
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0:  # input layer
                W = np.eye(layer_size)
                b = np.zeros((layer_size, 1))
                a = "linear"
            elif i == self.L - 1:  # output layer
                W = rng.normal(size=(layer_size, previous_layer_size)) * np.sqrt(
                    1.0 / previous_layer_size
                )
                b = np.zeros((layer_size, 1))
                a = self.output_activation
            else:  # hidden layer
                W = rng.normal(size=(layer_size, previous_layer_size)) * np.sqrt(
                    1.0 / previous_layer_size
                )
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

    def stack(self) -> np.ndarray:
        """Stack W, b into a single array.

        .. code-block::

            parameters.stack()
            >> np.array([[W1], [b1], [W2], [b2], [W3], [b3]])

        .. note::
            This method is used to convert the list format
            used by the neural net into a single array of stacked parameters
            for optimization.
        """
        stacks = self.stack_per_layer()
        return np.concatenate(stacks).reshape((-1, 1))

    def stack_per_layer(self) -> List[np.ndarray]:
        """Stack W, b into a single array for each layer.

        .. code-block::

            parameters.stack_per_layer()
            >> [np.array([[W1], [b1]]), [W2], [b2]]), np.array([[W3], [b3]])]
        """
        stacks = []
        for i in range(self.L):
            stack = np.concatenate([self.W[i].ravel(), self.b[i].ravel()]).reshape(
                (-1, 1)
            )
            stacks.append(stack)
        return stacks

    def stack_partials(self) -> np.ndarray:
        """Stack backprop partials dW, db.

        .. code-block::

            parameters.stack_partials()
            >> np.array([[dW1], [db1], [dW2], [db2], [dW3], [db3]])

        .. note::
            This method is used to convert the list format used by the neural
            net into a single array of stacked parameters for optimization.
        """
        stacks = self.stack_partials_per_layer()
        return np.concatenate(stacks).reshape((-1, 1))

    def stack_partials_per_layer(self) -> List[np.ndarray]:
        """Stack backprop partials dW, db per layer.

        .. code-block::

            parameters.stack_partials_per_layer()
            >> [np.array([[dW1], [db1]]), np.array([[dW2], [db2]]), np.array([[dW3], [db3]]),]
        """
        stacks = []
        for i in range(self.L):
            stack = np.concatenate(
                [
                    self.dW[i].ravel(),
                    self.db[i].ravel(),
                ]
            ).reshape((-1, 1))
            stacks.append(stack)
        return stacks

    def _column_to_stacks(self, params: np.ndarray) -> List[np.ndarray]:
        """Convert parameters from single stack to list of stacks.

        Neural net parameters are converted from single stack
        representation (for all layers) to a list of stacks (per layer).

        Parameters
        ----------
        params: np.ndarray
            Neural network parameters as single array where all layers
            are stacked on top of each other.
            e.g. np.array([[W1], [b1], [W2], [b2], [W3], [b3]])

        Returns
        -------
        params: List[np.ndarray]
            List of stacks (one per layer)
            e.g. [np.array([[W1], [b1]]), [W2], [b2]]), np.array([[W3], [b3]])]
        """
        stacks = []
        k = 0
        for i in range(self.L):  # single stack to many stacks (for each layer)
            n_w, p = self.W[i].shape
            n_b, _ = self.b[i].shape
            n = n_w * p + n_b
            stack = params[k : k + n]
            stacks.append(stack)
            k += n
        return stacks

    def unstack(self, parameters: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Unstack parameters W, b back into list of arrays.

        :param parameters: neural network parameters as either a single
            array where all layers are stacked on top of each other or a list of
            stacked parameters for each layer.

        .. code-block::

            # Unstack from single stack
            parameters.unstack(np.array([[W1], [b1], [W2], [b2], [W3], [b3]]))
            parameters.W, parameters.b
            >> [W1, W2, W3], [b1, b2, b3]

            # Unstack from list of stacks
            parameters.unstack([np.array([[W1], [b1]]), [W2], [b2]]), np.array([[W3], [b3]])])
            parameters.W, parameters.b
            >> [W1, W2, W3], [b1, b2, b3]

        .. note::
            This method is used to convert optimization results expressed
            as a single array of stacked parameters, back into the list format
            used by the neural net.
        """
        if isinstance(parameters, np.ndarray):  # single column
            parameters = self._column_to_stacks(parameters)
        for i, array in enumerate(parameters):  # stacks to params for each layer
            n, p = self.W[i].shape
            self.W[i][:] = array[: n * p].reshape(n, p)
            self.b[i][:] = array[n * p :].reshape(n, 1)

    def unstack_partials(self, partials: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Unstack backprop partials dW, db back into list of arrays.

        :param partials: neural network partials as either a single
            array where all layers are stacked on top of each other or a list of
            stacked parameters for each layer.

        .. code-block::

            # Unstack from single stack
            parameters.unstack(np.array([[dW1], [db1], [dW2], [db2], [dW3], [db3]]))
            parameters.dW, parameters.db
            >> [dW1, dW2, dW3], [db1, db2, db3]

            # Unstack from list of stacks
            parameters.unstack([np.array([[dW1], [db1]]), [dW2], [db2]]), np.array([[dW3], [db3]])])
            parameters.dW, parameters.db
            >> [dW1, dW2, dW3], [db1, db2, db3]

        .. note::
            This method is used to convert optimization results expressed
            as a single array of stacked parameters, back into the list format
            used by the neural net.
        """
        if isinstance(partials, np.ndarray):  # single column
            partials = self._column_to_stacks(partials)
        for i, array in enumerate(partials):
            n, p = self.dW[i].shape
            self.dW[i][:] = array[: n * p].reshape(n, p)
            self.db[i][:] = array[n * p :].reshape(n, 1)

    def _serialize(self) -> bytes:
        """Serialize parameters into byte stream for json."""
        keys = jsonpointer.JsonPointer("/properties").get(SCHEMA)
        data = {key: getattr(self, key) for key in keys}
        return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)

    def _deserialize(self, saved_parameters: bytes) -> None:
        """Deserialize and apply saved parameters."""
        params = orjson.loads(saved_parameters)
        jsonschema.validate(params, SCHEMA)
        self.W = [np.array(value) for value in params["W"]]
        self.b = [np.array(value) for value in params["b"]]
        self.a = params["a"]
        self.mu_x = np.array(params["mu_x"])
        self.mu_y = np.array(params["mu_y"])
        self.sigma_x = np.array(params["sigma_x"])
        self.sigma_y = np.array(params["sigma_y"])
        self.layer_sizes = [W.shape[0] for W in self.W]
        self.output_activation = self.a[-1]
        self.hidden_activation = self.a[-2]
        self.dW = [np.zeros(array.shape) for array in self.W]
        self.db = [np.zeros(array.shape) for array in self.b]
        assert (
            self.mu_x.size == self.layer_sizes[0]
        ), "mu_x size is different input layer size"
        assert (
            self.mu_y.size == self.layer_sizes[-1]
        ), "mu_y size is different output layer size"
        assert (
            self.sigma_x.size == self.layer_sizes[0]
        ), "sigma_x size is different input layer size"
        assert (
            self.sigma_y.size == self.layer_sizes[-1]
        ), "sigma_x size is different output layer size"
        assert (
            self.mu_x.shape == self.sigma_x.shape
        ), "mu_x and sigma_x have different shapes"
        assert (
            self.mu_y.shape == self.sigma_y.shape
        ), "mu_y and sigma_y have different shapes"
        m = self.layer_sizes[0]
        for i, n in enumerate(self.layer_sizes):
            assert (
                self.a[i] in ACTIVATIONS
            ), f"a[{i}] must be one of {list(ACTIVATIONS.keys())}"
            assert self.b[i].shape == (
                n,
                1,
            ), f"b[{i}] has the wrong shape (expected {(n, 1)})"
            assert self.W[i].shape == (
                n,
                m,
            ), f"W[{i}] has the wrong shape (expected {(n, m)})"
            m = n

    def save(self, binary_file: Union[str, Path] = "parameters.json") -> None:
        """Save parameters to specified json file."""
        with open(binary_file, "wb") as file:
            file.write(self._serialize())

    def load(self, binary_file: Union[str, Path] = "parameters.json") -> None:
        """Load parameters from specified json file."""
        with open(binary_file, "rb") as file:
            byte_stream = file.read()
        self._deserialize(byte_stream)
