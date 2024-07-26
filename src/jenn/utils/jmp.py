"""Load trained JMP model."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ..model import NeuralNet


def _is_float(string: str) -> bool:
    """Check is string can be converted to float."""
    tokens = string.split(".")
    for token in tokens:
        if not token.replace("-", "").isnumeric():
            return False
    return True


def _count_repetitions(char: str, string: str) -> int:
    """Count number of times a character is repeated."""
    count = 0
    for c in string:
        if c == char:
            count += 1
    return count


def _is_empty(items: List[list]) -> bool:
    """Return True is all items are empty lists."""
    if not isinstance(items, list):
        return False
    if all([item == [] for item in items]):
        return True
    if all([_is_empty(item) for item in items]):
        return True
    return False


def _expand(tokens: List[str]) -> List[str]:
    """Expand w * (a * x + b) into (w * a * x + w * b)."""
    updated = []
    number_open_brackets = 0
    factor = 1.0
    skip = 0
    N = len(tokens)
    for i, token in enumerate(tokens):
        if skip > 0:
            skip -= 1
            continue
        number_open_brackets += _count_repetitions("(", token) - _count_repetitions(
            ")", token
        )
        if number_open_brackets == 0:
            if _is_float(token):
                if i < N - 2:
                    if tokens[i + 1] == "*":
                        if tokens[i + 2] == "(":
                            factor = float(token)
                            number_open_brackets += 1
                            skip = 2
                            continue
            if token == ")":
                continue
        if number_open_brackets == 1:
            if tokens[i + 1] == ")":
                factor = 1.0
        if _is_float(token) and number_open_brackets == 1:
            updated.append(str(float(token) * factor))
        else:
            updated.append(token)
    return updated


def _remove_unnecessary_brackets(tokens: List[str]) -> List[str]:
    """Remove situations such as ["(", "-1.33806951259538", ")"] where brackets
    aren't needed."""
    updated = []
    skip = 0
    N = len(tokens)
    for i, token in enumerate(tokens):
        if skip > 0:
            skip -= 1
            continue
        if i < N - 2:
            if token == "(" and _is_float(tokens[i + 1]) and tokens[i + 2] == ")":
                updated.append(tokens[i + 1])
                skip = 2
                continue
        updated.append(token)
    return updated


def _get_node_parameters(
    tokens: List[str],
) -> Tuple[List[float], List[float], List[List[str]]]:
    """Return bias, weights, and activations associated with node in layer."""
    number_open_brackets = 0
    biases = []
    weights = []
    activations = []
    activation = []
    for i, token in enumerate(tokens):
        number_open_brackets += _count_repetitions("(", token) - _count_repetitions(
            ")", token
        )
        if number_open_brackets > 0:
            activation.append(token)
        if number_open_brackets == 0:
            if activation:
                expanded = _expand(
                    activation[1:]
                )  # remove "(" b/c only want inputs of Tanh(...)
                activations.append(expanded)
            activation = []
            if _is_float(token):
                if tokens[i + 1] == "+":
                    biases.append(float(token))  # there is only one per model in JMP
                elif tokens[i + 1] == "*":
                    weights.append(float(token))
    bias = [sum(biases)]
    return bias, weights, activations


def _get_layer_parameters(
    nodes: List[List[str]],
) -> Tuple[List[List[float]], List[List[float]], List[List[str]]]:
    """Return bias, weight, activations associated with current layer."""
    layer_biases = []
    layer_weights = []
    for node in nodes:  # for each node in layer
        (
            node_bias,
            node_weights,
            layer_activation_inputs,
        ) = _get_node_parameters(node)
        layer_biases.append(node_bias)
        layer_weights.append(node_weights)
    return layer_biases, layer_weights, layer_activation_inputs


def _get_tokens(equation: str) -> List[str]:
    """Parse equation into list of str."""
    equation = equation.replace("(", " ( ")
    equation = equation.replace(")", " ) ")
    equation = equation.replace("+", " + ")
    equation = equation.replace("*", " * ")
    return _remove_unnecessary_brackets(equation.split())


def _load_from_file(filename: Union[str, Path]) -> str:
    """Load JMP equation from local file."""
    with open(filename) as file:
        equation = "".join(file.readlines())
    return equation


def from_jmp(equation: Union[str, Path]) -> NeuralNet:
    """Load trained JMP model given formula.

    .. Note::
        Expected equation assumed to be obtained from JMP
        using the "Save Profile Formulas" method and copy/paste.

    .. Note::
        JMP yields a separate equation for each output. It does
        not provided a single equation that predicts all outputs
        at once. This function therefore yields NeuralNet objects
        that predict only a single output (consistent with JMP).

    .. Warning::
        Order of inputs matches order used in JMP. Burden is
        on user to keep track of variable ordering.

    :param equation: either the equation itself or a filename containing it
    :return: jenn.model.NeuralNet object preloaded with the JMP parameters
    """
    if isinstance(equation, Path) or "+" not in equation or "TanH" not in equation:
        equation = _load_from_file(equation)

    tokens = _get_tokens(equation)

    # Last layer
    (
        layer_bias,
        layer_weights,
        layer_activation_inputs,  # TanH(...) for each node in layer
    ) = _get_layer_parameters([tokens])

    # Initialize list of neural net parameters for each layer
    biases = [layer_bias]
    weights = [layer_weights]

    # Loop through previous layers
    while not _is_empty(layer_activation_inputs):
        (
            layer_bias,
            layer_weights,
            layer_activation_inputs,
        ) = _get_layer_parameters(nodes=layer_activation_inputs)
        biases.append(layer_bias)
        weights.append(layer_weights)

    # Parameters as arrays
    N = len(biases)  # number of layers
    b = [np.array(b) for b in reversed(biases)]
    W = [np.array(W) for W in reversed(weights)]

    # Load JMP parameters into NeuralNet object
    jmp_model = NeuralNet(
        layer_sizes=[W[0].shape[1]] + [W[i].shape[0] for i in range(N)],
        hidden_activation="tanh",
        output_activation="linear",
    )
    jmp_model.parameters.initialize()
    for i in range(1, N + 1):
        jmp_model.parameters.W[i][:] = W[i - 1]
        jmp_model.parameters.b[i][:] = b[i - 1]

    return jmp_model
