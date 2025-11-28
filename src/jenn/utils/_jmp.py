"""Load trained JMP model."""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from pathlib import Path

import numpy as np

from jenn.core.model import NeuralNet


def _is_float(string: str) -> bool:
    """Check is string can be converted to float."""
    items = string.split(".")
    return all(item.replace("-", "").isnumeric() for item in items)


def _count_repetitions(char: str, string: str) -> int:
    """Count number of times a character is repeated."""
    count = 0
    for c in string:
        if c == char:
            count += 1
    return count


def _is_empty(items: list[list]) -> bool:
    """Return True is all items are empty lists."""
    if not isinstance(items, list):
        return False
    if all(item == [] for item in items):
        return True
    return all(_is_empty(item) for item in items)


def _expand(items: list[str]) -> list[str]:
    """Expand w * (a * x + b) into (w * a * x + w * b)."""
    updated = []
    number_open_brackets = 0
    factor = 1.0
    skip = 0
    N = len(items)
    for i, item in enumerate(items):
        if skip > 0:
            skip -= 1
            continue
        number_open_brackets += _count_repetitions("(", item) - _count_repetitions(
            ")",
            item,
        )
        if number_open_brackets == 0:
            if (
                _is_float(item)
                and i < N - 2
                and items[i + 1] == "*"
                and items[i + 2] == "("
            ):
                factor = float(item)
                number_open_brackets += 1
                skip = 2
                continue
            if item == ")":
                continue
        if number_open_brackets == 1 and items[i + 1] == ")":
            factor = 1.0
        if _is_float(item) and number_open_brackets == 1:
            updated.append(str(float(item) * factor))
        else:
            updated.append(item)
    return updated


def _remove_unnecessary_brackets(items: list[str]) -> list[str]:
    """Remove situations such as ["(", "-1.33806951259538", ")"] where brackets
    aren't needed.
    """
    updated = []
    skip = 0
    N = len(items)
    for i, item in enumerate(items):
        if skip > 0:
            skip -= 1
            continue
        if (
            (i < N - 2)
            and item == "("
            and _is_float(items[i + 1])
            and items[i + 2] == ")"
        ):
            updated.append(items[i + 1])
            skip = 2
            continue
        updated.append(item)
    return updated


def _get_node_parameters(
    items: list[str],
) -> tuple[list[float], list[float], list[list[str]]]:
    """Return bias, weights, and activations associated with node in layer."""
    number_open_brackets = 0
    biases = []
    weights = []
    activations = []
    activation = []
    for i, item in enumerate(items):
        number_open_brackets += _count_repetitions("(", item) - _count_repetitions(
            ")",
            item,
        )
        if number_open_brackets > 0:
            activation.append(item)
        if number_open_brackets == 0:
            if activation:
                expanded = _expand(
                    activation[1:],
                )  # remove "(" b/c only want inputs of Tanh(...)
                activations.append(expanded)
            activation = []
            if _is_float(item):
                if items[i + 1] == "+":
                    biases.append(float(item))  # there is only one per model in JMP
                elif items[i + 1] == "*":
                    weights.append(float(item))
    bias = [sum(biases)]
    return bias, weights, activations


def _get_layer_parameters(
    nodes: list[list[str]],
) -> tuple[list[list[float]], list[list[float]], list[list[str]]]:
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


def _get_items(equation: str) -> list[str]:
    """Parse equation into list of str."""
    equation = equation.replace("(", " ( ")
    equation = equation.replace(")", " ) ")
    equation = equation.replace("+", " + ")
    equation = equation.replace("*", " * ")
    return _remove_unnecessary_brackets(equation.split())


def _load_from_file(filename: str | Path) -> str:
    """Load JMP equation from local file."""
    with Path(filename).open(encoding="utf-8") as file:
        equation = "".join(file.readlines())
    return equation


def from_jmp(equation: str | Path) -> NeuralNet:
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

    items = _get_items(equation)

    # Last layer
    (
        layer_bias,
        layer_weights,
        layer_activation_inputs,  # TanH(...) for each node in layer
    ) = _get_layer_parameters([items])

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
