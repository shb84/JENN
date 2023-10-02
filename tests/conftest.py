"""Test fixtures"""
import pytest
import jenn
import numpy as np
from typing import Tuple


@pytest.fixture
def xor() -> Tuple[jenn.Dataset, jenn.Parameters, jenn.Cache]:
    """Return XOR test data and model parameters"""
    data = jenn.Dataset(
        X=np.array([[0, 1, 0, 1], [0, 0, 1, 1]]),
        Y=np.array([[0, 1, 1, 0]]),
    )
    layer_sizes = [2, 2, 1]
    parameters = jenn.Parameters(layer_sizes, output_activation='relu')
    parameters.b[1][:] = np.array([[0], [-1]])        # layer 1
    parameters.W[1][:] = np.array([[1, 1], [1, 1]])   # layer 1
    parameters.b[2][:] = np.array([[0]])              # layer 2
    parameters.W[2][:] = np.array([[1, -2]])          # layer 2
    cache = jenn.Cache(layer_sizes, m=data.m)
    return data, parameters, cache


