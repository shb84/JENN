"""Test Parameter class."""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import pathlib
import tempfile

import numpy as np
import pytest

import jenn


class TestSerialization:
    """Check that parameters can be saved and reloaded."""

    @pytest.fixture
    def parameters(self) -> jenn.core.parameters.Parameters:
        """Return XOR parameters."""
        parameters = jenn.core.parameters.Parameters(
            layer_sizes=[2, 2, 1],
            output_activation="relu",
        )
        parameters.initialize()
        parameters.b[1][:] = np.array([[0], [-1]])  # layer 1
        parameters.W[1][:] = np.array([[1, 1], [1, 1]])  # layer 1
        parameters.b[2][:] = np.array([[0]])  # layer 2
        parameters.W[2][:] = np.array([[1, -2]])  # layer 2
        return parameters

    def test_serialization(self, parameters: jenn.core.parameters.Parameters) -> None:
        """Test that saved parameters can be reloaded into a new object."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpfile = pathlib.Path(tmpdirname) / "params.json"
            parameters.save(tmpfile)
            new_instance = jenn.core.parameters.Parameters(parameters.layer_sizes)
            new_instance.initialize()
            assert parameters != new_instance
            reloaded = jenn.core.parameters.Parameters.load(tmpfile)
            assert parameters == reloaded
