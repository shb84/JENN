"""Test Parameter class."""
import pytest 
import numpy as np 
import tempfile
import pathlib

import jenn 


class TestSerialization: 
    """Check that parameters can be saved and reloaded."""

    @pytest.fixture
    def params(self) -> jenn.core.parameters.Parameters:
        """Return XOR parameters."""
        parameters = jenn.core.parameters.Parameters(layer_sizes=[2, 2, 1], output_activation='relu')
        parameters.initialize()
        parameters.b[1][:] = np.array([[0], [-1]])        # layer 1
        parameters.W[1][:] = np.array([[1, 1], [1, 1]])   # layer 1
        parameters.b[2][:] = np.array([[0]])              # layer 2
        parameters.W[2][:] = np.array([[1, -2]])          # layer 2
        return parameters
    
    def test_serialization(self, params: jenn.core.parameters.Parameters) -> None: 
        """Test that saved parameters can be reloaded into a new object."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpfile = pathlib.Path(tmpdirname) / 'params.json'
            params.save(tmpfile)
            parameters = jenn.core.parameters.Parameters(params.layer_sizes) 
            assert params != parameters 
            parameters.load(tmpfile)
            assert params == parameters