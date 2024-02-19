"""Test training and prediction (including partials)."""
import jenn
import numpy as np

from ._utils import finite_difference


class TestSinusoid: 
    """Check model against 1D sinusoidal test function."""    
    
    def test_neural_net(self, m_train: int = 100, m_test: int = 30):
       """Train a neural net against 1D sinuidal data."""
       #########
       # Train #
       #########

       x_train, y_train, dydx_train = jenn.synthetic.Sinusoid.sample(0, m_train)
       nn = jenn.model.NeuralNet([1, 12, 1], 'tanh')
       nn.fit(x_train, y_train, 
              lambd=0.1, alpha=.01, max_iter=2000, is_normalize=True)

       #############################
       # Goodness of Fit: Training #
       #############################

       expected = y_train
       computed = nn.predict(x_train)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score > 0.95), f'r-square = {score} < 0.95'

       expected = dydx_train
       computed = nn.predict_partials(x_train)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score > 0.95), f'r-square = {score} < 0.95'

       expected = finite_difference(nn.predict, x_train)
       computed = nn.predict_partials(x_train)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score > 0.95), f'r-square = {score} < 0.95'

       ############################
       # Goodness of Fit: Testing #
       ############################

       x_test, y_test, dydx_test = jenn.synthetic.Sinusoid.sample(m_test)

       expected = y_test
       computed = nn.predict(x_test)
       assert np.all(jenn.utils.metrics.r_square(expected, computed) > 0.95), f'r-square = {score} < 0.95'

       expected = dydx_test
       computed = nn.predict_partials(x_test)
       assert np.all(jenn.utils.metrics.r_square(expected, computed) > 0.90), f'r-square = {score} < 0.90'

       expected = finite_difference(nn.predict, x_test)
       computed = nn.predict_partials(x_test)
       assert np.allclose(expected, computed, atol=1e-6)
       
    def test_gradient_enhanced_neural_net(self, m_train: int = 4, m_test: int = 30):
       """Verify that gradient-enhancement succeeds in training a neural net
       against 1D sinuidal data when insufficient number of points is provided
       for a regular neural net to succeed."""
       x_train, y_train, dydx_train = jenn.synthetic.Sinusoid.sample(0, m_train)
       x_test, y_test, dydx_test = jenn.synthetic.Sinusoid.sample(m_test)

       #########
       # Train # (regular neural net)
       #########

       nn = jenn.model.NeuralNet([1, 12, 1], 'tanh')
       nn.fit(x_train, y_train,
              lambd=0.1, alpha=.01, max_iter=2000, is_normalize=True)

       #############################
       # Goodness of Fit: Training # (regular neural net should generalize badly)
       #############################

       expected = y_test
       computed = nn.predict(x_test)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score < 0.5), f'r-square = {score} > 0.5'  

       #########
       # Train # (gradient-enhanced neural net)
       #########

       genn = jenn.model.NeuralNet([1, 12, 1], 'tanh')
       genn.fit(x_train, y_train, dydx_train,
              lambd=0.1, gamma=1.0, alpha=0.05, max_iter=500, is_normalize=True)

       #############################
       # Goodness of Fit: Training #
       #############################

       expected = y_train
       computed = genn.predict(x_train)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score > 0.95), f'r-square = {score} < 0.95'

       expected = dydx_train
       computed = genn.predict_partials(x_train)
       score = jenn.utils.metrics.r_square(expected, computed)
       assert np.all(score > 0.95), f'r-square = {score} < 0.95'

       ############################
       # Goodness of Fit: Testing #
       ############################

       expected = y_test
       computed = genn.predict(x_test)
       assert np.all(jenn.utils.metrics.r_square(expected, computed) > 0.9), f'r-square = {score} < 0.9'

       expected = dydx_test
       computed = genn.predict_partials(x_test)
       assert np.all(jenn.utils.metrics.r_square(expected, computed) > 0.9), f'r-square = {score} < 0.9'


class TestRastrigin: 
    """Check model against 2D sinusoidal test function."""    

    @classmethod
    def test_neural_net(self): 
        """TODO"""
        pass 

    @classmethod
    def test_gradient_enhanced_neural_net(self): 
        """TODO"""
        pass 