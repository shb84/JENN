.. jenn documentation master file, created by
   sphinx-quickstart on Sun Jan 21 13:58:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jenn's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Jacobian-Enhanced Neural Networks (JENN) are fully connected multi-layer
perceptrons, whose training process is modified to predict partial 
derivatives accurately. This is accomplished by minimizing a modified version 
of the Least Squares Estimator (LSE) which accounts for Jacobian prediction error.


The main benefit of jacobian-enhancement is better accuracy with
fewer training points, compared to standard fully connected neural nets. An example 
is shown below for a single-input, single-output function. JENN yields a near perfect 
prediction with only four training points (black dots), which is not the case without 
jacobian-enhancement.

.. image:: ../pics/JENN_vs_NN_1D.png
  :width: 300
  :class: with-border

Example: 1D Sinusoidal Function

.. image:: ../pics/JENN_vs_NN_2D.png
  :width: 900
  
Example: 2D Rastrigin Function

Mathematically, JENN solves the multi-task learning problem of predicting 
:math: \boldsymbol{y} = \hat{f}(\boldsymbol{x}) where the hypothesis 
:math: \hat{f} is a multi-layer perceptron and the Jacobian :math: J is given by: 

.. math::

   \boldsymbol{y} 
   =
   \left(
   \begin{matrix}
   y_1 \\
   \vdots \\
   y_K
   \end{matrix}
   \right)
   \qquad 
   \boldsymbol{x} 
   =
   \left(
   \begin{matrix}
   x_1 \\
   \vdots \\
   x_p
   \end{matrix}
   \right)
   \qquad
   J
   =
   \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}
   =
   \left(
   \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_p}  \\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_K}{\partial x_1} & \dots & \frac{\partial y_K}{\partial x_p}  \\
   \end{matrix}
   \right)

This particular implementation uses is fully vectorized and arrays updated in place.
It uses Adam optimization with L2-norm regularization. The core algorithm is written in 
Python 3 and requires only `numpy` and `orjson` (for serialization). Optionally, if `matplotlib` 
is installed, basic plotting utilities are offered to view sensivity profiles and check goodness of fit.  


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Model
-----

.. automodule:: jenn.model
   :members:

Synthetic Test Data
-------------------

.. automodule:: jenn.synthetic
   :members:

Core
----

.. automodule:: jenn.core.activation
   :members:

.. automodule:: jenn.core.cache
   :members:

.. automodule:: jenn.core.cost
   :members:

.. automodule:: jenn.core.data
   :members:

.. automodule:: jenn.core.optimization
   :members:

.. automodule:: jenn.core.parameters
   :members:

.. automodule:: jenn.core.propagation
   :members:

.. automodule:: jenn.core.training
   :members:

Utilities
---------

.. automodule:: jenn.utils
   :members: