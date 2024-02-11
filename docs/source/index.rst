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
perceptrons, whose training process was modified to accurately predict 
response values and partial derivatives. This accomplished by minimizing 
the Least Squares Estimator (LSE), modified to include prediction error 
of response values and partial derivatives.

Mathematically, the Jacobian is 
defined as: 

.. math::

   \left(
   \begin{matrix}
   y_1 \\
   \vdots \\
   y_K
   \end{matrix}
   \right)
   =
   f
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
   \left(
   \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_p}  \\
   \vdots & \ddots & \vdot \\
   \frac{\partial y_K}{\partial x_1} & \dots & \frac{\partial y_K}{\partial x_p}  \\
   \end{matrix}
   \right)



The main benefit of gradient-enhancement is better accuracy with
fewer training points, compared to full-connected neural nets without
gradient-enhancement. JENN applies to regression, but not classification since 
there is no gradient in that case. This particular implementation is fully 
vectorized and arrays updated in place. It uses Adam optimization with L2-norm 
regularization and mini-batch is available as an option.



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