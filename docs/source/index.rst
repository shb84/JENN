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
perceptrons, whose training process was modified to account for gradient
information. Specifically, the parameters are learned by minimizing the Least
Squares Estimator (LSE), modified to minimize prediction error of both 
response values and partial derivatives. 

The chief benefit of gradient-enhancement is better accuracy with
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