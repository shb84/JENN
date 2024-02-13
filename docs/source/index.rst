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
of the Least Squares Estimator (LSE) that accounts for Jacobian prediction error (see theory). 
The main benefit of jacobian-enhancement is better accuracy with
fewer training points compared to standard fully connected neural nets, as illustrated below. 
 

.. image:: ../pics/JENN_vs_NN_1D.png
  :width: 225
  :align: center 


.. image:: ../pics/JENN_vs_NN_2D.png
  :width: 750
  :align: center 

|

Installation 
------------

The core algorithm is written in Python 3 and requires only `numpy` and `orjson` (for serialization):: 

    pip install jenn 

The `matplotlib` library is used to offer basic plotting utilities, such as checking goodness of fit 
or viewing sensitivity profiles, but it is entirely optional. To install:: 

    pip install jenn[viz]

Data Structures
---------------

In order to use the library effectively, it is essential to understand 
its data structures. Mathematically, JENN is used to predict smooth, continuous functions 
of the form: 
 
.. math::

   \boldsymbol{y} = f(\boldsymbol{x}) 
   \qquad \Rightarrow \qquad 
   \dfrac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = f'(\boldsymbol{x}) 

where :math:`\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}` is the Jacobian. 
For a single example, the associated quantities are given by: 

.. math::

   \boldsymbol{x} 
   =
   \left(
   \begin{matrix}
   x_1 \\
   \vdots \\
   x_p
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{p}
   \quad 
   \boldsymbol{y} 
   =
   \left(
   \begin{matrix}
   y_1 \\
   \vdots \\
   y_K
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{K}
   \quad
   \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}
   =
   \left(
   \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_p}  \\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_K}{\partial x_1} & \dots & \frac{\partial y_K}{\partial x_p}  \\
   \end{matrix}
   \right) 
   \in 
   \mathbb{R}^{K \times p}

For multiple examples, denoted by :math:`m`, these quantities become vectorized as follows: 

.. math::

   \boldsymbol{X} 
   =
   \left(
   \begin{matrix}
   x_1^{(1)} & \dots & x_1^{(m)} \\
   \vdots & \ddots & \vdots \\
   x_p^{(1)} & \dots & x_p^{(m)} \\
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{p \times m}
   \qquad 
   \boldsymbol{Y} 
   =
   \left(
   \begin{matrix}
   y_1^{(1)} & \dots & y_1^{(m)} \\
   \vdots & \ddots & \vdots \\
   y_K^{(1)} & \dots & y_K^{(m)} \\
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{K \times m}

Similarly, the vectorized version of the Jacobian becomes: 

.. math::
   
   \boldsymbol{J} 
   =
   \left(
   \begin{matrix}
   {\left(
   \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_p}  \\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_K}{\partial x_1} & \dots & \frac{\partial y_K}{\partial x_p}  \\
   \end{matrix}
   \right)}^{(1)}
   & 
   \dots 
   & 
   {\left(
   \begin{matrix}
   \frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_p}  \\
   \vdots & \ddots & \vdots \\
   \frac{\partial y_K}{\partial x_1} & \dots & \frac{\partial y_K}{\partial x_p}  \\
   \end{matrix}
   \right)}^{(m)}
   \end{matrix}
   \right)
   \in
   \mathbb{R}^{K \times p \times m}

Programmatically, these data structures are exclusively represented using shaped `numpy` arrays:: 

    import numpy as np 

    # p = number of inputs 
    # K = number of outputs 
    # m = number of examples in dataset 

    x = np.array(
    [
       [11, 12, 13, 14], 
       [21, 22, 23, 24], 
       [31, 32, 33, 34], 
    ]
    )  # array of shape (p, m) = (3, 4)

    y = np.array(
    [
       [11, 12, 13, 14], 
       [21, 22, 23, 24], 
    ]
    )  # array of shape (K, m) = (2, 4)

    dydx = np.array(
    [
       [
          [111, 112, 113, 114],
          [121, 122, 123, 124],
          [131, 132, 133, 134],
       ],
       [
          [211, 212, 213, 214],
          [221, 222, 223, 224],
          [231, 232, 233, 234],
       ]
    ]
    )  # array of shape (K, p, m) = (2, 3, 4)

    p, m = x.shape 
    K, m = y.shape 
    K, p, m = dydx.shape

    assert y.shape[0] == dydx.shape[0]
    assert x.shape[0] == dydx.shape[1]
    assert x.shape[-1] == y.shape[-1] == dydx.shape[-1]

Usage
-----

This section provides a quick example to get started. Consider the task of fitting 
a simple 1D sinusoid using only three data points:: 

    import numpy as np 
    import jenn 

    # Example function to be learned 
    f = lambda x: np.sin(x)  
    f_prime = lambda x: np.cos(x).reshape((1, 1, -1))  # note: jacobian adds a dimension

    # Generate training data 
    x_train = np.linspace(-np.pi , np.pi, 3).reshape((1, -1))
    y_train = f(x_train)
    dydx_train = f_prime(x_train)

    # Generate test data 
    x_test = np.linspace(-np.pi , np.pi, 30).reshape((1, -1))
    y_test = f(x_test)
    dydx_test = f_prime(x_test)

    # Fit model
    nn = jenn.model.NeuralNet(
        layer_sizes=[
            x_train.shape[0],  # input layer 
            7, 7,              # hidden layer(s) -- user defined
            y_train.shape[0]   # output layer 
         ],  
        ).fit(
            x_train, y_train, dydx_train,
        )

    # Predict response only 
    y_pred = nn.predict(x_test)

    # Predict partials only 
    dydx_pred = nn.predict_partials(x_train)

    # Predict response and partials in one step 
    y_pred, dydx_pred = nn.evaluate(x_test) 

    # Check how well model generalizes 
    assert jenn.utils.metrics.r_square(y_pred, y_test) > 0.99
    assert jenn.utils.metrics.r_square(dydx_pred, dydx_test) > 0.99

Saving a model for later re-use::

    nn.save("parameters.json")

Reloading the parameters a previously trained model::

    new_model = jenn.model.NeuralNet(layer_sizes=[1, 12, 1]).load('parameters.json')

    y_reloaded, dydx_reloaded = new_model.evaluate(x_test) 

    assert np.allclose(y_reloaded, y_pred)
    assert np.allclose(dydx_reloaded, dydx_pred)

Optional plotting tools are available for convenience, provided `matplotlib` is installed:: 

    # Example: show goodness of fit of the partials 
    jenn.utils.plot.goodness_of_fit(
        y_true=dydx_test[0], 
        y_pred=nn.predict_partials(x_test)[0], 
        title="Partial Derivative: dy/dx (NN)"
    )

.. image:: ../pics/example_goodness_of_fit.png
  :width: 500

::

    # Example: visualize local trends
    jenn.utils.plot.sensitivity_profiles(
        f=[f, nn.predict], 
        x_min=x_train.min(), 
        x_max=x_train.max(), 
        x_true=x_train, 
        y_true=y_train, 
        resolution=100, 
        legend=['sin(x)', 'nn'], 
        xlabels=['x'], 
        ylabels=['y'],
        show_cursor=False
    )

.. image:: ../pics/example_sensitivity_profile.png
  :width: 250

Examples 
--------

Elaborated `demo``` notebooks can be found in the project's `repo <https://github.com/shb84/JENN.git>`_ under `docs/examples`. 

Runtime
-------

The algorithm was verified to scale as :math:`\mathcal{O}(n)`, as shown below. 

.. image:: ../pics/scalability.png
  :width: 750
  :align: center  

| 

Audience
--------

There exist many excellent deeplearning framework, such as `tensorflow`, which 
are more performant than `jenn`. However, gradient-enhancement is not inherently 
part of them and requires additional effort to implement. The present library is 
intended for those engineers in a rush with a need to accurately predict partials and 
seeking an api with a low-barrier to entry.  

Use Case(s)
-----------

JENN is primarily intended for the field of computer aided design, when there is often 
a need to replace computationally expensive, physics-based models with so-called `surrogate models` in
order to save time for further analysis down the line. The field of aerospace engineering is 
rich in examples with two important use-cases that come to mind: 

* Surrgate-based optimization 
* Uncertainty quantification

In both cases, the value proposition is that the computational expense of 
generating the training data to fit a surrogate is much less than the 
computational expense of performing the analysis with the original model itself. 
Since the `surrogate model` emulates the original model accurately 
in real time, it offers a speed benefit that can be used to carry out orders of magnitude 
more function calls quickly, enabling Monte Carlo simulations of computationally expensive functions for example. 

Limitations
-----------

Gradient-enhanced methods require responses to be continuous and smooth, 
but they are only beneficial if the cost of obtaining the gradient 
is not excessive in the first place, or if the need for accuracy outweighs the cost of 
computing the partials. The user should therefore carefully weigh the benefit of 
gradient-enhanced methods relative to the needs of their application. 

Acknowledgements
----------------

This code used the exercises by Prof. Andrew Ng in the
`Coursera Deep Learning Specialization <https://www.coursera.org/specializations/deep-learning>`_
as a starting point. It then built upon it to include additional features such
as line search and plotting but, most of all, it fundamentally changed the formulation 
to include gradient-enhancement and made sure all vectored were updated in place (data is never copied). 
The author would like to thank Andrew Ng for
offering the fundamentals of deep learning on Coursera, which took a complicated
subject and explained it in simple terms that made it accessible to laymen like the present author.

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