# Jacobian-Enhanced Neural Network (JENN)

Jacobian-Enhanced Neural Networks (JENN) are fully connected multi-layer
perceptrons, whose training process is modified to predict partial 
derivatives accurately. This is accomplished by minimizing a modified version 
of the Least Squares Estimator (LSE) that accounts for Jacobian prediction error (see theory). 
The main benefit of jacobian-enhancement is better accuracy with
fewer training points compared to standard fully connected neural nets, as illustrated below. 

|                  Example #1                    |      Example #2                 |             Example #3           |
|:----------------------------------------------:|:-------------------------------:|:--------------------------------:|
| ![](docs/pics/example_sensitivity_profile.png) | ![](docs/pics/JENN_vs_NN_1D.png)| ![](docs/pics/JENN_vs_NN_2D.png) |

 


----
# Main Features

* Multi-Task Learning : predict more than one output with same model Y = f(X) where Y = [y1, y2, ...]
* Jacobian prediction : analytically compute the Jacobian (_i.e._ forward propagation of dY/dX)
* Gradient-Enhancement: minimize prediction error of partials (_i.e._ back-prop accounts for dY/dX)

----

# Installation

    pip install jenn 

----

# Example Usage

_See [demo](./docs/examples/) notebooks for more details_

Import library:  

    import jenn

Generate example training and test data:  

    x_train, y_train, dydx_train = jenn.synthetic.Sinusoid.sample(
        m_lhs=0, 
        m_levels=4, 
        lb=-3.14, 
        ub=3.14,
    )
    x_test, y_test, dydx_test = jenn.synthetic.Sinusoid.sample(
        m_lhs=30, 
        m_levels=0, 
        lb=-3.14, 
        ub=3.14,
    )


Train a model: 

    nn = jenn.model.NeuralNet(
        layer_sizes=[1, 12, 1],
    ).fit(
        x=x_train,  
        y=y_train, 
        dydx=dydx_train,
        lambd=0.1,  # regularization parameter 
        is_normalize=True,  # normalize data before fitting it
    )
    
 Make predictions: 

    y, dydx = nn.evaluate(x)

    # OR 

    y = nn.predict(x)
    dydx = nn.predict_partials(x)


Save model (parameters) for later use: 

    nn.save('parameters.json')  

Reload saved parameters into new model: 

    reloaded = jenn.model.NeuralNet(layer_sizes=[1, 12, 1]).load('parameters.json')

Optionally, if `matplotlib` is installed, check goodness of fit: 

    jenn.utils.plot.goodness_of_fit(
        y_true=dydx_test[0], 
        y_pred=nn.predict_partials(x_test)[0], 
        title="Partial Derivative: dy/dx (JENN)"
    )

Optionally, if `matplotlib` is installed, show sensitivity profiles:

    jenn.utils.plot.sensitivity_profiles(
        f=[jenn.synthetic.Sinusoid.evaluate, nn.predict], 
        x_min=x_train.min(), 
        x_max=x_train.max(), 
        x_true=x_train, 
        y_true=y_train, 
        resolution=100, 
        legend=['true', 'pred'], 
        xlabels=['x'], 
        ylabels=['y'],
    )

--- 
# Documentation


* [API](shb84.github.io/JENN/ ) 
* [Theory](https://github.com/shb84/JENN/blob/master/docs/theory.pdf)
* [Example 1: sinusoid](https://github.com/shb84/JENN/blob/master/notebooks/demo_1_sinusoid.ipynb)  
* [Example 2: Rastrigin](https://github.com/shb84/JENN/blob/master/notebooks/demo_2_rastrigin.ipynb)  

----

# Use Case

JENN is primarily intended for the field of computer aided design, when there is often 
a need to replace computationally expensive, physics-based models with so-called `surrogate models` in
order to save time for further analysis down the line. The field of aerospace engineering is 
rich in examples with two important use-cases that come to mind: 

* Surrogate-based optimization 
* Uncertainty quantification

In both cases, the value proposition is that the computational expense of 
generating the training data to fit a surrogate is much less than the 
computational expense of performing the analysis with the original model itself. 
Since the `surrogate model` emulates the original model accurately 
in real time, it offers a speed benefit that can be used to carry out orders of magnitude 
more function calls quickly, such as enabling Monte Carlo simulations of computationally expensive functions 
or repeatedly running gradient-based optimization from different starting points to find local minima. 

----

# Limitations

Gradient-enhanced methods require responses to be continuous and smooth, 
but they are only beneficial if the cost of obtaining partials 
is not excessive in the first place (e.g. adjoint methods), or if the need for accuracy outweighs the cost of 
computing the partials. Users should therefore carefully weigh the benefit of 
gradient-enhanced methods relative to the needs of their application. 

--- 
# License
Distributed under the terms of the MIT License.

----

# Acknowledgement

This code used the code by Prof. Andrew Ng in the
[Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
as a starting point. It then built upon it to include additional features such
as line search and plotting but, most of all, it fundamentally changed the formulation 
to include gradient-enhancement and made sure all arrays were updated in place (data is never copied). 
The author would like to thank Andrew Ng for
offering the fundamentals of deep learning on Coursera, which took a complicated
subject and explained it in simple terms that even an aerospace engineer could understand.

