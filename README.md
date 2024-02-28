# Jacobian-Enhanced Neural Network (JENN)

Jacobian-Enhanced Neural Networks (JENN) are fully connected multi-layer
perceptrons, whose training process is modified to predict partial 
derivatives accurately. This is accomplished by minimizing a modified version 
of the Least Squares Estimator (LSE) that accounts for Jacobian prediction error (see [theory](https://github.com/shb84/JENN/blob/master/docs/theory.pdf)). 
The main benefit of jacobian-enhancement is better accuracy with
fewer training points compared to standard fully connected neural nets, as illustrated below. 

<div align="center">

|                  Example #1                    |      Example #2                 |
|:----------------------------------------------:|:-------------------------------:|
| ![](https://github.com/shb84/JENN/raw/master/docs/pics/example_sensitivity_profile.png) | ![](https://github.com/shb84/JENN/raw/master/docs/pics/JENN_vs_NN_1D.png)|

|             Example #3           |
|:--------------------------------:|
| ![](https://github.com/shb84/JENN/raw/master/docs/pics/JENN_vs_NN_2D.png) |

</div>
 


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

Optionally, if `matplotlib` is installed, import plotting utilities:  

    from jenn.utils import plot

Optionally, if `matplotlib` is installed, check goodness of fit: 

    plot.goodness_of_fit(
        y_true=dydx_test[0], 
        y_pred=nn.predict_partials(x_test)[0], 
        title="Partial Derivative: dy/dx (JENN)"
    )

Optionally, if `matplotlib` is installed, show sensitivity profiles:

    plot.sensitivity_profiles(
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

----

# Use Case

JENN is intended for the field of computer aided design, where there is often 
a need to replace computationally expensive, physics-based models with so-called _surrogate models_ in
order to save time down the line. Since the _surrogate model_ emulates the original model accurately 
in real time, it offers a speed benefit that can be used to carry out orders of magnitude 
more function calls quickly, opening the door to Monte Carlo simulation of expensive functions for example. 

In general, the value proposition of a surrogate is that the computational 
expense of generating training data to fit the model 
is much less than the computational expense of performing the analysis with the original physics-based model itself. 
However, in the special case of gradient-enhanced methods, there is the additional value proposition that partials 
are accurate which is a critical property for one important use-case: **surrogate-based optimization**. The field of 
aerospace engineering is rich in [applications](https://doi.org/10.1002/9780470686652.eae496) of such a use-case. 

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

