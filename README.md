# Jacobian-Enhanced Neural Network (JENN)

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

The core algorithm was written in Python 3. It requires only `numpy` and `orjson` for serialization; `matplotlib` is only optional. If installed, it is used to offer basic plotting utilities such as viewing sensivity profiles and checking goodness of fit.  

Jacobian-Enhanced Neural Net            |  Standard Neural Net
:-------------------------:|:-------------------------:
![](pics/JENN.png)  |  ![](pics/NN.png)

----

# Installation

    pip install jenn 

----
# Main Features

* Multi-Task Learning : predict more than one output with same model Y = f(X) where Y = [y1, y2, ...]
* Jacobian prediction : analytically compute the Jacobian (_i.e._ forward propagation of dY/dX)
* Gradient-Enhancement: minimize prediction error of partials (_i.e._ back-prop accounts for dY/dX)

--- 
# Documentation


* [API](shb84.github.io/JENN/ ) 
* [Theory](https://github.com/shb84/JENN/blob/master/docs/theory.pdf)
* [Example 1: sinusoid](https://github.com/shb84/JENN/blob/master/notebooks/demo_1_sinusoid.ipynb)  
* [Example 2: Rastrigin](https://github.com/shb84/JENN/blob/master/notebooks/demo_2_rastrigin.ipynb)  

----

# Usage

_See demo notebooks for more details_

    import jenn

    nn = jenn.model.NeuralNet(
        layer_sizes=[1, 12, 1],
    ).fit(
        x=x_train,  # assuming (x_train, y_train, dydx_train) is given
        y=y_train, 
        dydx=dydx_train,
    )
    
    y, dydx = nn.evaluate(x)

----

# Use Case

JENN is intended for the field of computer aided design, when there is a need to replace
computationally expensive, physics-based models with so-called “surrogate models” in
order to save time for further analysis down the line. The field of aerospace engineering is rich in examples with two important use-cases
that come to mind: 

* Surrgate-based optimization 
* Monte-Carlo simulation 

In both cases, the value proposition is that the computational expense of 
generating the training data to fit a surrogate is much smaller than the 
computational expense of performing the analysis with the original model itself.  

Aerospace engineers often train models according to the following process: 

1. Generate a Design Of Experiment (DOE) over the domain of interest
2. Evaluate the DOE by running the computationally expensive computer model at each DOE point
3. Use the results as training data to train a “surrogate model” (such as JENN)

Since the “surrogate model” emulates the original physics-based model accurately 
in real time, it offers a speed benefit that can be used to carry out additional 
analysis, such as uncertainty quantification where the surrogate model enable
Monte Carlo simulations, which would’ve been much too slow otherwise. 

----

# Limitations

Gradient-enhanced methods requires responses to be continuous and smooth (_i.e._ gradient is 
defined everywhere), but is only beneficial when  the cost of obtaining the gradient 
is not excessive in the first place or the need for accuracy outweighs the cost of 
computing partials. The user should therefore carefully weigh the benefit of 
gradient-enhanced methods relative to the needs of the application. For example, in the very special 
case of computational fluid dynamics, where adjoint design methods 
provide a scalable and efficient way to compute the gradient, gradient-enhanced methods 
 are almost always attractive if not compelling.

----

# Acknowledgement

This code used the code by Prof. Andrew Ng in the
[Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
as a starting point. It then built upon it to include additional features such
as line search and plotting but, most of all, it fundamentally changed the formulation 
to include gradient-enhancement. The author would like to thank Andrew Ng for
offering the fundamentals of deep learning on Coursera, which took a complicated
subject and explained it in simple terms that made it accessible to laymen, such as the present author.
