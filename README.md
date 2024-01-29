# Jacobian-Enhanced Neural Network (JENN)

Jacobian-Enhanced Neural Networks (JENN) are fully connected multi-layer
perceptrons, whose training process was modified to account for gradient
information. Specifically, the parameters are learned by minimizing the Least
Squares Estimator (LSE), modified to minimize prediction error of both 
response values and partial derivatives. 

The chief benefit of gradient-enhancement is better accuracy with
fewer training points, compared to full-connected neural nets without
gradient-enhancement. JENN applies to regression, but not classification since there is no gradient in that case.

This particular implementation is fully vectorized and uses Adam optimization,
mini-batch, and L2-norm regularization. Batch norm is not implemented and,
therefore, very deep networks might suffer from exploding and vanishing
gradients. This would be a useful addition for those who would like to
contribute. 

The core algorithm was written in Python 3 and requires only numpy. However, 
Matplotlib is required for plotting, some examples 
depend on pyDOE2 for generating synthetic data, and example notebooks 
require Jupyter to run. For now, documentation only exists in the form of a 
[PDF](https://github.com/shb84/JENN/blob/master/docs/theory.pdf) with the 
theory and [jupyter notebook examples](https://github.com/shb84/JENN/tree/master/demo) on the project website. 

Jacobian-Enhanced Neural Net            |  Standard Neural Net
:-------------------------:|:-------------------------:
![](pics/JENN.png)  |  ![](pics/NN.png)

> NOTE: this project was originally called GENN, but was renamed since a pypi package of that name already exists.

----

# Main Features

* Multi-Task Learning : predict more than one output with same model Y = f(X) where Y = [y1, y2, ...]
* Jacobian prediction : analytically compute the Jacobian (_i.e._ forward propagation of dY/dX)
* Gradient-Enhancement: minimize prediction error of partials (_i.e._ back-prop accounts for dY/dX)

----

# Installation

    pip install jenn 

# Contribute 

    conda env update --file environment.yml --name jenn
    conda activate jenn 
    pip install -e . 

----

# Usage

**Checkout demo for more detailed tutorials in the form of jupyter notebooks**

    import numpy as np
    from jenn import JENN
    import pickle

    def synthetic_data(): 
        f = lambda x: x * np.sin(x)
        df_dx = lambda x: np.sin(x) + x * np.cos(x) 

        # Domain 
        lb = -np.pi
        ub = np.pi

        # Training data 
        m = 4    # number of training examples
        n_x = 1  # number of inputs
        n_y = 1  # number of outputs
        X_train = np.linspace(lb, ub, m).reshape((m, n_x))
        Y_train = f(X_train).reshape((m, n_y))
        J_train = df_dx(X_train).reshape((m, n_y, n_x))

        # Test data 
        m = 30  # number of test examples
        X_test = lb + np.random.rand(m, 1).reshape((m, n_x)) * (ub - lb)
        Y_test = f(X_test).reshape((m, n_y))
        J_test = df_dx(X_test).reshape((m, n_y, n_x))

        return X_train, Y_train, J_train, X_test, Y_test, J_test

    # Generate synthetic data for this example 
    X_train, Y_train, J_train, X_test, Y_test, J_test = synthetic_data() 

    # Initialize model (gamma = 1 implies gradient enhancement)
    model = JENN(hidden_layer_sizes=(12,), activation='tanh',
                 num_epochs=1, max_iter=200, batch_size=None,
                 learning_rate='backtracking', random_state=None, tol=1e-6,
                 learning_rate_init=0.05, alpha=0.1, gamma=1, verbose=False)

    # Train neural net 
    model.fit(X_train, Y_train, J_train) 

    # Plot training history 
    history = model.training_history()

    # Visualize fit quality 
    r_square = model.goodness_fit(X_test, Y_test)

    # Predict
    Y_pred = model.predict(X_train)
    J_pred = model.jacobian(X_train)

    # Save as pkl file for re-use
    file = open('model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()

    # Assume you are starting a new script and want to reload a previously trained model:
    pkl_file = open('model.pkl', 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()

----

# Limitations

Gradient-enhanced methods requires responses to be continuous and smooth (_i.e._ gradient is 
defined everywhere), but is only beneficial when  the cost of obtaining the gradient 
is not excessive in the first place or the need for accuracy outweighs the cost of 
computing partials. The user should therefore carefully weigh the benefit of 
gradient-enhanced methods relative to the needs of the application.

----

# Use Case

JENN is unlikely to apply to real-world data since real data is usually
discrete, incomplete, and gradients are not available. However, in the field of
computer aided design, there exist a well known use case: the need to replace
computationally expensive computer models with so-called “surrogate models” in
order to save time for further analysis down the line. The field of aerospace
engineering and, more specifically, multi-disciplinary analysis and optimization
is rich in examples. In this scenario, the process typically consists of
generating a small Design Of Experiment (DOE), running the computationally
expensive computer model for each DOE point, and using the results as training
data to train a “surrogate model” (such as JENN). Since the “surrogate model”
emulates the original physics-based model accurately in real time, it offers a
speed benefit that can be used to carry out additional analysis such as
uncertainty quantification by means of Monte Carlo simulation, which would’ve
been computationally inefficient otherwise. Moreover, in the very special case
of computational fluid dynamics, adjoint design methods provide a scalable and 
efficient way to compute the gradient, making gradient-enhanced methods 
attractive (if not compelling). Otherwise, the cost of generating the gradient 
will have to be weighed against the benefit of improved accuracy depending on 
the needs of the application. 

----

# Acknowledgement

This code used the code by Prof. Andrew Ng in the
[Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
as a starting point. It then built upon it to include additional features such
as line search or plotting, but most of all, it fundamentally changed the software architecture
from pure functional programming to object oriented programming and modified the formulation 
to include a gradient-enhancement. The author would like to thank Andrew Ng for
offering the fundamentals of deep learning on Coursera, which took a complicated
subject and explained it in simple terms that made it accessible to laymen, such as the present author.
