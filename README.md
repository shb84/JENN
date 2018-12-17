# Gradient-Enhanced Neural Network (GENN)

Gradient-Enhanced Neural Networks (GENN) are fully connected multi-layer perceptrons, whose training process was
modified to account for gradient information. Specifically, the parameters are learned by minimizing the Least Squares
Estimator (LSE), modified to account for partial derivatives. The theory behind the algorithm is included in the docs,
but suffice it to say that the model is trained in such a way so as to minimize both the prediction error y - f(x) of
the response and the prediction error dydx - f’(x) of the partial derivatives. The chief benefit of gradient-enhancement
is better accuracy with fewer training points, compared to regular neural networks without gradient-enhancement. GENN
applies to regression (single-output or multi-output), but not classification since there is no gradient in that case.
This particular implementation is fully vectorized and uses Adam optimization, mini-batch, and L2-norm regularization.
Batch norm is not implemented and, therefore, very deep networks might suffer from exploding and vanishing gradients.
This would be a useful addition for those who would like to contribute.

----

# Installation

GENN is still in development mode. Therefore, in order to install it:

     pip install -e git+https://github.com/shb84/GENN.git#egg=genn

The algorithm was written in Python 3.6.4 :: Anaconda, Inc. and implemented using numpy=1.14.0. However, in addition,
certain support functions require pandas=0.23.4 and matplotlib=2.1.2 for reading CSV files and plotting.

----

# Usage

**Checkout demo for more detailed tutorials in the form of jupyter notebooks**


    from genn.model import GENN
    from genn.data import load_csv
    import pickle

    X_train, Y_train, J_train = load_csv(file='train_data.csv',
                                         inputs=["X[0]", "X[1]"],
                                         outputs=["Y[0]"],
                                         partials=[["J[0][0]", "J[0][1]"]])

    X_test, Y_test, J_test = load_csv(file='test_data.csv',
                                      inputs=["X[0]", "X[1]"],
                                      outputs=["Y[0]"],
                                      partials=[["J[0][0]", "J[0][1]"]])

    model = GENN.initialize(n_x=X_train.shape[0],
                            n_y=Y_train.shape[0],
                            deep=2,
                            wide=12)

    model.train(X=X_train,
                Y=Y_train,
                J=J_train,
                alpha=0.05,
                lambd=0.10,
                gamma=1.0,
                beta1=0.90,
                beta2=0.99,
                mini_batch_size=64,
                num_iterations=10,
                num_epochs=100,
                silent=True)

    model.plot_training_history()
    model.print_training_history()
    model.print_parameters()

    trained_parameters = model.parameters

    model.goodness_of_fit(X_test, Y_test)  # model.goodness_of_fit(X_test, Y_test, J_test, partial=1)

    Y_pred = model.evaluate(X_test)  # predict response
    J_pred = model.gradient(X_test)  # predict jacobian

    # Save as pkl file for re-use
    output = open('trained_parameters.pkl', 'wb')
    pickle.dump(trained_parameters, output)
    output.close()

    # Assume you are starting a new script and want to reload a previously trained model:
    pkl_file = open('trained_parameters.pkl', 'rb')
    trained_parameters = pickle.load(pkl_file)
    pkl_file.close()
    new_model = GENN.initialize().load_parameters(trained_parameters)  # new_model is now the same model

----

# Limitations

Gradient-enhanced methods only apply to the special use-case of computer aided design, where data is synthetically
generated using physics-based computer models, responses are continuous, and their gradient is defined. Furthermore,
gradient enhancement is only beneficial when the cost of obtaining the gradient is not excessive in the first place.
This is often true in computer-aided design with the advent of adjoint design methods for example, but it is not always
the case. The user should therefore carefully weigh the benefit of gradient-enhanced methods depending on the
application.

----

# Use Case

GENN is unlikely to apply to real data application since real data is usually discrete, incomplete, and gradients are
not available. However, in the field of computer aided design, there exist a well known use case: the need to replace
computationally expensive computer models with so-called “surrogate models” in order to save time for further analysis
down the line. The field of aerospace engineering and, more specifically, multi-disciplinary analysis and optimization
is rich in examples. In this scenario, the process typically consists of generating a small Design Of Experiment (DOE),
running the computationally expensive computer model for each DOE point, and using the results as training data to train
a “surrogate model” (such as GENN). Since the “surrogate model” emulates the original physics-based model accurately in
real time, it offers a speed benefit that can be used to carry out additional analysis such as uncertainty
quantification by means of Monte Carlo simulation, which would’ve been computationally inefficient otherwise.

----

# Acknowledgement

This code used the code by Prof. Andrew Ng in the
[Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
as a starting point. In then built
upon it to include additional features such as line search and some others, but most of all, it was modified to be
include a gradient-enhanced formulation. The author would like to thank Andrew Ng for offering the fundamentals of deep
learning on Coursera, which took a complicated subject and explained it in simple terms that made it accessible to laymen.
