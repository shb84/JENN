# Gradient-Enhanced Neural Network (GENN)

Gradient-Enhanced Neural Networks (GENN) are fully connected multi-layer perceptrons, whose training process was
modified to account for gradient information. Specifically, the parameters are learned by minimizing the Least Squares
Estimator (LSE), modified to account for partial derivatives. The theory behind the algorithm is included in the docs,
but suffice it to say that the model is trained in such a way so as to minimize both the prediction error y - f(x) of
the response and the prediction error dydx - f’(x) of the partial derivatives. The chief benefit of gradient-enhancement
is better accuracy with fewer training points, compared to regular neural networks without gradient-enhancement. GENN
applies to regression (single-output or multi-output), but not classification since there is no gradient in that case.
This particular implementation is fully vectorized and uses Adam optimization, mini-batch, and L2-norm regularization.

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

# ACKNOWLEDGEMENT

THIS CODE USED THE CODE BY ANDREW NG IN THE COURSERA DEEP LEARNING SPECIALIZATION AS A STARTING POINT. IT THEN BUILT
UPON IT TO INCLUDE ADDITIONAL FEATURES SUCH AS LINE SEARCH, OBJECT ORIENTED, ETC... BUT, MOST OF ALL, IT WAS MODIFIED TO
INCLUDE A GRADIENT-ENHANCED FORMULATION. THE AUTHOR WOULD LIKE TO THANK ANDREW NG FOR OFFERING THE FUNDAMENTALS OF DEEP
LEARNING ON COURSERA, WHICH TOOK SOMETHING COMPLICATED AND EXPLAINED IN CLEAR, SIMPLE TERMS THAT MADE IT ACCESSIBLE TO
NON-SPECIALISTS.
