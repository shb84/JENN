#!/bin/python

"""
MIT License

Copyright (c) 2018 Steven H. Berguin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

************************************ DISCLAIMER:***************************************
*                                                                                     *
* THIS CODE USED THE CODE BY ANDREW NG IN THE COURSERA DEEP LEARNING SPECIALIZATION   *
* AS A STARTING POINT. IT THEN BUILT UPON IT TO INCLUDE ADDITIONAL FEATURES SUCH AS   *
* K-FOLD CROSS-VALIDATION, LINE SEARCH, ETC... BUT, MOST OF ALL, IT WAS MODIFIED TO   *
* ADD A GRADIENT-ENHANCED FORMULATION!                                                *
*                                                                                     *
***************************************************************************************

ACKNOWLEDGEMENT: I WOULD LIKE TO THANK ANDREW NG FOR OFFERING THE FUNDAMENTALS OF DEEP
                 LEARNING ON COURSERA. THE COURSE IS WAS EXTREMELY INSIGHTFUL!

NOTE: THIS CODE WAS DEVELOPED USING PYTHON 3.6.1 AND TESTED USING BENCHMARK FUNCTIONS. IN
      ADDITION, FINITE DIFFERENCE WAS USED ALONG THE WAY IN ORDER TO VERIFY PROPER
      IMPLEMENTATION OF L_MODEL_BACKWARD() AND L_GRADS_FORWARD() (SEE CODE BELOW)

AUTHOR: STEVENBERGUIN@GATECH.EDU (SPRING 2018)
"""

import numpy as np
import pandas as pd
import math
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(1)  # fix random seed to ensure repeatability of results (feel free to comment out)

# ------------------------ D A T A   P R E - P R O C E S S I N G   M E T H O D S ---------------------------------------


def load_csv_data(file_name = "data.csv", inputs = None, outputs = None, partials = None):
    """
    Load training data from CSV file.

    Arguments:
    file_name -- the file name containing the training data (with headers)
                >> a string

    inputs -- column names (headers) to identify inputs
                >> a list of strings e.g. ["X1", "X2", "X3", ..., "XN"]

    outputs -- column names (headers) to identify outputs
                >> a list of strings e.g. ["Y1", "Y2", "Y3", ..., "YN"]

    partials -- column names (headers) to identify partials (i.e. the columns of the Jacobian matrix, J)
                >> a list of lists of strings e.g. [ ["J11", "J12", "J13"],
                                                     ["J21", "J22", "J23"],
                                                     ["J31", "J32", "J33"] ]

    Returns:
    X -- a numpy matrix of input features of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
    Y -- a numpy matrix of output labels of shape (n_y, m) where n_y = no. of outputs
    J -- a numpy array of size (n_y, n_x, m) representing the Jacobian of Y w.r.t. X:

        dY1/dX1 = J[0][0]  
        dY1/dX2 = J[0][1]
        ...
        dY2/dX1 = J[1][0]
        dY2/dX2 = J[1][1]
        ...

        N.B. To retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m
    
    """
    data = pd.read_csv(file_name)

    n_x = len(inputs)
    n_y = len(outputs)

    if n_x > 0:
        m = data[inputs[0]].size
        print("There are " + str(m) + " data points")
    else:
        print("no inputs specified")
        return None, None

    if n_y == 0:
        print("no outputs specified")
        return None, None
    
    X = np.zeros((m, n_x))
    for i in range(0, n_x):
        X[:, i] = data[inputs[i]]
    
    Y = np.zeros( (m, n_y) ) 
    for i in range(0, n_y):
        Y[:, i] = data[outputs[i]]
        
    J = np.zeros((n_y, n_x, m))
    if partials: 
        for i in range(0, n_y):
            for j in range(0, n_x):
                J[i][j] = data[partials[i][j]]
        
    return X.T, Y.T, J


def normalize_data(X, Y, J, options = {"problem_type": "classification"}):
    """
    Normalize training data to help with optimization, i.e. X_norm = (X - mu_x) / sigma_x where X is as below
                                                            Y_norm = (Y - mu_y) / sigma_y where Y is as below
                                                            J_norm = J * sigma_x/sigma_y

    Arguments:
    X -- a numpy matrix of input features of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
    Y -- a numpy matrix of output labels of shape (n_y, m) where n_y = no. of outputs
    J -- a numpy array of size (n_y, n_x, m) representing the Jacobian of Y w.r.t. X:

        dY1/dX1 = J[0][0]
        dY1/dX2 = J[0][1]
        ...
        dY2/dX1 = J[1][0]
        dY2/dX2 = J[1][1]
        ...

        N.B. To retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m
    

    Returns:
    data -- a tuple containing (X_norm, Y_norm, J_norm) representing normalized data
    scale_factors -- a dictionary containing the scale factors mu, sigma used for X and Y
    """
    scale_factors = {}

    # Copy to avoid overloading
    X_norm = np.copy(X)
    Y_norm = np.copy(Y)
    J_norm = np.copy(J)

    # Dimensions
    n_x, m = X.shape
    n_y, _ = Y.shape

    # Normalize inputs
    mu_x = np.zeros((n_x, 1))
    sigma_x = np.ones((n_x, 1))
    for i in range(0, n_x):
        mu_x[i] = np.mean(X[i])
        sigma_x[i] = np.std(X[i])
        X_norm[i] = (X[i] - mu_x[i]) / sigma_x[i]

    # Normalize outputs
    mu_y = np.zeros((n_y, 1))
    sigma_y = np.ones((n_y, 1))
    if options["problem_type"] == "classification":
        pass
    else: 
        for i in range(0, n_y):
            mu_y[i] = np.mean(Y[i])
            sigma_y[i] = np.std(Y[i])
            Y_norm[i] = (Y[i] - mu_y[i]) / sigma_y[i]

    # Normalize partials
    for i in range(0, n_y):
        for j in range(0, n_x):
            J_norm[i, j] = J[i, j] * sigma_x[j]/sigma_y[i]

    scale_factors["mu_x"] = mu_x
    scale_factors["sigma_x"] = sigma_x
    scale_factors["mu_y"] = mu_y
    scale_factors["sigma_y"] = sigma_y

    data = (X_norm, Y_norm, J_norm)
    
    return data, scale_factors

# ---------------------------- A C T I V A T I O N    F U N C T I O N S ------------------------------------------------


def sigmoid(z):
    """
    Compute the sigmoid of z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    a -- the sigmoid of z
    """
    a = 1. / (1. + np.exp(-z))
    return a


def sigmoid_grad(z):
    """
    Compute the gradient of sigmoid w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    da -- the gradient of sigmoid evaluated at z
    """
    a = sigmoid(z)
    da = a*(1.-a)
    return da


def sigmoid_second_derivative(z):
    """
    Compute the second derivative of sigmoid w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    dda -- the second derivative of sigmoid evaluated at z
    """
    a = sigmoid(z)
    da = sigmoid_grad(z)
    dda = da*(1-2*a)
    return dda


def tanh(z):
    """
    Compute the tanh of z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    a -- the tanh of z
    """
    numerator = np.exp(z) - np.exp(-z)
    denominator = np.exp(z) + np.exp(-z)
    a = np.divide(numerator, denominator)
    return a


def tanh_grad(z):
    """
    Compute the grad of tanh w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    da -- the gradient of tanh evaluated at z
    """
    a = tanh(z)
    da = 1 - np.square(a)
    return da


def tanh_second_derivative(z):
    """
    Compute the second derivative of tanh w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    dda -- the second derivative of tanh evaluated at z
    """
    a = tanh(z)
    da = tanh_grad(z)
    dda = -2*a*da
    return dda


def relu(z):
    """
    Compute the relu of z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    a -- the relu of z
    """
    a = (z > 0)*z
    return a


def relu_grad(z):
    """
    Compute the gradient of relu w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    da -- the gradient of relu evaluated at z
    """
    da = 1.0*(z > 0)
    return da


def relu_second_derivative(z):
    """
    Compute the second derivative of relu w.r.t. z

    Argument:
    z -- a scalar or numpy array of any size

    Return:
    dda -- the gradient of relu evaluated at z
    """
    dda = 0.0
    return dda
    
# ---------------------------- F O R W A R D    P R O P A G  A T I O N    M E T H O D S --------------------------------


def linear_activation_forward(A_prev, W, b, activation = 0):
    """
    Implement the linear part of a layer's forward propagation, followed by
    the activation function.

    Arguments:
    A_prev -- activations from previous layer
        >> numpy array of size (n[l-1], 1) where n[l-1] = no. nodes in previous layer

    W -- weights associated with current layer l
        >> numpy array of size (n_l, n[l-1]) where n_l = no. nodes in current layer

    b -- biases associated with current layer
        >> numpy array of size (n_l, 1)

    activation -- flag indicating what activation to use:
                    >> integer
                           -1 -- linear activation
                            0 -- sigmoid activation
                            1 -- tanh activation
                            2 -- relu activation

    Return:
    A -- a vector of post-activation values of current layer
    cache -- parameters that can be used in other functions:
            >> a tuple (A_prev, Z, W, b)    where       A_prev -- a numpy array of shape (n[l-1], m) containing previous
                                                                  layer post-activation values where:
                                                                            n[l-1] = no. nodes in previous layer
                                                                            m = no. of training examples
                                                        Z -- a numpy array of shape (n[l], m) containing linear forward
                                                             values where n_l = no. nodes in current layer
                                                        W -- a numpy array of shape (n[l], n[l-1]) containing weights of
                                                             the current layer
                                                        b -- a numpy array of shape (n[l], 1) containing biases of
                                                             the current layer
    """
    Z = np.dot(W, A_prev) + b

    if (activation == 0):
        A = sigmoid(Z)
    elif (activation == 1):
        A = tanh(Z)
    elif (activation == 2):
        A = relu(Z)
    else:
        A = Z

    cache = (A_prev, Z, W, b, activation)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implements forward propagation for the entire neural network.

    Arguments:
    X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    Returns:
    AL -- last post-activation value
        >> numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of training examples

    caches -- a list of tuples containing every cache of linear_activation_forward()
                Note: there are L-1 of them, indexed from 0 to L-2
            >> [(...), (A_prev, Z, W, b), (...)] where  A_prev -- a numpy array of shape (n[l-1], m) containing previous 
                                                                  layer post-activation values where: 
                                                                            n[l-1] = no. nodes in previous layer
                                                                            m = no. of training examples
                                                        Z -- a numpy array of shape (n[l], m) containing linear forward 
                                                             values where n_l = no. nodes in current layer
                                                        W -- a numpy array of shape (n[l], n[l-1]) containing weights of
                                                             the current layer
                                                        b -- a numpy array of shape (n[l], 1) containing biases of
                                                             the current layer
    """
    caches = []
    A = X

    L = len(parameters) // 3 # number of layers in the network (doesn't include input layer)

    for l in range(1, L+1):
        A_prev = A
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        a = parameters["a"+str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, a)
        caches.append(cache)

    AL = A
    
    return AL, caches


def L_grads_forward(X, parameters, options = {"problem_type": "classification"}):
    """
    Compute the gradient of the neural network evaluated at X.

    Argument:
    X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    options -- a dictionary containing options
               >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Return:
    JL -- numpy array of size (n_y, n_x, m) containing the Jacobian of w.r.t. X where n_y = no. of outputs

    J_caches -- list of caches containing every cache of L_grads_forward() where J stands for Jacobian
              >> a list containing [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                                          ------------------ input j --------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
    """
    J_caches = []

    if options["problem_type"] == "classification":

        dAL = None

    else:

        # Dimensions
        L = len(parameters) // 3  # number of layers in network
        n_y = parameters["W" + str(L)].shape[0]  # number of outputs
        try:
            n_x, m = X.shape  # number of inputs, number of examples
        except ValueError:
            n_x = X.size
            m = 1
            X = X.reshape(n_x, m)

        # Initialize Jacobian for layer 0 (one example)
        I = np.eye(n_x, dtype=float)

        # Initialize Jacobian for layer 0 (all m examples)
        J0 = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)

        # Initialize Jacobian for last layer
        JL = np.zeros((n_y, n_x, m))

        # Initialize caches
        for l in range(0, L):
            J_caches.append([])

        # Loop over partials
        for j in range(0, n_x):

            # Initialize (first layer)
            A = np.copy(X).reshape(n_x, m)
            A_prime_j = J0[:, j, :]

            # Loop over layers
            for l in range(1, L + 1):

                # Previous layer
                A_prev = A
                A_prime_j_prev = A_prime_j

                # Get parameters for this layer
                W = parameters["W" + str(l)]
                b = parameters["b" + str(l)]
                activation = parameters["a" + str(l)]

                # Linear
                Z = np.dot(W, A_prev) + b

                # The following is not needed here, but it is needed later, during backprop.
                # We will thus compute it here and store it as a cache for later use.
                Z_prime_j = np.dot(W, A_prime_j_prev)

                # Activation
                if (activation == -1):  # linear
                    A = Z
                    G_prime = np.ones(Z.shape)
                if (activation == 0):  # sigmoid
                    A = sigmoid(Z)
                    G_prime = sigmoid_grad(Z)
                elif (activation == 1):  # tanh
                    A = tanh(Z)
                    G_prime = tanh_grad(Z)
                elif (activation == 2):  # relu
                    A = relu(Z)
                    G_prime = relu_grad(Z)

                # Current layer output gradient
                A_prime_j = G_prime * np.dot(W, A_prime_j_prev)

                # Store cache
                J_caches[l - 1].append((j, Z_prime_j, A_prime_j_prev))

            # Store partial
            JL[:, j, :] = A_prime_j

        if m == 1:
            JL = JL[:, :, 0]

    return JL, J_caches


def L_grads_forward_FD(X, parameters, options, dx=1e-7):
    """
    Compute the gradient of the neural network evaluated at x using central difference.

    Argument:
    X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    options -- a dictionary containing options
               >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    dx -- step size to be used for finite differencing, float

    Return:
    J -- numpy array of size (n_y, n_x, m) containing the Jacobian of w.r.t. X where n_y = no. of outputs
    """
    if options["problem_type"] == "classification":

        J = None

    else:

        # Dimensions
        L = len(parameters) // 3  # number of layers in network
        n_y = parameters["W" + str(L)].shape[0]  # number of outputs
        try:
            n_x, m = X.shape  # number of inputs, number of examples
        except ValueError:
            n_x = X.size
            m = 1
            X = X.reshape(n_x, m)

        # Initialize
        J = np.zeros((n_y, n_x, m))

        # Loop
        for j in range(0, n_x):
            # Initialize
            X_plus = np.copy(X).reshape(n_x, m)
            X_minus = np.copy(X).reshape(n_x, m)

            # Step forward
            X_plus[j, :] = X[j, :] + dx
            Y_plus = predict(X_plus, parameters, options)

            # Step backward
            X_minus[j, :] = X[j, :] - dx
            Y_minus = predict(X_minus, parameters, options)

            # Central difference
            J_tmp = (Y_plus - Y_minus) / (2 * dx)
            J[:, j, :] = J_tmp

        if m == 1:
            J = J[:, :, 0]

    return J


def L_grads_forward_check(X, parameters, options, tol=1e-6):
    """
    Compares analytical gradients vs. finite difference. Mostly useful for debugging during code development.

    Argument:
    X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    options -- a dictionary containing options
               >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"
    """

    # Small number to avoid division by zero
    epsilon = 1e-6

    # As computed analytically
    J, _ = L_grads_forward(X, parameters, options)

    # As computed using finite difference
    J_FD = L_grads_forward_FD(X, parameters, options)

    # Difference
    numerator = np.linalg.norm(J - J_FD)
    denominator = np.linalg.norm(J) + np.linalg.norm(J_FD)
    difference = numerator / (denominator + epsilon)

    # Message
    if difference <= tol or numerator <= tol:
        print("The forward propagation gradient d(Y)/dX is correct :-)")
    else:
        print("The forward propagation gradient d(Y)/dX is wrong :-(")

# ---------------------------- B A C K P R O P A G A  T I O N   M E T H O D S ------------------------------------------


def initialize_backprop(AL, Y, AL_prime, Y_prime, options={"problem_type": "classification"}):
    """
    Initialize backward propagation

    Arguments:
    AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    options -- a dictionary containing options
               >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Returns:
    dAL -- gradient of the loss function w.r.t. last layer activations: d(L)/dAL
           >> a numpy array of shape (n_y, m)

    dAL_prime -- gradient of the loss function w.r.t. last layer activations derivatives: d(L)/dAL' where AL' = d(AL)/dX
                 >> a numpy array of shape (n_y, n_x, m)
    """

    # Extract options needed for this method
    problem_type = options["problem_type"]

    # Some terms needed
    epsilon = 1e-12  # small number to avoid division by zero
    n_y, _ = AL.shape  # number layers, number examples
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Gradient
    dAL = None
    dAL_prime = None
    if problem_type == "classification":
        if n_y == 1:  # binary classification
            dAL = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
        else:  # softmax
            # placeholder for softmax
            pass
    elif problem_type == "regression":
        dAL = AL - Y  # derivative of loss function w.r.t. to activations: dAL = d(L)/dAL
        dAL_prime = AL_prime - Y_prime  # derivative of loss function w.r.t. to partials: dAL_prime = d(L)/d(AL_prime)

    return dAL, dAL_prime


def linear_activation_backward(dA, dA_prime, cache, J_cache, hyperparameters = {"lambd": 0.0, "gamma": 0.0}
                                                            , options         = {"problem_type": "classification"}):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient w.r.t. A for current layer l, dA = d(L)/dA where L is the loss function
            >> a numpy array of shape (n_1, m) where n_l = no. nodes in current layer, m = no. of examples

    dA_prime -- post-activation gradient w.r.t. A' for current layer l, dA' = d(L)/dA' where L is the loss function
                                                                                        and A' = d(AL) / dX
            >> a numpy array of shape (n_l, n_x, m) where n_l = no. nodes in current layer
                                                          n_x = no. of inputs (X1, X2, ...)
                                                          m = no. of examples

    cache -- tuple of values stored in linear_activation_forward()
              >> a tuple containing (A_prev, Z, W, b, activation)
                      where
                            A_prev -- activations from previous layer
                                      >> a numpy array of shape (n_prev, m) where n_prev is the no. nodes in layer L-1
                            Z -- input to activation functions for current layer
                                      >> a numpy array of shape (n, m) where n is the no. nodes in layer L
                            W -- weight parameters for current layer
                                 >> a numpy array of shape (n, n_prev)
                            b -- bias parameters for current layer
                                 >> a numpy array of shape (n, 1)

    J_cache -- list of caches containing every cache of L_grads_forward() where J stands for Jacobian
              >> a list containing [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                                          ------------------ input j --------------------
                      where
                            j -- input variable associated with current cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # Extract information from current layer cache (avoids recomputing what was previously computed)
    A_prev, Z, W, b, activation = cache

    # Some dimensions that will be useful
    m = A_prev.shape[1]  # number of examples
    n = len(J_cache)    # number of inputs

    # Hyper-parameters that will be needed to compute cost function
    lambd = hyperparameters["lambd"]  # regularization
    gamma = hyperparameters["gamma"]  # gradient-enhancement

    # Convenient intermediate variables to clean-up notation
    gamma_0 = 1. / m        # this term will multiply the 0th order terms in the cost function (hence, 0)
    gamma_1 = gamma / m     # this term will multiply the 1st order terms in the cost function (hence, 1)
    lambd_a = lambd / m     # this term will multiply the regularization penalty term in the cost function

    # 1st derivative of activation function A = G(Z)
    if activation == -1:  # linear activation
        G_prime = np.ones(Z.shape)
    if activation == 0:   # sigmoid activation
        G_prime = sigmoid_grad(Z)
    elif activation == 1: # tanh activation
        G_prime = tanh_grad(Z)
    elif activation == 2: # relu activation
        G_prime = relu_grad(Z)

    # Compute the contribution due to the 0th order terms (where regularization only affects dW)
    dW = gamma_0 * np.dot(G_prime*dA, A_prev.T) + lambd_a * W  # dW = d(J)/dW where J is the cost function
    db = gamma_0 * np.sum(G_prime*dA, axis=1, keepdims=True)   # db = d(J)/db
    dA_prev = np.dot(W.T, G_prime*dA)                          # dA_prev = d(L)/dA_prev where A_prev = previous layer activation

    # Initialize dA_prime_prev = d(J)/dA_prime_prev
    dA_prime_prev = np.zeros((W.shape[1], n, m))

    # Gradient enhancement
    if gamma == 0 or options["problem_type"] != "regression":
        pass
    else:

        # 2nd derivative of activation function A = G(Z)
        if activation == -1:  # linear activation
            G_prime_prime = np.zeros(Z.shape)
        if activation == 0:  # sigmoid activation
            G_prime_prime = sigmoid_second_derivative(Z)
        elif activation == 1:  # tanh activation
            G_prime_prime = tanh_second_derivative(Z)
        elif activation == 2:  # relu activation
            G_prime_prime = relu_second_derivative(Z)

        # Loop over partials, d()/dX_j
        for j_cache in J_cache:

            # Extract information from current layer cache associated with derivative of A w.r.t. j^th input
            j, Z_prime_j, A_prime_j_prev = j_cache

            # Extract partials of A w.r.t. to j^th input, i.e. A_prime_j = d(A)/dX_j
            dA_prime_j = dA_prime[:, j, :].reshape(Z_prime_j.shape)
                
            # Compute contribution to cost function gradient, db = d(J)/db, dW = d(J)/dW, d(L)/dA, d(L)/dA'
            dW += gamma_1 * (np.dot(dA_prime_j * G_prime_prime * Z_prime_j, A_prev.T) +
                             np.dot(dA_prime_j * G_prime, A_prime_j_prev.T))
            db += gamma_1 * np.sum(dA_prime_j * G_prime_prime * Z_prime_j, axis=1, keepdims=True)
            dA_prev += gamma * np.dot(W.T, dA_prime_j * G_prime_prime * Z_prime_j)
            dA_prime_prev[:, j, :] = gamma * np.dot(W.T, dA_prime_j * G_prime)

    return dA_prev, dW, db, dA_prime_prev


def L_model_backward(AL, Y, AL_prime, Y_prime, caches, J_caches
                                                     , hyperparameters = {"lambd": 0.0, "gamma": 0.0}
                                                     , options         = {"problem_type": "classification"}):
    """
    Implement backward propagation

    Arguments:
    AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    caches -- list of caches containing every cache of L_model_forward()
              >> a tuple containing {(A_prev, Z, W, b, activation), ..., (A_prev, Z, W, b, activation)}
                                      -------- layer 1 -----------        -------- layer L ----------
                      where
                            A_prev -- activations from previous layer
                                      >> a numpy array of shape (n_prev, m) where n_prev is the no. nodes in layer L-1
                            Z -- input to activation functions for current layer
                                      >> a numpy array of shape (n, m) where n is the no. nodes in layer L
                            W -- weight parameters for current layer
                                 >> a numpy array of shape (n, n_prev)
                            b -- bias parameters for current layer
                                 >> a numpy array of shape (n, 1)

    J_caches -- a list of lists containing every cache of L_grads_forward() for each layer (where J stands for Jacobian)
              >> a tuple [ [[...], ..., [...]], ..., [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...], ...]
                            --- layer 1 ------        ------------------ layer l, partial j ---------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Returns:
    grads -- A dictionary with the gradients of the cost function w.r.t. to parameters:
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
    """
    # Initialize grads
    grads = {}

    # Some quantities needed
    L = len(caches)  # the number of layers
    _, m = AL.shape
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dA, dA_prime = initialize_backprop(AL, Y, AL_prime, Y_prime, options)

    # Loop over each layer
    for l in reversed(range(L)):

        # Get cache
        cache = caches[l]
        J_cache = J_caches[l]

        # Backprop step
        dA, dW, db, dA_prime = linear_activation_backward(dA, dA_prime, cache, J_cache, hyperparameters, options)

        # Store result
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def L_model_backward_FD(Y, Y_prime, X, parameters, hyperparameters = {"lambd": 0.0, "gamma": 0.0}
                                                 , options = {"problem_type": "classification"}, epsilon=1e-6):
    """
    Implement the backward propagation using finite difference (this function is useful for debugging, but inefficient)

    Arguments:
    X -- inputs to neural network
        >> a numpy array of shape (n_x, m) where n_x = no. inputs, m = no. training examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX (Jacobian)
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Returns:
    gradapprox -- A dictionary with the gradients of the cost function w.r.t. to parameters:..
                    gradapprox["dW" + str(l)] = ...
                    gradapprox["db" + str(l)] = ...
    """
    # Initialize
    gradapprox = {}
    L = len(parameters) // 3  # the number of layers in the network

    # Loop for each layer
    for l in range(1, L+1):

        # Finite difference W, b parameters
        W = np.copy(parameters["W" + str(l)])
        b = np.copy(parameters["b" + str(l)])
        n = W.shape[0]
        p = W.shape[1]
        dW = np.zeros((n, p))
        db = np.zeros((n, 1))

        for i in range(0, n):

            # Weight, W
            # ---------
            for j in range(0, p):

                # (Step forward)
                parameters["W" + str(l)][i, j] = W[i, j] + epsilon
                AL_plus, _ = L_model_forward(X, parameters)
                JL_plus, _ = L_grads_forward(X, parameters, options)
                J_plus = compute_cost(AL_plus, Y, JL_plus, Y_prime, parameters, hyperparameters, options)
                parameters["W" + str(l)] = np.copy(W)

                # (Step backward)
                parameters["W" + str(l)][i, j] = W[i, j] - epsilon
                AL_minus, _ = L_model_forward(X, parameters)
                JL_minus, _ = L_grads_forward(X, parameters, options)
                J_minus = compute_cost(AL_minus, Y, JL_minus, Y_prime, parameters, hyperparameters, options)
                parameters["W" + str(l)] = np.copy(W)

                # (Central difference)
                dW[i, j] = (J_plus - J_minus) / (2*epsilon)

            # Bias, b
            # ---------
            # (Step forward)
            parameters["b" + str(l)][i] = b[i] + epsilon
            AL_plus, _ = L_model_forward(X, parameters)
            JL_plus, _ = L_grads_forward(X, parameters, options)
            J_plus = compute_cost(AL_plus, Y, JL_plus, Y_prime, parameters, hyperparameters, options)
            parameters["b" + str(l)] = np.copy(b)

            # (Step backward)
            parameters["b" + str(l)][i] = b[i] - epsilon
            AL_minus, _ = L_model_forward(X, parameters)
            JL_minus, _ = L_grads_forward(X, parameters, options)
            J_minus = compute_cost(AL_minus, Y, JL_minus, Y_prime, parameters, hyperparameters, options)
            parameters["b" + str(l)] = np.copy(b)

            # (Central difference)
            db[i] = (J_plus - J_minus) / (2*epsilon)
                    
        # Store information for this layer 
        gradapprox["dW"+str(l)] = dW
        gradapprox["db"+str(l)] = db

    return gradapprox    


def L_model_backward_check(AL, Y, AL_prime, Y_prime, X, parameters, caches, J_caches
                                                      , hyperparameters = {"lambd": 0.0, "gamma": 0.0}
                                                      , options = {"problem_type": "classification"}, tol=1e-6):
    """
    Check analytical gradients by comparing to finite difference

    Arguments:
    X -- inputs to neural network
        >> a numpy array of shape (n_x, m) where n_x = no. inputs, m = no. training examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX (Jacobian)
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    caches -- list of caches containing every cache of L_model_forward()
              >> a tuple containing {(A_prev, Z, W, b, activation), ..., (A_prev, Z, W, b, activation)}
                                      -------- layer 1 -----------        -------- layer L ----------
                      where
                            A_prev -- activations from previous layer
                                      >> a numpy array of shape (n_prev, m) where n_prev is the no. nodes in layer L-1
                            Z -- input to activation functions for current layer
                                      >> a numpy array of shape (n, m) where n is the no. nodes in layer L
                            W -- weight parameters for current layer
                                 >> a numpy array of shape (n, n_prev)
                            b -- bias parameters for current layer
                                 >> a numpy array of shape (n, 1)

    J_caches -- a list of lists containing every cache of L_grads_forward() for each layer (where J stands for Jacobian)
              >> a tuple [ [[...], ..., [...]], ..., [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...], ...]
                            --- layer 1 ------        ------------------ layer l, partial j ---------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"
    
    lambd -- regularization parameter
    caches -- list of caches containing every cache of linear_activation_forward()
    tol -- acceptable error tolerance between numeric and analytic gradients

    Return:
    grads -- A dictionary of finite difference gradients of the cost function w.r.t.
             to parameters: grads["dW" + str(l)] = ...
                            grads["db" + str(l)] = ...
    """
    # A small number to avoid division by zero
    epsilon = 1e-8
    
    # Compute analytical gradients
    grads = L_model_backward(AL, Y, AL_prime, Y_prime, caches, J_caches, hyperparameters, options)
    
    # Compute numerical gradients
    gradapprox = L_model_backward_FD(Y, Y_prime, X, parameters, hyperparameters, options)

    # Check the gradients for the parameters of each layer
    number_layers = len(parameters) // 3 
    for l in range(1, number_layers + 1):
        
        # Weight, W
        numerator = np.linalg.norm((grads["dW"+str(l)] - gradapprox["dW"+str(l)]))

        denominator = np.linalg.norm(grads["dW"+str(l)]) + np.linalg.norm(gradapprox["dW"+str(l)])
        difference = numerator / (denominator + epsilon)
        if difference <= tol or numerator <= tol:
            print("The gradient of W" + str(l) + " is correct!")
        else:
            print("The gradient of W" + str(l) + " is wrong!")
        
        # Bias, b
        numerator = np.linalg.norm((grads["db"+str(l)] - gradapprox["db"+str(l)]))
        denominator = np.linalg.norm(grads["db"+str(l)]) + np.linalg.norm(gradapprox["db"+str(l)])
        difference = numerator / (denominator + epsilon)
        if difference <= tol or numerator <= tol:

            print("The gradient of b" + str(l) + " is correct!")
        else:
            print("The gradient of b" + str(l) + " is wrong!")

    return gradapprox

# ---------------------------- O P T I M I Z A T I O N   M E T H O D S -------------------------------------------------


def compute_cost(AL, Y, AL_prime, Y_prime, parameters, hyperparameters={"lambd": 0.0, "gamma": 0.0}
                                                     , options        ={"problem_type": "classification"}):
    """
    Computes the cost function based on type. This method only applies to regression (i.e. can't
    predict gradients for a classification problem).

    Arguments:
    AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Returns:
    cost -- cost function value
            >> float
    """
    # Get some information about the network
    L = len(parameters) // 3  # number of layers in the network (doesn't include input layer)
    K, m = Y.shape  # number of outputs, training examples
    N = parameters["W1"].shape[1]  # number of nodes in input layer

    # Extract hyperparameters needed for this method
    lambd = hyperparameters["lambd"]
    gamma = hyperparameters["gamma"]

    gamma_0 = 1. / m
    gamma_1 = gamma / m
    lambd_a = lambd / m

    # A small number to help avoid division by zero or log(0)
    eps = 1e-8

    # Cost before regularization
    J0 = 0.
    J1 = 0.
    if options["problem_type"] == "classification":
        if K == 1:  # binary classification
            J0 = - gamma_0 * np.sum(Y * np.log(AL + eps) + (1. - Y) * np.log(1. - AL + eps), axis=1, keepdims=True)
        else:
            # Place holder for softmax
            pass
    elif options["problem_type"] == "regression":

        # 0th derivative terms
        for k in range(0, K):
            J0 += 0.5 * gamma_0 * np.dot((AL[k, :] - Y[k, :]), (AL[k, :] - Y[k, :]).T)

        # 1st derivative terms
        if gamma_1 != 0:
            for k in range(0, K):
                for j in range(0, N):
                    dY_j_pred = AL_prime[k, j, :].reshape(1, m)
                    dY_j_true = Y_prime[k, j, :].reshape(1, m)
                    J1 += 0.5 * gamma_1 * np.dot((dY_j_pred - dY_j_true), (dY_j_pred - dY_j_true).T)
    else:
        pass

    # Regularization using L2 penalty
    regularization = 0.0
    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        regularization += 0.5 * lambd_a * np.sum(np.square(W))

    # Cost function after regularization
    cost = J0 + J1 + regularization
    cost = np.squeeze(cost)
    return cost


def update_parameters_with_gd(parameters, grads, alpha = 0.1):
    """
    Update parameters using gradient descent.

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    grads -- python dictionary containing your gradients, output of L_model_backward()
                        >> a dictionary containing: {"dW1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"dW2": a numpy array of shape (n[2], n[1])}
                                                    {"dW3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"dWL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"db1": a numpy array of shape (n[1], 1)}
                                                    {"db2": a numpy array of shape (n[2], 1)}
                                                    {"db3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"dbL": a numpy array of shape (n[L], 1)}

    alpha -- learning rate
            >> float

    Returns:
    parameters -- updated parameters
    """
    # Avoid overloading by copying numpy array
    parameters_copy = parameters.copy()

    L = len(parameters) // 3  # number of layers in the neural network (doesn't include input layer)

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L+1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]
        parameters_copy["W" + str(l)] = W - alpha*dW
        parameters_copy["b" + str(l)] = b - alpha*db
        
    return parameters_copy


def initialize_momentum(parameters):
    """
    Initialize velocity parameters for gradient descent with momentum

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    Return:
    v -- python dictionary containing the current velocity:
         v["dW1"] = velocity of dWl, float
         v["db1"] = velocity of dbl, float
         v["dW2"] = velocity of dW2, float
         v["db2"] = velocity of db2, float
         ...
    """
    v = {}

    L = len(parameters) // 3

    for l in range(1, L + 1):
        v["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, alpha = 0.1, beta = 0.9):
    """
    Update parameters using gradient descent with momentum.

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    grads -- python dictionary containing your gradients, output of L_model_backward()
                        >> a dictionary containing: {"dW1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"dW2": a numpy array of shape (n[2], n[1])}
                                                    {"dW3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"dWL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"db1": a numpy array of shape (n[1], 1)}
                                                    {"db2": a numpy array of shape (n[2], 1)}
                                                    {"db3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"dbL": a numpy array of shape (n[L], 1)}

    v -- python dictionary containing the current velocity (moving average of the first derivative):
         v["dW1"] = velocity of dWl, float
         v["db1"] = velocity of dbl, float
         v["dW2"] = velocity of dW2, float
         v["db2"] = velocity of db2, float
         ...

    alpha -- learning rate hyper-parameter
            >> float

    beta -- momentum decay hyper-parameter
            >> float

    Returns:
    parameters -- updated parameters
    v -- updated v
    """
    # Avoid overloading by copying numpy arrays
    v_copy = v.copy()
    parameters_copy = parameters.copy()
    
    L = len(parameters) // 3  # number of layers in the neural network (doesn't include input layer)

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L+1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]
        vdW = v["dW" + str(l)]
        vdb = v["db" + str(l)]
        vdW = beta*vdW + (1.-beta)*dW
        vdb = beta*vdb + (1.-beta)*db
        parameters_copy["W" + str(l)] = W - alpha*vdW
        parameters_copy["b" + str(l)] = b - alpha*vdb
        v_copy["dW" + str(l)] = vdW
        v_copy["db" + str(l)] = vdb
        
    return parameters_copy, v_copy


def initialize_adam(parameters):
    """
    Initialize adam optimization

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    Return:
    v -- python dictionary containing the current velocity:
         v["dW1"] = velocity of dWl, integer
         v["db1"] = velocity of dbl, integer
         v["dW2"] = velocity of dW2, integer
         v["db2"] = velocity of db2, integer
         ...
    s -- python dictionary containing the current square velocity:
         s["dW1"] = square velocity of dWl, float
         s["db1"] = square velocity of dbl, float
         s["dW2"] = square velocity of dW2, float
         s["db2"] = square velocity of db2, float
    """

    v = {}
    s = {}

    L = len(parameters) // 3

    for l in range(1, L + 1):
        v["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)
        s["dW"+str(l)] = np.zeros(parameters["W"+str(l)].shape)
        s["db"+str(l)] = np.zeros(parameters["b"+str(l)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, alpha = 0.1, beta1 = 0.9, beta2 = 0.99):
    """
    Update parameters using gradient descent with momentum.

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    grads -- python dictionary containing your gradients, output of L_model_backward()
                        >> a dictionary containing: {"dW1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"dW2": a numpy array of shape (n[2], n[1])}
                                                    {"dW3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"dWL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"db1": a numpy array of shape (n[1], 1)}
                                                    {"db2": a numpy array of shape (n[2], 1)}
                                                    {"db3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"dbL": a numpy array of shape (n[L], 1)}

    v -- Adam variable, moving average of the gradient, python dictionary:
         v["dW1"] = velocity of dWl, integer
         v["db1"] = velocity of dbl, integer
         v["dW2"] = velocity of dW2, integer
         v["db2"] = velocity of db2, integer
         ...

    s -- Adam variable, moving average of the squared gradient, python dictionary:
         s["dW1"] = square velocity of dWl, float
         s["db1"] = square velocity of dbl, float
         s["dW2"] = square velocity of dW2, float
         s["db2"] = square velocity of db2, float

    t -- current optimizer iteration
        >> integer

    alpha -- learning rate (hyper-parameter)
            >> float

    beta1 -- momentum decay (hyper-parameter)
            >> float

    beta2 -- adam decay (hyper-parameter)
            >> float

    Returns:
    parameters -- updated parameters
    v -- updated v
    s -- updated s
    """
    # Avoid overloading by copying numpy arrays
    v_copy = v.copy()
    s_copy = s.copy()
    parameters_copy = parameters.copy()
    
    L = len(parameters) // 3 # number of layers in the neural network (doesn't include input layer)

    # Update rule for each parameter. Use a for loop.
    epsilon = 1e-8
    for l in range(1, L+1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]
        vdW = v["dW" + str(l)]
        vdb = v["db" + str(l)]
        sdW = s["dW" + str(l)]
        sdb = s["db" + str(l)]
        vdW = beta1*vdW + (1.-beta1)*dW
        vdb = beta1*vdb + (1.-beta1)*db
        sdW = beta2*sdW + (1.-beta2)*np.square(dW)
        sdb = beta2*sdb + (1.-beta2)*np.square(db)
        vdW_corrected = vdW / (1.-beta1**t)
        vdb_corrected = vdb / (1.-beta1**t)
        sdW_corrected = sdW / (1.-beta2**t)
        sdb_corrected = sdb / (1.-beta2**t)
        parameters_copy["W" + str(l)] = W - alpha*vdW_corrected/(np.sqrt(sdW_corrected)+epsilon)
        parameters_copy["b" + str(l)] = b - alpha*vdb_corrected/(np.sqrt(sdb_corrected)+epsilon)
        v_copy["dW" + str(l)] = vdW
        v_copy["db" + str(l)] = vdb
        s_copy["dW" + str(l)] = sdW
        s_copy["db" + str(l)] = sdb
        
    return parameters_copy, v_copy, s_copy


def initialize_optimizer(parameters, options):
    """
    Initializes optimizer parameters

    Arguments:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    Return:
    optimizer_parameters -- python dictionary containing initial optimizer parameters:
                    v -- python dictionary, moving average of the first gradient (applies to GD with Adam or Momentum)
                    s -- python dictionary, moving average of the squared gradient (applies to GD with Adam)
                    i -- iteration number 
    """
    v = None
    s = None
    if options["optimizer"] == "gd":
        pass
    elif options["optimizer"] == "momentum":
        v = initialize_momentum(parameters)
    elif options["optimizer"] == "adam":
        v, s = initialize_adam(parameters)

    optimizer_parameters = {"v": v, "s": s, "i": 0}
    
    return optimizer_parameters


def update_with_backtracking_linesearch(X, Y, Y_prime, cost, grads, parameters, optimizer_parameters, hyperparameters,
                                        options, tau=0.5):
    """
    Perform backtracking line search

    Arguments:
    X -- inputs to neural network
        >> a numpy array of shape (n_x, m) where n_x = no. inputs, m = no. training examples

    Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    cost -- float, current value of cost function at the start of line search

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation


    optimizer_parameters -- python dictionary containing optimizer parameters from initialize_optimizer()
                                v -- Adam variable, moving average of the gradient, python dictionary:
                                 v["dW1"] = velocity of dWl, integer
                                 v["db1"] = velocity of dbl, integer
                                 v["dW2"] = velocity of dW2, integer
                                 v["db2"] = velocity of db2, integer
                                 ...
                                s -- Adam variable, moving average of the squared gradient, python dictionary:
                                     s["dW1"] = square velocity of dWl, float
                                     s["db1"] = square velocity of dbl, float
                                     s["dW2"] = square velocity of dW2, float
                                     s["db2"] = square velocity of db2, float
                                i -- current optimizer iteration number, integer

    hyperparameters -- a dictionary containing hyper-parameters
                        >> {"alpha": float}, learning rate
                        >> {"lambd": float}, regularization parameter
                        >> {"gamma": float}, gradient-enhancement parameter

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    tau -- a hyper-parameter between 0 and 1 used to reduce alpha during backtracking line search, float


    Return:
    new_parameters -- python dictionary containing updated neural net parameters
    opt_parameters -- python dictionary containing updated optimizer parameters
    """
    # Avoid overloading by copying array
    parameters = parameters.copy()
    opt_parameters = optimizer_parameters.copy()

    # Store initial learning rate, alpha
    initial_alpha = hyperparameters["alpha"]

    # Line search
    converged = False
    while not (converged):
        new_parameters, opt_parameter = update_parameters(grads, parameters, optimizer_parameters, hyperparameters,
                                                          options)
        AL, _ = L_model_forward(X, new_parameters)
        AL_prime, _ = L_grads_forward(X, new_parameters, options)
        new_cost = compute_cost(AL, Y, AL_prime, Y_prime, new_parameters, hyperparameters, options)
        if new_cost < cost:
            converged = True
        elif hyperparameters["alpha"] == 0.0:
            converged = True
        else:
            hyperparameters["alpha"] = hyperparameters["alpha"] * tau
            if hyperparameters["alpha"] < 1e-6:
                hyperparameters["alpha"] = 0.0

    # Restore initial learning rate, alpha (to avoid overloading)
    hyperparameters["alpha"] = initial_alpha

    return new_parameters, opt_parameters


def update_parameters(grads, parameters, optimizer_parameters = {"v": {}, "s": {}, "i": 0}
                                       , hyperparameters = {"alpha": 0.1, "beta1": 0.9, "beta2": 0.99}
                                       , options = {"optimizer": "gd"}):
    """
    Update parameters using gradient descent. Specifically, performs on step (update) of gradient descent.

    Arguments
    grads -- python dictionary containing your gradients, output of L_model_backward()
                        >> a dictionary containing: {"dW1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"dW2": a numpy array of shape (n[2], n[1])}
                                                    {"dW3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"dWL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"db1": a numpy array of shape (n[1], 1)}
                                                    {"db2": a numpy array of shape (n[2], 1)}
                                                    {"db3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"dbL": a numpy array of shape (n[L], 1)}

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    optimizer_parameters -- python dictionary containing optimizer parameters
                            v -- Adam variable, moving average of the gradient, python dictionary:
                                 v["dW1"] = velocity of dWl, integer
                                 v["db1"] = velocity of dbl, integer
                                 v["dW2"] = velocity of dW2, integer
                                 v["db2"] = velocity of db2, integer
                                 ...
                            s -- Adam variable, moving average of the squared gradient, python dictionary:
                                 s["dW1"] = square velocity of dWl, float
                                 s["db1"] = square velocity of dbl, float
                                 s["dW2"] = square velocity of dW2, float
                                 s["db2"] = square velocity of db2, float
                            i -- current optimizer iteration number, integer
    
    hyperparameters -- a dictionary containing hyper-parameters:
    
                    alpha -- float, learning rate, float
                    beta1 -- float  momentum decay, float
                    beta2 -- float, adam decay, float
                       
    options -- a dictionary containing options 
    
                    optimizer -- a string indicating what optimizer to use:
                                 --> "gd" -- gradient descent (worst)
                                 --> "momentum" -- gradient descent with momentum (better)
                                 --> "adam" -- gradient descent with adam (best)

    Return:
    new_parameters -- updated neural network parameters
    opt_parameters -- updated optimizer parameters
    """
    # Avoid overloading by copying array
    old_parameters = parameters.copy()
    opt_parameters = optimizer_parameters.copy()

    # Extract hyper-parameters needed for this method
    alpha = hyperparameters["alpha"]
    beta1 = hyperparameters["beta1"]
    beta2 = hyperparameters["beta2"]

    # Extract optimizer parameters needed for this method
    v = optimizer_parameters["v"]
    s = optimizer_parameters["s"]
    i = optimizer_parameters["i"]

    # Extract options needed for this method
    optimizer = options["optimizer"]

    # Select update method
    if optimizer == "gd" or i == 0: # always use GD for 1st iteration (helps initialize v, s) 
        new_parameters = update_parameters_with_gd(old_parameters, grads, alpha)
    elif optimizer == "momentum":
        new_parameters, v = update_parameters_with_momentum(old_parameters, grads, v, alpha, beta1)
    elif optimizer == "adam":
        t = i + 1  # adam iteration counter
        new_parameters, v, s = update_parameters_with_adam(old_parameters, grads, v, s, t, alpha, beta1, beta2)

    # Update optimizer parameters
    opt_parameters["v"] = v
    opt_parameters["s"] = s
    
    return new_parameters, opt_parameters

# ---------------------------- P R E D I C T I O N   M E T H O D S -----------------------------------------------------


def predict(X, parameters, options = {"problem_type": "classification"}):
    """
    Predicts label (response) given input feature X. In other words, it evaluates the neural network given X.

    Arguments:
    X -- inputs to neural network
        >> a numpy array of shape (n_x, m) where n_x = no. inputs, m = no. training examples

    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Return:
    predictions -- a numpy array of neural net predictions of shape (n_y, m) where n_y = no. of outputs
                                                                                   m = no. of examples to evaluate
    """
    AL, _ = L_model_forward(X, parameters)

    if options["problem_type"] == "classification": 
        predictions = (AL > 0.5)
    elif options["problem_type"] == "regression": 
        predictions = AL
    else:
        pass

    return predictions


# ----------------------- N E U R A L   N E T   T R A I N I N G   M E T H O D S ----------------------------------------


def initialize_parameters(data , hyperparameters = {"hidden_layer_activation": "relu",
                                                    "hidden_layer_dimensions": [2]}
                                , options        = {"problem_type": "classification"}):
    """
    Initialize neural network given topology

    Arguments:
    data -- a tuple of input-output value pairs (X, Y, J):
    
            X -- a numpy array of size (n_x, m) containing input features of the training data
            Y -- a numpy array of size (n_y, m) containing output values of the training data
            J -- a numpy array of size (n_y, n_x, m) where m = number of examples
                                                           n_y = number of outputs
                                                           n_x = number of inputs
            
    hyperparameters -- a dictionary containing hyperparameters:
    
               {"hidden_layer_activation": string} -- string indicating hidden layer activation function:
                                                      --> "sigmoid"
                                                      --> "tanh"
                                                      --> "relu"
               {"hidden_layer_dimensions": list} -- list of integers indicating number of nodes per layer:
                                                    --> e.g. [10, 5, 2] = Layer 1: 10 nodes
                                                                          Layer 2: 5 nodes
                                                                          Layer 3: 2 nodes
                       
    options -- a dictionary containing options
                >> {"problem_type": string}, indicates whether neural net is for "classification" or "regression"

    Return:
    parameters -- python dictionary containing the neural net parameters:
                    ...
                    Wl -- matrix of weights associated with layer l
                    bl -- vector of biases associated with layer l
                    a1 -- activation function type for per layer where:

                           -1 -- linear activation
                            0 -- sigmoid activation
                            1 -- tanh activation
                            2 -- relu activation
    """
    # Extract training data
    X, Y, _ = data
    
    # Extract hyperparameters needed for this method
    hidden_layer_dimensions = hyperparameters["hidden_layer_dimensions"]
    hidden_layer_activation = hyperparameters["hidden_layer_activation"]

    # Extract options needed for this method
    problem_type = options["problem_type"]

    # Network topology 
    number_inputs = X.shape[0]
    number_output = Y.shape[0]
    layer_dims = [number_inputs] + hidden_layer_dimensions + [number_output]
    number_layers = len(layer_dims) - 1 # input layer doesn't count

    # Parameters
    parameters = {}
    for l in range(1, number_layers + 1):
        if hidden_layer_activation == "relu": 
            parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2./layer_dims[l-1])
        else:
            parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(1./layer_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
        if l < number_layers: # hidden layer
            if hidden_layer_activation == "sigmoid":  # not recommended
                parameters["a"+str(l)] = 0 
            elif hidden_layer_activation == "tanh":   # classic choice
                parameters["a"+str(l)] = 1
            elif hidden_layer_activation == "relu":   # modern choice 
                parameters["a"+str(l)] = 2
        else: # output layer
            if problem_type == "classification": 
                parameters["a"+str(l)] =  0 # sigmoid for classification
            elif problem_type == "regression":
                parameters["a"+str(l)] = -1 # linear for regression
    
    return parameters
    
    
def train(data, parameters
              , hyperparameters  = {"alpha": 0.01,
                                    "lambd": 0.0,
                                    "gamma": 0.0,
                                    "beta1": 0.9,
                                    "beta2": 0.99}
              , options          = {"finite_difference": False,
                                    "grad_check": False,
                                    "optimizer": "gd",
                                    "num_iterations": 100, 
                                    "problem_type": "classification",
                                    "print_iter": True}):
    """
    Train the neural network using gradient descent 

    Arguments:
    data -- a tuple of input-output value pairs (X, Y, J):

            X -- a numpy array of size (n_x, m) containing input features of the training data
            Y -- a numpy array of size (n_y, m) containing output values of the training data
            J -- a numpy array of size (n_y, n_x, m) where m = number of examples
                                                           n_y = number of outputs
                                                           n_x = number of inputs
                            
    hyperparameters -- a dictionary containing hyperparameters:
    
            alpha -- float, learning rate 
            lambd -- float, regularization parameter
            gamma -- float, weight factor for gradient terms
            beta1 -- float, momentum decay rate 
            beta2 -- float, adam decay rate 
             
    options -- a dictionary containing options
    
            finite_difference -- boolean flag to use FD instead of analytical gradients
            grad_check -- boolean flag to check analytical gradients against FD (for debugging)
            optimizer -- string indicating what optimizer to use:
                         --> "gd" -- gradient descent
                         --> "momentum" -- gradient descent with momentum
                         --> "adam" -- gradient descent with adam
            num_iterations -- integer, number of optimizer iterations (if not using minibatch)
            problem_type -- string indicating type of problem
                            --> "classification"
                            --> "regression"
            print_iter -- boolean, print the cost at each iteration):
    """
    # Extract training data
    X, Y, dY = data

    # I/O dimensions
    n_y, _ = Y.shape

    # Extract options needed for this method
    num_iterations = options["num_iterations"]
    is_finite_diff = options["finite_difference"]
    is_grad_check  = options["grad_check"]
    
    # Initialize optimizer
    optimizer_parameters = initialize_optimizer(parameters, options)

    # Stopping criteria (Vanderplaats, ch. 3, p. 121)
    converged = False
    N1 = 0
    N1_max = 100                # number of required consecutive passes over which
                                # absolute convergence criteria must be satisfied before stopping
    N2 = 0
    N2_max = 100                # number of required consecutive passes over which
                                # relative convergence criteria must be satisfied before stopping
    epsilon_absolute = 1e-5     # absolute error criterion
    epsilon_relative = 1e-5     # relative error criterion
        
    # Gradient descent
    cost_history = []
    for i in range(0, num_iterations):

        # Update optimizer iteration counter
        optimizer_parameters["i"] = i

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        if is_finite_diff:
            JL  = L_grads_forward_FD(X, parameters, options)
        else: 
            JL, J_caches  = L_grads_forward(X, parameters, options)

        # Cost
        current_cost = compute_cost(AL, Y, JL, dY, parameters, hyperparameters, options)
        cost_history.append(current_cost)
        if options["print_iter"]:
            print("iteration = " + str(i) + ", cost = " + str(current_cost))
            

        # Back propagation
        if is_finite_diff:
            grads = L_model_backward_FD(Y, dY, X, parameters, hyperparameters, options)
        else:
            grads = L_model_backward(AL, Y, JL, dY, caches, J_caches, hyperparameters, options)

        # Optional gradient check (mostly useful for debugging)
        if is_grad_check and not(is_finite_diff):
            _ = L_model_backward_check(AL, Y, JL, dY, X, parameters, caches, J_caches, hyperparameters, options, tol = 1e-6)

        # Parameter update with backtracking linesearch
        parameters, optimizer_parameter = update_with_backtracking_linesearch(X, Y, dY, current_cost
                                                                                      , grads, parameters
                                                                                      , optimizer_parameters
                                                                                      , hyperparameters, options)

        # Checks for optimizer convergence
        if i == 0:
            initial_cost = current_cost
            previous_cost = current_cost
        else:

            # Absolute convergence criterion
            dF1 = abs(current_cost - previous_cost)
            if dF1 < epsilon_absolute*initial_cost:
                N1 += 1
            else:
                N1 = 0
            if N1 > N1_max:
                converged = True
                print("Absolute stopping criteria satisfied")

            # Relative convergence criterion
            dF2 = abs(current_cost - previous_cost) / max(abs(current_cost), 1e-6)
            if dF2 < epsilon_relative:
                N2 += 1
            else:
                N2 = 0
            if N2 > N2_max:
                converged = True
                print("Relative stopping criteria satisfied")

            if converged:
                break
            else: 
                previous_cost = current_cost

    return parameters, current_cost, cost_history


def random_mini_batches(X, Y, J, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- a numpy array of size (n_x, m) containing input features of the training data
    Y -- a numpy array of size (n_y, m) containing output values of the training data
    J -- a numpy array of size (n_y, n_x, m) where m = number of examples
                                                   n_y = number of outputs
                                                   n_x = number of inputs
    mini_batch_size -- size of the mini-batches, integer

    Return:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y, mini_batch_J)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y, J)
    permutations = list(np.random.permutation(m))
    shuffled_X = X[:, permutations].reshape(X.shape)
    shuffled_Y = Y[:, permutations].reshape(Y.shape)
    shuffled_J = J[:, :, permutations].reshape(J.shape)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_J = shuffled_J[:, :, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_J)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0: 
        mini_batch_X = shuffled_X[:, (k+1)*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size:]
        mini_batch_J = shuffled_J[:, :, (k+1)*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_J)
        mini_batches.append(mini_batch)

    assert(mini_batch_X.shape[1] > 0)
    
    return mini_batches


def train_with_minibatch(data, parameters, hyperparameters = {"alpha": 0.01,
                                                              "lambd": 0.0,
                                                              "gamma": 0.0,
                                                              "beta1": 0.9,
                                                              "beta2": 0.99,
                                                              "hidden_layer_activation": "relu",
                                                              "hidden_layer_dimensions": [10],
                                                              "mini_batch_size": 0}
                                         , options         = {"finite_difference": False,
                                                              "grad_check": False,
                                                              "optimizer": "adam",
                                                              "num_epochs": None,
                                                              "num_iterations": 200, 
                                                              "problem_type": "classification"}):
    """
    Train the neural network using gradient descent methods with mini-batch

    Arguments:
    data -- a tuple of input-output value pairs (X, Y, J):

            X -- a numpy array of size (n_x, m) containing input features of the training data
            Y -- a numpy array of size (n_y, m) containing output values of the training data
            J -- a numpy array of size (n_y, n_x, m) where m = number of examples
                                                           n_y = number of outputs
                                                           n_x = number of inputs
                            
    hyperparameters -- a dictionary containing hyperparameters:
    
            alpha -- float, learning rate 
            lambd -- float, regularization parameter
            gamma -- float, weight factor for gradient terms
            beta1 -- float, momentum decay rate 
            beta2 -- float, adam decay rate 
            hidden_layer_activation -- string, hidden layer activation function type
                                       --> "sigmoid"
                                       --> "tanh"
                                       --> "relu"
            hidden_layer_dimensions -- list of integers indicating number of nodes per layer:
                                       --> e.g. [10, 5, 2] = Layer 1: 10 nodes --> Layer 2: 5 nodes --> Layer 3: 2 nodes  
            mini_batch_size -- int, the size of each minibatch (set to None if batch run)
             
    options -- a dictionary containing options
    
            finite_difference -- boolean flag to use FD instead of analytical gradients
            grad_check -- boolean flag to check analytical gradients against FD (for debugging)
            optimizer -- string indicating what optimizer to use:
                         --> "gd" -- gradient descent
                         --> "momentum" -- gradient descent with momentum
                         --> "adam" -- gradient descent with adam
            num_epochs -- integer, number of epochs (if minibatch)
            num_iterations -- number of iterations (if not minibatch)
            problem_type -- string inidcating type of problem
                            --> "classification"
                            --> "regression"):
    """
    # Extract data
    X, Y, J = data

    # Extract hyperparameters needed for this method
    mini_batch_size = hyperparameters["mini_batch_size"]

    # Extract options needed for this method
    num_epochs = options["num_epochs"]
    
    # Check that mini-batch size not greater than batch (otherwise, use entire batch)
    m = Y.shape[1]
    if mini_batch_size >= m:
        mini_batch_size = m
    
    # Train
    if not(mini_batch_size):
        parameters, cost, cost_history = train(data, parameters, hyperparameters, options)
    else:     

        # Turn off print iteration
        options["print_iter"] = False

        # Loop 1: one epoch = one pass through data
        cost_history = []
        for epoch in range(0, num_epochs):
            
            # Create mini-batches (or use the whole batch)
            mini_batches = random_mini_batches(X, Y, J, mini_batch_size, epoch)

            # Loop 2: repeat for each mini-batch
            for mini_batch in mini_batches:
                parameters, cost, history = train(mini_batch, parameters, hyperparameters, options)

            # Print average cost for this epoch
            avg_cost = np.mean(history)
            std_cost = np.std(history)
            cost_history.append(avg_cost)
            if options["print_iter"]:
                print("epoch " + str(epoch) + ", avg cost = " + str(avg_cost) + ", std cost = " + str(std_cost))

    return parameters, cost_history


# ---------------------------- G O O D N E S S   O F   F I T    M E T H O D S ------------------------------------------


def compute_precision(Y_pred, Y_true):
    """
    Compute precision = True positives / Total Number of Predicted Positives
                      = True positives / (True Positives + False Positives)

    NOTE: This method applies to binary classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    P -- precision, numpy array of (n_y, 1)
        >> P is a number between 0 and 1 where 0 is bad and 1 is good
    """
    true_positives = np.sum(((Y_pred + Y_true) == 2).astype(float), axis=1, keepdims=True)
    false_positives = np.sum(((Y_pred - Y_true) == 1).astype(float), axis=1, keepdims=True)
    if true_positives == 0:
        P = 0
    else:
        P = true_positives / (true_positives + false_positives)
    return P


def compute_recall(Y_pred, Y_true):
    """
    Compute recall = True positives / Total Number of Actual Positives
                   = True positives / (True Positives + False Negatives)

    NOTE: This method applies to classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    R -- recall, numpy array of (n_y, 1)
        >> R is a number between 0 and 1 where 0 is bad and 1 is good
    """
    true_positives = np.sum(((Y_pred + Y_true) == 2).astype(float), axis=1, keepdims=True)
    false_negatives = np.sum(((Y_true - Y_pred) == 1).astype(float), axis=1, keepdims=True)
    if true_positives == 0:
        R = 0
    else:
        R = true_positives / (true_positives + false_negatives)
    return R


def compute_Fscore(Y_pred, Y_true):
    """
    Compute F-scoare = 2*P*R / (P + R) where P = precision
                                             R = recall

    NOTE: This method applies to classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    F -- F-score, numpy array of (n_y, 1)
        >> F is a number between 0 and 1 where 0 is bad and 1 is good
    """
    P = compute_precision(Y_pred, Y_true)
    R = compute_recall(Y_pred, Y_true)
    if (P + R) == 0:
        F = 0
    else:
        F = 2 * P * R / (P + R)
    return F


def goodness_fit_regression(Y_pred, Y_true):
    """
    Compute goodness of fit metrics: R2, std(error), avg(error).

    Note: these metrics only apply to regression

    Arguments:
    Y_pred -- numpy array of size (K, m) where K = num outputs, n = num examples
    Y_true -- numpy array of size (K, m) where K = num outputs, m = num examples

    Return:
    R2 -- float, RSquare value
    sig -- numpy array of shape (K, 1), standard deviation of error
    mu -- numpy array of shape (K, 1), avg value of error expressed
    """
    K = Y_true.shape[0]

    R2 = Rsquare(Y_pred, Y_true)
    sig = np.std(Y_pred - Y_true)
    mu = np.mean(Y_pred - Y_true)

    return R2.reshape(K, 1), sig.reshape(K, 1), mu.reshape(K, 1)


def Rsquare(Y_pred, Y_true):
    """
    Compute R-square for a single response.

    NOTE: If you have more than one response, then you'll either have to modify this method to handle many responses at
          once or wrap a for loop around it (i.e. treat one response at a time).

    Arguments:
    Y_pred -- predictions,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples

    Return:
    R2 -- the R-square value,  numpy array of shape (K, 1)
    """
    epsilon = 1e-8  # small number to avoid division by zero
    Y_bar = np.mean(Y_true)
    SSE = np.sum(np.square(Y_pred - Y_true), axis=1)
    SSTO = np.sum(np.square(Y_true - Y_bar) + epsilon, axis=1)
    R2 = 1 - SSE / SSTO
    return R2


def random_k_folds(X, Y, J, num_folds=10, seed=0):
    """
    Creates a list of random k-folds from (X, Y, J)

    Arguments:
    X -- input data, numpy array of shape (n_x, m)  where n_x = no. of inputs, m = no. of examples
    Y -- output data, numpy array of shape (n_y, m) where n_y = no. of outputs
    J -- Jacobian, numpy array of shape (n_y, n_x, m)
    num_folds -- number of folds, integer

    Return:
    folds -- list of synchronous (fold_X, fold_Y, fold_J)
    """
    # Set random seed
    np.random.seed(seed)

    # Number of training examples
    m = X.shape[1]

    # Figure out fold size
    if num_folds > m:
        fold_size = 1
    else:
        fold_size = int(math.floor(m / num_folds))

    # Initialize
    training = []
    crossval = []

    # Shuffle (X, Y)
    permutations = list(np.random.permutation(m))
    shuffled_X = X[:, permutations].reshape(X.shape)
    shuffled_Y = Y[:, permutations].reshape(Y.shape)
    shuffled_J = J[:, :, permutations].reshape(J.shape)

    # Partition (shuffled_X, shuffled_Y, shuffled_J). Minus the end case.
    num_complete_folds = int(math.floor(m / fold_size))
    for k in range(0, num_complete_folds):
        test_X = shuffled_X[:, k * fold_size:(k + 1) * fold_size]
        test_Y = shuffled_Y[:, k * fold_size:(k + 1) * fold_size]
        test_J = shuffled_J[:, :, k * fold_size:(k + 1) * fold_size]
        train_X = np.concatenate([shuffled_X[:, :k * fold_size], shuffled_X[:, (k + 1) * fold_size:]], axis=1)
        train_Y = np.concatenate([shuffled_Y[:, :k * fold_size], shuffled_Y[:, (k + 1) * fold_size:]], axis=1)
        train_J = np.concatenate([shuffled_J[:, :, :k * fold_size], shuffled_J[:, :, (k + 1) * fold_size:]], axis=2)
        training.append((train_X, train_Y, train_J))
        crossval.append((test_X, test_Y, test_J))

    # Handling the end case (last fold_size < fold_size)
    if m % fold_size != 0:
        test_X = shuffled_X[:, (k + 1) * fold_size:]
        test_Y = shuffled_Y[:, (k + 1) * fold_size:]
        test_J = shuffled_J[:, :, (k + 1) * fold_size:]
        train_X = shuffled_X[:, :k * fold_size]
        train_Y = shuffled_Y[:, :k * fold_size]
        train_J = shuffled_J[:, :, :k * fold_size:]
        training.append((train_X, train_Y, train_J))
        crossval.append((test_X, test_Y, test_J))

    return training, crossval


def k_fold_crossvalidation(data, hyperparameters={"alpha": 0.01,
                                                  "lambd": 0.0,
                                                  "gamma": 0.0,
                                                  "beta1": 0.9,
                                                  "beta2": 0.99,
                                                  "hidden_layer_activation": "relu",
                                                  "hidden_layer_dimensions": [10],
                                                  "mini_batch_size": None}
                           , options={"finite_difference": False,
                                      "grad_check": False,
                                      "optimizer": "adam",
                                      "num_epochs": None,
                                      "num_iterations": 200,
                                      "num_folds": None,
                                      "problem_type": "classification"}):
    """
    Train the neural network using gradient descent methods with mini-batch and K-fold cross validation.

    Arguments:
    data -- a tuple of input-output value pairs (X, Y):

            X -- a numpy array containing input features of training data of shape (n_x, m)
            Y -- a numpy array containing output labels (response values) of training data of shape (n_y, m)

                    where n_x = no. of inputs
                          n_y = no. of outputs
                          m = number of training examples

    hyperparameters -- a dictionary containing hyperparameters:

            alpha -- float, learning rate
            lambd -- float, regularization parameter
            gamma -- float, weight factor for gradient terms
            beta1 -- float, momentum decay rate
            beta2 -- float, adam decay rate
            hidden_layer_activation -- string, hidden layer activation function type
                                       --> "sigmoid"
                                       --> "tanh"
                                       --> "relu"
            hidden_layer_dimensions -- list of integers indicating number of nodes per layer:
                                       --> e.g. [10, 5, 2] = Layer 1: 10 nodes --> Layer 2: 5 nodes --> Layer 3: 2 nodes
            mini_batch_size -- int, the size of each minibatch (set to None if batch run)

    options -- a dictionary containing options

            finite_difference -- boolean flag to use FD instead of analytical gradients
            grad_check -- boolean flag to check analytical gradients against FD (for debugging)
            optimizer -- string indicating what optimizer to use:
                         --> "gd" -- gradient descent
                         --> "momentum" -- gradient descent with momentum
                         --> "adam" -- gradient descent with adam
            num_epochs -- number of epochs (if minibatch)
            num_iterations -- number of iterations (if not minibatch)
            num_folds -- number of K-folds for cross-validation
            problem_type -- string indicating type of problem
                            --> "classification"
                            --> "regression"

    Return:
    parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}
                                                    {"a1": an integer}
                                                    {"a2": an integer}
                                                    {"a3": an integer}
                                                    ...
                                                    {"aL": an integer}

                                                    where the integers can be: -1 -- linear activation
                                                                                0 -- sigmoid activation
                                                                                1 -- tanh activation
                                                                                2 -- relu activation

    crossval_history -- cross-validation history
        >> a numpy array of shape (num_folds, n_y) containing F-scores for classification or RSquares for regression
    """
    # Extract training data
    X, Y, J = data  # X = inputs, Y = outputs, J = Jacobian

    # Extract options needed for this method
    num_folds = options["num_folds"]
    problem_type = options["problem_type"]

    # Create K-folds for cross-validation
    if num_folds == None or num_folds == 1:
        training = [data]
        crossval = [data]
        num_folds = 1
    else:
        training, crossval = random_k_folds(X, Y, J, num_folds)
        num_folds = len(training)  # the number of folds changes

    # Initialize training and validation error
    num_outputs = Y.shape[0]
    goodness_fit = {"FScore_val": np.zeros((num_outputs, num_folds)),
                    "FScore_train": np.zeros((num_outputs, num_folds)),
                    "R2_val": np.zeros((num_outputs, num_folds)),
                    "R2_train": np.zeros((num_outputs, num_folds)),
                    "sig_train": np.zeros((num_outputs, num_folds)),
                    "mu_train": np.zeros((num_outputs, num_folds)),
                    "sig_val": np.zeros((num_outputs, num_folds)),
                    "mu_val": np.zeros((num_outputs, num_folds))}

    # Loop over each fold
    for k in range(0, num_folds):

        print("*** K-fold " + str(k) + " ***")

        # Initialize neural net
        parameters = initialize_parameters(training[k], hyperparameters, options)

        # Train neural net for current fold
        parameters, _ = train_with_minibatch(training[k], parameters, hyperparameters, options)

        # Get current cross-validation and training data
        X_crossval, Y_crossval, _ = crossval[k]
        X_train, Y_train, _ = training[k]

        # Predict using cross-validation data
        Y_crossval_pred = predict(X_crossval, parameters, options)
        Y_train_pred = predict(X_train, parameters, options)

        # Compute cross-validation accuracy
        if problem_type == "classification":

            F_crossval = compute_Fscore(Y_crossval_pred, Y_crossval)
            goodness_fit["FScore_val"][:, k] = F_crossval

            F_train = compute_Fscore(Y_train_pred, Y_train)
            goodness_fit["FScore_train"][:, k] = F_train

        elif problem_type == "regression":

            R2, sig, mu = goodness_fit_regression(Y_crossval_pred, Y_crossval)
            goodness_fit["R2_val"][:, k] = R2
            goodness_fit["sig_val"][:, k] = sig
            goodness_fit["mu_val"][:, k] = mu

            R2, sig, mu = goodness_fit_regression(Y_train_pred, Y_train)
            goodness_fit["R2_train"][:, k] = R2
            goodness_fit["sig_train"][:, k] = sig
            goodness_fit["mu_train"][:, k] = mu

    # Now that cross-validation is complete, return parameters using all data
    if num_folds > 1:
        print("*** K-fold complete. Training with all data. ***")
        parameters, _ = train_with_minibatch(data, parameters, hyperparameters, options)

    return parameters, goodness_fit


def save_parameters(parameters, scale_factors, directory):
    """
    Write trained neural net parameters to CSV file. Result is written to a local
    directory (which is overwritten if already existing). 

    Argument:
    parameters -- trained neural net parameters
    scale_factors -- scale factors used to normalize data during training
    directory -- name of directory to be created that will contain the csv files
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    # Save parameters numpy format
    filename = os.path.join(directory, "parameters.pkl")
    f = open(filename, 'wb')
    pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # Save numpy format
    filename = os.path.join(directory, "scale_factors.pkl")
    f = open(filename, 'wb')
    pickle.dump(scale_factors, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # Save parameters csv format
    L = len(parameters) // 3
    for l in range(1, L+1):
        W = pd.DataFrame(parameters["W"+str(l)].tolist())
        b = pd.DataFrame(parameters["b"+str(l)].tolist())
        if parameters["a"+str(l)] == -1:
            W.to_csv(os.path.join(directory, "layer_" + str(l) + "_W_linear.csv"), index=None)
            b.to_csv(os.path.join(directory, "layer_" + str(l) + "_b_linear.csv"), index=None)
        elif parameters["a"+str(l)] == 0:
            W.to_csv(os.path.join(directory, "layer_" + str(l) + "_W_sigmoid.csv"), index=None)
            b.to_csv(os.path.join(directory, "layer_" + str(l) + "_b_sigmoid.csv"), index=None)
        elif parameters["a"+str(l)] == 1:
            W.to_csv(os.path.join(directory, "layer_" + str(l) + "_W_tanh.csv"), index=None)
            b.to_csv(os.path.join(directory, "layer_" + str(l) + "_b_tanh.csv"), index=None)
        elif parameters["a"+str(l)] == 2:
            W.to_csv(os.path.join(directory, "layer_" + str(l) + "_W_relu.csv"), index=None)
            b.to_csv(os.path.join(directory, "layer_" + str(l) + "_b_relu.csv"), index=None)

    # Save scale factors csv format
    mu_x = pd.DataFrame(scale_factors["mu_x"])
    mu_y = pd.DataFrame(scale_factors["mu_y"])
    sigma_x = pd.DataFrame(scale_factors["sigma_x"])
    sigma_y = pd.DataFrame(scale_factors["sigma_y"])
    mu_x.to_csv(os.path.join(directory, "mu_x.csv"), index=None)
    mu_y.to_csv(os.path.join(directory, "mu_y.csv"), index=None)
    sigma_x.to_csv(os.path.join(directory, "sigma_x.csv"), index=None)
    sigma_y.to_csv(os.path.join(directory, "sigma_y.csv"), index=None)

def load_parameters(model_name):
    """
    Load parameters from numpy file

    Argument:
    directory -- name of directory containing parameters.npy
    """
    filename = os.path.join(model_name, "parameters.pkl")
    f = open(filename, 'rb')
    parameters = pickle.load(f)
    f.close()

    filename = os.path.join(model_name, "scale_factors.pkl")
    f = open(filename, 'rb')
    scale_factors = pickle.load(f)
    f.close()
        
    return parameters, scale_factors


def plot_actual_by_predicted(saveas, X_test, Y_test, X_train, Y_train, parameters, options, scale_factors, show_plot=True):
    """
    Make actual by predicted plot.

    Arguments:
    saveas -- String, name to save as name.png
    X -- numpy array of shape (p, m) containing points at which to evaluate
    Y -- numpy array of shape (1, m) containing true output values
    parameters -- the trained parameters of the neural network
    options -- the options associated with the network
    scale_factors -- the neural net is trained over a normalized data set. The scale
                     factors used for normalization must be provided so that they can
                     be applied to the new samples that will be generated:

                         scale_factors["mu_x"] -- a list containing the mean of the training inputs along each dimension
                         scale_factors["mu_y"] -- a list containing the mean of the training output along each dimension
                         scale_factors["sigma_x"] -- a list containing the std of the training inputs along each dim
                         scale_factors["sigma_y"] -- a list containing the std of the training outputs along each dim

    show_plot -- flag to show plot

    Return:
    nothing -- just a plot
    """
    # Normalize test data (using same scale factors used for training!)
    mu_x = scale_factors["mu_x"]
    mu_y = scale_factors["mu_y"]
    sigma_x = scale_factors["sigma_x"]
    sigma_y = scale_factors["sigma_y"]
    X_norm_test = (X_test - mu_x) / sigma_x
    Y_norm_test = (Y_test - mu_y) / sigma_y
    X_norm_train = (X_train - mu_x) / sigma_x
    Y_norm_train = (Y_train - mu_y) / sigma_y

    # Predicted response using neural net
    Y_pred_test = predict(X_norm_test, parameters, options)
    Y_pred_train = predict(X_norm_train, parameters, options)

    # Compute goodness of fit
    R2, sig, mu = goodness_fit_regression(Y_pred_test, Y_norm_test)
    R2 = np.round(R2.squeeze(), 2)
    sig = np.round(sig.squeeze(), 2)
    mu = np.round(mu.squeeze(), 2)

    # Reference line
    y = np.linspace(np.min(Y_norm_test), np.max(Y_norm_test), 100)

    # Prepare to plot
    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.25)

    # Prepare to plot
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.plot(y, y)
    ax1.scatter(Y_norm_test, Y_pred_test, s=20, c='r')
    ax1.scatter(Y_norm_train, Y_pred_train, s=100, c='k', marker="+")
    plt.legend(["perfect", "test", "train"])
    plt.xlabel("Y_norm (actual)")
    plt.ylabel("Y_norm (predicted)")
    plt.title("RSquare = " + str(R2))

    ax2 = fig.add_subplot(spec[0, 1])
    error = (Y_pred_test - Y_norm_test).T
    weights = np.ones(error.shape) / Y_pred_test.shape[1]
    ax2.hist(error, weights=weights, facecolor='g', alpha=0.75)
    plt.xlabel('Absolute Prediction Error')
    plt.ylabel('Probability')
    plt.title('$\mu$=' + str(mu) + ', $\sigma=$' + str(sig))
    plt.grid(True)

    plt.savefig(saveas + '.png')
    plt.interactive(True)
    if show_plot:
        plt.show()
    plt.close()

def plot_learning_history(saveas, cost_history, show_plot=True):
    """
    Plot the convergence history of the neural network learning algorithm

    Argument:
    saveas -- String, name to save as name.png
    cost_history -- a list containing the cost history
    show_plot -- flag to show plot

    Return:
    saveas -- String, name to save as name.png
    nothing -- just a plot
    """
    n = len(cost_history)
    plt.plot(range(0, n), cost_history)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Neural Net Convergence History')
    plt.savefig(saveas + '.png')
    plt.interactive(True)
    if show_plot:
        plt.show()
    plt.close()
