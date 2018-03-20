#!/bin/python

"""
This program will fit the 2D Rastrigin function using NN and GENN and plot a comparison.

"""

import genn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------- S U P P O R T   F U N C T I O N S -------------------------------------------------

def plot_2D_rastrigin(X_train, genn_parameters, parameters, options, scale_factors, lb = -1., ub = 1.5, m = 100):
    """
    Make contour plots of 2D Rastrigin function and compare to Neural Net prediction. 
    
    Arguments: 
    X_train -- numpy array of size (2,m) containing training data inputs (used to show samples used for training)
    ge_parameters -- the trained parameters of the neural network using gradient enhancement
    parameters -- the trained parameters of the neural network without gradient enhancement
    scale_factors -- the neural net is trained over a normalized data set. The scale 
                     factors used for normalization must be provided so that they can 
                     be applied to the new samples that will be generated: 
                     
                         scale_factors["mu_x"] -- a list containing the mean of the training inputs along each dimension
                         scale_factors["mu_y"] -- a list containing the mean of the training output along each dimension
                         scale_factors["sigma_x"] -- a list containing the std of the training inputs along each dim
                         scale_factors["sigma_y"] -- a list containing the std of the training outputs along each dim
                         
    lb = float, lower bound (same for both dimensions)
    ub = float, upper bound (same for both dimensions)
    m = int, the number of samples along each dimensions 
    
    Return: 
    nothing -- just a plot
    """
            
    # Domain
    x1 = np.linspace(lb, ub, m)
    x2 = np.linspace(lb, ub, m)
    X1, X2 = np.meshgrid(x1, x2)
    
    # True response
    pi = 3.1459
    Y_true = np.power(X1, 2) - 10*np.cos(2*pi*X1) + 10 + np.power(X2, 2) - 10*np.cos(2*pi*X2) + 10
    
    # Predicted response (with GE)
    mu_x = scale_factors["mu_x"]
    mu_y = scale_factors["mu_y"]
    sigma_x = scale_factors["sigma_x"]
    sigma_y = scale_factors["sigma_y"]
    Y_pred_ge = np.zeros((m, m))
    for i in range(0, m):
        for j in range(0, m):
            x1 = (X1[i, j] - mu_x[0])/sigma_x[0]
            x2 = (X2[i, j] - mu_x[1])/sigma_x[0]
            y = nn.predict(np.array([x1, x2]).reshape(2, 1), genn_parameters, options)
            Y_pred_ge[i, j] = y*sigma_y + mu_y

    # Predicted response (with GE)
    mu_x = scale_factors["mu_x"]
    mu_y = scale_factors["mu_y"]
    sigma_x = scale_factors["sigma_x"]
    sigma_y = scale_factors["sigma_y"]
    Y_pred = np.zeros((m, m))
    for i in range(0, m):
        for j in range(0, m):
            x1 = (X1[i, j] - mu_x[0])/sigma_x[0]
            x2 = (X2[i, j] - mu_x[1])/sigma_x[0]
            y = nn.predict(np.array([x1, x2]).reshape(2, 1), parameters, options)
            Y_pred[i, j] = y*sigma_y + mu_y

    # Prepare to plot
    fig = plt.figure(figsize=(9, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1, wspace=0)

    # Plot Truth model
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.contour(X1, X2, Y_true, 20, cmap='RdGy')
    anno_opts = dict(xy=(0.5, 1.075), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('True', **anno_opts)
    anno_opts = dict(xy=(-0.075, 0.5), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('X2', **anno_opts)
    anno_opts = dict(xy=(0.5, -0.05), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('X1', **anno_opts)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.scatter(X_train[0, :], X_train[1, :], s=2)

    # Plot prediction with gradient enhancement
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.contour(X1, X2, Y_pred_ge, 20, cmap='RdGy')
    anno_opts = dict(xy=(0.5, 1.075), xycoords='axes fraction', va='center', ha='center')
    ax2.annotate('GENN', **anno_opts)
    anno_opts = dict(xy=(0.5, -0.05), xycoords='axes fraction', va='center', ha='center')
    ax2.annotate('X1', **anno_opts)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot prediction without gradient enhancement
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.contour(X1, X2, Y_pred, 20, cmap='RdGy')
    anno_opts = dict(xy=(0.5, 1.075), xycoords='axes fraction', va='center', ha='center')
    ax3.annotate('NN', **anno_opts)
    anno_opts = dict(xy=(0.5, -0.05), xycoords='axes fraction', va='center', ha='center')
    ax3.annotate('X1', **anno_opts)
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.savefig('Rastrigin.png')

    plt.interactive(False)
    plt.show()

# ---------------------------------- U S E R   I N P U T S -------------------------------------------------------------


retrain_genn = True  # retrain an existing model
retrain_nn = True    # retrain an existing model

# Training data
file_name = "rastrigin_2D.csv"
inputs    = ["X1", "X2"]
outputs   = ["Y1"]
partials  = [["J11", "J12"]]

hyperparameters = {"alpha": 0.5,
                   "lambd": 0.1,
                   "gamma": 1.0,
                   "beta1": 0.9,
                   "beta2": 0.99,
                   "hidden_layer_activation": "tanh",
                   "hidden_layer_dimensions": [24, 12],
                   "mini_batch_size": 0}

options         = {"finite_difference": False,
                   "grad_check": False,
                   "optimizer": "adam",
                   "num_epochs": None,
                   "num_iterations": 2000,
                   "num_folds": None,
                   "problem_type": "regression",
                   "model_name": "genn_rastrigin",
                   "print_iter": True}

# ---------------------------------- N O R M A L I Z E   T R A I N I N G   D A T A -------------------------------------


# Get data and normalize
X, Y, J = nn.load_csv_data(file_name, inputs, outputs, partials)

# Normalize
data, scale_factors = nn.normalize_data(X, Y, J, options)

# ---------------------------------- G R A D I E N T - E N H A N C E D   N E U R A L   N E T ---------------------------


options["model_name"] = "genn_rastrigin"

if retrain_genn:

    # Perform K-fold cross-validation with gradient-enhancement
    genn_parameters, genn_goodness_fit = nn.k_fold_crossvalidation(data, hyperparameters, options)

    # Save parameters and scale factors
    nn.save_parameters(genn_parameters, scale_factors, options["model_name"])

else:

    genn_parameters, scale_factors = nn.load_parameters(options["model_name"])

# ---------------------------------- S T A N D A R D   N E U R A L   N E T ---------------------------------------------


options["model_name"] = "nn_rastrigin"

if retrain_nn:

    # Turn off gradient enhancement
    hyperparameters["gamma"] = 0.0

    # Perform K-fold cross-validation without gradient-enhancement
    parameters, goodness_fit = nn.k_fold_crossvalidation(data, hyperparameters, options)

    # Save parameters and scale factors
    nn.save_parameters(parameters, scale_factors, options["model_name"])

else:

    parameters, scale_factors = nn.load_parameters(options["model_name"])

# ---------------------------------- P L O T   2 D   C O M P A R I S O N -----------------------------------------------


# Print cross-validation error
if retrain_nn:
    print("NN avg cross-validation RSquare: " + str(np.mean(goodness_fit["R2_val"], axis=1)))

if retrain_genn:
    print("GENN avg cross-validation RSquare: " + str(np.mean(genn_goodness_fit["R2_val"], axis=1)))

plot_2D_rastrigin(X, genn_parameters, parameters, options, scale_factors)
