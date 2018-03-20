#!/bin/python

import genn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------- S U P P O R T   F U N C T I O N S -------------------------------------------------


def plot_1D_rastrigin(saveas, X_train, Y_train, genn_parameters, parameters,
                      options, scale_factors, lb=-1., ub=1.5, m=100):


    # Domain
    x = np.linspace(lb, ub, m)

    # True response
    pi = 3.1459
    y_true = np.power(x, 2) - 10 * np.cos(2 * pi * x) + 10

    # Scale factors
    mu_x = scale_factors["mu_x"]
    mu_y = scale_factors["mu_y"]
    sigma_x = scale_factors["sigma_x"]
    sigma_y = scale_factors["sigma_y"]
    x_norm = (x - mu_x) / sigma_x

    # Predicted response (with GE)
    y_norm = nn.predict(x_norm, genn_parameters, options)
    y_pred_ge = y_norm * sigma_y + mu_y

    # Predicted response (without GE)
    y_norm = nn.predict(x_norm, parameters, options)
    y_pred = y_norm * sigma_y + mu_y

    # Prepare to plot
    plt.plot(x, y_true.reshape(x.shape), x, y_pred_ge.reshape(x.shape), 'r--', x, y_pred.reshape(x.shape), 'k:')
    plt.scatter(X_train, Y_train, s=20, c='k')
    plt.legend(["True", "GENN", "NN", "Train"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(saveas + '.png')
    plt.interactive(False)
    plt.show()
    plt.close()

# ---------------------------------- U S E R   I N P U T S -------------------------------------------------------------


retrain_genn = True  # retrain an existing model
retrain_nn = True    # retrain an existing model

file_name = "rastrigin_1D.csv"
inputs    = ["X1"]
outputs   = ["Y1"]
partials  = [["J11"]]

hyperparameters = {"alpha": 0.5,
                   "lambd": 0.1,
                   "gamma": 1.0,
                   "beta1": 0.9,
                   "beta2": 0.99,
                   "hidden_layer_activation": "tanh",
                   "hidden_layer_dimensions": [24, 12]}

options         = {"finite_difference": False,
                   "grad_check": False,
                   "optimizer": "adam",
                   "num_iterations": 1000,
                   "problem_type": "regression",
                   "model_name": "rastrigin_1D",
                   "print_iter": True}

# ---------------------------------- N O R M A L I Z E   T R A I N I N G   D A T A -------------------------------------


# Get data and normalize
X, Y, J = nn.load_csv_data(file_name, inputs, outputs, partials)

# Normalize data
data, scale_factors = nn.normalize_data(X, Y, J, options)

# ---------------------------------- G R A D I E N T - E N H A N C E D   N E U R A L   N E T -----------------------


print("*** GENN *** ")

options["model_name"] = "genn_rastrigin_1D"

if retrain_genn:

    # Initialize network
    genn_parameters = nn.initialize_parameters(data, hyperparameters, options)

    # Train
    genn_parameters, genn_cost, genn_cost_history = nn.train(data, genn_parameters, hyperparameters, options)

    # Save parameters and scale factors
    nn.save_parameters(genn_parameters, scale_factors, options["model_name"])

else:

    genn_parameters, scale_factors = nn.load_parameters(options["model_name"])

# ---------------------------------- S T A N D A R D   N E U R A L   N E T -----------------------------------------

print("*** NN *** ")

options["model_name"] = "nn_rastrigin_1D"

if retrain_nn:

    # Turn off gradient enhancement
    hyperparameters["gamma"] = 0.0

    # Initialize network
    parameters = nn.initialize_parameters(data, hyperparameters, options)

    # Train
    parameters, cost, cost_history = nn.train(data, parameters, hyperparameters, options)

    # Save parameters and scale factors
    nn.save_parameters(parameters, scale_factors, options["model_name"])

else:

    parameters, scale_factors = nn.load_parameters(options["model_name"])

# ---------------------------------- P L O T   1 D   C O M P A R I S O N -------------------------------------------

plot_1D_rastrigin(options["model_name"], X, Y, genn_parameters, parameters, options, scale_factors)
