#!/bin/python

import genn as nn
import pandas as pd
import multiprocessing as mp
from functools import partial

# ---------------------------------- S U P P O R T   F U N C T I O N S -------------------------------------------------


def train_model(job_id, jobs):
    """
    Helper function for multi-processing. Trains the neural net corresponding to the job_id in the jobs list.

    Argument:
    job_id -- the job id corresponding to the model to train
    jobs -- a list of jobs to run in parallel, where each item is a dictionary containing model information:

        jobs = [job_1, job_2, ...] and for each job: job["training_data"] = (X_train, Y_train, J_train)
                                                     job["test_data"] = (X_test, Y_test, J_test)
                                                     job["hyperparameters"] = dictionary of hyper-parameters
                                                     job["options"] = dictionary of options

    Return:
    parameters -- dictionary of trained neural net parameters
    cost_history -- list of cost function convergence history
    """
    job = jobs[job_id]

    train = job["training_data"]
    test = job["test_data"]
    options = job["options"]
    hyperparameters = job["hyperparameters"]

    # Normalize training data
    X_train, Y_train, J_train = train
    data, scale_factors = nn.normalize_data(X_train, Y_train, J_train, options)
    
    # Train neural net
    parameters = nn.initialize_parameters(data, hyperparameters, options)
    parameters, cost_history = nn.train_with_minibatch(data, parameters, hyperparameters, options)
    
    # Save parameters (and scale_factors)
    nn.save_parameters(parameters, scale_factors, options["model_name"])

    # Plot convergence history
    saveas = "convhis_" + options["model_name"]
    nn.plot_learning_history(saveas, cost_history, False)

    # Goodness of fit
    X_test, Y_test, _ = test
    saveas = "goodfit_" + options["model_name"]
    nn.plot_actual_by_predicted(saveas, X_test, Y_test, X_train, Y_train, parameters, options, scale_factors, False)
    
    return parameters, cost_history


def parallel_runs(jobs, num_cpus=1):
    """
    Trains neural net models in parallel.Each case to run is allocated it's own separate processor. Hence,
    the number of items in "cases_to_run" should not exceed the number of processors available on the local
    machine. 

    Arguments:
    list_of_cases -- list containing all neural net cases in the DOE. Each item in the list is a dictionary
                     containing the following information about the model:

                        {"data": data} -- the training data (X, Y, J) (see genn.load_csv_data() )
                        {"scale_factors": scale_factors} -- scale factors used to normalize data
                                                            (see genn.normalize_data() )
                        {"initial_parameters": parameters} -- the initial parameters of the network
                                                              (see genn.initialize_parameters() )
                        {"hyperparameters": hyperparameters} -- the hyperparameters used for training
                                                               (see genn.train_with_minibatch() )
                        {"options": options} -- training options (see genn.train_with_minibatch() )

    Return:
    results -- a list containing the output of traing_with_minibtach() for each item in "cases_to_run"
                     
    """
    num_jobs = len(jobs)
    job_id = range(0, num_jobs)
    num_cpus = min(num_cpus, num_jobs)

    # Run the jobs
    print("Running " + str(num_jobs) + " jobs across " + str(num_cpus) + " cpus")
    pool = mp.Pool(processes=num_cpus)
    run_job = partial(train_model, jobs=jobs)
    result = pool.map(run_job, job_id)
    pool.close()
    pool.join()
        
    return result
      
# ---------------------------------- M A I N   P R O G R A M -----------------------------------------------------------

if __name__ == '__main__':

    # Read content of CSV file
    DOE = pd.read_csv("DOE.csv")

    # Loop
    jobs = []
    for s in range(0, len(DOE)):

        # ---------------------------------- E X T R A C T   D A T A   F R O M   C S V ---------------------------------

        # Convert string to int
        hidden_layer_dimensions = []
        list_str = DOE["hidden_layer_dimensions"][s].split(",")
        for item in list_str:
            hidden_layer_dimensions.append(int(item))

        hyperparameters = {"alpha": float(DOE["alpha"][s]),
                           "lambd": float(DOE["lambda"][s]),
                           "gamma": float(DOE["gamma"][s]),
                           "beta1": float(DOE["beta1"][s]),
                           "beta2": float(DOE["beta2"][s]),
                           "hidden_layer_activation": DOE["hidden_layer_activation"][s],
                           "hidden_layer_dimensions": hidden_layer_dimensions,
                           "mini_batch_size": int(DOE["mini_batch_size"][s])}

        options = {"finite_difference": False,
                   "grad_check": False,
                   "optimizer": "adam",
                   "num_epochs": int(DOE["num_epochs"][s]),
                   "num_iterations": int(DOE["num_iterations"][s]),
                   "problem_type": "regression",
                   "model_name": DOE["model_name"][s],
                   "print_iter": True}

        # Input file name
        training = "data/" + DOE["training"][s]
        test = "data/" + DOE["test"][s]

        # Input headers
        inputs = []
        for i in range(0, int(DOE["num_inputs"][s])):
            inputs.append("X"+str(i+1))
            tmp = []
        print(inputs)

        # Partial headers
        partials = []
        for o in range(0, int(DOE["num_outputs"][s])):
            for i in range(0, int(DOE["num_inputs"][s])):
                tmp.append("J" + str(o+1) + str(i+1))
            partials.append(tmp)
        print(partials)

        # Output headers
        outputs = []
        for o in range(0, int(DOE["num_outputs"][s])):
            outputs.append("Y"+str(o+1))
        print(outputs)

        # Get training data
        X_train, Y_train, J_train = nn.load_csv_data(training, inputs, outputs, partials)

        # Get test data
        X_test, Y_test, J_test = nn.load_csv_data(test, inputs, outputs, partials)

    # ---------------------------------- S E T U P   P A R A L L E L   J O B S -----------------------------------------

        job = {}
        job["training_data"] = (X_train, Y_train, J_train)
        job["test_data"] = (X_test, Y_test, J_test)
        job["hyperparameters"] = hyperparameters
        job["options"] = options
        jobs.append(job)

    # ---------------------------------- T R A I N   M O D E L S   I N   P A R A L L E L -------------------------------

    results = parallel_runs(jobs,  mp.cpu_count())
