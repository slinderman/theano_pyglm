# Run as script using 'python -m test.synth_map'
import cPickle
import os
import numpy as np
import copy

from population import Population
from inference.coord_descent import coord_descent
from plotting.plot_results import plot_results
from synth_harness import initialize_test_harness, get_xv_models
from models.model_factory import make_model
from utils.io import segment_data

def run_synth_test():
    """ Run a test with synthetic data and MAP inference with cross validation
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()
    
    # Get the list of models for cross validation
    base_model = make_model(options.model, N=data['N'])
    models = get_xv_models(base_model)

    # TODO Segment data into training and cross validation sets
    train_frac = 0.75
    T_split = data['T'] * train_frac
    train_data = segment_data(data, (0,T_split))
    xv_data = segment_data(data, (T_split,data['T']))
    
    # Sample random initial state
    x0 = popn.sample()

    # Track the best model and parameters
    best_ind = -1
    best_xv_ll = -np.Inf
    best_x = x0
    best_model = None

    # Fit each model using the optimum of the previous models
    train_lls = np.zeros(len(models))
    xv_lls = np.zeros(len(models))
    total_lls = np.zeros(len(models))
    for (i,model) in enumerate(models):
        print "Training model %d" % i
        x0 = copy.deepcopy(best_x)
        popn.set_hyperparameters(model)
        popn.set_data(train_data)
        ll0 = popn.compute_log_p(x0)
        print "Training LL0: %f" % ll0

        # Perform inference
        x_inf = coord_descent(popn, data, x0=x0, maxiter=1,
                              use_hessian=False,
                              use_rop=False)
        ll_train = popn.compute_log_p(x_inf)
        print "Training LL_inf: %f" % ll_train
        train_lls[i] = ll_train

        
        # Compute log lkhd on xv data
        popn.set_data(xv_data)
        ll_xv = popn.compute_ll(x_inf)
        print "Cross Validation LL: %f" % ll_xv
        xv_lls[i] = ll_xv

        # Compute log lkhd on total dataset
        popn.set_data(data)
        ll_total = popn.compute_ll(x_inf)
        print "Tota LL: %f" % ll_total
        total_lls[i] = ll_total

        # Update best model
        if ll_xv > best_xv_ll:
            best_ind = i
            best_xv_ll = ll_xv
            best_x = copy.deepcopy(x_inf)
            best_model = copy.deepcopy(model)
        
    # Create a population with the best model
    popn.set_hyperparameters(best_model)
    popn.set_data(data)

    # Fit the best model on the full training data
    best_x = coord_descent(popn, data, x0=x0, maxiter=1,
                           use_hessian=False,
                           use_rop=False)

    # Print results summary
    for i in np.arange(len(models)):
        print "Model %d:\tTrain LL: %.1f\tXV LL: %.1f\tTotal LL: %.1f" % (i, train_lls[i], xv_lls[i], total_lls[i])
    print "Best model: %d" % best_ind
    print "Best Total LL: %f" % popn.compute_ll(best_x)
    print "True LL: %f" % popn_true.compute_ll(x_true)

    # Save results
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(best_x, f)

    # Plot results
    plot_results(popn, best_x, popn_true, x_true, resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()

