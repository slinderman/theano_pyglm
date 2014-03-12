# Run as script using 'python -m test.synth_map'
import cPickle
import os
import numpy as np
import copy
import scipy.io

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
    best_ll = -np.Inf
    best_x = x0
    best_model = None

    # Fit each model using the optimum of the previous models
    for model in models:
        x0 = best_x
        popn.set_hyperparameters(model)
        popn.set_data(train_data)
        ll0 = popn.compute_log_p(x0)
        print "Training LL0: %f" % ll0

        # Perform inference
        x_inf = coord_descent(popn, data, x0=x0, maxiter=1,
                              use_hessian=False,
                              use_rop=False)
        ll_inf = popn.compute_log_p(x_inf)
        print "Training LL_inf: %f" % ll_inf
        
        # Compute log lkhd on xv data
        popn.set_data(xv_data)
        ll_xv = popn.compute_ll(x_inf)
        print "Cross Validation LL: %f" % ll_xv
        
        # Update best model
        if ll_xv > best_ll:
            best_ll = ll_xv
            best_x = copy.deepcopy(x_inf)
            best_model = model
        
    # Create a population with the best model
    popn = Population(best_model)
    popn.set_data(data)

    # Save results
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(best_x, f)

    # Plot results
    plot_results(popn, best_x, popn_true, x_true, resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()

