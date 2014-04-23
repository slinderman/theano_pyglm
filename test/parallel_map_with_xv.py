# Run as script using 'python -m test.synth_map'
import cPickle
import numpy as np
import copy
import os
import time

from inference.parallel_coord_descent import parallel_coord_descent
from parallel_harness import initialize_parallel_test_harness
from utils.parallel_util import set_data_on_engines, \
                                set_hyperparameters_on_engines, \
                                parallel_compute_ll
from plotting.plot_results import plot_results
from models.model_factory import make_model
from synth_harness import get_xv_models

from utils.parallelutil import *
from utils.io import segment_data

def run_parallel_map():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, client, popn_true, x_true = initialize_parallel_test_harness()

    # Get the list of models for cross validation
    base_model = make_model(options.model, N=data['N'])
    models = get_xv_models(base_model)

    # Segment data into training and cross validation sets
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

    use_existing = False

    start_time = time.clock()

    # Fit each model using the optimum of the previous models
    train_lls = np.zeros(len(models))
    xv_lls = np.zeros(len(models))
    total_lls = np.zeros(len(models))
    for (i,model) in enumerate(models):
        print "Evaluating model %d" % i
        set_hyperparameters_on_engines(client[:], model)
        set_data_on_engines(client[:], train_data)

        if use_existing and  \
           os.path.exists(os.path.join(options.resultsDir, 'results.partial.%d.pkl' % i)):
            print "Found existing results for model %d" % i
            with open(os.path.join(options.resultsDir, 'results.partial.%d.pkl' % i)) as f:
                (x_inf, ll_train, ll_xv, ll_total) = cPickle.load(f)
                train_lls[i] = ll_train
                xv_lls[i] = ll_xv
                total_lls[i] = ll_total

        else:
            x0 = copy.deepcopy(best_x)
            # set_data_on_engines(client[:], train_data)
            ll0 = parallel_compute_ll(client[:], x0, data['N'])
            print "Training LL0: %f" % ll0

            # Perform inference
            x_inf = parallel_coord_descent(client, data['N'], x0=x0, maxiter=1,
                                           use_hessian=False,
                                           use_rop=False)

            ll_train = parallel_compute_ll(client[:], x_inf, data['N'])
            print "Training LL_inf: %f" % ll_train
            train_lls[i] = ll_train

            # Compute log lkhd on xv data
            set_data_on_engines(client[:], xv_data)
            ll_xv = parallel_compute_ll(client[:], x_inf, data['N'])
            print "Cross Validation LL: %f" % ll_xv
            xv_lls[i] = ll_xv

            # Compute log lkhd on total dataset
            set_data_on_engines(client[:], data)
            ll_total = parallel_compute_ll(client[:], x_inf, data['N'])
            print "Total LL: %f" % ll_total
            total_lls[i] = ll_total

            print "Saving partial results"
            with open(os.path.join(options.resultsDir, 'results.partial.%d.pkl' % i),'w') as f:
                cPickle.dump((x_inf, ll_train, ll_xv, ll_total) ,f, protocol=-1)

        # Update best model
        if ll_xv > best_xv_ll:
            best_ind = i
            best_xv_ll = ll_xv
            best_x = copy.deepcopy(x_inf)
            best_model = copy.deepcopy(model)

    print "Training the best model (%d) with the full dataset" % best_ind
    # Set the best hyperparameters
    set_hyperparameters_on_engines(client[:], best_model)
    set_data_on_engines(client[:], data)

    # Fit the best model on the full training data
    best_x = parallel_coord_descent(client, data['N'], x0=best_x, maxiter=1,
                                    use_hessian=False,
                                    use_rop=False)

    # Print results summary
    for i in np.arange(len(models)):
        print "Model %d:\tTrain LL: %.1f\tXV LL: %.1f\tTotal LL: %.1f" % (i, train_lls[i], xv_lls[i], total_lls[i])
    print "Best model: %d" % best_ind
    print "Best Total LL: %f" % parallel_compute_ll(client[:], best_x, data['N'])
    print "True LL: %f" % popn_true.compute_ll(x_true)


    stop_time = time.clock()

    # Save results
    with open(os.path.join(options.resultsDir, 'results.pkl'),'w') as f:
        cPickle.dump(best_x, f, protocol=-1)

    # Save runtime
    with open(os.path.join(options.resultsDir, 'runtime.pkl'),'w') as f:
        cPickle.dump(stop_time-start_time, f, protocol=-1)

    # Plot results
    # plot_results(popn, best_x,
    #              popn_true, x_true,
    #              do_plot_imp_responses=(model['N']<64),
    #              resdir=options.resultsDir)

if __name__ == "__main__":
    run_parallel_map()

