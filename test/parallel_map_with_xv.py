# Run as script using 'python -m test.synth_map'
import cPickle
import numpy as np
import copy
import os

from inference.parallel_coord_descent import parallel_coord_descent
from parallel_harness import initialize_parallel_test_harness, \
                             set_data_on_engines, \
                             set_hyperparameters_on_engines
from plotting.plot_results import plot_results
from models.model_factory import make_model
from synth_harness import get_xv_models

from utils.parallelutil import *
from utils.io import segment_data

from IPython.parallel.util import interactive

# TODO move this to a helper function
def parallel_compute_ll(dview,
                        v,
                        N):
    """ Compute the log prob in parallel
    """
    ll_tot = 0
    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_ll(n, vs):
        print "Computing log lkhd for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.ll,
                   syms,
                   nvars)
        return lp

    ll_glms = dview.map_async(_compute_glm_ll,
                              range(N),
                              [v]*N)

    ll_tot += sum(ll_glms.get())
    return ll_tot

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

    # Fit each model using the optimum of the previous models
    train_lls = np.zeros(len(models))
    xv_lls = np.zeros(len(models))
    total_lls = np.zeros(len(models))
    for (i,model) in enumerate(models):
        print "Training model %d" % i
        x0 = copy.deepcopy(best_x)
        #popn.set_hyperparameters(model)
        set_hyperparameters_on_engines(client[:], model)
        #popn.set_data(train_data)
        set_data_on_engines(client[:], train_data)
        ll0 = popn.compute_log_p(x0)
        print "Training LL0: %f" % ll0

        # Perform inference
        x_inf = parallel_coord_descent(client, data['N'], x0=x0, maxiter=1,
                                       use_hessian=False,
                                       use_rop=False)
        ll_train = parallel_compute_ll(client[:], x_inf, data['N'])
        print "Training LL_inf: %f" % ll_train
        train_lls[i] = ll_train


        # Compute log lkhd on xv data
        #popn.set_data(xv_data)
        set_data_on_engines(client[:], xv_data)
        ll_xv = parallel_compute_ll(client[:], x_inf, data['N'])
        print "Cross Validation LL: %f" % ll_xv
        xv_lls[i] = ll_xv

        # Compute log lkhd on total dataset
        #popn.set_data(data)
        set_data_on_engines(client[:], data)
        ll_total = parallel_compute_ll(client[:], x_inf, data['N'])
        print "Total LL: %f" % ll_total
        total_lls[i] = ll_total

        # Update best model
        if ll_xv > best_xv_ll:
            best_ind = i
            best_xv_ll = ll_xv
            best_x = copy.deepcopy(x_inf)
            best_model = copy.deepcopy(model)

    # Set the best hyperparameters
    set_hyperparameters_on_engines(client[:], best_model)
    #popn.set_data(data)
    set_data_on_engines(client[:], data)

    # Fit the best model on the full training data
    best_x = parallel_coord_descent(client, data['N'], x0=x0, maxiter=1,
                                    use_hessian=False,
                                    use_rop=False)

    # Print results summary
    for i in np.arange(len(models)):
        print "Model %d:\tTrain LL: %.1f\tXV LL: %.1f\tTotal LL: %.1f" % (i, train_lls[i], xv_lls[i], total_lls[i])
    print "Best model: %d" % best_ind
    print "Best Total LL: %f" % parallel_compute_ll(client[:], best_x, data['N'])
    print "True LL: %f" % popn_true.compute_ll(x_true)


    # Save results
    with open(os.path.join(options.resultsDir, 'results.pkl'),'w') as f:
        cPickle.dump(x_inf,f, protocol=-1)

    # Plot results
    plot_results(popn, x_inf,
                 popn_true, x_true,
                 do_plot_imp_responses=False,
                 resdir=options.resultsDir)

if __name__ == "__main__":
    run_parallel_map()

