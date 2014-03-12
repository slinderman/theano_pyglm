# Run as script using 'python -m test.parallel_coord_descent'
import os
import cPickle

from inference.parallel_coord_descent import parallel_coord_descent
from plotting.plot_results import plot_results
from parallel_harness import initialize_parallel_test_harness

def run_synth_test():
    """ Run a test with synthetic data and MAP inference via
        parallel coordinate descent.
    """
    options, popn, data, client, popn_true, x_true = initialize_parallel_test_harness()

    print "Performing parallel inference"
    x_inf = parallel_coord_descent(client, data['N'], maxiter=1)
    ll_inf = popn.compute_log_p(x_inf)
    print "LL_inf: %f" % ll_inf

    # Save results
    with open(os.path.join(options.resultsDir, 'results.pkl'),'w') as f:
        cPickle.dump(x_inf,f, protocol=-1)
    
    # Plot results
    plot_results(popn, x_inf, 
                 popn_true, x_true, 
                 do_plot_imp_responses=False,
                 resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()
