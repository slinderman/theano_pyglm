# Run as script using 'python -m test.synth_map'
import cPickle
import os
import scipy.io

from inference.coord_descent import coord_descent
from plotting.plot_results import plot_results
from synth_harness import initialize_test_harness

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()

    # Sample random initial state
    x0 = popn.sample()
    ll0 = popn.compute_log_p(x0)
    print "LL0: %f" % ll0

    # Perform inference
    x_inf = coord_descent(popn, data, x0=x0, maxiter=1,
                          use_hessian=False,
                          use_rop=False)
    ll_inf = popn.compute_log_p(x_inf)
    print "LL_inf: %f" % ll_inf

    # Save results
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(x_inf, f)

    # Plot results
    plot_results(popn, x_inf, popn_true, x_true, resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()

