# Run as script using 'python -m test.parallel_coord_descent'
from inference.parallel_coord_descent import parallel_coord_descent
from plotting.plot_results import plot_results

from parallel_harness import initialize_parallel_test_harness

def run_synth_test():
    """ Run a test with synthetic data and MAP inference via
        parallel coordinate descent.
    """
    # Make a population with N neurons
    model = 'standard_glm'
    popn, data, client = initialize_parallel_test_harness(model)

    print "Performing parallel inference"
    # HACK: Figure out why the second iteration sometimes hangs
    x_inf = parallel_coord_descent(client, data['N'], maxiter=1)

    ll_inf = popn.compute_log_p(x_inf)
    print "LL_inf: %f" % ll_inf

    # TODO Save results
    
    # Plot results
    x_true = None
    if 'vars' in data:
        x_true = data['vars']
    plot_results(popn, x_inf, x_true=x_true)

if __name__ == "__main__":
    run_synth_test()
