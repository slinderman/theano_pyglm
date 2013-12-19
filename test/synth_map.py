# Run as script using 'python -m test.synth_map'
import cPickle
import scipy.io

from inference.coord_descent import coord_descent
from plotting.plot_results import plot_results

from synth_harness import initialize_test_harness

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    # Make a population with N neurons
    model = 'standard_glm'
    N = 10
    population, data, x_true = initialize_test_harness(N, model)
    
    # Sample random initial state
    x0 = population.sample()
    ll0 = population.compute_log_p(x0)
    print "LL0: %f" % ll0

    # Perform inference
    x_inf = coord_descent(population, data, x0=x0, maxiter=1,
                          use_hessian=False,
                          use_rop=False)
    ll_inf = population.compute_log_p(x_inf)
    print "LL_inf: %f" % ll_inf

    # TODO Save results
    
    # Plot results
    plot_results(population, x_inf, x_true)

if __name__ == "__main__":
    run_synth_test()

