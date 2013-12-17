# Run as script using 'python -m test.synth'

from inference.parallel_gibbs import parallel_gibbs_sample
from parallel_harness import initialize_parallel_test_harness
from plotting.plot_results import plot_results

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    # Make a population with N neurons
    model = 'simple_sparse_model'
    popn, data, client = initialize_parallel_test_harness(model)

    # Perform inference
    print "Performing parallel inference"
    N_samples = 100
    x_smpls = parallel_gibbs_sample(client, data, N_samples=N_samples)

    # TODO Save results

    # Plot average of last 20% of samples
    smpl_frac = 0.2
    x_true = None
    if 'vars' in data:
        x_true = data['vars']
    plot_results(popn,
                 x_smpls[-1*int(smpl_frac*N_samples):],
                 x_true=x_true)

if __name__ == "__main__":
    run_synth_test()
