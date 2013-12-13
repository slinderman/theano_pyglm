# Run as script using 'python -m test.synth'
import cPickle
import scipy.io

from models.model_factory import *
from inference.gibbs import gibbs_sample
from utils.avg_dicts import average_list_of_dicts
from synth_harness import initialize_test_harness
from plotting.plot_results import plot_results

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    # Make a population with N neurons
    model = 'simple_sparse_model'
    N = 2
    population, data, x_true = initialize_test_harness(N, model)

    # Perform inference
    N_samples = 100
    x_smpls = gibbs_sample(population, data, N_samples=N_samples)

    # TODO Save results
    
    # Plot average of last 20% of samples
    smpl_frac = 0.2
    plot_results(population, 
                 x_smpls[-1*int(smpl_frac*N_samples):],
                 x_true )

if __name__ == "__main__":
    run_synth_test()
