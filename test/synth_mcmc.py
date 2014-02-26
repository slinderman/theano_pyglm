# Run as script using 'python -m test.synth'
import cPickle
import os
import scipy.io

from models.model_factory import *
from inference.gibbs import gibbs_sample
from utils.avg_dicts import average_list_of_dicts
from synth_harness import initialize_test_harness
from plotting.plot_results import plot_results

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()

    # If x0 specified, load x0 from file
    x0 = popn.sample()
    if options.x0_file is not None:
        with open(options.x0_file, 'r') as f:
            mle_x0 = cPickle.load(f)

            mle_model = make_model('standard_glm', N=data['N'])
            from population import Population
            mle_popn = Population(mle_model)
            mle_popn.set_data(data)

            from models.model_factory import convert_model
            x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

    # Perform inference
    N_samples = 100
    x_smpls = gibbs_sample(popn, data, x0=x0, N_samples=N_samples)

    # Save results
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(x_smpls, f)

    # Plot average of last 20% of samples
    smpl_frac = 0.2
    plot_results(popn, 
                 x_smpls[-1*int(smpl_frac*N_samples):],
                 popn_true=popn_true,
                 x_true=x_true,
                 resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()
