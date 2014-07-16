# Run as script using 'python -m test.synth'
import os
import cPickle
import time
import numpy as np

from generate_synth_data import gen_synth_data
from models.model_factory import make_model, convert_model
from population import Population
from inference.parallel_gibbs import parallel_gibbs_sample
from parallel_harness import initialize_parallel_test_harness
from plotting.plot_results import plot_results

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, client, popn_true, x_true = initialize_parallel_test_harness()

    # If x0 specified, load x0 from file
    x0 = None
    if options.x0_file is not None:
        with open(options.x0_file, 'r') as f:
            print "Initializing with state from: %s" % options.x0_file
            prev_x0 = cPickle.load(f)
            if isinstance(prev_x0, list):

                x0 = prev_x0[-1]
            else:
                mle_x0 = prev_x0
                # HACK: We're assuming x0 came from a standard GLM
                mle_model = make_model('standard_glm', N=data['N'])
                mle_popn = Population(mle_model)
                mle_popn.set_data(data)

                x0 = popn.sample()
                x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

    use_existing = False
    
    if use_existing and  \
       os.path.exists(os.path.join(options.resultsDir, 'marginal_lkhd.pkl')):

        print "Found existing results"
        with open(os.path.join(options.resultsDir, 'marginal_lkhd.pkl')) as f:
            marg_lkhd = cPickle.load(f)
    else:
        N_samples = 1000
        popn_true.set_data(data)

        # Estimate the marginal log likelihood
        print "Performing parallel inference"
        marg_lkhd = parallel_gibbs_sample(client, data,
                                          x0=x0, N_samples=N_samples,
                                          )

        # Save results
        print "Saving results to %s" % os.path.join(options.resultsDir, 'results.pkl')
        with open(os.path.join(options.resultsDir, 'marginal_lkhd.pkl'),'w') as f:
            cPickle.dump(marg_lkhd, f, protocol=-1)

if __name__ == "__main__":
    run_synth_test()
