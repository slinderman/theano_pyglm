# Run as script using 'python -m test.synth'
import os
import cPickle

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
            mle_x0 = cPickle.load(f)
            # HACK: We're assuming x0 came from a standard GLM
            mle_model = make_model('standard_glm', N=data['N'])
            mle_popn = Population(mle_model)
            mle_popn.set_data(data)

            x0 = popn.sample()
            x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

    # Perform inference
    print "Performing parallel inference"
    N_samples = 1000
    x_smpls = parallel_gibbs_sample(client, data, x0=x0, N_samples=N_samples)

    # Save results
    print "Saving results to %s" % os.path.join(options.resultsDir, 'results.pkl')
    with open(os.path.join(options.resultsDir, 'results.pkl'),'w') as f:
        cPickle.dump(x_smpls, f, protocol=-1)

    # Plot average of last 20% of samples
    print "Plotting results"
    smpl_frac = 0.5

    # Only plot the impulse response matrix for small N
    do_plot_imp_responses = data['N'] < 30

    plot_results(popn,
                 x_smpls[-1*int(smpl_frac*N_samples):],
                 popn_true,
                 x_true,
                 do_plot_imp_responses=do_plot_imp_responses,
                 resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()
