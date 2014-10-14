# Run as script using 'python -m test.synth'
import cPickle
import os

import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

from pyglm.plotting.plot_results import plot_results
from pyglm.models.model_factory import *
from pyglm.inference.kayak_gibbs import gibbs_sample
# from plotting.plot_results import plot_results
from pyglm.plotting.plotting import NetworkPlotProvider, LocationPlotProvider
from pyglm.population import KayakPopulation
from pyglm.utils.io import load_data, parse_cmd_line_args


def initialize_test_harness():
    """ Initialize a model with N neurons. Use the data if specified on the
        command line, otherwise sample new data from the model.
        Return a population object, the data, and a set of true parameters
        which is expected for synthetic tests
    """
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    # Load data from file or create synthetic test dataset
    data = load_data(options)

    print "Creating master population object"
    model = make_model(options.model, N=data['N'], dt=0.001)
    stabilize_sparsity(model)
    popn = KayakPopulation(model)
    popn.add_data(data)

    # Initialize the GLM with the data
    popn_true = None
    x_true = None
    if 'vars' in data:
        x_true = data['vars']

        # Load the true model
        data_dir = os.path.dirname(options.dataFile)
        model_file = os.path.join(data_dir, 'model.pkl')
        print "Loading true model from %s" % model_file
        with open(model_file) as f:
            popn_true = cPickle.load(f)

        popn_true.set_parameters(x_true)

        popn_true.add_data(data)
        popn_true.set_data(data)


    return options, popn, data, popn_true, x_true


def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    N_samples = 1000

    if os.path.exists(results_file):
        print "Results found. Loading from file."
        with open(results_file) as f:
            x_smpls = cPickle.load(f)
            N_samples = len(x_smpls)

        # TODO: Check that the results are from the same model?
    else:
        print "Results not found. Running MCMC inference."
        # If x0 specified, load x0 from file
        x0 = None
        if options.x0_file is not None:
            with open(options.x0_file, 'r') as f:
                print "Initializing with state from: %s" % options.x0_file
                mle_x0 = cPickle.load(f)
                # HACK: We're assuming x0 came from a standard GLM
                mle_model = make_model('standard_glm', N=data['N'])
                mle_popn = KayakPopulation(mle_model)
                mle_popn.set_data(data)

                x0 = popn.sample()
                x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

        # DEBUG: Initialize with true params
        # x0 = x_true

        # Perform inference
        # raw_input('Press any key to begin inference...\n')
        x_smpls = gibbs_sample(popn, x0=x0, N_samples=N_samples,
                               init_from_mle=False,
                               callback=None)

        # Save results
        print "Saving results to %s" % results_file
        with open(results_file, 'w') as f:
            cPickle.dump(x_smpls, f, protocol=-1)

    # Plot average of last 20% of samples
    smpl_frac = 0.2
    plot_results(popn, 
                 x_smpls[-1*int(smpl_frac*N_samples):],
                 popn_true=popn_true,
                 x_true=x_true,
                 resdir=options.resultsDir)

run_synth_test()
