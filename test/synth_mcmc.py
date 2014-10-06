# Run as script using 'python -m test.synth'
import cPickle
import os

import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

from pyglm.plotting.plot_results import plot_results
from pyglm.models.model_factory import *
from pyglm.inference.gibbs import gibbs_sample
from synth_harness import initialize_test_harness
# from plotting.plot_results import plot_results
from pyglm.plotting.plotting import NetworkPlotProvider, LocationPlotProvider
from pyglm.population import TheanoPopulation


def initialize_plotting(popn_true, x_true, popn_inf):
    fig = plt.figure(figsize=(5,3))

    # Make a list of plotters and axes for inferred results to update
    plotters = []

    # Plot the true and inferred network
    true_network_plotter = NetworkPlotProvider(popn_true)
    ax_true_network = fig.add_subplot(2,2,1)
    true_network_plotter.plot(x_true, ax=ax_true_network)

    inf_network_plotter = NetworkPlotProvider(popn_inf)
    ax_inf_network = fig.add_subplot(2,2,2)

    plotters.append((ax_inf_network, inf_network_plotter))

    # Plot the true and inferred locations
    true_loc_plotter = LocationPlotProvider(popn_true)
    ax_true_locs = fig.add_subplot(2,2,3)
    true_loc_plotter.plot(x_true, ax=ax_true_locs)

    inf_loc_plotter = LocationPlotProvider(popn_inf)
    ax_inf_locs = fig.add_subplot(2,2,4)

    plotters.append((ax_inf_locs, inf_loc_plotter))

    return plotters

def plot_sample_callback(x, plotters):
    """
    Plot a single sample
    """
    for ax, plotter in plotters:
        ax.cla()
        plotter.plot(x, ax=ax)

    plt.pause(0.001)

def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    N_samples = 100

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
                mle_popn = TheanoPopulation(mle_model)
                mle_popn.set_data(data)

                x0 = popn.sample()
                x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

        # # Prepare for online plotting
        # plt.ion()
        # plotters = initialize_plotting(popn_true, x_true, popn)
        # plt.show()
        # cbk = lambda x: plot_sample_callback(x, plotters)
        cbk = None

        # Perform inference
        raw_input('Press any key to begin inference...\n')
        x_smpls = gibbs_sample(popn, x0=x0, N_samples=N_samples,
                               init_from_mle=False,
                               callback=cbk)

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
