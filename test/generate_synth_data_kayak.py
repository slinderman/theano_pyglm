# Run as script using 'python -m test.synth_map'
import cPickle
import os

import numpy as np

from pyglm.population import TheanoPopulation, KayakPopulation

from pyglm.models.model_factory import make_model, \
                                 stabilize_sparsity, \
                                 check_stability
from pyglm.plotting.plot_results import plot_results
from pyglm.utils.io import create_unique_results_folder


def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", default='standard_glm',
                      help="Type of model to use. See model_factory.py for available types.")

    parser.add_option("-r", "--resultsDir", dest="resultsDir", default='.',
                      help="Save the results to this directory.")
    
    parser.add_option("-N", "--N", dest="N", default=1,
                      help="Number of neurons.")

    parser.add_option("-T", "--T_stop", dest="T_stop", default=60.0,
                      help="Length of simulation (sec).")

    parser.add_option("-u", "--unique_result", dest="unique_results", default="true",
                      help="Whether or not to create a unique results directory.")

    (options, args) = parser.parse_args()
    
    # Make sure parameters are of the correct type
    if not isinstance(options.N, int):
        options.N = int(options.N)

    if not isinstance(options.T_stop, float):
        options.T_stop = float(options.T_stop)

    # Check if specified files exist
    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid results folder specified: %s" % options.resultsDir)

    if not( options.unique_results == "0" or \
            options.unique_results.lower() == "false"):
        options.resultsDir = create_unique_results_folder(options.resultsDir)
    return (options, args)


def gen_synth_data(N, T_stop, popn, x_true, dt=0.001, dt_stim=0.1, D_stim=1, stim=None):
    # Simulate spikes
    S,X = popn.simulate(x_true, (0, T_stop), dt, stim, dt_stim)

    # Package data into dict
    data = {"S": S,
            "X": X,
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars' : x_true}

    return data

def run_gen_synth_data():
    """ Run a test with synthetic data and MCMC inference
    """
    options, args = parse_cmd_line_args()
    
    # Create the model
    dt = 0.001
    model = make_model(options.model, N=options.N, dt=dt)
    # Set the sparsity level to minimize the risk of unstable networks
    stabilize_sparsity(model)

    print "Creating master population object"
    popn = KayakPopulation(model)

    # Sample random parameters from the model
    x_true = popn.sample()

    # Check stability of matrix
    assert check_stability(model, x_true, options.N), "ERROR: Sampled network is unstable!"

    # Save the model so it can be loaded alongside the data
    fname_model = os.path.join(options.resultsDir, 'model.pkl')
    print "Saving population to %s" % fname_model
    with open(fname_model,'w') as f:
        cPickle.dump(popn, f, protocol=-1)

    print "Generating synthetic data with %d neurons and %.2f seconds." % \
          (options.N, options.T_stop)

    # Set simulation parametrs
    dt_stim = 0.1
    D_stim = (5,5)
    # D_stim = model['bkgd']['D_stim'] if 'D_stim' in model['bkgd'] else 0
    if isinstance(D_stim, int):
        D_stim = [D_stim]
    stim = np.random.randn(options.T_stop/dt_stim, *D_stim)

    data = gen_synth_data(options.N, options.T_stop, popn, x_true, dt, dt_stim, D_stim, stim)

    # Set the data so that the population state can be evaluated
    popn.add_data(data)

    # DEBUG Evaluate the firing rate and the simulated firing rate
    state = popn.eval_state(x_true)
    for n in np.arange(options.N):
        lam_true = state['glms'][n]['lam'].ravel()
        lam_sim =  popn.glms[n].nlin_model.f_nlin(data['X'][:,n])
        assert np.allclose(lam_true, lam_sim)

    # Pickle the data so we can open it more easily
    fname_pkl = os.path.join(options.resultsDir, 'data.pkl')
    print "Saving data to %s" % fname_pkl
    with open(fname_pkl,'w') as f:
        cPickle.dump(data, f, protocol=-1)

    # Plot firing rates, stimulus responses, etc
    do_plot_imp_resonses = int(options.N) <= 16
    plot_results(popn, data['vars'],
                 resdir=options.resultsDir,
                 do_plot_stim_resp=True,
                 do_plot_imp_responses=do_plot_imp_resonses)
    
if __name__ == "__main__":
    run_gen_synth_data()
