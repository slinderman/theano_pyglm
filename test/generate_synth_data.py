# Run as script using 'python -m test.synth_map'
import cPickle
import os
import scipy.io
import numpy as np

from population import Population
from models.model_factory import make_model, \
                                 stabilize_sparsity
from plotting.plot_results import plot_results
from utils.io import create_unique_results_folder

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

    (options, args) = parser.parse_args()
    
    # Make sure parameters are of the correct type
    if not isinstance(options.N, int):
        options.N = int(options.N)

    if not isinstance(options.T_stop, float):
        options.T_stop = float(options.T_stop)

    # Check if specified files exist
    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid results folder specified: %s" % options.resultsDir)
    
    options.resultsDir = create_unique_results_folder(options.resultsDir)
    return (options, args)

def gen_synth_data():
    """ Run a test with synthetic data and MCMC inference
    """
    options, args = parse_cmd_line_args()
    
    # Create the model
    model = make_model(options.model, N=options.N)
    # Set the sparsity level to minimize the risk of unstable networks
    stabilize_sparsity(model)

    print "Creating master population object"
    popn = Population(model)

    print "Generating synthetic data with %d neurons and %.2f seconds." % \
          (options.N, options.T_stop)

    # Set simulation parametrs
    dt = 0.001
    dt_stim = 0.1
    D_stim = 1
    T_start = 0

    # Sample random parameters from the model
    x_true = popn.sample()
    
    # Check stability of matrix
    if model['network']['weight']['type'].lower() == 'gaussian':
        Weff = x_true['net']['graph']['A'] * np.reshape(x_true['net']['weights']['W'], (options.N,options.N))
        maxeig = np.amax(np.abs(np.linalg.eig(Weff)[0]))
        print "Max eigenvalue of Weff: %.2f" % maxeig
        assert maxeig < 1, "ERROR: For stability, maxeig must be less than 1"
        
    # Generate random white noise stimulus
    stim = np.random.randn(options.T_stop/dt_stim,D_stim)

    # Initialize the GLMs with just the stimulus
    temp_data = {"S": np.zeros((options.T_stop/dt, options.N)),
                 "N": options.N,
                 "dt": dt,
                 "T": np.float(options.T_stop),
                 "stim": stim,
                 'dt_stim': dt_stim}
    popn.set_data(temp_data)

    # Simulate spikes
    S,X = popn.simulate(x_true, (0, options.T_stop), dt)

    # Save the model so it can be loaded alongside the data
    fname_model = os.path.join(options.resultsDir, 'model.pkl')
    print "Saving data to %s" % fname_model
    with open(fname_model,'w') as f:
        cPickle.dump(model,f)

    # Package data into dict
    data = {"S": S,
            "X": X,
            "N": options.N,
            "dt": dt,
            "T": np.float(options.T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars': x_true,
            'model' : 'model.pkl'}

    # Set the data so that the population state can be evaluated
    popn.set_data(data)
    
    # DEBUG Evaluate the firing rate and the simulated firing rate
    state = popn.eval_state(x_true)
    for n in np.arange(options.N):
        lam_true = state['glms'][n]['lam']
        lam_sim =  popn.glm.nlin_model.f_nlin(X[:,n])
        assert np.allclose(lam_true, lam_sim)

    # Save the data for reuse
    fname_mat = os.path.join(options.resultsDir, 'data.mat')
    print "Saving data to %s" % fname_mat
    scipy.io.savemat(fname_mat, data, oned_as='row')
        
    # Pickle the data so we can open it more easily
    fname_pkl = os.path.join(options.resultsDir, 'data.pkl')
    print "Saving data to %s" % fname_pkl
    with open(fname_pkl,'w') as f:
        cPickle.dump(data,f)

    # Plot firing rates, stimulus responses, etc
    plot_results(popn, data['vars'], resdir=options.resultsDir,
                 plot_stim_resp=False,
                 plot_imp_resp=False)
    
if __name__ == "__main__":
    gen_synth_data()
