# Run as script using 'python -m test.synth_map'
import cPickle
import scipy.io
import numpy as np

from glm_shared import *
from models.model_factory import *
from inference.coord_descent import coord_descent
from plotting.plot_results import plot_results
from utils.theano_func_wrapper import seval

def generate_synth_data(glm,
                        resultsDir,
                        T_start=0, T_stop=60,
                        dt=0.001,
                        dt_stim=0.01):
    """ Generate synthetic data from the given model.
    """

    D_stim = model['bkgd']['D_stim']
    N = model['N']

    # Sample random parameters from the model
    x_true = glm.sample()
    
    # Generate random stimulus
    stim = np.random.randn(T_stop/dt_stim,D_stim)

    # Initialize the GLMs with the stimulus
    temp_data = {"S": np.zeros((T_stop/dt,N)),
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim}
    glm.set_data(temp_data)

    # Simulate spikes
    S,X = glm.simulate(x_true, (T_start, T_stop), dt)

    # Package data into dict
    data = {"S": S,
            "X": X,
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars': x_true}

    # Save the data so we don't have to continually simulate!
    import os
    scipy.io.savemat(os.path.join(resultsDir, 'data.mat'), data)

    # Pickle the data so we can open it more easily
    with open(os.path.join(resultsDir, 'data.pkl'),'w') as f:
        cPickle.dump(data,f)

    return data

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--dataFile", dest="dataFile", default=None,
                      help="Use this data file. If not specified, simulate from model.")

    parser.add_option("-s", "--sampleFile", dest="sampleFile", default=None,
                      help="Use this sample file, either as filename in the config directory, or as a path.")

    parser.add_option("-r", "--resultsDir", dest="resultsDir", default='.',
                      help="Save the results to this directory.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if not options.dataFile is None and not os.path.exists(options.dataFile):
        raise Exception("Invalid data file specified: %s" % options.dataFile)

    if not options.sampleFile is None and not os.path.exists(options.sampleFile):
        raise Exception("Invalid sample file specified: %s" % options.sampleFile)

    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid sample file specified: %s" % options.resultsDir)

    return (options, args)


if __name__ == "__main__":
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    print "Initializing GLM"
    N=2
    model = make_model('spatiotemporal_glm', N=N)
    # model = make_model('standard_glm', N=N)
    glm = NetworkGlm(model)
    # Load data
    if not options.dataFile is None:
        if options.dataFile.endswith('.mat'):
            print "Loading data from %s" % options.dataFile
            #data = scipy.io.loadmat(options.dataFile)
            # Scipy's IO is weird -- we can save dicts as structs but its hard to reload them
            raise Exception('Loading from .mat file is not implemented!')
        elif options.dataFile.endswith('.pkl'):
            print "Loading data from %s" % options.dataFile
            with open(options.dataFile,'r') as f:
                data = cPickle.load(f)
        else:
            raise Exception("Unrecognized file type: %s" % options.dataFile)

    else:
        print "Generating synthetic data"
        data = generate_synth_data(glm, options.resultsDir,
                                   T_stop=120)
        

    # Initialize the GLM with the data
    x_true = data['vars']
    glm.set_data(data)

    # DEBUG Compare rate from model and np.exp(X) from sim
    for n in np.arange(N):
        syms = glm.get_variables()
        nvars = glm.extract_vars(x_true,n)
        if not np.allclose(seval(glm.glm.lam,
                                 syms,
                                 nvars),
                           np.exp(data['X'][:,n])):
            import pdb
            pdb.set_trace()
            raise Exception("Model and simulated firing rates do not match for neuron %d!" % n)
    # END DEBUG

    ll_true = glm.compute_log_p(x_true)
    print "true LL: %f" % ll_true

    # Sample random initial state
    x0 = glm.sample()
    # # DBG Set x0 to zero
    #for xi in x0:
    #    for xj in xi:
    #        xj *= 0
    #print x0

    ll0 = glm.compute_log_p(x0)
    print "LL0: %f" % ll0

#    x_inf = map_estimate(glm, x0)
    x_inf = coord_descent(glm, data, x0=x0, maxiter=0)
    ll_inf = glm.compute_log_p(x_inf)
    print "LL_inf: %f" % ll_inf

    # Plot results
    plot_results(glm, x_true, x_inf)
