import cPickle
import scipy.io
import numpy as np
import os

from population import Population
from models.model_factory import *
from plotting.plot_results import plot_results
from utils.theano_func_wrapper import seval

def generate_synth_data(population,
                        resultsDir,
                        T_start=0, T_stop=60,
                        dt=0.001,
                        dt_stim=0.01):
    """ Generate synthetic data from the given model.
    """
    model = population.model
    D_stim = model['bkgd']['D_stim']
    N = model['N']

    # Sample random parameters from the model
    x_true = population.sample()
    # DEBUG print x_true
    print x_true
    
    # Generate random stimulus
    stim = np.random.randn(T_stop/dt_stim,D_stim)

    # Initialize the GLMs with just the stimulus
    temp_data = {"S": np.zeros((T_stop/dt,N)),
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim}
    population.set_data(temp_data)

    # Simulate spikes
    S,X = population.simulate(x_true, (T_start, T_stop), dt)

    # Package data into dict
    data = {"S": S,
            "X": X,
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars': x_true}

    # Save the data so we don't have to continually simulate
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


def initialize_test_harness(N=2,
                            model_type='standard_glm'):
    """ Initialize a model with N neurons. Use the data if specified on the
        command line, otherwise sample new data from the model.
        Return a population object, the data, and a set of true parameters
        which is expected for synthetic tests 
    """
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    # Initialize a model with N neurons
    print "Initializing GLM"
    model = make_model(model_type, N=N)
    population = Population(model)
    
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
        data = generate_synth_data(population,
                                   options.resultsDir,
                                   T_stop=180)
        

    # Initialize the GLM with the data
    x_true = data['vars']
    population.set_data(data)
    ll_true = population.compute_log_p(x_true)
    print "true LL: %f" % ll_true

    return population, data, x_true
