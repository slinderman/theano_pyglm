import cPickle
import scipy.io
import numpy as np
import os

from population import Population
from models.model_factory import *
from plotting.plot_results import plot_results
from utils.theano_func_wrapper import seval
from utils.io import parse_cmd_line_args, load_data


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
    model = make_model(options.model, N=data['N'])
    popn = Population(model)
    popn.set_data(data) 
    
    # Initialize the GLM with the data
    popn_true = None
    x_true = None
    if 'vars' in data:
        x_true = data['vars']
        
        # Load the true model 
        model_true = None
        data_dir = os.path.dirname(options.dataFile)
        model_file = os.path.join(data_dir, 'model.pkl')
        print "Loading true model from %s" % model_file
        with open(model_file) as f:
            model_true = cPickle.load(f) 
            popn_true = Population(model_true)
            popn_true.set_data(data)
            ll_true = popn_true.compute_log_p(x_true)
            print "true LL: %f" % ll_true

    return options, popn, data, popn_true, x_true
