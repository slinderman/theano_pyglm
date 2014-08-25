import cPickle
import scipy.io
import numpy as np
import os

from population import Population
from models.model_factory import *
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
    stabilize_sparsity(model)
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
            # HACK FOR EXISTING DATA!
            if 'N_dims' not in model_true['network']['graph']:
                model_true['network']['graph']['N_dims'] = 1
            if 'location_prior' not in model_true['network']['graph']:
                model_true['network']['graph']['location_prior'] = \
                         {
                             'type' : 'gaussian',
                             'mu' : 0.0,
                             'sigma' : 1.0
                         }
            if 'L' in x_true['net']['graph']:
                x_true['net']['graph']['L'] = x_true['net']['graph']['L'].ravel()
            # END HACK
            popn_true = Population(model_true)
            popn_true.set_data(data)
            ll_true = popn_true.compute_log_p(x_true)
            print "true LL: %f" % ll_true

    return options, popn, data, popn_true, x_true

def get_xv_models(model):
    """ Get a set of models for cross validation. Each model
        has a different set of hyperparameters (eg variances 
        for priors)
    """
    # Create a set of parameters and values
    # prms = {('bias','sigma') : (0.05, 0.5, 1.0, 2.0),
    #         ('bkgd','sigma') : (0.05, 0.5, 1.0, 2.0),
    #         ('bkgd','sigma') : (0.05, 0.5, 1.0, 2.0),
    #         ('bkgd','sigma_x') : (0.05, 0.5, 1.0, 2.0),
    #         ('bkgd','sigma_t') : (0.05, 0.5, 1.0, 2.0),
    #         ('impulse', 'sigma') : (0.05, 0.5, 1.0, 2.0),
    #         ('network', 'weight', 'sigma') : (0.05, 0.5, 1.0, 2.0),
    #         ('network', 'weight', 'sigma_refractory') : (0.05, 0.5, 1.0, 2.0)}

    prms = {('impulse', 'prior', 'lam') : (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)}
    # prms = {('impulse', 'prior', 'lam') : (15.0, 10.0, 5.0, 2.0, 1.0, 0.5)}

    # Only keep those settings which exist in the model
    def check_key(d,tk):
        # d is a dict
        # tk is a tuple of subkeys
        if len(tk) == 0:
            return True
        k = tk[0]
        if k in d:
            return check_key(d[k], tk[1:])
        else:
            return False

    for key in prms.keys():
        if not check_key(model, key):
            del prms[key]
    
    # Take the Cartesian product of the remaining key vals
    prms_items = prms.items()
    prms_keys = [k for (k,v) in prms_items]
    prms_vals = [v for (k,v) in prms_items]
    import itertools
    prms_vals_combos = list(itertools.product(*prms_vals))
    print "Number of cross validation combos: %d" % len(prms_vals_combos)
    
    # Create a list of models, one for each param combo
    def set_key(d, tk, v):
        k = tk[0]
        if len(tk) == 1:
            d[k] = v
        else:
            set_key(d[k], tk[1:], v)

    import copy
    models = []
    for v_combo in prms_vals_combos:
        m = copy.deepcopy(model)
        for (tk,v) in zip(prms_keys, v_combo):
            set_key(m, tk, v)
        models.append(m)

    return models
