from IPython.parallel import Client

import os
import cPickle

from utils.parallel_util import initialize_imports, create_population_on_engines
from utils.io import parse_cmd_line_args, load_data
from population import Population
from models.model_factory import *

def initialize_parallel_test_harness():
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

    # Create a client with direct view to all engines
    if options.json is not None:
        client = Client(options.json)
    else:
        client = Client(profile=options.profile)
    dview = client[:]
    print "Found %d engines." % len(dview)

    print "Initializing imports on each engine"
    initialize_imports(dview)

    print "Creating population objects on each engine"
    create_population_on_engines(dview, data, options.model)

    return options, popn, data, client, popn_true, x_true
