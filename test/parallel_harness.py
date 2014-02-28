from IPython.parallel import Client

import os
import cPickle

from utils.io import parse_cmd_line_args, load_data
from population import Population
from models.model_factory import *
from inference.parallel_coord_descent import parallel_coord_descent

def initialize_imports(dview):
    """ Import required model code on the clients.
        Initialize global variable for each client's local population obj
    """
    dview.execute('from population import Population')
    dview.execute('from models.model_factory import make_model')

def create_population_on_engines(dview,
                                 data,
                                 model_type
                                 ):
    """ Initialize a model with N neurons. Use the data if specified on the
        command line, otherwise sample new data from the model.
        Return a population object, the data, and a set of true parameters
        which is expected for synthetic tests 
    """
    # Initialize a model with N neurons
    N = data['N']
    model = make_model(model_type, N=N)
    dview['model'] = model


    # Create a population object on each engine
    dview.execute('popn = Population(model)', block=True)

    # Initialize the GLM with the data
    dview['data'] = data
    dview.execute("popn.set_data(data)", block=True)

def initialize_parallel_test_harness():
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
