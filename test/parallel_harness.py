from IPython.parallel import Client
from IPython.parallel.util import interactive

import os
import cPickle

from utils.io import parse_cmd_line_args, load_data
from population import Population
from models.model_factory import *

def initialize_imports(dview):
    """ Import required model code on the clients.
        Initialize global variable for each client's local population obj
    """
    dview.execute('from population import Population')
    dview.execute('from models.model_factory import make_model')
    dview.execute('import cPickle')

def set_data_on_engines(dview, d):
    """ Send the data to each engine
    """
    # Initialize the GLM with the data
    # TODO: This seems to be the bottleneck for large datasets!
    # The data is sent to each worker sequentially (not broadcast)
    #dview['data'] = d

    # Pickle the data and save it to the shared file system
    with open('.temp_data.pkl', 'w') as f:
        cPickle.dump(d, f, protocol=-1)

    # Initialize the data globa
    dview['data'] = 'tmp'
    @interactive
    def _load_data():
        global data
        with open('.temp_data.pkl', 'r') as f:
            data = cPickle.load(f)

    dview.apply_sync(_load_data)

    # Delete the temp data
    os.remove('.temp_data.pkl')

    dview.execute("popn.set_data(data)", block=True)


def set_hyperparameters_on_engines(dview, m):
    """ Send the hyperparameters to each engine
    """
    dview['model'] = m

    #@interactive
    #def _set_hyperparameters():
    #    popn.set_hyperparameters(model)

    dview.execute("popn.set_hyperparameters(model)", block=True)
    #dview.apply(_set_hyperparameters, block=True)

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
    m = make_model(model_type, N=N)
    dview['model'] = m


    # Create a population object on each engine
    #@interactive
    #def _create_population():
    #    global popn
    #    popn = Population(model)
    #dview.apply(_create_population, block=True)

    dview.execute('popn = Population(model)', block=True)

    # Initialize the GLM with the data
    #dview['data'] = data
    #dview.execute("popn.set_data(data)", block=True)
    set_data_on_engines(dview, data)

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
            #ll_true = popn_true.compute_log_p(x_true)
            #print "true LL: %f" % ll_true

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
