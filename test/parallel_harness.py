from IPython.parallel import Client
from IPython.parallel.util import interactive

import os
import cPickle

from utils.io import parse_cmd_line_args, load_data
from population import Population
from models.model_factory import *

def parallel_compute_ll(dview,
                        v,
                        N):
    """ Compute the log prob in parallel
    """
    ll_tot = 0
    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_ll(n, vs):
        print "Computing log lkhd for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.ll,
                   syms,
                   nvars)
        return lp

    ll_glms = dview.map_async(_compute_glm_ll,
                              range(N),
                              [v]*N)

    ll_tot += sum(ll_glms.get())
    return ll_tot


def parallel_compute_log_p(dview,
                           master,
                           v,
                           N):
    """ Compute the log prob in parallel
    """

    # Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp_tot = 0

    @interactive
    def _compute_network_lp(vs):
        print "Computing log prob for network"
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs,0)
        lp = seval(popn.network.log_p,
                   syms,
                   nvars)
        #lp = popn.network.log_p.eval(dict(zip(_flatten(tmpsyms),
                                     #        _flatten(tmpnvars))),
                                     #on_unused_input='ignore')
        return lp

    lp_tot += master.apply_sync(_compute_network_lp, v)

    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_lp(n, vs):
        print "Computing lp for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.log_p,
                   syms,
                   nvars)
        return lp

    lp_glms = dview.map_async(_compute_glm_lp,
                              range(N),
                              [v]*N)
    # print lp_glms.get()
    # lp_glms.display_outputs()

    lp_tot += sum(lp_glms.get())
    return lp_tot


def initialize_imports(dview):
    """ Import required model code on the clients.
        Initialize global variable for each client's local population obj
    """
    dview.execute('from population import Population')
    dview.execute('from models.model_factory import make_model')
    dview.execute('from utils.theano_func_wrapper import seval')
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
    if isinstance(model_type, dict):
        m = model_type
    elif isinstance(model_type, str):
        m = make_model(model_type, N=N)
    else:
        raise Exception("Create_population_on_engines requires model to be either str type or dict")
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
