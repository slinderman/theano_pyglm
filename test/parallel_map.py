# Run as script using 'python -m test.synth_map'
import cPickle
import scipy.io


from synth_harness import parse_cmd_line_args
from inference.coord_descent import coord_descent
from plotting.plot_results import plot_results
from models.model_factory import make_model
from population import Population

from utils.parallelutil import *
from IPython.parallel import Client

def load_data(options):
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
        raise Exception("Data is required for parallel MAP")

    return data

def load_imports_on_client(dview):
    """ Import required modules on client
    """
    dview.execute('from population import Population')
    dview.execute('popn = None')

def initialize_client(model, data):
    """ Initialize a population objsect on the client
    """
    # Initialize a model with N neurons
    print "Initializing GLM"
    global popn
    popn = Population(model)

    # Initialize the GLM with the data
    popn.set_data(data)

def run_parallel_map():
    """ Run a test with synthetic data and MCMC inference
    """
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    # Load the data
    data = load_data(options)
    # Get a model for the data
    model_type = 'standard_glm'
    model = make_model(model_type, N=data['N'])

    # Get parallel clients
    dview = get_engines(n_workers=8)

    # Load imports on the client
    load_imports_on_client(dview)

    # Initialize population objects on the clients
    dview.apply_sync(initialize_client, (model_type,N,data))

    ## Sample random initial state
    #x0 = population.sample()
    #ll0 = population.compute_log_p(x0)
    #print "LL0: %f" % ll0
    #
    ## Perform inference
    #x_inf = coord_descent(population, data, x0=x0, maxiter=1,
    #                      use_hessian=False,
    #                      use_rop=False)
    #ll_inf = population.compute_log_p(x_inf)
    #print "LL_inf: %f" % ll_inf
    #
    ## TODO Save results
    #
    ## Plot results
    #plot_results(population, x_inf, x_true)

if __name__ == "__main__":
    run_parallel_map()

