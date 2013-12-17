import cPickle
import os
from IPython.parallel import Client

from population import Population
from models.model_factory import *
from inference.parallel_coord_descent import parallel_coord_descent


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

    parser.add_option("-p", "--profile", dest="profile", default='default',
                      help="IPython parallel profile to use.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if not options.dataFile is None and not os.path.exists(options.dataFile):
        raise Exception("Invalid data file specified: %s" % options.dataFile)

    if not options.sampleFile is None and not os.path.exists(options.sampleFile):
        raise Exception("Invalid sample file specified: %s" % options.sampleFile)

    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid sample file specified: %s" % options.resultsDir)

    return (options, args)

def initialize_imports(dview):
    """ Import required model code on the clients.
        Initialize global variable for each client's local population obj
    """
    dview.execute('from population import Population')
    dview.execute('from models.model_factory import make_model')


def create_population_on_engines(dview,
                                 data,
                                 model_type='standard_glm'
                                 ):
    """ Initialize a model with N neurons. Use the data if specified on the
        command line, otherwise sample new data from the model.
        Return a population object, the data, and a set of true parameters
        which is expected for synthetic tests 
    """
    # Initialize a model with N neurons
    print "Initializing GLM"
    N = data['N']
    model = make_model(model_type, N=N)
    dview['model'] = model

    # Create a population object on each engine
    dview.execute('popn = Population(model)', block=True)

    # Initialize the GLM with the data
    dview['data'] = data
    dview.execute("popn.set_data(data)", block=True)

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
        raise Exception("Data must be specified!")

    return data

def initialize_parallel_test_harness(model_type='standard_glm'):
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    data = load_data(options)

    print "Creating master population object"
    model = make_model(model_type, N=data['N'])
    popn = Population(model)
    popn.set_data(data)

    # Create a client with direct view to all engines
    client = Client(profile=options.profile)
    dview = client[:]

    print "Initializing imports on each engine"
    initialize_imports(dview)

    print "Creating population objects on each engine"
    create_population_on_engines(dview, data, model_type='standard_glm')

    return popn, data, client



