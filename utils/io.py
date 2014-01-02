import cPickle
import os
import numpy as np

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", default='standard_glm',
                      help="Type of model to use. See model_factory.py for available types.")

    parser.add_option("-d", "--dataFile", dest="dataFile", default=None,
                      help="Use this data file. If not specified, simulate from model.")

    parser.add_option("-s", "--sampleFile", dest="sampleFile", default=None,
                      help="Use this sample file, either as filename in the config directory, or as a path.")

    parser.add_option("-r", "--resultsDir", dest="resultsDir", default='.',
                      help="Save the results to this directory.")

    # Parallel-specific options for loading IPython profiles
    parser.add_option("-p", "--profile", dest="profile", default='default',
                      help="IPython parallel profile to use.")

    parser.add_option("-j", "--json", dest="json", default=None,
                      help="IPython parallel json file specifying which controller to connect to.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if not options.dataFile is None and not os.path.exists(options.dataFile):
        raise Exception("Invalid data file specified: %s" % options.dataFile)

    if not options.sampleFile is None and not os.path.exists(options.sampleFile):
        raise Exception("Invalid sample file specified: %s" % options.sampleFile)
    
    # Make a unique results folder for this run
    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid results folder specified: %s" % options.resultsDir)
    
    options.resultsDir = create_unique_results_folder(options.resultsDir)
    return (options, args)


def create_unique_results_folder(results_dir):
    """ Create a unique results folder for results.
        The folder will be named based on the current date and time
    """
    from datetime import datetime 
    d = datetime.now()
    unique_name = d.strftime('%Y_%m_%d-%H_%M')
    full_res_dir = os.path.join(results_dir, unique_name)

    # Create the results directory
    if not os.path.exists(full_res_dir):
        print "Creating results directory: %s" % full_res_dir
        os.makedirs(full_res_dir)

    return full_res_dir

def load_data(options):
    """ Load data from the specified file or generate synthetic data if necessary.
    """
    # Load data
    if not options.dataFile is None:
        if options.dataFile.endswith('.mat'):
            print "Loading data from %s" % options.dataFile
            print "WARNING: true parameters for synthetic data " \
                  "will not be loaded properly"

            import scipy.io
            data = scipy.io.loadmat(options.dataFile,
                                    squeeze_me=True)

            # Scipy IO is a bit weird... ints like 'N' are saved as arrays
            # and the dictionary of parameters doesn't get reloaded as a dictionary
            # but rather as a record array. Do some cleanup here.
            data['N'] = int(data['N'])
            data['T'] = np.float(data['T'])
            
        elif options.dataFile.endswith('.pkl'):
            print "Loading data from %s" % options.dataFile
            with open(options.dataFile,'r') as f:
                data = cPickle.load(f)

                # Print data stats
                N = data['N']
                Ns = np.sum(data['S'])
                T = data['S'].shape[0]
                fr = 1.0/data['dt']
                print "Data has %d neurons, %d spikes, " \
                      "and %d time bins at %.3fHz sample rate" % \
                      (N,Ns,T,fr)

        else:
            raise Exception("Unrecognized file type: %s" % options.dataFile)

    else:
        print "Generating synthetic data and saving to results directory."
        T_stop = 180
        data = generate_synth_data(population,
                                   T_stop=T_stop)

        # Save the data for reuse
        import scipy.io
        import os
        scipy.io.savemat(os.path.join(options.resultsDir, 'data.mat'), data)
        
        # Pickle the data so we can open it more easily
        with open(os.path.join(options.resultsDir, 'data.pkl'),'w') as f:
            cPickle.dump(data,f)
    
    return data
