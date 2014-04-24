import os
import cPickle

def plot_residuals(popn, x, popn_true, x_true, resdir='.'):
    pass

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    
    parser.add_option("-d", "--dataFile", dest="dataFile",
                      help="Data file to load")

    parser.add_option("-r", "--resultsFile", dest="resultsFile", default='.',
                      help="Results file to plot.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if options.resultsFile is None or not os.path.exists(options.resultsFile):
        raise Exception("Invalid results file: %s" % options.resultsFile)

    return (options, args)


if __name__ == "__main__":
    from test.synth_harness import initialize_test_harness
    options, popn, data, popn_true, x_true = initialize_test_harness()
    
    # Load the results
    with open(options.x0_file, 'r') as f:
        print "Loading results from: %s" % options.x0_file
        x = cPickle.load(f)
        # If x is a list of samples, only keep the last (burned-in) fraction
        if isinstance(x, list):
            smpl_frac = 1.0
            x = x[-1*int(smpl_frac*len(x)):]

    print "Plotting results"
    plot_residuals(popn,
                   x,
                   popn_true=popn_true,
                   x_true=x_true,
                   resdir=options.resultsDir)
