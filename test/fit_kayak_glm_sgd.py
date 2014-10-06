# Run as script using 'python -m test.synth_map'
import cPickle
import os

from pyglm.population import KayakPopulation

from pyglm.inference.coord_descent import coord_descent
from pyglm.plotting.plot_results import plot_results
from pyglm.models.model_factory import make_model, stabilize_sparsity
from pyglm.utils.io import parse_cmd_line_args, load_data


def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    # Load data from file or create synthetic test dataset
    data = load_data(options)

    print "Creating master population object"
    model = make_model(options.model, N=data['N'], dt=0.001)
    stabilize_sparsity(model)
    popn = KayakPopulation(model)
    popn.add_data(data)

    # Initialize the GLM with the true data
    popn_true = None
    x_true = None
    if 'vars' in data:
        x_true = data['vars']

        # Load the true population
        data_dir = os.path.dirname(options.dataFile)
        model_file = os.path.join(data_dir, 'model.pkl')
        print "Loading true model from %s" % model_file
        with open(model_file) as f:
            popn_true = cPickle.load(f)
            popn_true.set_parameters(x_true)
            popn_true.add_data(data)
            ll_true = popn_true.compute_log_p(x_true)
            print "true LL: %f" % ll_true

    # Sample random initial state
    x_inf = popn.sample()
    ll0 = popn.compute_log_p(x_inf)
    print "LL0: %f" % ll0

    # A helper function to take a gradient descent step for each parameter.
    learn = 0.001
    def _sgd_step(d):
        """ Iterate over the dict and take SGD steps for each value
        """
        x_next = {}
        for (name,prm) in d.items():
            if isinstance(prm, dict):
                sub_x_next = _sgd_step(prm)
                x_next[name] = sub_x_next
            else:
                grad = popn.log_p.grad(prm)
                prm.value += learn * grad
                x_next[name] = prm.value

        return x_next

    # Perform inference with SGD
    prms = popn.get_parameters()
    for ii in xrange(1000):
        x_inf = _sgd_step(prms)
        print "Iteration %d:\t\tLP:%.2f" % (ii, popn.log_p.value)

    ll_inf = popn.log_p.value
    print "LL_inf: %f" % ll_inf

    # Save results
    results_file = os.path.join(options.resultsDir, 'results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(x_inf, f, protocol=-1)

    # Plot results
    plot_results(popn, x_inf, popn_true, x_true, resdir=options.resultsDir)

run_synth_test()

