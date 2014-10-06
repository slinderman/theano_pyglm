# Run as script using 'python -m test.synth_map'
import cPickle
import os
import numpy as np
import scipy.optimize as opt

from pyglm.population import KayakPopulation

from pyglm.inference.coord_descent import coord_descent
from pyglm.plotting.plot_results import plot_results
from pyglm.models.model_factory import make_model, stabilize_sparsity
from pyglm.utils.io import parse_cmd_line_args, load_data
from pyglm.utils.packvec import dictshapes, packdict, unpackdict


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
for n in range(popn.N):
    x_inf['glm_%d' % n]['imp']['w_ir'] *= 0
ll0 = popn.compute_log_p(x_inf)
print "LL0: %f" % ll0

# WARNING: Assuming that we have a standard GLM
for n in range(popn.N):
    xn_inf = x_inf['glm_%d' % n]
    xn_shapes = dictshapes(xn_inf)
    prms = popn.glms[n].get_variables()
    objective = -popn.glms[n].log_p

    # Helper function to compute the objective (neg log prob) for a vector of values
    def _compute_obj(x_vec):
        x_dict = unpackdict(x_vec, xn_shapes)
        popn._set_parameter_helper(prms, x_dict)

        # Handle NaNs
        if np.isnan(objective.value):
            return -np.Inf

        return objective.value

    # A helper function to compute the gradients of
    def _compute_grad_dict(d):
        """ Iterate over the dict and take SGD steps for each value
        """
        grad = {}
        for (name,prm) in d.items():
            if isinstance(prm, dict):
                sub_grad = _compute_grad_dict(prm)
                grad[name] = sub_grad
            else:
                grad[name] = objective.grad(prm)

        return grad

    def _compute_grad(x_vec):
        x_dict = unpackdict(x_vec, xn_shapes)
        # popn.set_parameters(x_dict)
        popn._set_parameter_helper(prms, x_dict)
        grad_dict = _compute_grad_dict(prms)
        return packdict(grad_dict)[0]

    # Callback to print progress. In order to count iters, we need to
    # pass the current iteration via a list
    ncg_iter_ls = [0]
    def progress_report(x_curr, ncg_iter_ls):
        ll = -1.0 * _compute_obj(x_curr)
        print "Newton iter %d.\t Neuron %d\tLL: %.1f" % (ncg_iter_ls[0], n, ll)
        ncg_iter_ls[0] += 1
    cbk = lambda x_curr: progress_report(x_curr, ncg_iter_ls)

    # Call the appropriate scipy optimization function
    res = opt.minimize(_compute_obj, packdict(xn_inf)[0],
                       method="bfgs",
                       jac=_compute_grad,
                       options={'disp': True,
                                'maxiter' : 200},
                       callback=cbk)
    xn_inf = unpackdict(res.x, xn_shapes)
    x_inf['glm_%d' % n] = xn_inf

# Finished optimizing the neurons. Compute the total log prob.
ll_inf = popn.log_p.value
print "LL_inf: %f" % ll_inf

# Save results
results_file = os.path.join(options.resultsDir, 'results.pkl')
print "Saving results to %s" % results_file
with open(results_file, 'w') as f:
    cPickle.dump(x_inf, f, protocol=-1)

# Plot results
plot_results(popn, x_inf, popn_true, x_true, resdir=options.resultsDir)


