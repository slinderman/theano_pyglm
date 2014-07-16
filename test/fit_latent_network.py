# Run as script using 'python -m test.synth'
import os
import cPickle
import numpy as np
import copy

from models.model_factory import make_model, convert_model
from population import Population
from synth_harness import initialize_test_harness

from inference.gibbs import LatentDistanceNetworkUpdate

def fit_latent_network_given_A(x0, loc_sampler, N_samples=1000):
    """
    Fit the parameters of a latent network model
    """
    x = x0
    smpls = [x0]
    for s in np.arange(N_samples):
        if np.mod(s, 100) == 0:
            print "Iteration %d" % s
        x = copy.deepcopy(x)
        try:
            x = loc_sampler.update(x)
        except Exception as e:
            print "ERROR! Sampled invalid location"
            x = copy.deepcopy(smpls[-1])
        smpls.append(x)

    return smpls

def fit_latent_network_to_mle():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, popn_true, x_true = initialize_test_harness()

    import pdb; pdb.set_trace()
    # Load MLE parameters from command line
    mle_x = None
    if options.x0_file is not None:
        with open(options.x0_file, 'r') as f:
            print "Initializing with state from: %s" % options.x0_file
            mle_x = cPickle.load(f)

            mle_model = make_model('standard_glm', N=data['N'])
            mle_popn = Population(mle_model)
            mle_popn.set_data(data)

    # Create a location sampler
    print "Initializing latent location sampler"
    loc_sampler = LatentDistanceNetworkUpdate()
    loc_sampler.preprocess(popn)

    # Convert the mle results into a weighted adjacency matrix
    x_aw = popn.sample()
    x_aw = convert_model(mle_popn, mle_model, mle_x, popn, popn.model, x_aw)

    # Get rid of unnecessary keys
    del x_aw['glms']

    # Fit the latent distance network to a thresholded adjacency matrix
    ws = np.sort(np.abs(x_aw['net']['weights']['W']))

    wperm = np.argsort(np.abs(x_aw['net']['weights']['W']))
    nthrsh = 20
    threshs = np.arange(ws.size, step=ws.size/nthrsh)

    res = []

    N = popn.N
    for th in threshs:
        print "Fitting network for threshold: %.3f" % th
        A = np.zeros_like(ws, dtype=np.int8)
        A[wperm[th:]] = 1
        A = A.reshape((N,N))
        # A = (np.abs(x_aw['net']['weights']['W']) >= th).astype(np.int8).reshape((N,N))

        # Make sure the diag is still all 1s
        A[np.diag_indices(N)] = 1

        x = copy.deepcopy(x_aw)
        x['net']['graph']['A'] = A
        smpls = fit_latent_network_given_A(x, loc_sampler)

        # Index the results by the overall sparsity of A
        key = (np.sum(A)-N) / (np.float(np.size(A))-N)
        res.append((key, smpls))

    # Save results
    results_file = os.path.join(options.resultsDir, 'fit_latent_network_results.pkl')
    print "Saving results to %s" % results_file
    with open(results_file, 'w') as f:
        cPickle.dump(res, f)

if __name__ == "__main__":
    fit_latent_network_to_mle()
