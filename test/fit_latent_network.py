# Run as script using 'python -m test.synth'
import os
import sys
import cPickle
import numpy as np
import copy
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import brewer2mpl

from utils.theano_func_wrapper import seval
from models.model_factory import make_model, convert_model
from population import Population
from synth_harness import initialize_test_harness

from inference.gibbs import LatentDistanceNetworkUpdate, MetropolisHastingsUpdate

def sample_network_from_prior(model):
    """
    Sample a network from the prior
    """
    popn = Population(model)
    x_true = popn.sample()
    A_true = x_true['net']['graph']['A']

    return popn, x_true, A_true

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

def ais_latent_network_given_A(x0, graph_model, graph_sampler, N_samples=1000, B=100,
                               steps_per_B=11):
    """
    Use AIS to approximate the marginal likelihood of a latent network model
    """
    import pdb; pdb.set_trace()
    betas = np.linspace(0,1,B)

    # Sample m points
    log_weights = np.zeros(N_samples)
    for m in range(N_samples):
        # Sample a new set of graph parameters from the prior
        x = copy.deepcopy(x0)

        # print "M: %d" % m
        # Sample mus from each of the intermediate distributions,
        # starting with a draw from the prior.
        samples = []

        # Ratios correspond to the 'f_{n-1}(x_{n-1})/f_{n}(x_{n-1})' values in Neal's paper
        ratios = np.zeros(B-1)

        # Sample the intermediate distributions
        for (n,beta) in zip(range(1,B), betas[1:]):
            # print "M: %d\tBeta: %.3f" % (m,beta)
            sys.stdout.write("M: %d\tBeta: %.3f \r" % (m,beta))
            sys.stdout.flush()
            # Set the likelihood scale (beta) in the graph model
            graph_model.lkhd_scale.set_value(beta)

            # Take 100 steps per beta
            for s in range(steps_per_B):
                x = graph_sampler.update(x)

            # Compute the ratio of this sample under this distribution and the previous distribution
            curr_lkhd = seval(graph_model.log_p,
                              graph_model.get_variables(),
                              x['net']['graph'])

            graph_model.lkhd_scale.set_value(betas[n-1])
            prev_lkhd = seval(graph_model.log_p,
                              graph_model.get_variables(),
                              x['net']['graph'])

            ratios[n-1] = curr_lkhd - prev_lkhd

        # Compute the log weight of this sample
        log_weights[m] = np.sum(ratios)

        print ""
        print "W: %f" % log_weights[m]

    # Compute the mean of the weights to get an estimate of the normalization constant
    log_Z = -np.log(N_samples) + logsumexp(log_weights)
    return log_Z

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

def rotate_sample(L):
    """
    Rotate such that the first neuron is at 0'
    """
    N,D = L.shape
    assert D==2, 'L must be 2d'
    c = np.mean(L,axis=0)
    x = (L-c)[:,0]
    y = (L-c)[:,1]

    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)

    # Rotate
    th -= th[0]
    xn = r * np.cos(th)
    yn = r * np.sin(th)

    Ln = np.concatenate((xn[:,None], yn[:,None]), axis=1)
    return Ln


def plot_latent_distance_samples(L_true, L_smpls_o, A_true=None, ax=None):
    """
    Visualize the latent distance samples
    """
    # L_true = x_true['net']['graph']['L']
    # L_smpls_o = np.array([s['net']['graph']['L'] for s in smpls])
    N_smpls = len(L_smpls_o)

    if L_true.ndim == 1:
        D = 1
        N = L_true.shape[0]
    elif L_true.ndim == 2:
        N,D = L_true.shape
        L_true = rotate_sample(L_true)
        L_smpls = np.array([rotate_sample(Ls) for Ls in L_smpls_o])
    else:
        raise Exception('Can only visualize 1 or 2-D locations')

    # Only plot the burned in samples
    start = max(N_smpls-100, 0)

    N_colors = min(9, N)
    colors = brewer2mpl.get_map('Set1', 'Qualitative', N_colors).mpl_colors
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)

    # Plot the true location and the inferred location
    if D == 1:
        # Histogram for 1D?
        # Scatter at fixed height
        for n in range(N):
            color = colors[np.mod(n,N_colors)]
            Ln_true = L_true[n]
            Ln_smpls = L_smpls[start:,n]

            ax.scatter(Ln_true, n/float(N), marker='o',
                       facecolor=color, edgecolor=color, s=40)
            ax.scatter(Ln_smpls, n/float(N) * np.ones_like(Ln_smpls),
                       marker='o', facecolor='w', edgecolor=color, s=20)

    elif D == 2:
        # Scatter for 2D
        for n in range(N):
            color = colors[np.mod(n,N_colors)]
            Ln_true = L_true[n,:]
            Ln_smpls = L_smpls[start:,n]

            ax.scatter(Ln_true[0], Ln_true[1], marker='o',
                       facecolor=color, edgecolor=color, s=40)
            ax.scatter(Ln_smpls[:,0], Ln_smpls[:,1],
                       marker='o', facecolor='w', edgecolor=color,
                       alpha=0.25, s=20)

        # Plot the network
        if A_true is not None:
            for n1 in range(N):
                for n2 in range(N):
                    if A_true[n1,n2]:
                        ax.plot([L_true[n1,0], L_true[n2,0]],
                                [L_true[n1,1], L_true[n2,1]],
                                '-k')

    # Show the plot
    plt.show()

    return fig, ax

def compute_distances(Ls):
    """
    Compute pairwise distances for the given set of locations
    """
    if not isinstance(Ls, list):
        Ls = [Ls]

    dists = []
    for L in Ls:
        N,D = L.shape
        # 1xNxD - Nx1xD (L1 distance)
        dist = (np.abs(L[None,:,:] - L[:,None,:])).sum(axis=2)
        dists.append(dist)

    return dists

def compute_diff_of_dists(L_true, L_smpls):
    N_smpls = len(L_smpls)

    assert L_true.ndim == 2

    true_dist = compute_distances(L_true)[0]
    inf_dists = compute_distances(L_smpls)

    error_dists = [(inf_dist - true_dist)**2 for inf_dist in inf_dists]
    mse = [error_dist.mean() for error_dist in error_dists]
    medse = [np.median(error_dist.ravel()) for error_dist in error_dists]
    stdse = [error_dist.std() for error_dist in error_dists]

    fig = plt.figure()
    # plt.errorbar(np.arange(N_smpls), mse, yerr=stdse)
    plt.errorbar(np.arange(N_smpls), mse, color='b')
    plt.errorbar(np.arange(N_smpls), medse, color='r')
    plt.show()
    return mse, stdse

def test_latent_distance_network_sampler(N, N_samples=10000):
    """
    Generate a bunch of latent distance networks, run the sampler
    on them to see how well we mix over latent locations.

    :param N: Number of neurons in the network
    """
    true_model_type = 'latent_distance'
    if true_model_type == 'erdos_renyi':
        true_model = make_model('sparse_weighted_model', N)
    elif true_model_type == 'latent_distance':
        true_model = make_model('distance_weighted_model', N)

    distmodel = make_model('distance_weighted_model', N)
    D = distmodel['network']['graph']['N_dims']
    trials = 1
    for t in range(trials):
        # Generate a true random network
        popn_true, x_true, A_true = sample_network_from_prior(true_model)
        dist_popn, x_inf, _ = sample_network_from_prior(distmodel)

        # Seed the inference population with the true network
        x_inf['net']['graph']['A'] = A_true

        # Create a location sampler
        print "Initializing latent location sampler"
        loc_sampler = LatentDistanceNetworkUpdate()
        loc_sampler.preprocess(dist_popn)

        # Run the sampler
        N_samples = 1000
        smpls = fit_latent_network_given_A(x_inf, loc_sampler, N_samples=N_samples)

        if true_model_type == 'latent_distance':
            # Evaluate the state
            L_true = x_true['net']['graph']['L'].reshape((N,D))
            L_smpls = [x['net']['graph']['L'].reshape((N,D)) for x in smpls]

            # Visualize the results
            plot_latent_distance_samples(L_true, L_smpls, A_true=A_true)

            # Plot errors in relative distance over time
            compute_diff_of_dists(L_true, L_smpls)

        # Compute marginal likelihood of erdos renyi with the same sparsity
        nnz_A = float(A_true.sum())
        N_conns = A_true.size
        # Ignore the diagonal
        nnz_A -= N
        N_conns -= N
        # Now compute density
        er_rho = nnz_A / N_conns
        true_er_marg_lkhd = nnz_A * np.log(er_rho) + (N_conns-nnz_A)*np.log(1-er_rho)
        print "True ER Marg Lkhd: ", true_er_marg_lkhd

        # DEBUG: Make sure AIS gives the same answer as what we just computed
        # er_model = make_model('sparse_weighted_model', N)
        # er_model['network']['graph']['rho'] = er_rho
        # er_popn, x_inf, _ = sample_network_from_prior(er_model)
        # # Make a dummy update for the ER model
        # er_sampler = MetropolisHastingsUpdate()
        # er_x0 = er_popn.sample()
        # er_x0['net']['graph']['A'] = A_true
        # er_marg_lkhd = ais_latent_network_given_A(er_x0,
        #                                           er_popn.network.graph,
        #                                           er_sampler
        #                                           )
        #
        # print "AIS ER Marg Lkhd: ", er_marg_lkhd



        # Approximate the marginal log likelihood of the distance mode
        dist_x0 = dist_popn.sample()
        dist_x0['net']['graph']['A'] = A_true
        dist_marg_lkhd = ais_latent_network_given_A(dist_x0,
                                                    dist_popn.network.graph,
                                                    loc_sampler
                                                    )
        print "Dist Marg Lkhd: ", dist_marg_lkhd

if __name__ == "__main__":
    fit_latent_network_to_mle()
    # test_latent_distance_network_sampler(16)

