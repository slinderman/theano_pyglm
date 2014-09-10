"""
Estimate the marginal log likelihood of the data under a given model.
"""
import sys

import numpy as np
from scipy.misc import logsumexp
from IPython.parallel.util import interactive

from pyglm.utils.parallel_util import parallel_compute_log_p


def initialize_imports(dview):
    """ Import functions req'd for coordinate descent
    """
    dview.execute('from pyglm.utils.theano_func_wrapper import seval, _flatten')
    dview.execute('from pyglm.utils.packvec import *')
    dview.execute('from pyglm.inference.gibbs import *')
    dview.execute('from pyglm.log_sum_exp import log_sum_exp_sample')
    dview.execute('from hips.inference.hmc import hmc')

def concatenate_parallel_updates(xs, x):
    # Concatenate results into x
    for (n, xn) in enumerate(xs):
        x['glms'][n] = xn['glms'][n]

        # Copy over the network 
        if 'A' in xn['net']['graph']:
            x['net']['graph']['A'][:,n] = xn['net']['graph']['A'][:,n]
        if 'W' in xn['net']['weights']:
            N = len(xs)
            W_inf = np.reshape(xn['net']['weights']['W'], (N,N))
            W_curr = np.reshape(x['net']['weights']['W'], (N,N))
            W_curr[:,n] = W_inf[:,n]
            x['net']['weights']['W'] = np.ravel(W_curr)

def concatenate_network_results(x_net, x, N):
    """ Concatenate the list of results from the parallel
        sampling of network columns
    """
    for (n, xn) in enumerate(x_net):
        if 'A' in xn['graph']:
            x['net']['graph']['A'][:,n] = xn['graph']['A'][:,n]
        if 'W' in xn['weights']:
            W_inf = np.reshape(xn['weights']['W'], (N,N))
            W_curr = np.reshape(x['net']['weights']['W'], (N,N))
            W_curr[:,n] = W_inf[:,n]
            x['net']['weights']['W'] = np.reshape(W_curr, (N**2,))

def periodically_save_results(x,d,results_dir):
    """ Periodically save the MCMC samples
    """
    fname = "ais_results.partial.%d.pkl" % (d)
    import os
    import cPickle
    print "Saving partial results to %s" % os.path.join(results_dir, fname)
    with open(os.path.join(results_dir, fname),'w') as f:
        cPickle.dump(x, f, protocol=-1)

def check_convergence(x):
    """ Check for convergence of the sampler
    """
    return False

def parallel_ais(client,
                 data,
                 N_samples=1000,
                 x0=None,
                 B=200,
                 steps_per_B=10,
                 resdir='.'):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    dview = client[:]
    master = client[client.ids[0]]
    N = data['N']

    # Import req'd functions on engines
    initialize_imports(dview)

    # Draw initial state from prior if not given
    # if x0 is None:
    #     master.execute('x0 = popn.sample()', block=True)
    #     x0 = master['x0']

    # Create parallel samplers
    @interactive
    def _create_samplers():
        global serial_updates
        global parallel_updates
        serial_updates, parallel_updates = initialize_updates(popn)

        # Return the number of parallel_updates 
        return len(serial_updates), len(parallel_updates)

    n_serial_updates, n_parallel_updates = dview.apply(_create_samplers).get()[0]

    # Create map-able functions to sample in parallel
    @interactive
    def _parallel_update(i, x, n):
        return parallel_updates[i].update(x, n)
        
    @interactive
    def _serial_update(i, x):
        return serial_updates[i].update(x)

    def _set_beta(dview, beta):
        dview.execute('popn.glm.lkhd_scale.set_value(%f)' % beta, block=True)

    #betas = np.linspace(0,1,B)
    betas = np.concatenate((np.linspace(0,0.01,B/5), 
                            np.logspace(-2,0,4*B/5)))
    B = len(betas)

    # Sample m points
    log_weights = np.zeros((N_samples,B))
    for m in range(N_samples):
        # Sample a new set of graph parameters from the prior
        # x = copy.deepcopy(x0)
        master.execute('x0 = popn.sample()', block=True)
        x = master['x0']

        # print "M: %d" % m
        # Sample mus from each of the intermediate distributions,
        # starting with a draw from the prior.

        # Ratios correspond to the 'f_{n-1}(x_{n-1})/f_{n}(x_{n-1})' values in Neal's paper
        ratios = np.zeros(B-1)

        # Sample the intermediate distributions
        for (n,beta) in zip(range(1,B), betas[1:]):
            # print "M: %d\tBeta: %.3f" % (m,beta)

            # Set the likelihood scale (beta) in the graph model
            _set_beta(dview, beta)

            # Take many samples to mix over this beta
            for s in range(steps_per_B):
                sys.stdout.write("M: %d\tBeta: %.3f\tSample: %d \r" % (m,beta,s))
                sys.stdout.flush()

                # Go through variables, sampling one at a time, in parallel where possible
                for i in range(n_parallel_updates):
                    xs = dview.map_async(_parallel_update,
                                         [i]*N,
                                         [x]*N,
                                         range(N))

                    # wait_watching_stdout(xs, interval=interval)
                    concatenate_parallel_updates(xs.get(), x)

                # Sample serial updates
                for i in range(n_serial_updates):
                    x = master.apply(_serial_update, i, x).get()


            # Compute the ratio of this sample under this distribution and the previous distribution
            curr_lkhd = parallel_compute_log_p(dview, master, x, N)

            # import pdb; pdb.set_trace()
            _set_beta(dview, betas[n-1])
            prev_lkhd = parallel_compute_log_p(dview, master, x, N)

            print ""
            print "curr_lkhd", curr_lkhd
            print "prev_lkhd", prev_lkhd

            ratios[n-1] = curr_lkhd - prev_lkhd
            log_weights[m,n] = curr_lkhd - prev_lkhd

        if np.mod(m, 10) == 0:
            periodically_save_results(log_weights[:(m+1)], m, resdir)

        print "W: %f" % log_weights[m,:].sum()

    # Compute the mean of the weights to get an estimate of the normalization constant
    log_Z = -np.log(N_samples) + logsumexp(log_weights.sum(axis=1))
    return log_Z, log_weights
