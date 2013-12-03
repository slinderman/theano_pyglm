""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy
import theano
import theano.tensor as T
import numpy as np

from utils.grads import *

def gibbs_sample(network_glm, x0=None, N_samples=1000):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = network_glm.model['N']
    network = network_glm.network

    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample()

    # Set up for Gibbs sampling? 
    # Compute gradients for HMC, etc. 

    # Alternate fitting the network and fitting the GLMs
    x_smpls = []
    x = x0
    for smpl in np.arange(N_samples):
        # Go through variables, sampling one at a time, in parallel where possible
        x[0] = network.gibbs_step(x, network_glm)
        
        # Sample the GLM parameters
        # TODO Parallelize this!
        for n in np.arange(N):
            n = np.int32(n)
            x[n+1] = network_glm.glm.gibbs_step(x, network_glm, n)
        x_smpls.append(copy.deepcopy(x))

        # Print the current log likelihood
        lp = network_glm.compute_log_p(x)
        print "Iter %d: Log prob: %.3f" % (smpl,lp)

    return x_smpls

