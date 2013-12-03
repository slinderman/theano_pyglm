""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy
import theano
import theano.tensor as T
import numpy as np

import scipy.optimize as opt

from utils.grads import *
from components.graph import CompleteGraphModel

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
    x = x0
    converged = False
    for smpl in np.arange(N_samples):
        x_prev = copy.deepcopy(x)
        # Go through variables, sampling one at a time, in parallel where possible
        x[0] = network.gibbs_step(x)
        
        # Sample the GLM parameters
        # TODO Parallelize this!
        for n in np.arange(N):
            n = np.int32(n)
            x[n+1] = network.glm.gibbs_step(x, n)
        
    return x

