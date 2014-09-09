""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import scipy.optimize as opt

from pyglm.utils.packvec import *
from components.graph import CompleteGraphModel
from pyglm.components.weights import ConstantWeightModel

def map_estimate(network_glm, x0=None):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = network_glm.model['N']
    network = network_glm.network

    # Make sure the network is a complete adjacency matrix with constant weights
    # and basis function impulse responses
    if not isinstance(network.graph, CompleteGraphModel):
        raise Exception("MAP inference can only be performed with the complete graph model.")
    if not isinstance(network.weights, ConstantWeightModel):
        raise Exception("MAP inference can only be performed with the constant weight model.")
    
    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample(None)

    # Fit the GLMs.
    # TODO Parallelize this!
    x = x0
    for n in np.arange(N):
        print "Fitting GLM %d" % n
        n = np.int32(n)
        x_net = x[0]
        x_glm_0, shapes = pack(x[n+1])
        nll = lambda xn: -1.0 * network_glm.glm.f_lp(*([n] + x_net + unpack(xn, shapes)))
        grad_nll = lambda xn: -1.0 * network_glm.glm.g_lp(*([n] + x_net + unpack(xn, shapes)))
        hess_nll = lambda xn: -1.0 * network_glm.glm.H_lp(*([n] + x_net + unpack(xn, shapes)))

        xn_opt = opt.fmin_ncg(nll, x_glm_0,
                              fprime=grad_nll,
                              fhess=hess_nll,
                              disp=True)
        x[n+1] = unpack(xn_opt, shapes)
        

    return x
