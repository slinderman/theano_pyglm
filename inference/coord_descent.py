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

def coord_descent(network_glm, x0=None):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = network_glm.model['N']
    network = network_glm.network

    # Make sure the network is a complete adjacency matrix because we
    # do not do integer programming
    if not isinstance(network.graph, CompleteGraphModel):
        raise Exception("MAP inference via coordinate descent can only be performed with the complete graph model.")
    
    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample()

    # Compute the log prob, gradient, and Hessian wrt to the network
    if not network.vars == []:
        # If the network is deterministic then there are no params to optimize
        # Otherwise compute the joint log probability

        # Compute the prior
        net_prior = theano.function(network.vars, network.log_p)
        # Compute the gradient of the joint log prob wrt the network
        g_prior, g_list = grad_wrt_list(network.log_p, network.vars)
        net_g_prior = theano.function(network.vars, g_prior)
        # Compute the Hessian of the joint log prob wrt the network
        H_prior = hessian_wrt_list(network.log_p, network.vars, g_list)
        net_H_prior = theano.function(network.vars, H_prior)

        all_vars = [network_glm.glm.n] + network.vars + [network_glm.glm.vars]
        net_lp = theano.function(all_vars, network_glm.glm.log_p)
        # Compute the gradient of the joint log prob wrt the network
        g, g_list = grad_wrt_list(network_glm.glm.log_p, network.vars)
        net_g = theano.function(all_vars, g)
        # Compute the Hessian of the joint log prob wrt the network
        H = hessian_wrt_list(network_glm.glm.log_p, network.vars, g_list)
        net_H = theano.function(all_vars, H)

    # Alternate fitting the network and fitting the GLMs
    x = x0
    x_prev = copy.deepcopy(x0)
    converged = False
    iter = 0
    maxiter = 50
    while not converged and iter < maxiter:
        iter += 1

        print "MAP Iteration %d." % iter
        if not network.vars == []:
            # Fit the network

            # Reduce the log prob, gradient, and Hessian across all GLM nodes.
            # We can do this because the log prob is a sum of log probs from each GLM,
            # plus the log prior from the network model.

            # TODO: Parallelize the reduction
            x_net_0, shapes = pack(x[0])
            x_glms = x[1:]
            nll = lambda x_net: -1.0 * reduce(lambda lp_acc,n: lp_acc + net_lp(*([n] + unpack(x_net,shapes) + x_glms[n])),
                                              np.arange(N),
                                              net_prior(x_net))
            grad_nll = lambda x_net: -1.0 * reduce(lambda g_acc,n: g_acc + net_g(*([n] + unpack(x_net,shapes) + x_glms[n])),
                                                   np.arange(N),
                                                   net_g_prior(x_net))
            hess_nll = lambda x_net: -1.0 * reduce(lambda H_acc,n: H_acc + net_H(*([n] + unpack(x_net,shapes) + x_glms[n])),
                                                   np.arange(N),
                                                   net_H_prior(x_net))

            x_net_opt = opt.fmin_ncg(nll, x_net_0,
                                     fprime=grad_nll,
                                     fhess=hess_nll,
                                     disp=True)
            x[0] = unpack(x_net_opt, shapes)

        # Fit the GLMs.
        # TODO Parallelize this!
        for n in np.arange(N):
            n = np.int32(n)
            x_net = x[0]
            x_glm_0, shapes = pack(x[n + 1])
            nll = lambda xn: -1.0 * network_glm.glm.f_lp(*([n] + x_net + unpack(xn, shapes)))
            grad_nll = lambda xn: -1.0 * network_glm.glm.g_lp(*([n] + x_net + unpack(xn, shapes)))
            hess_nll = lambda xn: -1.0 * network_glm.glm.H_lp(*([n] + x_net + unpack(xn, shapes)))

            xn_opt = opt.fmin_ncg(nll, x_glm_0,
                                  fprime=grad_nll,
                                  fhess=hess_nll,
                                  disp=True)
            x[n + 1] = unpack(xn_opt, shapes)

        diffs = np.zeros(len(x))
        for i in range(len(x)):
            xi, _ = pack(x[i])
            xip, _ = pack(x_prev[i])
            if len(xi) == 0:
                dxi = 0.0
            else:
                dxi = np.mean((xi - xip) ** 2)
            diffs[i] = dxi
        maxdiff = np.max(diffs)

        print diffs
        converged = maxdiff < 1e-5
        x_prev = copy.deepcopy(x)
    return x


def pack(var_list):
    """ Pack a list of variables (as numpy arrays) into a single vector
    """
    vec = np.zeros((0,))
    shapes = []
    for var in var_list:
        assert isinstance(vec, np.ndarray), "Can only pack numpy arrays!"
        sz = var.size
        shp = var.shape
        assert sz == np.prod(shp), "Just making sure the size matches the shape"
        shapes.append(shp)
        vec = np.concatenate((vec, np.reshape(var, (sz,))))
    return vec, shapes


def unpack(vec, shapes):
    """ Unpack a vector of variables into an array
    """
    off = 0
    var_list = []
    for shp in shapes:
        sz = np.prod(shp)
        var_list.append(np.reshape(vec[off:off + sz], shp))
        off += sz
    assert off == len(vec), "Unpack was called with incorrect shapes!"
    return var_list
