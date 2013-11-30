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


def map_estimate(network_glm, x0=None):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = network_glm.model['N']
    network = network_glm.network

    # Make sure the network is a complete adjacency matrix
    # We don't do integer programming
    if not isinstance(network.graph, CompleteGraphModel):
        raise Exception("MAP inference can only be performed with the complete graph model.")

    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample()

    # Compute the log prob, gradient, and Hessian wrt to the network
    if not network.vars == []:
        # If the network is deterministic then there are no params to optimize
        # Otherwise compute the joint log probability
        net_lp = theano.function(network_glm.vars, network_glm.log_p)
        # Compute the gradient of the joint log prob wrt the network
        g, g_list = grad_wrt_list(network_glm.log_p, network.vars)
        net_g = theano.function(network_glm.vars, g)
        # Compute the Hessian of the joint log prob wrt the network
        H = hessian_wrt_list(network_glm.log_p, network.vars, g_list)
        net_H = theano.function(network_glm.vars, H)


    # Compute log probs, gradients, and Hessians for each GLM
    log_probs = []
    gradients = []
    hessians = []
    for (n, glm) in enumerate(network_glm.glms):
        # In order to compute the log probability we need the set of
        # variables on which it depends
        all_vars = network.vars + [glm.vars]
        glm_vars = [glm.vars]
        f_lp = theano.function(all_vars, glm.log_p)
        log_probs.append(f_lp)

        # Compute the gradient
        g, g_list = grad_wrt_list(glm.log_p, glm_vars)
        f_g = theano.function(all_vars, g)
        gradients.append(f_g)

        # And finally, compute the Hessian
        H = hessian_wrt_list(glm.log_p, glm_vars, g_list)
        f_H = theano.function(all_vars, H)
        hessians.append(f_H)

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
            x_net_0, shapes = pack(x[0])
            x_glms = reduce(lambda x_acc, x_glm: x_acc + x_glm, x[1:], [])
            nll = lambda x_net: -1.0 * net_lp(x_net, *x_glms)
            grad_nll = lambda x_net: -1.0 * net_g(x_net, *x_glms)
            hess_nll = lambda x_net: -1.0 * net_H(x_net, *x_glms)

            x_net_opt = opt.fmin_ncg(nll, x_net_0,
                                     fprime=grad_nll,
                                     fhess=hess_nll,
                                     disp=True)
            x[0] = unpack(x_net_opt, shapes)

        # Fit the GLMs.
        # TODO Parallelize this!
        for n in np.arange(N):
            x_net = x[0]
            x_glm_0, shapes = pack(x[n + 1])
            nll = lambda xn: -1.0 * log_probs[n](*(x_net + unpack(xn, shapes)))
            grad_nll = lambda xn: -1.0 * gradients[n](*(x_net + unpack(xn, shapes)))
            hess_nll = lambda xn: -1.0 * hessians[n](*(x_net + unpack(xn, shapes)))

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
