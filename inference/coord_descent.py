""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy
import theano
import theano.tensor as T
import numpy as np

import scipy.optimize as opt

from utils.theano_func_wrapper import seval, _flatten
from utils.packvec import *
from utils.grads import *
from components.graph import CompleteGraphModel

def coord_descent(network_glm, x0=None, maxiter=50, atol=1e-5):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = network_glm.model['N']
    network = network_glm.network
    glm = network_glm.glm
    syms = network_glm.get_variables()

    # Make sure the network is a complete adjacency matrix because we
    # do not do integer programming
    if not isinstance(network.graph, CompleteGraphModel):
        raise Exception("MAP inference via coordinate descent can only be performed \
                        with the complete graph model.")
    
    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample()

    # TODO Check whether the number of network variables is greater than zero,
    #      and, if so, fit the network
    #fit_network = (not network.vars == [])
    fit_network = False

    # Compute the log prob, gradient, and Hessian wrt to the network
    if fit_network:
        # If the network is deterministic then there are no params to optimize
        # Otherwise compute the joint log probability

        # Compute the prior
        #net_prior = theano.function(network.vars, network.log_p)
        ## Compute the gradient of the joint log prob wrt the network
        #g_prior, g_list = grad_wrt_list(network.log_p, network.vars)
        #net_g_prior = theano.function(network.vars, g_prior)
        ## Compute the Hessian of the joint log prob wrt the network
        #H_prior = hessian_wrt_list(network.log_p, network.vars, g_list)
        #net_H_prior = theano.function(network.vars, H_prior)
        #
        #all_vars = [network_glm.glm.n] + network.vars + [network_glm.glm.vars]
        #net_lp = theano.function(all_vars, network_glm.glm.log_p)
        ## Compute the gradient of the joint log prob wrt the network
        #g, g_list = grad_wrt_list(network_glm.glm.log_p, network.vars)
        #net_g = theano.function(all_vars, g)
        ## Compute the Hessian of the joint log prob wrt the network
        #H = hessian_wrt_list(network_glm.glm.log_p, network.vars, g_list)
        #net_H = theano.function(all_vars, H)

        print "Computing log probabilities, gradients, and Hessians for network variables"
        net_prior = network.log_p
        g_net_prior = grad_wrt_list(net_prior, syms['net'])
        H_net_prior = hessian_wrt_list(net_prior, syms['net'])
        _,net_shapes = pack(x0['net'])

        glm_logp = glm.log_p
        g_glm_logp_wrt_net = grad_wrt_list(glm_logp, syms['net'])
        H_glm_logp_wrt_net = hessian_wrt_list(glm_logp, syms['net'])

        # Private function to compute the log probability (or grads and Hessians thereof)
        # of the log probability given new network variables
        def net_helper(x_net_vec, x, net_expr, glm_expr):
            """ Compute the negative log probability (or gradients and Hessians thereof)
            of the given network variables
            """
            x_net = pack(x_net_vec, net_shapes)
            lp = seval(net_expr,
                       syms['net'],
                       x_net)

            # Reduce the log prob, gradient, and Hessian across all GLM nodes.
            # We can do this because the log prob is a sum of log probs from each GLM,
            # plus the log prior from the network model.
            # TODO Parallelize this loop!
            for n in np.arange(N):
                # Get the variables associated with the n-th GLM
                nvars = network_glm.extract_vars(x, n)
                # Override the network vars
                nvars['net'] = x_net
                lp += seval(glm_expr,
                            syms,
                            nvars)
            return -1.0*lp

    # Compute gradients of the log prob wrt the GLM parameters
    print "Computing log probabilities, gradients, and Hessians for GLM variables"
    import pdb
    pdb.set_trace()
    glm_syms = differentiable(syms['glm'])
    glm_logp = glm.log_p
    g_glm_logp_wrt_glm, g_list = grad_wrt_list(glm_logp, _flatten(glm_syms))

    ## DEBUG
    #g = T.grad(glm.log_p,_flatten(syms['glm']['imp']))
    #
    #xx = _flatten(syms['glm']['imp'])
    #H = T.hessian(glm.log_p,xx)
    ##H =  T.grad(g[0],_flatten(syms['glm']['imp']['ir0']))
    #theano.scan(lambda i, gy, x: T.grad(gy[i], x),
    #            sequences=T.arange(g[0].shape[0]),
    #            non_sequences=[g[0], _flatten(syms['glm']['imp'])])
    ## END DEBUG

    H_glm_logp_wrt_glm = hessian_wrt_list(glm_logp, _flatten(glm_syms), g_list)
    #v = T.dvector()
    #H_glm_logp_wrt_glm = hessian_rop_wrt_list(glm_logp,
    #                                          _flatten(glm_syms),
    #                                          v,
    #                                          g_vec=g_glm_logp_wrt_glm)

    # Augment the syms with v
    #symsH = copy.deepcopy(syms)
    #symsH['v'] = v
    # Test
    nvars = network_glm.extract_vars(x0, 0)
    print seval(H_glm_logp_wrt_glm, syms, nvars)

    # TODO Replace this hack with a function that evaluates shapes
    _,glm_shapes = pack(_flatten(x0['glms'][0]))

    # Private function to compute the log probability (or grads and Hessians thereof)
    # of the log probability given new network variables
    def glm_helper(x_glm_vec, x, glm_expr):
        """ Compute the negative log probability (or gradients and Hessians thereof)
        of the given glm variables
        """
        x_glm = pack(x_glm_vec, glm_shapes)
        x['glm'] = x_glm
        lp = seval(glm_expr,
                   syms,
                   x)
        return -1.0*lp

    # Alternate fitting the network and fitting the GLMs
    x = x0
    x_prev = copy.deepcopy(x0)
    converged = False
    iter = 0
    while not converged and iter < maxiter:
        iter += 1

        print "Coordinate descent iteration %d." % iter
        if fit_network:
            # Fit the network
            x_net_0, shapes = pack(_flatten(x['net']))
            #x_glms = x[1:]
            #nll = lambda x_net: -1.0 * reduce(lambda lp_acc,n: lp_acc +
            #                                                   net_lp(*([n] +
            #                                                            unpack(x_net,shapes) +
            #                                                            x_glms[n])),
            #                                  np.arange(N),
            #                                  net_prior(x_net))
            #grad_nll = lambda x_net: -1.0 * reduce(lambda g_acc,n: g_acc +
            #                                                       net_g(*([n] +
            #                                                               unpack(x_net,shapes) +
            #                                                               x_glms[n])),
            #                                       np.arange(N),
            #                                       net_g_prior(x_net))
            #hess_nll = lambda x_net: -1.0 * reduce(lambda H_acc,n: H_acc +
            #                                                       net_H(*([n] +
            #                                                               unpack(x_net,shapes) +
            #                                                               x_glms[n])),
            #                                       np.arange(N),
            #                                       net_H_prior(x_net))
            nll = lambda x_net_vec: net_helper(x_net_vec, x, net_prior, glm_logp)
            grad_nll = lambda x_net_vec: net_helper(x_net_vec, x, g_net_prior, g_glm_logp_wrt_net)
            hess_nll = lambda x_net_vec: net_helper(x_net_vec, x, H_net_prior, H_glm_logp_wrt_net)

            x_net_opt = opt.fmin_ncg(nll, x_net_0,
                                     fprime=grad_nll,
                                     fhess=hess_nll,
                                     disp=True)
            x['net'] = unpack(x_net_opt, shapes)

        # Fit the GLMs.
        # TODO Parallelize this!
        for n in np.arange(N):
            #x_net = x['net']
            #x_glm_0, shapes = pack(x[n + 1])
            #nll = lambda xn: -1.0 * network_glm.glm.f_lp(*([n] + x_net + unpack(xn, shapes)))
            #grad_nll = lambda xn: -1.0 * network_glm.glm.g_lp(*([n] + x_net + unpack(xn, shapes)))
            #hess_nll = lambda xn: -1.0 * network_glm.glm.H_lp(*([n] + x_net + unpack(xn, shapes)))

            nvars = network_glm.extract_vars(x, n)
            x_glm_0, shapes = pack(_flatten(nvars['glm']))

            #try:
            xn_opt = opt.fmin_ncg(nll, x_glm_0,
                                  fprime=grad_nll,
                                  fhess=hess_nll,
                                  disp=True)
            #except Exception as e:
            #    import pdb
            #    pdb.set_trace()
            #    raise e
            #x[n + 1] = unpack(xn_opt, shapes)
            x['glms'][n] = unpack(xn_opt, shapes)

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
        converged = maxdiff < atol
        x_prev = copy.deepcopy(x)
    return x

