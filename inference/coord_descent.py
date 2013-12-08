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

def coord_descent(network_glm, 
                  x0=None, 
                  maxiter=50, 
                  atol=1e-5,
                  use_hessian=False,
                  use_rop=False):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    import pdb
    pdb.set_trace()
    
    N = network_glm.model['N']
    network = network_glm.network
    glm = network_glm.glm
    syms = network_glm.get_variables()

    # Parameter checking
    # We only use Rops if use_hessian is False
    use_rop = use_rop and not use_hessian
    
    # Make sure the network is a complete adjacency matrix because we
    # do not do integer programming
    if not isinstance(network.graph, CompleteGraphModel):
        raise Exception("MAP inference via coordinate descent can only be performed \
                        with the complete graph model.")
    
    # Draw initial state from prior if not given
    if x0 is None:
        x0 = network_glm.sample()

    # TODO Remove this temporary hack 
    fit_network = False

    # Compute the log prob, gradient, and Hessian wrt to the network
    if fit_network:
        # TODO Determine the differentiable network parameters in the same way
        # we do for the GLM parameters
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
            x_net = unpack(x_net_vec, net_shapes)
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
    glm_syms = differentiable(syms['glm'])
    glm_logp = glm.log_p
    g_glm_logp_wrt_glm, g_list = grad_wrt_list(glm_logp, _flatten(glm_syms))
    if not use_rop:
        H_glm_logp_wrt_glm = hessian_wrt_list(glm_logp, _flatten(glm_syms), g_list)

    else:
        # Alternatively, we could just use an Rop to compute Hessian-vector prod       
        v = T.dvector()
        H_glm_logp_wrt_glm = hessian_rop_wrt_list(glm_logp,
                                                  _flatten(glm_syms),
                                                  v,
                                                  g_vec=g_glm_logp_wrt_glm)

    # TODO: Replace this with a function that just gets the shapes?
    nvars = network_glm.extract_vars(x0, 0)
    dnvars = get_vars(glm_syms, nvars['glm'])
    _,glm_shapes = packdict(dnvars)

    # Private function to compute the log probability (or grads and Hessians thereof)
    # of the log probability given new network variables
    def glm_helper(x_glm_vec, x, glm_expr):
        """ Compute the negative log probability (or gradients and Hessians thereof)
        of the given glm variables
        """
        x_glm = unpackdict(x_glm_vec, glm_shapes)
        #x['glm'] = x_glm
        set_vars(glm_syms, x['glm'], x_glm)
        lp = seval(glm_expr,
                    syms,
                    x)
        return -1.0*lp

    if use_rop:
        rop_syms = copy.copy(syms)
        rop_syms['v'] = v
        def glm_rop_helper(x_glm_vec, v_vec, x, glm_expr):
            """ Compute the Hessian vector product for the GLM
            """
            import pdb
            pdb.set_trace()
            x_glm = unpackdict(x_glm_vec, glm_shapes)
            #x['glm'] = x_glm
            set_vars(glm_syms, x['glm'], x_glm)
            defaults = {'v' : v_vec}
            Hv = seval(glm_expr,
                       rop_syms,
                       x,
                       defaults)
            return -1.0*Hv
    
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
            x_net_0, shapes = packdict(x['net'])

            nll = lambda x_net_vec: net_helper(x_net_vec, x, net_prior, glm_logp)
            grad_nll = lambda x_net_vec: net_helper(x_net_vec, x, g_net_prior, g_glm_logp_wrt_net)
            hess_nll = lambda x_net_vec: net_helper(x_net_vec, x, H_net_prior, H_glm_logp_wrt_net)

            x_net_opt = opt.fmin_ncg(nll, x_net_0,
                                     fprime=grad_nll,
                                     fhess=hess_nll,
                                     disp=True)
            x['net'] = unpackdict(x_net_opt, shapes)

        # Fit the GLMs.
        # TODO Parallelize this!
        for n in np.arange(N):
            # Get the differentiable variables for the n-th GLM
            nvars = network_glm.extract_vars(x, n)
            dnvars = get_vars(glm_syms, nvars['glm'])
            x_glm_0, shapes = packdict(dnvars)

            # Create lambda functions to compute the nll and its gradient and Hessian
            nll = lambda x_glm_vec: glm_helper(x_glm_vec, nvars, glm_logp)
            grad_nll = lambda x_glm_vec: glm_helper(x_glm_vec, nvars, g_glm_logp_wrt_glm)
            if use_hessian:
                hess_nll = lambda x_glm_vec: glm_helper(x_glm_vec, nvars, H_glm_logp_wrt_glm)
            elif use_rop:
                hess_nll = lambda x_glm_vec, v_vec: glm_rop_helper(x_glm_vec, v_vec, nvars, H_glm_logp_wrt_glm)

            # Callback to print progress
            def progress_report(x_curr):
                ll = -1.0*nll(x_curr)
                print "Iter %d.\tNeuron %d. LL: %.1f" % (iter,n,ll) 
                
            if use_hessian:
                xn_opt = opt.fmin_ncg(nll, x_glm_0,
                                      fprime=grad_nll,
                                      fhess=hess_nll,
                                      disp=True,
                                      callback=progress_report)
            elif use_rop:
                xn_opt = opt.fmin_ncg(nll, x_glm_0,
                                  fprime=grad_nll,
                                  fhess_p=hess_nll,
                                  disp=True,
                                  callback=progress_report)
            else:
                xn_opt = opt.fmin_ncg(nll, x_glm_0,
                                  fprime=grad_nll,
                                  disp=True,
                                  callback=progress_report)
            x_glm_n = unpackdict(xn_opt, shapes)
            set_vars(glm_syms, x['glms'][n], x_glm_n)
            
        diffs = np.zeros(N)
        for n in np.arange(N):
            nvars = network_glm.extract_vars(x, n)
            dnvars = get_vars(glm_syms, nvars['glm'])
            xn_curr, shapes = packdict(dnvars)

            nvars = network_glm.extract_vars(x_prev, n)
            dnvars = get_vars(glm_syms, nvars['glm'])
            xn_prev, shapes = packdict(dnvars)
            
            
            diffs[n] = np.mean((xn_curr - xn_prev) ** 2)
        maxdiff = np.max(diffs)

        print diffs
        converged = maxdiff < atol
        x_prev = copy.deepcopy(x)
    return x

