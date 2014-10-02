""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy

import scipy.optimize as opt

from pyglm.utils.theano_func_wrapper import seval, _flatten
from pyglm.utils.packvec import *
from pyglm.utils.grads import *
from pyglm.components.graph import TheanoCompleteGraphModel
from pyglm.inference.smart_init import initialize_with_data

def prep_first_order_glm_inference(population):
    """ Initialize functions that compute the gradient and Hessian of
        the log probability with respect to the differentiable GLM
        parameters, e.g. the weight matrix if it exists.
    """
    glm = population.glm
    syms = population.get_variables()

    # Compute gradients of the log prob wrt the GLM parameters
    glm_syms = differentiable(syms['glm'])

    print "Computing gradient of the prior w.r.t. the differentiable GLM parameters"
    g_glm_logprior, _ = grad_wrt_list(glm.log_prior, _flatten(glm_syms))

    print "Computing gradient of the GLM likelihood w.r.t. the differentiable GLM parameters"
    g_glm_ll, _ = grad_wrt_list(glm.ll, _flatten(glm_syms))

    # TODO: Replace this with a function that just gets the shapes?
    x0 = population.sample()
    nvars = population.extract_vars(x0, 0)
    dnvars = get_vars(glm_syms, nvars['glm'])
    _,glm_shapes = packdict(dnvars)

    # Private function to compute the log probability and its gradient
    # with respect to a set of parameters
    def nlp(x_glm_vec, x):
        """
        Helper function to compute the negative log posterior for a given set
        of GLM parameters. The parameters are passed in as a vector.
        """
        x_glm = unpackdict(x_glm_vec, glm_shapes)
        set_vars(glm_syms, x['glm'], x_glm)
        lp = seval(glm.log_prior,
                   syms,
                   x)

        # Add the likelihood of each data sequence
        for data in population.data_sequences:
            # Set the data
            population.set_data(data)
            lp += seval(glm.ll,
                        syms,
                        x)

        return -1.0 * lp

    def grad_nlp(x_glm_vec, x):
        """
        Helper function to compute the gradient of negative log posterior for
        a given set of GLM parameters. The parameters are passed in as a vector.
        """
        x_glm = unpackdict(x_glm_vec, glm_shapes)
        set_vars(glm_syms, x['glm'], x_glm)
        glp = seval(g_glm_logprior,
                   syms,
                   x)

        # Add the likelihood of each data sequence
        for data in population.data_sequences:
            # Set the data
            population.set_data(data)
            glp += seval(g_glm_ll,
                        syms,
                        x)

        return -1.0 * glp

    return glm_syms, nlp, grad_nlp

def prep_first_order_network_inference(population):
    """ Initialize functions that compute the gradient and Hessian of
        the log probability with respect to the differentiable GLM
        parameters, e.g. the weight matrix if it exists.
    """
    network = population.network
    syms = population.get_variables()

    # Compute gradients of the log prob wrt the GLM parameters
    network_syms = differentiable(syms['net'])

    print "Computing gradient of the network prior w.r.t. the differentiable GLM parameters"
    g_network_logprior, _ = grad_wrt_list(network.log_p, _flatten(network_syms))

    # TODO: Replace this with a function that just gets the shapes?
    x0 = population.sample()
    nvars = population.extract_vars(x0, 0)
    dnvars = get_vars(network_syms, nvars['net'])
    _,network_shapes = packdict(dnvars)

    # Private function to compute the log probability and its gradient
    # with respect to a set of parameters
    def nlp(x_network_vec, x):
        """
        Helper function to compute the negative log posterior for a given set
        of GLM parameters. The parameters are passed in as a vector.
        """
        x_network = unpackdict(x_network_vec, network_shapes)
        set_vars(network_syms, x['net'], x_network)
        lp = seval(network.log_prior,
                   syms,
                   x)

        return -1.0 * lp

    def grad_nlp(x_glm_vec, x):
        """
        Helper function to compute the gradient of negative log posterior for
        a given set of GLM parameters. The parameters are passed in as a vector.
        """
        x_network = unpackdict(x_glm_vec, network_shapes)
        set_vars(network_syms, x['net'], x_network)
        glp = seval(g_network_logprior,
                   syms,
                   x)

        return -1.0 * glp

    return network_syms, nlp, grad_nlp

def fit_network(x, 
                (net_syms, net_nll, g_net_nll)):
    """ Fit the GLM parameters in state dict x
    """
    dx_net = get_vars(net_syms, x['net'])
    x_net_0, shapes = packdict(dx_net)
    
    if x_net_0.size > 0:
        nll = lambda x_net_vec: net_nll(x_net_vec, x)
        grad_nll = lambda x_net_vec: g_net_nll(x_net_vec, x)
        
        # Callback to print progress. In order to count iters, we need to
        # pass the current iteration via a list
        ncg_iter_ls = [0]
        def progress_report(x_curr, ncg_iter_ls):
            ll = -1.0*nll(x_curr)
            print "Newton iter %d.\tNetwork. LL: %.1f" % (ncg_iter_ls[0],ll)
            ncg_iter_ls[0] += 1
        cbk = lambda x_curr: progress_report(x_curr, ncg_iter_ls)
    
        x_net_opt = opt.fmin_ncg(nll, x_net_0,
                                 fprime=grad_nll,
                                 disp=True,
                                 callback=cbk)
        x_net = unpackdict(x_net_opt, shapes)
        set_vars(net_syms, x['net'], x_net)

def fit_glm(xn, n, 
            (glm_syms, glm_nll, g_glm_nll)):
    """ Fit the GLM parameters in state dict x
    """
    # Get the differentiable variables for the n-th GLM
    dnvars = get_vars(glm_syms, xn['glm'])
    x_glm_0, shapes = packdict(dnvars)
    
    # Create lambda functions to compute the nll and its gradient and Hessian
    def nll(x_glm_vec):
        y = glm_nll(x_glm_vec, xn)
        if np.isnan(y):
            y = 1e16

        return y

    def grad_nll(x_glm_vec):
        g = g_glm_nll(x_glm_vec, xn)
        if np.any(np.isnan(g)):
            g = np.zeros_like(g)

        return g
    
    # Callback to print progress. In order to count iters, we need to
    # pass the current iteration via a list
    ncg_iter_ls = [0]
    def progress_report(x_curr, ncg_iter_ls):
        ll = -1.0*nll(x_curr)
        print "Newton iter %d.\tNeuron %d. LL: %.1f" % (ncg_iter_ls[0],n,ll)
        ncg_iter_ls[0] += 1
    cbk = lambda x_curr: progress_report(x_curr, ncg_iter_ls)

    # Call the appropriate scipy optimization function
    res = opt.minimize(nll, x_glm_0,
                       method="bfgs",
                       jac=grad_nll,
                       options={'disp': True,
                                'maxiter' : 225},
                       callback=cbk)
    xn_opt = res.x

    # Unpack the optimized parameters back into the state dict
    x_glm_n = unpackdict(xn_opt, shapes)
    set_vars(glm_syms, xn['glm'], x_glm_n)

def coord_descent(population,
                  x0=None, 
                  maxiter=50, 
                  atol=1e-5):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()

    # Make sure the network is a complete adjacency matrix because we
    # do not do integer programming
    if not isinstance(network.graph, TheanoCompleteGraphModel):
        print " WARNING: MAP inference via coordinate descent can only be performed "\
              "with the complete graph model."
              
    # Draw initial state from prior if not given
    if x0 is None:
        x0 = population.sample()

    # Also initialize with intelligent parameters from the data
    initialize_with_data(population, population.data_sequences[-1], x0)

    lp = population.compute_log_p(x0)
    print "Initial LP=%.2f." % (lp)

    # Compute log prob, gradient, and hessian wrt network parameters
    net_inf_prms = prep_first_order_network_inference(population)
    
    # Compute gradients of the log prob wrt the GLM parameters
    glm_inf_prms = prep_first_order_glm_inference(population)
    
    # Alternate fitting the network and fitting the GLMs
    x = x0
    x_prev = copy.deepcopy(x0)
    lp_prev = population.compute_log_p(x)
    converged = False
    iter = 0
    while not converged and iter < maxiter:
        iter += 1
        print "Coordinate descent iteration %d." % iter
                
        # Fit the GLMs.
        for n in np.arange(N):
            nvars = population.extract_vars(x, n)
            fit_glm(nvars, n, glm_inf_prms)
            x['glms'][n] = nvars['glm']
        
        # Fit the network
        fit_network(x, net_inf_prms)
    
        # Check for convergence 
        lp = population.compute_log_p(x)
        print "Iteration %d: LP=%.2f. Change in LP: %.2f" % (iter, lp, lp-lp_prev)
        
        converged = np.abs(lp-lp_prev) < atol
        lp_prev = lp
    return x

