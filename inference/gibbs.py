""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy
import theano
import theano.tensor as T
import numpy as np

from utils.theano_func_wrapper import seval, _flatten
from utils.packvec import *
from utils.grads import *

from components.graph import CompleteGraphModel

from hmc import hmc
from coord_descent import coord_descent
from smart_init import initialize_with_data

def prep_network_inference(population,
                           use_hessian=False,
                           use_rop=False):
    """ Initialize functions that compute the gradient and Hessian of 
        the log probability with respect to the differentiable network 
        parameters, e.g. the weight matrix if it exists.
    """
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()
    
    # Determine the differentiable network parameters
    print "Computing log probabilities, gradients, and Hessians for network variables"
    net_syms = differentiable(syms['net'])
    net_prior = network.log_p
    g_net_prior = grad_wrt_list(net_prior, _flatten(net_syms))
    H_net_prior = hessian_wrt_list(net_prior, _flatten(net_syms))
    
    # TODO: Replace this with a function that just gets the shapes?
    x0 = population.sample()
    net_vars = get_vars(net_syms, x0['net'])
    _,net_shapes = packdict(net_vars)
    
    # Get the likelihood of the GLM wrt the net variables
    glm_logp = glm.log_p
    g_glm_logp = grad_wrt_list(glm_logp, _flatten(net_syms))
    
    if use_hessian:
        H_glm_logp = hessian_wrt_list(glm_logp, _flatten(net_syms))
        
    # Private function to compute the log probability (or grads and Hessians thereof)
    # of the log probability given new network variables
    def net_helper(x_net_vec, x, net_expr, glm_expr):
        """ Compute the negative log probability (or gradients and Hessians thereof)
        of the given network variables
        """
        x_net = unpackdict(x_net_vec, net_shapes)
        set_vars(net_syms, x['net'], x_net)
        lp = seval(net_expr,
                   syms['net'],
                   x['net'])
        
        # Reduce the log prob, gradient, and Hessian across all GLM nodes.
        # We can do this because the log prob is a sum of log probs from each GLM,
        # plus the log prior from the network model.
        # TODO Parallelize this loop!
        for n in np.arange(N):
            # Get the variables associated with the n-th GLM
            nvars = population.extract_vars(x, n)
            # Override the network vars
            set_vars(net_syms, nvars['net'], x_net)
            lp += seval(glm_expr,
                        syms,
                        nvars)
            return -1.0*lp

    # Create simple functions that take in a vector representing only the 
    # flattened network parameters, and a dictionary x representing the 
    # state of the population.
    nll = lambda x_net_vec, x: net_helper(x_net_vec, 
                                          x, 
                                          net_prior, 
                                          glm_logp)
    grad_nll = lambda x_net_vec, x: net_helper(x_net_vec, 
                                               x, 
                                               g_net_prior, 
                                               g_glm_logp)
    hess_nll = lambda x_net_vec, x: net_helper(x_net_vec, 
                                               x, 
                                               H_net_prior, 
                                               H_glm_logp)
        
    # Return the symbolic expressions and a function that evaluates them
    # for given vector.
    return net_syms, nll, grad_nll, hess_nll

def prep_glm_inference(population,
                       use_hessian=False,
                       use_rop=False):
    """ Initialize functions that compute the gradient and Hessian of 
        the log probability with respect to the differentiable GLM 
        parameters, e.g. the weight matrix if it exists.
    """
    
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()
    
    # Compute gradients of the log prob wrt the GLM parameters
    print "Computing log probabilities, gradients, and Hessians for GLM variables"
    glm_syms = differentiable(syms['glm'])
    glm_logp = glm.log_p
    g_glm_logp_wrt_glm, g_list = grad_wrt_list(glm_logp, _flatten(glm_syms))
    if use_hessian:
        H_glm_logp_wrt_glm = hessian_wrt_list(glm_logp, _flatten(glm_syms), g_list)

    elif use_rop:
        # Alternatively, we could just use an Rop to compute Hessian-vector prod       
        v = T.dvector()
        H_glm_logp_wrt_glm = hessian_rop_wrt_list(glm_logp,
                                                  _flatten(glm_syms),
                                                  v,
                                                  g_vec=g_glm_logp_wrt_glm)

    # TODO: Replace this with a function that just gets the shapes?
    x0 = population.sample()
    nvars = population.extract_vars(x0, 0)
    dnvars = get_vars(glm_syms, nvars['glm'])
    _,glm_shapes = packdict(dnvars)

    # Private function to compute the log probability (or grads and Hessians thereof)
    # of the log probability given new network variables
    def glm_helper(x_glm_vec, x, glm_expr):
        """ Compute the negative log probability (or gradients and Hessians thereof)
        of the given glm variables
        """
        x_glm = unpackdict(x_glm_vec, glm_shapes)
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
            set_vars(glm_syms, x['glm'], x_glm)
            defaults = {'v' : v_vec}
            Hv = seval(glm_expr,
                       rop_syms,
                       x,
                       defaults)
            return -1.0*Hv
    
    nll = lambda x_glm_vec, x: glm_helper(x_glm_vec, 
                                          x, 
                                          glm_logp)
    grad_nll = lambda x_glm_vec, x: glm_helper(x_glm_vec, 
                                               x, 
                                               g_glm_logp_wrt_glm)
    if use_rop:
        hess_nll = lambda x_glm_vec, v_vec, x: glm_rop_helper(x_glm_vec, 
                                                              v_vec, 
                                                              x, 
                                                              H_glm_logp_wrt_glm)
    else:
        hess_nll = lambda x_glm_vec, x: glm_helper(x_glm_vec, 
                                                   x, 
                                                   H_glm_logp_wrt_glm)        
    return glm_syms, nll, grad_nll, hess_nll

def sample_A(self, state, network_glm, n_pre, n_post):
        """
        Sample a specific entry in A
        """
        # Compute the log prob with and without this connection
        x_net = state[0]
        x_glm = state[n_post+1]

        x_net['A'][n_pre,n_post] = 0
        log_pr_noA = network_glm.glm.f_lp(*([n_post]+x_net+x_glm)) + np.log(1.0-self.rho)
        
        x_net['A'][n_pre,n_post] = 1
        log_pr_A = network_glm.glm.f_lp(*([n_post]+x_net+x_glm)) + np.log(self.rho)

        # Sample A[n_pre,n_post]
        x_net['A'][n_pre,n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])
        
        return x_net
    

def network_gibbs_step(x, 
                       (net_syms, net_nll, g_net_nll, H_net_nll)):
    """ Gibbs sample the network by collapsing out the weights to 
        sample the binary adjacency matrix and then sampling the 
        weights using HMC or slice sampling. 

        We must also sample the parameters of the network prior.
    """
    pass

def glm_gibbs_step(xn, n,
                   (glm_syms, glm_nll, g_glm_nll, H_glm_nll)):
    """ Gibbs sample the GLM parameters. These are mostly differentiable
        so we use HMC wherever possible.
    """
    # Get the differentiable variables suitable for HMC
    dxn = get_vars(glm_syms, xn['glm'])
    x_glm_0, shapes = packdict(dxn)

    # Create lambda functions to compute the nll and its gradient
    nll = lambda x_glm_vec: glm_nll(x_glm_vec, xn)
    grad_nll = lambda x_glm_vec: g_glm_nll(x_glm_vec, xn)
    hess_nll = lambda x_glm_vec: H_glm_nll(x_glm_vec, xn)
    
    # Call HMC
    # TODO Automatically tune these parameters
    epsilon = 0.01
    L = 10
    x_glm = hmc(nll, grad_nll, epsilon, L, x_glm_0)

    # Unpack the optimized parameters back into the state dict
    x_glm_n = unpackdict(x_glm, shapes)
    set_vars(glm_syms, xn['glm'], x_glm_n)

def gibbs_sample(population, 
                 data, 
                 N_samples=1000,
                 x0=None, 
                 init_from_mle=True):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()

    # Draw initial state from prior if not given
    if x0 is None:
        if init_from_mle:
            print "Initializing with coordinate descent"
            x0 = coord_descent(population, data, maxiter=1)
        else:
            x0 = network_glm.sample()
    
            # Also initialize with intelligent parameters from the data
            initialize_with_data(population, data, x0)

    # Compute log prob, gradient, and hessian wrt network parameters
    net_inf_prms = prep_network_inference(population)
    
    # Compute gradients of the log prob wrt the GLM parameters
    glm_inf_prms = prep_glm_inference(population)
    
    # Alternate fitting the network and fitting the GLMs
    x_smpls = []
    x = x0
    for smpl in np.arange(N_samples):
        # Go through variables, sampling one at a time, in parallel where possible
        network_gibbs_step(x, net_inf_prms)
        
        # Sample the GLM parameters
        # TODO Parallelize this!
        for n in np.arange(N):
            nvars = population.extract_vars(x, n)
            glm_gibbs_step(nvars, n, glm_inf_prms)
            x['glms'][n] = nvars['glm']
            
        x_smpls.append(copy.deepcopy(x))

        # Print the current log likelihood
        lp = population.compute_log_p(x)
        print "Iter %d: Log prob: %.3f" % (smpl,lp)

    return x_smpls
