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
from log_sum_exp import log_sum_exp_sample
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

    # Helper functions to sample A
    def lp_A(A, x, n_post):
        """ Compute the log probability for a given column A[:,n_post] 
        """
        # Set A in state dict x
        set_vars('A', x['net']['graph'], A)

        # Get the prior probability of A
        lp = seval(network.log_p,
                   syms['net'],
                   x['net'])

        # Get the likelihood of the GLM under A
        nvars = population.extract_vars(x, n_post)

        # At some point it may become beneficial to cache a function
        # with the GLM parameters as given. This requires us to pay once to
        # compile an optimized function that should precompute the unweighted currents
        # and then benefit from increased performance on subsequent calls.

        #givens = zip(_flatten(syms['glm']),
        #             _flatten(get_vars(syms['glm'],nvars['glm'])))
        #lp += seval(glm.log_p,
        #            syms['net'],
        #            nvars['net'],
        #            givens=givens)
        lp += seval(glm.log_p,
                    syms,
                    nvars)

        return lp

    # Helper functions to sample W
    W_gibbs_prms = {}
    if 'W' in syms['net']['weights']:
        def lp_W(W, x, n_post):
            """ Compute the log probability for a given column W[:,n_post] 
            """
            # Set A in state dict x
            set_vars('W', x['net']['weights'], W)

            # Get the prior probability of A
            lp = seval(network.log_p,
                       syms['net'],
                       x['net'])

            # Get the likelihood of the GLM under W
            nvars = population.extract_vars(x, n_post)
            lp += seval(glm.log_p,
                        syms,
                        nvars)

            return lp

        g_netlp_wrt_W = T.grad(network.log_p, syms['net']['weights']['W'])
        g_glmlp_wrt_W = T.grad(glm.log_p, syms['net']['weights']['W'])
        def grad_lp_W(W, x, n_post):
            """ Compute the log probability for a given column W[:,n_post] 
            """
            # Set A in state dict x
            set_vars('W', x['net']['weights'], W)

            # Get the prior probability of A
            g_lp = seval(g_netlp_wrt_W,
                         syms['net'],
                         x['net'])

            # Get the likelihood of the GLM under W
            nvars = population.extract_vars(x, n_post)
            g_lp += seval(g_glmlp_wrt_W,
                          syms,
                          nvars)

            g_mask = np.zeros((N,N))
            g_mask[:,n_post] = 1
            g_lp *= g_mask.flatten()
            return g_lp

        W_gibbs_prms['lp_W'] = lp_W
        W_gibbs_prms['g_lp_W'] = grad_lp_W
        W_gibbs_prms['avg_accept_rate'] = 0.9
        W_gibbs_prms['step_sz'] = 0.05

    return lp_A, W_gibbs_prms

def prep_glm_inference(population):
    """ Initialize functions that compute the gradient and Hessian of 
        the log probability with respect to the differentiable GLM 
        parameters, e.g. the weight matrix if it exists.
    """
    glm_gibbs_prms = {}
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()
    
    # Compute gradients of the log prob wrt the GLM parameters
    print "Computing log probabilities, gradients, and Hessians for GLM variables"
    glm_syms = differentiable(syms['glm'])
    glm_logp = glm.log_p
    g_glm_logp_wrt_glm, g_list = grad_wrt_list(glm_logp, _flatten(glm_syms))
 
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
        return lp

    lp = lambda x_glm_vec, x: glm_helper(x_glm_vec, 
                                         x, 
                                         glm_logp)
    g_lp = lambda x_glm_vec, x: glm_helper(x_glm_vec, 
                                           x, 
                                           g_glm_logp_wrt_glm)
    
    glm_gibbs_prms['lp'] = lp
    glm_gibbs_prms['g_lp'] = g_lp
    glm_gibbs_prms['avg_accept_rate'] = 0.9
    glm_gibbs_prms['step_sz'] = 0.05
    return glm_syms, glm_gibbs_prms

def sample_network_column(n_post,
                          x,
                          (lp_A, W_gibbs_prms)):
    """ Sample a single column of the network (all the incoming
        coupling filters). This is a parallelizable chunk.
    """
    # TODO Check for Gaussian weights with Bernoulli A and do
    #      collapsed Gibbs.
    # import pdb; pdb.set_trace()
    # Sample the adjacency matrix if it exists
    if 'A' in x['net']['graph']:
        print "Sampling A"
        A = x['net']['graph']['A']
        N = A.shape[0]

        # Sample coupling filters from other neurons
        for n_pre in np.arange(N):
            # print "Sampling A[%d,%d]" % (n_pre,n_post)
            # WARNING Setting A is somewhat of a hack. It only works
            # because nvars copies x's pointer to A rather than making
            # a deep copy of the adjacency matrix.
            A[n_pre,n_post] = 0
            log_pr_noA = lp_A(A, x, n_post)

            A[n_pre,n_post] = 1
            log_pr_A = lp_A(A, x, n_post)

            # Sample A[n_pre,n_post]
            A[n_pre,n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])

    # Sample W if it exists
    if 'W' in x['net']['weights']:
        print "Sampling W"
        nll = lambda W: -1.0 * W_gibbs_prms['lp_W'](W, x, n_post)
        grad_nll = lambda W: -1.0 * W_gibbs_prms['g_lp_W'](W, x, n_post)

        # Automatically tune these parameters
        n_steps = 10
        (W, new_step_sz, new_accept_rate) = hmc(nll,
                                                grad_nll, 
                                                W_gibbs_prms['step_sz'], 
                                                n_steps,
                                                x['net']['weights']['W'],
                                                adaptive_step_sz=True,
                                                avg_accept_rate=W_gibbs_prms['avg_accept_rate'])

        # Update step size and accept rate
        W_gibbs_prms['step_sz'] = new_step_sz
        W_gibbs_prms['avg_accept_rate'] = new_accept_rate
        print "W step sz: %.3f\tW_accept rate: %.3f" % (new_step_sz, new_accept_rate)
        
        # Update current W
        x['net']['weights']['W'] = W

def network_gibbs_step(x, 
                       net_inf_prms):
    """ Gibbs sample the network by collapsing out the weights to 
        sample the binary adjacency matrix and then sampling the 
        weights using HMC or slice sampling. 

        We must also sample the parameters of the network prior.
    """
    # TODO Check for Gaussian weights with Bernoulli A and do 
    #      collapsed Gibbs.
    for n_post in np.arange(len(x['glms'])):
        # Sample coupling filters from other neurons
        sample_network_column(n_post,
                              x,
                              net_inf_prms)
                
def glm_gibbs_step(xn, n,
                   (glm_syms, glm_gibbs_prms)):
    """ Gibbs sample the GLM parameters. These are mostly differentiable
        so we use HMC wherever possible.
    """
    # Get the differentiable variables suitable for HMC
    dxn = get_vars(glm_syms, xn['glm'])
    x_glm_0, shapes = packdict(dxn)

    # Create lambda functions to compute the nll and its gradient
    nll = lambda x_glm_vec: -1.0*glm_gibbs_prms['lp'](x_glm_vec, xn)
    grad_nll = lambda x_glm_vec: -1.0*glm_gibbs_prms['g_lp'](x_glm_vec, xn)
    
    # Call HMC
    # TODO Automatically tune these parameters
    n_steps = 10
    x_glm, new_step_sz, new_accept_rate = hmc(nll, 
                                              grad_nll, 
                                              glm_gibbs_prms['step_sz'],
                                              n_steps, 
                                              x_glm_0,
                                              adaptive_step_sz=True,
                                              avg_accept_rate=glm_gibbs_prms['avg_accept_rate'])

    # Update step size and accept rate
    glm_gibbs_prms['step_sz'] = new_step_sz
    glm_gibbs_prms['avg_accept_rate'] = new_accept_rate
    print "GLM step sz: %.3f\tGLM_accept rate: %.3f" % (new_step_sz, new_accept_rate)


    # Unpack the optimized parameters back into the state dict
    x_glm_n = unpackdict(x_glm, shapes)
    set_vars(glm_syms, xn['glm'], x_glm_n)

    return xn['glm']

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
        x0 = population.sample()
        
        if init_from_mle:
            print "Initializing with coordinate descent"
            from models.model_factory import make_model
            from population import Population
            mle_model = make_model('standard_glm', N=N)
            mle_popn = Population(mle_model)
            mle_popn.set_data(data)
            mle_x0 = mle_popn.sample()

            x0 = coord_descent(mle_popn, data, x0=mle_x0, maxiter=1)

            import pdb; pdb.set_trace()
            # TODO Create a population with standard GLM models 
            # x0 = coord_descent(population, data, x0=x0, maxiter=1)

            # TODO Convert between inferred parameters of the standard GLM
            # and the parameters of this model. Eg. Convert unweighted 
            # networks to weighted networks with normalized impulse responses.

    # Compute log prob, gradient, and hessian wrt network parameters
    net_inf_prms = prep_network_inference(population)
    
    # Compute gradients of the log prob wrt the GLM parameters
    glm_inf_prms = prep_glm_inference(population)

    # DEBUG Profile the Gibbs sampling loop
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    # Alternate fitting the network and fitting the GLMs
    x_smpls = [x0]
    x = x0
    
    for smpl in np.arange(N_samples):
        # Print the current log likelihood
        lp = population.compute_log_p(x)
        print "Iter %d: Log prob: %.3f" % (smpl,lp)

        # Go through variables, sampling one at a time, in parallel where possible
        network_gibbs_step(x, net_inf_prms)

        # Sample the GLM parameters
        print "Sampling GLMs"
        for n in np.arange(N):
            # print "Gibbs step for GLM %d" % n
            nvars = population.extract_vars(x, n)
            glm_gibbs_step(nvars, n, glm_inf_prms)
            x['glms'][n] = nvars['glm']

        x_smpls.append(copy.deepcopy(x))

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    #with open('mcmc.prof.txt', 'w') as f:
    #    f.write(s.getvalue())
    #    f.close()

    return x_smpls
