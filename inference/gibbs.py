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

# Define constants for Sampling
DEG_GAUSS_HERMITE = 20
GAUSS_HERMITE_ABSCISSAE, GAUSS_HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(DEG_GAUSS_HERMITE)

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

        # Helper functions to sample A

    def precompute_currents(x, n_post):
        """ Precompute currents for sampling A and W
        """
        nvars = population.extract_vars(x, n_post)

        I_bias = seval(glm.bias_model.I_bias,
                       syms,
                       nvars)

        I_stim = seval(glm.bkgd_model.I_stim,
                       syms,
                       nvars)

        I_imp = seval(glm.imp_model.I_imp,
                      syms,
                      nvars)

        return I_bias, I_stim, I_imp

    def lp_A_new(A, x, n_post, I_bias, I_stim, I_imp):
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
        s = [network.graph.A] + \
             _flatten(syms['net']['weights']) + \
            [glm.n,
             glm.bias_model.I_bias,
             glm.bkgd_model.I_stim,
             glm.imp_model.I_imp] + \
            _flatten(syms['glm']['nlin'])

        xv = [A] + \
             _flatten(x['net']['weights']) + \
             [n_post,
              I_bias,
              I_stim,
              I_imp] + \
            _flatten(x['glms'][n_post]['nlin'])

        lp += glm.ll.eval(dict(zip(s, xv)))

        return lp

    # Helper functions to sample W
    W_gibbs_prms = {}
    if 'W' in syms['net']['weights']:
        def lp_W(W, x, n_post, I_bias, I_stim, I_imp):
            """ Compute the log probability for a given column W[:,n_post] 
            """
            # Set A in state dict x
            set_vars('W', x['net']['weights'], W)

            # Get the prior probability of A
            lp = seval(network.log_p,
                       syms['net'],
                       x['net'])

            # Get the likelihood of the GLM under W
            s = _flatten(syms['net']['graph']) + \
                [network.weights.W_flat,
                 glm.n,
                 glm.bias_model.I_bias,
                 glm.bkgd_model.I_stim,
                 glm.imp_model.I_imp] + \
                _flatten(syms['glm']['nlin'])

            xv = _flatten(x['net']['graph']) + \
                 [W,
                  n_post,
                  I_bias,
                  I_stim,
                  I_imp] + \
                 _flatten(x['glms'][n_post]['nlin'])

            lp += glm.ll.eval(dict(zip(s, xv)))

            return lp

        g_netlp_wrt_W = T.grad(network.log_p, syms['net']['weights']['W'])
        g_glmlp_wrt_W = T.grad(glm.ll, syms['net']['weights']['W'])
        def grad_lp_W(W, x, n_post, I_bias, I_stim, I_imp):
            """ Compute the log probability for a given column W[:,n_post] 
            """
            # Set A in state dict x
            set_vars('W', x['net']['weights'], W)

            # Get the prior probability of A
            g_lp = seval(g_netlp_wrt_W,
                         syms['net'],
                         x['net'])

            # Get the likelihood of the GLM under W
            s = _flatten(syms['net']['graph']) + \
                [network.weights.W_flat,
                 glm.n,
                 glm.bias_model.I_bias,
                 glm.bkgd_model.I_stim,
                 glm.imp_model.I_imp] + \
                _flatten(syms['glm']['nlin'])

            xv = _flatten(x['net']['graph']) + \
                 [W,
                  n_post,
                  I_bias,
                  I_stim,
                  I_imp] + \
                 _flatten(x['glms'][n_post]['nlin'])
            #g_lp += seval(g_glmlp_wrt_W,
            #              syms,
            #              nvars)
            g_lp += g_glmlp_wrt_W.eval(dict(zip(s, xv)))

            # Ignore gradients wrt columns other than n_post
            g_mask = np.zeros((N,N))
            g_mask[:,n_post] = 1
            g_lp *= g_mask.flatten()
            return g_lp

        W_gibbs_prms['lp_W'] = lp_W
        W_gibbs_prms['g_lp_W'] = grad_lp_W
        W_gibbs_prms['avg_accept_rate'] = 0.9
        W_gibbs_prms['step_sz'] = 0.05

    return lp_A_new, W_gibbs_prms, precompute_currents

def prep_collapsed_network_inference(population):
    """ Initialize functions that compute the gradient and Hessian of
        the log probability with respect to the differentiable network
        parameters, e.g. the weight matrix if it exists.
    """
    N = population.model['N']
    network = population.network
    glm = population.glm
    syms = population.get_variables()

    # Helper functions to sample A and W:
    #   precompute_currents
    #   log_prior_A(n_pre, n_post)
    #   log_prior_noA(n_pre, n_post)
    #   glm_ll_A(n_pre, n_post, W, x, Is)
    #   glm_ll_noA(n_pre, n_post, x, Is)

    def precompute_currents(x, n_post):
        """ Precompute currents for sampling A and W
        """
        nvars = population.extract_vars(x, n_post)

        I_bias = seval(glm.bias_model.I_bias,
                       syms,
                       nvars)

        I_stim = seval(glm.bkgd_model.I_stim,
                       syms,
                       nvars)

        I_imp = seval(glm.imp_model.I_imp,
                      syms,
                      nvars)

        return I_bias, I_stim, I_imp

    def lp_A(n_pre, n_post, v, x):
        """ Compute the log probability for a given column A[:,n_post]
        """

        # Update A[n_pre, n_post]
        A = x['net']['graph']['A']
        A[n_pre, n_post] = v

        # Get the prior probability of A
        lp = seval(network.log_p,
                   syms['net'],
                   x['net'])
        return lp

    def glm_ll_A(n_pre, n_post, w, x, I_bias, I_stim, I_imp):
        """ Compute the log likelihood of the GLM with A=True and given W
        """
        # Set A in state dict x
        A = x['net']['graph']['A']
        A[n_pre, n_post] = 1

        # Set W in state dict x
        W = x['net']['weights']['W'].reshape(A.shape)
        W[n_pre, n_post] = w


        # Get the likelihood of the GLM under A and W
        s = [network.graph.A] + \
             _flatten(syms['net']['weights']) + \
            [glm.n,
             glm.bias_model.I_bias,
             glm.bkgd_model.I_stim,
             glm.imp_model.I_imp] + \
            _flatten(syms['glm']['nlin'])

        xv = [A] + \
             [W.ravel()] + \
             [n_post,
              I_bias,
              I_stim,
              I_imp] + \
            _flatten(x['glms'][n_post]['nlin'])

        ll = glm.ll.eval(dict(zip(s, xv)))

        return ll

    def glm_ll_noA(n_pre, n_post, x, I_bias, I_stim, I_imp):
        """ Compute the log likelihood of the GLM with A=True and given W
        """
        # Set A in state dict x
        A = x['net']['graph']['A']
        A[n_pre, n_post] = 0

        # Get the likelihood of the GLM under A and W
        s = [network.graph.A] + \
             _flatten(syms['net']['weights']) + \
            [glm.n,
             glm.bias_model.I_bias,
             glm.bkgd_model.I_stim,
             glm.imp_model.I_imp] + \
            _flatten(syms['glm']['nlin'])

        xv = [A] + \
             _flatten(x['net']['weights']) + \
             [n_post,
              I_bias,
              I_stim,
              I_imp] + \
            _flatten(x['glms'][n_post]['nlin'])

        ll = glm.ll.eval(dict(zip(s, xv)))

        return ll

    return precompute_currents, lp_A, glm_ll_A, glm_ll_noA


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

def sample_column_of_A(n_post, x, lp_A_new, I_bias, I_stim, I_imp):
    # Sample the adjacency matrix if it exists
    if 'A' in x['net']['graph']:
        # print "Sampling A"
        A = x['net']['graph']['A']
        N = A.shape[0]

        # Sample coupling filters from other neurons
        for n_pre in np.arange(N):
            # print "Sampling A[%d,%d]" % (n_pre,n_post)
            # WARNING Setting A is somewhat of a hack. It only works
            # because nvars copies x's pointer to A rather than making
            # a deep copy of the adjacency matrix.
            A[n_pre,n_post] = 0
            log_pr_noA = lp_A_new(A, x, n_post, I_bias, I_stim, I_imp)

            A[n_pre,n_post] = 1
            log_pr_A = lp_A_new(A, x, n_post, I_bias, I_stim, I_imp)

            # Sample A[n_pre,n_post]
            A[n_pre,n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])

def sample_column_of_W(n_post, x, W_gibbs_prms, I_bias, I_stim, I_imp):
    # Sample W if it exists
    if 'W' in x['net']['weights']:
        # print "Sampling W"
        nll = lambda W: -1.0 * W_gibbs_prms['lp_W'](W, x, n_post, I_bias, I_stim, I_imp)
        grad_nll = lambda W: -1.0 * W_gibbs_prms['g_lp_W'](W, x, n_post, I_bias, I_stim, I_imp)

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
        # print "W step sz: %.3f\tW_accept rate: %.3f" % (new_step_sz, new_accept_rate)

        # Update current W
        x['net']['weights']['W'] = W

def collapsed_sample_AW(n_pre, n_post, x,
                        lp_A, glm_ll_A, glm_ll_noA,
                        I_bias, I_stim, I_imp):
    """
    Do collapsed Gibbs sampling for an entry A_{n,n'} and W_{n,n'} where
    n = n_pre and n' = n_post.
    """
    # import pdb; pdb.set_trace()
    # TODO: Set sigma_w and mu_w
    if n_pre == n_post:
        sigma_w = 0.5
        mu_w = -2.0
    else:
        mu_w = 0.0
        sigma_w = 2.0

    A = x['net']['graph']['A']
    W = x['net']['weights']['W'].reshape(A.shape)

    # Approximate G = \int_0^\infty p({s,c} | A, W) p(W_{n,n'}) dW_{n,n'}
    log_L = np.zeros(DEG_GAUSS_HERMITE)
    W_nns = np.sqrt(2) * sigma_w * GAUSS_HERMITE_ABSCISSAE + mu_w
    for i in np.arange(DEG_GAUSS_HERMITE):
        w = GAUSS_HERMITE_WEIGHTS[i]
        W_nn = W_nns[i]
        log_L[i] = np.log(w/np.sqrt(np.pi)) + glm_ll_A(n_pre, n_post, W_nn,
                                                       x, I_bias, I_stim, I_imp)

    # compute log pr(A_nn) and log pr(\neg A_nn) via log G
    from scipy.misc import logsumexp
    log_G = logsumexp(log_L)

    # Compute log Pr(A_nn=1) given prior and estimate of log lkhd after integrating out W
    log_pr_A = lp_A(n_pre, n_post, 1, x) + log_G
    # Compute log Pr(A_nn = 0 | {s,c}) = log Pr({s,c} | A_nn = 0) + log Pr(A_nn = 0)
    log_pr_noA = lp_A(n_pre, n_post, 0, x) + glm_ll_noA(n_pre, n_post, x,
                                                        I_bias, I_stim, I_imp)

    # Sample A
    A[n_pre,n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])
    set_vars('A', x['net']['graph'], A)

    # Sample W from its posterior, i.e. log_L with denominator log_G
    # If A_nn = 0, we don't actually need to resample W since it has no effect
    if A[n_pre,n_post] == 1:
        log_p_W = log_L - log_G
        # Compute the log CDF
        log_F_W = [logsumexp(log_p_W[:i]) for i in range(1,DEG_GAUSS_HERMITE)] + [0]
        # Sample via inverse CDF
        W[n_pre, n_post] = np.interp(np.log(np.random.rand()),
                                     log_F_W,
                                     W_nns)
    else:
        # Sample W from the prior
        W[n_pre, n_post] = mu_w + sigma_w * np.random.randn()

    # Set W in state dict x
    x['net']['weights']['W'] = W.ravel()

def sample_network_column(n_post,
                          x,
                          (lp_A, W_gibbs_prms, precompute_currents)):
    """ Sample a single column of the network (all the incoming
        coupling filters). This is a parallelizable chunk.
    """
    # TODO Check for Gaussian weights with Bernoulli A and do
    #      collapsed Gibbs.

    # Precompute the filtered currents from other GLMs
    I_bias, I_stim, I_imp = precompute_currents(x, n_post)
    sample_column_of_A(n_post, x, lp_A, I_bias, I_stim, I_imp)
    sample_column_of_W(n_post, x, W_gibbs_prms, I_bias, I_stim, I_imp)
    return x['net']

def collapsed_sample_network_column(n_post,
                                    x,
                                    (precompute_currents, lp_A, glm_ll_A, glm_ll_noA)):
    """ Collapsed Gibbs sample a column of A and W
    """
    A = x['net']['graph']['A']
    N = A.shape[0]
    I_bias, I_stim, I_imp = precompute_currents(x, n_post)

    order = np.arange(N)
    np.random.shuffle(order)
    for n_pre in order:
        collapsed_sample_AW(n_pre, n_post, x, lp_A, glm_ll_A, glm_ll_noA,
                            I_bias, I_stim, I_imp)

    return x['net']

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
        # sample_network_column(n_post,
        #                       x,
        #                       net_inf_prms)
        collapsed_sample_network_column(n_post,
                              x,
                              net_inf_prms)
    return x['net']

def single_glm_gibbs_step(xn, n,
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
    
    # HMC with automatic parameter tuning
    n_steps = 2
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
    # print "GLM step sz: %.3f\tGLM_accept rate: %.3f" % (new_step_sz, new_accept_rate)


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
            from models.model_factory import make_model, convert_model
            from population import Population
            mle_model = make_model('standard_glm', N=N)
            mle_popn = Population(mle_model)
            mle_popn.set_data(data)
            mle_x0 = mle_popn.sample()

            # Initialize with MLE under standard GLM
            mle_x0 = coord_descent(mle_popn, data, x0=mle_x0, maxiter=1)

            # Convert between inferred parameters of the standard GLM
            # and the parameters of this model. Eg. Convert unweighted 
            # networks to weighted networks with normalized impulse responses.
            x0 = convert_model(mle_popn, mle_model, mle_x0, population, population.model, x0)

    # Compute log prob, gradient, and hessian wrt network parameters
    # net_inf_prms = prep_network_inference(population)
    net_inf_prms = prep_collapsed_network_inference(population)

    # Compute gradients of the log prob wrt the GLM parameters
    glm_inf_prms = prep_glm_inference(population)

    # DEBUG Profile the Gibbs sampling loop
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    # Alternate fitting the network and fitting the GLMs
    x_smpls = [x0]
    x = x0

    import time
    start_time = time.clock()

    for smpl in np.arange(N_samples):
        # Print the current log likelihood
        lp = population.compute_log_p(x)

        # Compute iters per second
        stop_time = time.clock()
        if stop_time - start_time == 0:
            print "Gibbs iteration %d. Iter/s exceeds time resolution. Log prob: %.3f" % (smpl, lp)
        else:
            print "Gibbs iteration %d. Iter/s = %f. Log prob: %.3f" % (smpl,
                                                                       1.0/(stop_time-start_time),
                                                                       lp)
        start_time = stop_time

        # Go through variables, sampling one at a time, in parallel where possible
        network_gibbs_step(x, net_inf_prms)

        # Sample the GLM parameters
        for n in np.arange(N):
            # print "Gibbs step for GLM %d" % n
            nvars = population.extract_vars(x, n)
            single_glm_gibbs_step(nvars, n, glm_inf_prms)
            x['glms'][n] = nvars['glm']

        x_smpls.append(copy.deepcopy(x))

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open('mcmc.prof.txt', 'w') as f:
        f.write(s.getvalue())
        f.close()

    return x_smpls
