""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""
import copy
import numpy as np

from IPython.parallel.util import interactive

def initialize_imports(dview):
    """ Import functions req'd for coordinate descent
    """
    dview.execute('from utils.theano_func_wrapper import seval')
    dview.execute('from utils.packvec import *')
    dview.execute('from inference.gibbs import *')
    dview.execute('from log_sum_exp import log_sum_exp_sample')
    dview.execute('from hmc import hmc')


def parallel_compute_log_p(dview,
                           v,
                           N):
    """ Compute the log prob in parallel
    """

    # TODO Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp_tot = 0

    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_lp(n, vs):
        print "Computing lp for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.log_p,
                   syms,
                   nvars)
        return lp

    lp_glms = dview.map_async(_compute_glm_lp,
                              range(N),
                              [v]*N)
    # print lp_glms.get()
    # lp_glms.display_outputs()

    lp_tot += sum(lp_glms.get())
    return lp_tot


def parallel_gibbs_sample(client,
                          data,
                          N_samples=1000,
                          x0=None,
                          init_from_mle=True):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    import pdb
    pdb.set_trace()
    dview = client[:]
    N = data['N']

    # Import req'd functions on engines
    initialize_imports(dview)

    # Draw initial state from prior if not given
    if x0 is None:
        # x0 = population.sample()
        #
        # if init_from_mle:
        #     print "Initializing with coordinate descent"
        #
        #     # Also initialize with intelligent parameters from the data
        #     initialize_with_data(population, data, x0)
        #
        #     # If we are using a sparse network, set it to complete
        #     # before computing the initial state
        #     if 'A' in x0['net']['graph']:
        #         x0['net']['graph']['A'] = np.ones((N,N), dtype=np.bool)
        #
        #     x0 = coord_descent(population, data, x0=x0, maxiter=1)
        client[0].execute('x0 = popn.sample()', block=True)
        x0 = client[0]['x0']

    # Compute log prob, gradient, and hessian wrt network parameters
    dview.execute('net_inf_prms = prep_network_inference(popn)',
                  block=True)
    
    # Compute gradients of the log prob wrt the GLM parameters
    dview.execute('glm_inf_prms = prep_glm_inference(popn)',
                  block=True)


    # Create map-able functions to sample in parallel
    # Parallel function to sample network
    @interactive
    def _parallel_sample_network_col(n_post, x):
        return sample_network_column(n_post,
                                     x,
                                     net_inf_prms)

    # Parallel function to sample GLMs
    @interactive
    def _parallel_sample_glm(n, x, use_hessian=False, use_rop=False):
        nvars = popn.extract_vars(x, n)
        glm_gibbs_step(nvars, n, glm_inf_prms)
        return nvars['glm']

    ## DEBUG Profile the Gibbs sampling loop
    # import cProfile, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    ## END DEBUG

    # Alternate fitting the network and fitting the GLMs
    x_smpls = []
    x = x0

    for smpl in np.arange(N_samples):
        # Print the current log likelihood
        lp = parallel_compute_log_p(dview,
                                    x,
                                    N)
        print "Iter %d: Log prob: %.3f" % (smpl,lp)

        # Go through variables, sampling one at a time, in parallel where possible
        x_net = dview.map_async(_parallel_sample_network_col,
                                range(N),
                                [x]*N)

        # TODO Incorporate the results back into a single network

        # Sample the GLM parameters
        x_glms = dview.map_async(_parallel_sample_glm,
                                 range(N),
                                 [x]*N)
        x['glms'] = x_glms.get()
        x_glms.display_outputs()

        x_smpls.append(copy.deepcopy(x))

    ## DEBUG Profile the Gibbs sampling loop
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    #
    # with open('mcmc.prof.txt', 'w') as f:
    #     f.write(s.getvalue())
    #     f.close()
    ## END DEBUG


    return x_smpls
