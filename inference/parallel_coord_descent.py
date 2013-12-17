import numpy as np

from IPython.parallel.util import interactive

from coord_descent import prep_glm_inference, \
                          prep_network_inference, \
                          fit_glm, \
                          fit_network

def initialize_imports(dview):
    """ Import functions req'd for coordinate descent
    """
    dview.execute('from utils.theano_func_wrapper import seval')
    dview.execute('from inference.coord_descent import *')

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

def parallel_coord_descent(client,
                           N,
                           x0=None,
                           maxiter=50,
                           atol=1e-5,
                           use_hessian=True,
                           use_rop=False):
    """
    Compute the maximum a posterior parameter estimate using Theano to compute
    gradients of the log probability.
    """
    dview = client[:]

    # Import req'd functions on engines
    initialize_imports(dview)

    # Parameter checking
    # We only use Rops if use_hessian is False
    use_rop = use_rop and not use_hessian
    
    # Make sure the network is a complete adjacency matrix because we
    # do not do integer programming
    # if not isinstance(network.graph, CompleteGraphModel):
    #     print "WARNING: MAP inference via coordinate descent can only be performed "\
    #           "with the complete graph model."
              
    # Draw initial state from prior if not given
    if x0 is None:
        # TODO Initialize with STA

        client[0].execute('x0 = popn.sample()', block=True)
        x0 = client[0]['x0']

    # Also initialize with intelligent parameters from the data
    # initialize_with_data(population, data, x0)

    # Compute log prob, gradient, and hessian wrt network parameters
    dview.execute('net_inf_prms = prep_network_inference(popn,'
                  'use_hessian=%s,'
                  'use_rop=%s)' % (str(use_hessian), str(use_rop)),
                  block=True)

    # Compute gradients of the log prob wrt the GLM parameters
    dview.execute('glm_inf_prms = prep_glm_inference(popn,'
                  'use_hessian=%s,'
                  'use_rop=%s)' % (str(use_hessian), str(use_rop)),
                  block=True)

    # Parallel function to fit GLMs
    @interactive
    def _parallel_fit_glm(n, x, use_hessian=False, use_rop=False):
        nvars = popn.extract_vars(x, n)
        fit_glm(nvars, n, glm_inf_prms, use_hessian, use_rop)
        return nvars['glm']

    # Alternate fitting the network and fitting the GLMs
    x = x0
    lp_prev = parallel_compute_log_p(dview, x, N)
    converged = False
    iter = 0
    while not converged and iter < maxiter:
        iter += 1
        print "Coordinate descent iteration %d." % iter
        
        # TODO Fit the network on the first engine
        # fit_network(x, net_inf_prms, use_hessian, use_rop)
        
        # Fit the GLMs in parallel
        x_glms = dview.map_async(_parallel_fit_glm,
                                 range(N),
                                 [x]*N)

        x['glms'] = x_glms.get()
        x_glms.display_outputs()

        # Check for convergence
        lp = parallel_compute_log_p(dview, x, N)
        print "Iteration %d: LP=%.2f. Change in LP: %.2f" % (iter, lp, lp-lp_prev)

        converged = np.abs(lp-lp_prev) < atol
        lp_prev = lp
    return x

