import copy

from coord_descent import prep_glm_inference, \
                          prep_network_inference, \
                          fit_glm, \
                          fit_network

def parallel_compute_log_p(client,
                           vars,
                           N):
    """ Compute the log prob in parallel
    """
    dview = client[:]

    # TODO Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp = 0


    # Compute the log probability in parallel
    dview['vars'] = vars
    def _compute_glm_lp(n):
        print "Computing lp for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vars, n)
        lp = seval(popn.glm.log_p,
                   syms,
                   nvars)
        return lp

    lp_glms = dview.map_async(_compute_glm_lp, range(N))
    # print lp_glms.get()
    # lp_glms.display_outputs()

    lp += sum(lp_glms.get())
    return lp

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
    # N = data['N']
    # network = population.network
    # glm = population.glm
    # syms = population.get_variables()

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
        client[0].execute('x0 = popn.sample()', block=True)
        x0 = client[0]['x0']
    # Also initialize with intelligent parameters from the data
    # initialize_with_data(population, data, x0)

    # Compute log prob, gradient, and hessian wrt network parameters
    # net_inf_prms = prep_network_inference(population,
    #                                       use_hessian=use_hessian,
    #                                       use_rop=use_rop)

    dview.execute('net_inf_prms = prep_network_inference(popn,'
                  'use_hessian=use_hessian,'
                  'use_rop=use_rop)')

    # Compute gradients of the log prob wrt the GLM parameters
    dview.execute('glm_inf_prms = prep_glm_inference(population,'
                  'use_hessian=use_hessian,'
                  'use_rop=use_rop)')
    
    # Alternate fitting the network and fitting the GLMs
    x = x0
    x_prev = copy.deepcopy(x0)
    lp_prev = parallel_compute_log_p(client, x, N)
    converged = False
    iter = 0
    while not converged and iter < maxiter:
        iter += 1
        print "Coordinate descent iteration %d." % iter
        
        # TODO Fit the network on the first engine
        fit_network(x, net_inf_prms, use_hessian, use_rop)
        
        # TODO Fit the GLMs in parallel
        for n in np.arange(N):
            nvars = population.extract_vars(x, n)
            fit_glm(nvars, n, glm_inf_prms, use_hessian, use_rop)
            x['glms'][n] = nvars['glm']
            
        # Check for convergence 
        lp = parallel_compute_log_p(client, x, N)
        print "Iteration %d: LP=%.2f. Change in LP: %.2f" % (iter, lp, lp-lp_prev)
        
        converged = np.abs(lp-lp_prev) < atol
        lp_prev = lp
    return x

