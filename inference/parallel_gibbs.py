""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""
import copy
import numpy as np

from IPython.parallel.util import interactive
from utils.progress_report import wait_watching_stdout

def initialize_imports(dview):
    """ Import functions req'd for coordinate descent
    """
    dview.execute('from utils.theano_func_wrapper import seval, _flatten')
    dview.execute('from utils.packvec import *')
    dview.execute('from inference.gibbs import *')
    dview.execute('from log_sum_exp import log_sum_exp_sample')
    dview.execute('from hmc import hmc')


def parallel_compute_log_p(dview,
                           master,
                           v,
                           N):
    """ Compute the log prob in parallel
    """

    # Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp_tot = 0

    @interactive
    def _compute_network_lp(vs):
        print "Computing log prob for network"
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs,0)
        lp = seval(popn.network.log_p,
                   syms,
                   nvars)
        #lp = popn.network.log_p.eval(dict(zip(_flatten(tmpsyms),
                                     #        _flatten(tmpnvars))),
                                     #on_unused_input='ignore')
        return lp

    lp_tot += master.apply_sync(_compute_network_lp, v)

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

def concatenate_parallel_updates(xs, x):
    # Concatenate results into x
    for (n, xn) in enumerate(xs):
        x['glms'][n] = xn['glms'][n]

        # Copy over the network 
        if 'A' in xn['net']['graph']:
            x['net']['graph']['A'][:,n] = xn['net']['graph']['A'][:,n]
        if 'W' in xn['net']['weights']:
            N = len(xs)
            W_inf = np.reshape(xn['net']['weights']['W'], (N,N))
            W_curr = np.reshape(x['net']['weights']['W'], (N,N))
            W_curr[:,n] = W_inf[:,n]
            x['net']['weights']['W'] = np.ravel(W_curr)

# TODO: Remove this
def concatenate_network_results(x_net, x, N):
    """ Concatenate the list of results from the parallel
        sampling of network columns
    """
    for (n, xn) in enumerate(x_net):
        if 'A' in xn['graph']:
            x['net']['graph']['A'][:,n] = xn['graph']['A'][:,n]
        if 'W' in xn['weights']:
            W_inf = np.reshape(xn['weights']['W'], (N,N))
            W_curr = np.reshape(x['net']['weights']['W'], (N,N))
            W_curr[:,n] = W_inf[:,n]
            x['net']['weights']['W'] = np.reshape(W_curr, (N**2,))

def periodically_save_results(x, start, stop, results_dir):
    """ Periodically save the MCMC samples
    """
    fname = "results.partial.%d-%d.pkl" % (start,stop)
    import os
    import cPickle
    print "Saving partial results to %s" % os.path.join(results_dir, fname)
    with open(os.path.join(results_dir, fname),'w') as f:
        cPickle.dump(x[start:stop], f, protocol=-1)

def check_convergence(x):
    """ Check for convergence of the sampler
    """
    return False

def parallel_gibbs_sample(client,
                          data,
                          N_samples=1000,
                          x0=None,
                          init_from_mle=True,
                          save_interval=-1,
                          results_dir='.'):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    dview = client[:]
    master = client[client.ids[0]]
    N = data['N']

    # Import req'd functions on engines
    initialize_imports(dview)

    # Draw initial state from prior if not given
    if x0 is None:
        client[0].execute('x0 = popn.sample()', block=True)
        x0 = client[0]['x0']

    # Create parallel samplers
    @interactive
    def _create_samplers():
        global serial_updates
        global parallel_updates
        serial_updates, parallel_updates = initialize_updates(popn)

        # Return the number of parallel_updates 
        return len(serial_updates), len(parallel_updates)

    n_serial_updates, n_parallel_updates = dview.apply(_create_samplers).get()[0]

    # Create map-able functions to sample in parallel
    @interactive
    def _parallel_update(i, x, n):
        return parallel_updates[i].update(x, n)
        
    @interactive
    def _serial_update(i, x):
        return serial_updates[i].update(x)

    ## DEBUG Profile the Gibbs sampling loop
    # import cProfile, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    ## END DEBUG

    # Alternate fitting the network and fitting the GLMs
    lp_smpls = np.zeros(N_samples+1)
    lp_smpls[0] = parallel_compute_log_p(dview, master, x0, N)
    x_smpls = [x0]
    x = x0

    import time
    start_time = time.clock()

    for smpl in np.arange(N_samples):
        # Print the current log likelihood
        lp = parallel_compute_log_p(dview,
                                    master,
                                    x,
                                    N)
        lp_smpls[smpl+1] = lp

        # DEBUG: Look for sharp drops in LP
        if lp - lp_smpls[smpl] > 500:
            import pdb; pdb.set_trace()
        # END DEBUG

        # Compute iters per second
        stop_time = time.clock()
        if stop_time - start_time == 0:
            print "Gibbs iteration %d. Iter/s exceeds time resolution. Log prob: %.3f" % (smpl, lp)
        else:
            print "Gibbs iteration %d. Iter/s = %f. Log prob: %.3f" % (smpl,
                                                                       1.0/(stop_time-start_time),
                                                                       lp)
        start_time = stop_time

        # Periodically save results
        if save_interval > 0 and np.mod(smpl+1, save_interval)==0:
            periodically_save_results(x_smpls, smpl+1-save_interval, smpl+1, results_dir)

        # Go through variables, sampling one at a time, in parallel where possible
        interval = 0.1
        for i in range(n_parallel_updates):
            xs = dview.map_async(_parallel_update,
                                 [i]*N,                                     
                                 [x]*N,
                                 range(N))
            
            wait_watching_stdout(xs, interval=interval)
            
            concatenate_parallel_updates(xs.get(), x)

        # Sample serial updates
        for i in range(n_serial_updates):
            x = master.apply(_serial_update, i, x).get()

        # DEBUG
        print "Num changes in A: %d" % np.abs(x['net']['graph']['A']-x_smpls[-1]['net']['graph']['A'])
        x_smpls.append(copy.deepcopy(x))

    ## DEBUG Profile the Gibbs sampling loop
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()p
    #
    # with open('mcmc.prof.txt', 'w') as f:
    #     f.write(s.getvalue())
    #     f.close()
    ## END DEBUG
        

    return x_smpls
