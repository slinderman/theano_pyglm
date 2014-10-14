# Run as script using 'python -m test.synth'
import cPickle
import os

import matplotlib.pyplot as plt

from pyglm.models.model_factory import *
# from pyglm.inference.gibbs import initialize_updates
from pyglm.inference.kayak_gibbs import initialize_updates, CollapsedGibbsNetworkColumnUpdate
from pyglm.population import KayakPopulation


def geweke_test(population,
                data,
                N_samples=1000):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    N = population.model['N']

    # Draw initial state from prior
    x0 = population.sample()

    # Create updates for this population
    serial_updates, parallel_updates = initialize_updates(population)

    # DEBUG Profile the Gibbs sampling loop
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    # Alternate fitting the network and fitting the GLMs
    x_smpls = [x0]
    x = x0

    import time
    start_time = time.time()

    for smpl in np.arange(N_samples):
        # Print the current log likelihood
        lp = population.compute_log_p(x)

        # Compute iters per second
        stop_time = time.time()
        if stop_time - start_time == 0:
            print "Geweke iteration %d. Iter/s exceeds time resolution. Log prob: %.3f" % (smpl, lp)
        else:
            print "Geweke iteration %d. Iter/s = %f. Log prob: %.3f" % (smpl,
                                                                       1.0/(stop_time-start_time),
                                                                       lp)
        start_time = stop_time

        for parallel_update in parallel_updates:
            if isinstance(parallel_update, CollapsedGibbsNetworkColumnUpdate):
                for n in np.arange(N):
                    parallel_update.update(x, n)

        # # Go through each parallel MH update
        # for parallel_update in parallel_updates:
        #     for n in np.arange(N):
        #         parallel_update.update(x, n)
        #
        # # Sample the serial updates
        # for serial_update in serial_updates:
        #     serial_update.update(x)

        # Geweke step: Sample new data
        # import pdb; pdb.set_trace()
        data = gen_synth_data(population, x, N, data['T'] )
        population.data_sequences.pop()
        population.add_data(data)

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


def gen_synth_data(popn, x_true, N, T_stop=15):
    # Set simulation parametrs
    dt = 0.001
    dt_stim = 0.1
    D_stim = 1

    # Generate random white noise stimulus
    stim = np.random.randn(T_stop/dt_stim, D_stim)

    # Initialize the GLMs with just the stimulus
    # temp_data = {"S": np.zeros((T_stop/dt, N)),
    #              "N": N,
    #              "dt": dt,
    #              "T": np.float(T_stop),
    #              "stim": stim,
    #              'dt_stim': dt_stim}
    # popn.set_data(temp_data)
    #
    # # Simulate spikes
    # S,X = popn.simulate(x_true, (0, T_stop), dt)
    #
    # # Package data into dict
    # data = {"S": S,
    #         "X": X,
    #         "N": N,
    #         "dt": dt,
    #         "T": np.float(T_stop),
    #         "stim": stim,
    #         'dt_stim': dt_stim}

    # Simulate spikes
    S,X = popn.simulate(x_true, (0, T_stop), dt, stim, dt_stim)

    # Package data into dict
    data = {"S": S,
            "X": X,
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars' : x_true}

    return data

    return data


def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--force_recompute", dest="force_recompute",
                      default=False, action='store_true',
                      help="Force a recomputation of the ROC curve, even if previous results exist.")

    parser.add_option("-T", "--T", dest="T", default=-1)

    parser.add_option("-N", "--N", dest="N", default=-1)

    parser.add_option("-m", "--model", dest="model", default="sparse_weighted_model")

    parser.add_option("-r", "--resultsDir", dest="resultsDir", default='.',
                      help="Save the results to this directory.")

    (options, args) = parser.parse_args()
    options.N = int(options.N)
    options.T = int(options.T)
    return (options, args)

def plot_geweke_results(popn, x_smpls, model, resdir='.'):
    """
    Plot a histogram of samples vs the prior
    """
    import matplotlib.mlab as mlab
    N = model['N']

    # Evaluate the state
    s_smpls = []
    for x in x_smpls:
        s_smpls.append(popn.eval_state(x))

    # Plot the adjacency probability
    As = [s['net']['graph']['A'] for s in s_smpls]
    As = np.array(As)
    pA = np.mean(As, axis=0)
    f = plt.figure()
    plt.imshow(np.kron(pA,np.ones((10,10))),
               vmin=0,vmax=1,
               extent=[0,1,0,1],
               interpolation='nearest')
    plt.colorbar()
    plt.title('P(A_{n1,n2})')
    f.savefig(os.path.join(resdir,'geweke_A.pdf'))
    plt.close(f)

    # Plot the weight histogram
    if isinstance(popn, KayakPopulation):
        mu_w = popn.network.weights.mu
        sigma_w = popn.network.weights.sigma
    else:
        mu_w = popn.network.weights.prior.mu.get_value()
        sigma_w = popn.network.weights.prior.sigma.get_value()

        if hasattr(popn.network.weights, 'refractory_prior'):
            mu_w_ref = popn.network.weights.refractory_prior.mu.get_value()
            sigma_w_ref = popn.network.weights.refractory_prior.sigma.get_value()
        else:
            mu_w_ref = mu_w
            sigma_w_ref = sigma_w

    Ws = [s['net']['weights']['W'] for s in s_smpls]
    Ws = np.array(Ws)
    f = plt.figure()
    for n1 in range(N):
        for n2 in range(N):
            ax = f.add_subplot(N,N,1+n1*N+n2)
            n, bins, patches = ax.hist(np.squeeze(Ws[:,n1,n2]), 20, normed=1)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            if isinstance(popn, KayakPopulation):
                y = mlab.normpdf(bincenters, mu_w[n1,n2], sigma_w[n1,n2])
            else:
                if n1==n2:
                    y = mlab.normpdf(bincenters, mu_w_ref, sigma_w_ref)
                else:
                    y = mlab.normpdf(bincenters, mu_w, sigma_w)
            ax.plot(bincenters, y, 'r--', linewidth=1)
    f.savefig(os.path.join(resdir,'geweke_W.pdf'))

    # Plot the background rates
    biases = [[s['glms'][n]['bias']['bias'][0] for n in range(N)] for s in s_smpls]
    biases = np.array(biases)
    from scipy.stats import norm
    mu_bias = model['bias']['mu']
    sig_bias = model['bias']['sigma']


    f = plt.figure()
    for n in range(N):
        ax = f.add_subplot(1,N,1+n)
        c, bins, patches = ax.hist(biases[:,n], 20, normed=1)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        pbias = norm(mu_bias, sig_bias).pdf(bincenters)
        ax.plot(bincenters, pbias, 'r--', linewidth=1)
        plt.title("x_{%d}~N(%.1f,%.1f)" % (n,mu_bias,sig_bias))
    f.savefig(os.path.join(resdir,'geweke_bias.pdf'))

    # Plot the gamma distributed latent vars of the normalized impulse resp
    # gs = [[np.exp(x['glms'][n]['imp']['w_lng']) for n in range(N)] for x in x_smpls]
    if isinstance(popn, KayakPopulation):
        gs = np.concatenate([np.concatenate([x['glm_%d'%n]['imp']['g_%d'%n]
                                             for n in range(N)], axis=1)
                             for x in x_smpls], axis=0)
    else:
        gs = [[x['glms'][n]['imp']['g_%d'%n] for n in range(N)] for x in x_smpls]
    gs = np.array(gs)
    gs = np.abs(gs)
    (_,N,B) = gs.shape

    # Get the true dist
    from scipy.stats import gamma
    g_alpha = model['impulse']['alpha']

    f = plt.figure()
    for n in range(N):
        for b in range(B):
            ax = f.add_subplot(N,B,1 + n*B +b)
            c, bins, patches = ax.hist(gs[:,n,b], 20, normed=1)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            pg = gamma(g_alpha).pdf(bincenters)
            ax.plot(bincenters, pg, 'r--', linewidth=1)
            plt.title("G_{%d,%d}~Gamma(%.1f,1)" % (n,b,g_alpha))
    f.savefig(os.path.join(resdir,'geweke_g.pdf'))


def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    # Parse command line args
    (options, args) = parse_cmd_line_args()
    print "Creating master population object"
    model = make_model(options.model, N=options.N)
    # popn = TheanoPopulation(model)
    popn = KayakPopulation(model)

    results_file = os.path.join(options.resultsDir, 'geweke_results.pkl')
    if os.path.exists(results_file) and not options.force_recompute:
        with open(results_file) as f:
            x_smpls = cPickle.load(f)

    else:
        x0 = popn.sample()
        data = gen_synth_data(popn, x0, options.N, options.T)
        popn.add_data(data)

        # Perform inference
        N_samples = 10000
        x_smpls = geweke_test(popn, data, N_samples=N_samples)

        # Save results
        print "Saving results to %s" % results_file
        with open(results_file, 'w') as f:
            cPickle.dump(x_smpls, f, protocol=-1)

    # Plot empirical parameter distributions
    print "Plotting results."
    plot_geweke_results(popn, x_smpls, model, resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()
