import os
import cPickle

import numpy as np

import matplotlib
matplotlib.use('Agg')       # To enable saving remotely
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.model_factory import make_model
from population import Population

from utils.avg_dicts import average_list_of_dicts, std_list_of_dicts

def make_rwb_cmap():
    # Get a red-white-black cmap
    cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
    cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap

def compute_weights(s, dt):
    # Get the weights of the impulse responses
    W = s['net']['weights']['W'] * s['net']['graph']['A']
    N = W.shape[0]

    s_imps = []
    s_imps_std = []
    for n_post in np.arange(N):
        s_imp_row = []
        s_imp_std_row = []

        s_imp_n = s['glms'][n_post]['imp']['impulse']
        for n_pre in np.arange(N):
            w = W[n_pre,n_post]
            s_imp_row.append(w*s_imp_n[n_pre,:])

        s_imps.append(s_imp_row)
        s_imps_std.append(s_imp_std_row)

    s_imps = np.array(s_imps)
    t_imp = dt*np.arange(s_imps.shape[2])
    W_imp = np.trapz(s_imps,t_imp, axis=2)
    return W_imp

def compute_pr_A(s_smpls):
    """ Compute the probability of an edge.
    """
    As = np.array([s['net']['graph']['A'] for s in s_smpls])
    pr_A = np.sum(As, axis=0) / float(As.shape[0])
    return pr_A

def scatter_plot_weight_residuals(dt, s_inf, s_true, ax=None, color='r', sz=20, figsize=(2,2), resdir='.', label=None):
    """
    Plot residuals in the weights of true connections
    """
    # Compute the weights
    W_inf = compute_weights(s_inf, dt)
    W_true = compute_weights(s_true, dt)

    ax_given = ax is not None
    if not ax_given:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    conn_thresh = 1e-2
    true_conns = np.abs(W_true) > conn_thresh
    ax.scatter(W_true[true_conns].ravel(), W_inf[true_conns].ravel(), s=sz,
               marker='.', c=color, linewidths=0,
               label=label)

    # Set up the axes
    ax.set_xlabel('True W')
    ax.set_ylabel('Inferred W')

    # Set equal x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    alim = np.amax(np.abs([xlim[0], xlim[1], ylim[0], ylim[1]]))
    ax.set_xlim([-alim, alim])
    ax.set_ylim([-alim, alim])

    # Only have a few ticks
    atick = np.round(alim, decimals=1)
    ax.set_xticks(np.array([-atick, 0, atick]))
    ax.set_yticks(np.array([-atick, 0, atick]))

    if not ax_given:
        if label is not None:
            plt.legend(loc='lower right')

    if not ax_given:
        fig.savefig(os.path.join(resdir, 'W_resid.pdf'))
        plt.close(fig)

def compute_pr_A_tp(s_infs_mcmc, s_trues, ax=None, color='r', sz=20, figsize=(2,2), resdir='.', label=None):
    """
    Compute the probability of a false negative vs true weight
    """
    pr_fn = np.array([])
    W_true_tp = np.array([])

    for d,s_inf_mcmc in enumerate(s_infs_mcmc):
        pr_A = compute_pr_A(s_inf_mcmc)
        A_true = s_trues[d]['net']['graph']['A']
        W_true = s_trues[d]['net']['weights']['W']

        tpi, tpj = np.nonzero(A_true)
        pr_fn = np.concatenate((pr_fn, 1.0 - pr_A[tpi, tpj].ravel()))
        W_true_tp = np.concatenate((W_true_tp, W_true[tpi, tpj].ravel()))

    ax_given = ax is not None
    if not ax_given:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # Compute the mean and standard deviation of pr_A as a function of
    # the true weight
    nbins = 11
    n, _ = np.histogram(W_true_tp, bins=nbins)
    sy, _ = np.histogram(W_true_tp, bins=nbins, weights=pr_fn)
    sy2, _ = np.histogram(W_true_tp, bins=nbins, weights=pr_fn**2)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    ax.scatter(W_true_tp.ravel(), pr_fn.ravel(), s=sz,
           marker='.', c=color, linewidths=0,
           label=label)
    plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='k-')

    xlim = ax.get_xlim()
    alim = np.amax(np.abs(xlim))
    ax.set_xticks(np.array([-alim, -alim/2.0,  0, alim/2.0, alim]).round(decimals=2))

    ax.set_ylim([0,1])
    ax.set_xlabel('True W')
    ax.set_ylabel('Pr(FN)')

    if not ax_given:
        fig.savefig(os.path.join(resdir, 'pr_fn.pdf'))
        plt.close(fig)

def compute_pr_W_fp(s_infs_mcmc, s_trues, ax=None, color='r', figsize=(2,2), resdir='.'):
    """
    Compute the probability of a false negative vs true weight
    """
    Ws_tn = np.array([])
    As_tn = np.array([], dtype=np.bool)

    for d,s_inf_mcmc in enumerate(s_infs_mcmc):
        A_true = s_trues[d]['net']['graph']['A']
        tni, tnj = np.nonzero(np.bitwise_not(A_true))
        Ws = np.array([s['net']['weights']['W'] for s in s_inf_mcmc])
        As = np.array([s['net']['graph']['A'] for s in s_inf_mcmc])
        Ws_tn = np.concatenate((Ws_tn, Ws[:, tni, tnj].ravel()))
        As_tn = np.concatenate((As_tn, As[:, tni, tnj].ravel().astype(np.bool)))

    W_fp = Ws_tn[As_tn].ravel()
    ax_given = ax is not None
    if not ax_given:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # Compute the mean and standard deviation of pr_A as a function of
    # the true weight
    # import pdb; pdb.set_trace()
    # nbins = 15
    # pr_W_fp, bins = np.histogram(W_fp, bins=nbins, normed=True)

    ax.hist(W_fp, bins=20, normed=True, color='r')

    ax.set_ylim([0,1])
    xlim = ax.get_xlim()
    alim = np.amax(np.abs(xlim))
    ax.set_xticks(np.array([-alim, -alim/2.0,  0, alim/2.0, alim]).round(decimals=1))

    ax.set_xlabel('Inferred W (FP)')
    ax.set_ylabel('p(W)')

    if not ax_given:
        fig.savefig(os.path.join(resdir, 'pr_W_fp.pdf'))
        plt.close(fig)

def make_pr_A_v_W_plot(s_infs_mcmc, s_trues, resdir='.'):

    # Scatter plot the average weights vs true weights
    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.gca()

    compute_pr_A_tp(s_infs_mcmc, s_trues, ax=ax, color='r', resdir=resdir)

    # Make room for the axis labels
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Save and close
    fig.savefig(os.path.join(resdir, 'pr_A_tp.pdf'))
    plt.close(fig)

def make_pr_W_fp_plot(s_infs_mcmc, s_trues, resdir='.'):
    # Scatter plot the average weights vs true weights
    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.gca()

    compute_pr_W_fp(s_infs_mcmc, s_trues, ax=ax, color='r', resdir=resdir)

    # Make room for the axis labels
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Save and close
    fig.savefig(os.path.join(resdir, 'pr_W_fp.pdf'))
    plt.close(fig)

def make_weight_residual_plot(N, dt, s_infs_mcmc, s_trues, s_infs_map=None, resdir='.'):
    # Scatter plot the average weights vs true weights
    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.gca()

    for d,s_inf_mcmc in enumerate(s_infs_mcmc):
        s_avg = average_list_of_dicts(s_inf_mcmc)
        label = None if d >0 else "N-GLM"
        scatter_plot_weight_residuals(dt, s_avg, s_trues[d], ax=ax, sz=20, color='r', label=label)

    if s_infs_map is not None:
        for d, s_inf_map in enumerate(s_infs_map):
            label = None if d >0 else "L1-GLM"
            scatter_plot_weight_residuals(dt, s_inf_map, s_trues[d], ax=ax, sz=15, color='b', label=label)

    # Plot identity line
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, '-k', linewidth=0.5)

    ax.legend(loc='upper left', prop={'size' : 8}, scatterpoints=1)

    # Make room for the axis labels
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Save and close
    fig.savefig(os.path.join(resdir, 'W_resid.pdf'))
    plt.close(fig)


def plot_residuals(popn, x_inf, popn_true, x_true, popn2=None, x2_inf=None, resdir='.'):
    """
    Plot each of the residuals
    """
    N = popn.N
    dt = popn.glm.dt.get_value()

    # Make sure we have a list of x's
    if not isinstance(x_inf, list):
        x_inf = [x_inf]

    # Evaluate the state for each of the parameter settings
    s_inf = []
    for x in x_inf:
        s_inf.append(popn.eval_state(x))


    if popn2 is not None and x2_inf is not None:
        x2_given = True

    if x2_given:
        if not isinstance(x2_inf, list):
            x2_inf = [x2_inf]
        # Evaluate the state for each of the parameter settings
        s2_inf = []
        for x in x2_inf:
            s2_inf.append(popn2.eval_state(x))

    s_true = popn_true.eval_state(x_true)

    # Make the individual plots
    make_pr_W_fp_plot(s_inf, s_true, s2_inf=s2_inf, resdir=resdir)
    make_pr_A_v_W_plot(N, dt, s_inf, s_true, s2_inf=s2_inf, resdir=resdir)
    make_weight_residual_plot(N, dt, s_inf, s_true, s2_inf=s2_inf, resdir=resdir)


def load_set_of_results(N, T, graph_model='er', sample_frac=0.1):
    data_dir = os.path.join('/group', 'hips', 'scott', 'pyglm', 'data', 'synth', graph_model, 'N%dT%d' % (N, T))

    # Evaluate the state for each of the parameter settings
    s_infs_mcmc = []
    s_infs_map = []
    s_trues = []

    # Enumerate the subdirectories containing the data
    subdirs = os.listdir(data_dir)
    subdirs = reduce(lambda sd, d: sd + [d] \
                                   if os.path.isdir(os.path.join(data_dir, d)) \
                                   else sd,
                     subdirs, [])

    # For each data subdirectory, load the true data, the MAP estimate, and the MCMC results
    print "WARNING: Make sure we sample all subdirs"
    # import pdb; pdb.set_trace()
    for d in subdirs:
        print "Loading data and results from %s" % d
        print "Loading true data"
        with open(os.path.join(data_dir, d, 'data.pkl'), 'r') as f:
            data = cPickle.load(f)

        print "Loading model"
        with open(os.path.join(data_dir, d, 'model.pkl'), 'r') as f:
            model_data = cPickle.load(f)
            #HACK
            if 'N_dims' not in model_data['network']['graph']:
                model_data['network']['graph']['N_dims'] = 1
            if 'location_prior' not in model_data['network']['graph']:
                model_data['network']['graph']['location_prior'] = \
                         {
                             'type' : 'gaussian',
                             'mu' : 0.0,
                             'sigma' : 1.0
                         }
            if 'L' in data['vars']['net']['graph']:
                data['vars']['net']['graph']['L'] = data['vars']['net']['graph']['L'].ravel()
        popn_data = Population(model_data)
        popn_data.set_data(data)
        s_trues.append(popn_data.eval_state(data['vars']))

        try:
            print "Loading map estimate"
            with open(os.path.join(data_dir, d, 'map', 'results.pkl'), 'r') as f:
                x_map = cPickle.load(f)

            model_map = make_model('standard_glm', N=data['N'])
            popn_map = Population(model_map)
            popn_map.set_data(data)
            print "Evaluating MAP state"
            s_infs_map.append(popn_map.eval_state(x_map))

        except Exception as e:
            print "ERROR: Failed to load MAP estimate"

        try:
            print "Loading mcmc estimate"
            with open(os.path.join(data_dir, d, 'mcmc', 'results.pkl'), 'r') as f:
                x_mcmc = cPickle.load(f)

            model_mcmc = make_model('sparse_weighted_model', N=data['N'])
            popn_mcmc = Population(model_mcmc)
            popn_mcmc.set_data(data)

            # Now compute the true and false positive rates for MCMC
            # For MCMC results, only consider the tail of the samples
            print "Evaluating MCMC states"
            N_samples = len(x_mcmc)
            start_smpl = int(np.floor(N_samples - sample_frac*N_samples))

            # Evaluate the state
            this_s_mcmc = []
            for i in range(start_smpl, N_samples):
                this_s_mcmc.append(popn_mcmc.eval_state(x_mcmc[i]))
            s_infs_mcmc.append(this_s_mcmc)
        except Exception as e:
            print "ERROR: Failed to load MCMC estimate"

    return s_trues, s_infs_map, s_infs_mcmc


if __name__ == "__main__":
    from test.synth_harness import initialize_test_harness
    options, popn, data, popn_true, x_true = initialize_test_harness()

    smpl_frac = 0.1

    # # Load the results
    # with open(options.x0_file, 'r') as f:
    #     print "Loading results from: %s" % options.x0_file
    #     x1 = cPickle.load(f)
    #     # If x is a list of samples, only keep the last (burned-in) fraction
    #     if isinstance(x1, list):
    #         x1 = x1[-1*int(smpl_frac*len(x1)):]
    #
    # # Load secondary results. Assume these came from a standard glm
    # with open(options.x1_file, 'r') as f:
    #     from models.model_factory import make_model
    #     from population import Population
    #     popn2 = Population(make_model("standard_glm", popn.N))
    #     popn2.set_data(data)
    #
    #     print "Loading secondary results from: %s" % options.x1_file
    #     x2 = cPickle.load(f)
    #     # If x is a list of samples, only keep the last (burned-in) fraction
    #     if isinstance(x2, list):
    #         x2 = x2[-1*int(smpl_frac*len(x1)):]

    N = 16
    T = 15
    dt = 0.001
    s_trues, s_infs_map, s_infs_mcmc = load_set_of_results(N, T, sample_frac=smpl_frac)

    print "Plotting results"
    # plot_residuals(popn,
    #                x1,
    #                popn_true=popn_true,
    #                x_true=x_true,
    #                popn2=popn2,
    #                x2_inf=x2,
    #                resdir=options.resultsDir)
    # make_weight_residual_plot(N, dt, s_infs_mcmc, s_trues, s_infs_map=s_infs_map, resdir=options.resultsDir)
    # make_pr_W_fp_plot(s_infs_mcmc, s_trues, resdir=options.resultsDir)
    make_pr_A_v_W_plot(s_infs_mcmc, s_trues, resdir=options.resultsDir)