""" Plot the inference results
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')       # To enable saving remotely
import matplotlib.pyplot as plt

import cPickle

def plot_results(network_glm, x_inf, resdir=None):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    if not resdir:
        resdir = '.'

    # Get a red-gray cmap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('RdGy')

    inf_state = network_glm.get_state(x_inf)

    N = network_glm.N

    # Plot the inferred connectivity matrix
    f = plt.figure()
    W_inf = inf_state['net']
    W_max = np.amax(np.abs(W_inf))
    px_per_node = 10
    plt.imshow(np.kron(W_inf,np.ones((px_per_node,px_per_node))),
               vmin=-W_max,vmax=W_max,
               extent=[0,1,0,1],
               cmap=cmap,
               interpolation='nearest')
    plt.colorbar()
    plt.title('Inferred Network')

    f.savefig(os.path.join(resdir,'conn.pdf'))

    # Plot the stimulus tuning curve
    for n in np.arange(N):
        f = plt.figure()
        if 'stim_t' in inf_state[n].keys() and \
            'stim_x' in inf_state[n].keys():
            plt.subplot(1,2,1)

            # Spatial tuning curve is 2D
            stim_x = np.reshape(inf_state[n]['stim_x'],[10,10])
            plt.imshow(np.kron(stim_x,np.ones((px_per_node,px_per_node))),
                       extent=[0,10,0,10],
                       interpolation='nearest')
            plt.colorbar()
            plt.title('GLM[%d]: Spatial stimulus filter' % n)

            plt.subplot(1,2,2)
            plt.plot(inf_state[n]['stim_t'],'--r')
            plt.title('GLM[%d]: Temporal stimulus filter' % n)
        elif 'stim' in inf_state[n].keys():
            plt.plot(inf_state[n]['stim'],'--r')
            plt.title('GLM[%d]: stimulus filter' % n)
        f.savefig(os.path.join(resdir,'stim_resp_%d.pdf' % n))

    # Plot the impulse responses
    imps = []
    W_inf = inf_state['net']
    f = plt.figure()
    for n_pre in np.arange(N):
        imp_row = []
        for n_post in np.arange(N):
            imp_row.append(W_inf[n_pre,n_post]*inf_state[n_post]['ir'][n_pre,:])
        imps.append(imp_row)
    imps = np.array(imps)

    imp_max = np.amax(np.abs(imps))
    W_imp = np.sum(imps,2)
    W_imp_max = np.amax(W_imp)
    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            # Set background color based on weight of impulse
            color = cmap((W_imp[n_pre,n_post] -(-W_imp_max))/(2*W_imp_max))
            # Set it slightly transparent
            tcolor = list(color)
            tcolor[3] = 0.5
            tcolor = tuple(tcolor)
            plt.subplot(N,N,n_pre*N+n_post + 1, axisbg=tcolor)
            plt.plot(np.squeeze(imps[n_pre,n_post,:]),'k')
            plt.xlabel("")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("")
            plt.ylim(-imp_max,imp_max)

    f.savefig(os.path.join(resdir,'imp_resp.pdf'))

    # Infer the firing rates

    for n in np.arange(N):
        f = plt.figure()
        plt.plot(inf_state[n]['lam'],'r')

        # Plot the spike times
        St = np.nonzero(network_glm.glm.S.get_value()[:,n])[0]
        plt.plot(St,0.1*np.ones_like(St),'kx')
        plt.title('Firing rate %d' % n)

        # Zoom in on a small fraction
        plt.xlim(10000,12000)
        f.savefig(os.path.join(resdir,'firing_rate_%d.pdf' % n))


def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option("-r", "--resultsFile", dest="resultsFile", default='.',
                      help="Results file to plot.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if options.resultsFile is None or not os.path.exists(options.resultsFile):
        raise Exception("Invalid results file: %s" % options.resultsFile)

    return (options, args)


if __name__ == "__main__":
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    # Load results
    if options.dataFile.endswith('.pkl'):
        print "Loading results from %s" % options.resultsFile
        with open(options.dataFile,'r') as f:
            (glm,x_inf) = cPickle.load(f)
    else:
        raise Exception("Unrecognized file type: %s" % options.resultsFile)

    print "Plotting results"
    plot_results(glm, x_inf)
