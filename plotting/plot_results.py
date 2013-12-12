""" Plot the inference results
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')       # To enable saving remotely
import matplotlib.pyplot as plt

import cPickle

def plot_results(network_glm, x_inf, x_true=None, resdir=None):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    if not resdir:
        resdir = '.'

    # Get a red-gray cmap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('RdGy')

    true_given = x_true is not None
    if true_given:
        true_state = network_glm.eval_state(x_true)
    opt_state = network_glm.eval_state(x_inf)

    N = network_glm.N
    
    # Plot the inferred connectivity matrix
    print "Plotting connectivity matrix"
    f = plt.figure()
    W_inf = opt_state['net']['weights']['W'] * opt_state['net']['graph']['A']
    
    if true_given:
        plt.subplot(1,2,1)
        W_true = true_state['net']['weights']['W'] * true_state['net']['graph']['A']
        W_max = np.amax(np.maximum(np.abs(W_true),np.abs(W_inf)))
    else:
        W_max = np.amax(np.abs(W_inf))
        
    px_per_node = 10

    if true_given:
        plt.imshow(np.kron(W_true,np.ones((px_per_node,px_per_node))),
                   vmin=-W_max,vmax=W_max,
                   extent=[0,1,0,1],
                   interpolation='nearest')
        plt.colorbar()
        plt.title('True Network')
        plt.subplot(1,2,2)

    # Plot the inferred network
    plt.imshow(np.kron(W_inf,np.ones((px_per_node,px_per_node))),
               vmin=-W_max,vmax=W_max,
               extent=[0,1,0,1],
               interpolation='nearest')
    plt.colorbar()
    plt.title('Inferred Network')
    f.savefig(os.path.join(resdir,'conn.pdf'))
    plt.close(f)

    # Plot the stimulus tuning curve
    print "Plotting stimulus response curves"
    for n in np.arange(N):
        f = plt.figure()
        opt_state_n = opt_state['glms'][n]
        if true_given:
            true_state_n = true_state['glms'][n]
        if 'stim_response_t' in opt_state_n['bkgd'].keys() and \
            'stim_response_x' in opt_state_n['bkgd'].keys():
            
            # Get the stimulus responses
            opt_stim_x = opt_state_n['bkgd']['stim_response_x']
            opt_stim_t = opt_state_n['bkgd']['stim_response_t']
            if true_given:
                true_state_n = true_state['glms'][n]
                true_stim_x = true_state_n['bkgd']['stim_response_x']
                true_stim_t = true_state_n['bkgd']['stim_response_t']

            plt.subplot(1,2,1)
            plt.plot(opt_stim_x,'--r')
            plt.hold(True)
            if true_given:
                plt.plot(true_stim_x,'b')
            plt.title('GLM[%d]: Spatial stimulus filter' % n)

            plt.subplot(1,2,2)
            plt.plot(opt_stim_t,'--r')
            plt.hold(True)
            if true_given: 
                plt.plot(true_stim_t,'b')
            plt.title('GLM[%d]: Temporal stimulus filter' % n)
        elif 'stim_response' in opt_state_n['bkgd'].keys():
            opt_stim_t = opt_state_n['bkgd']['stim_response']
            plt.plot(opt_stim_t,'--r')
            plt.hold(True)
            if true_given:
                true_stim_t = true_state_n['bkgd']['stim_response']
                plt.plot(true_stim_t,'b')
            plt.title('GLM[%d]: stimulus filter' % n)
        f.savefig(os.path.join(resdir,'stim_resp_%d.pdf' % n))
        plt.close(f)
        
    # Plot the impulse responses
    print "Plotting impulse responses"
    true_imps = []
    opt_imps = []
    f = plt.figure()
    for n_post in np.arange(N):
        true_imp_row = []
        opt_imp_row = []
        
        opt_state_n = opt_state['glms'][n_post]
        opt_imp = opt_state_n['imp']['impulse']
        if true_given:
            true_state_n = true_state['glms'][n_post]
            true_imp = true_state_n['imp']['impulse']
        
        for n_pre in np.arange(N):
            opt_imp_row.append(W_inf[n_pre,n_post]*opt_imp[n_pre,:])
            if true_given:
                true_imp_row.append(W_true[n_pre,n_post]*true_imp[n_pre,:])
        
        opt_imps.append(opt_imp_row)
        if true_given:
            true_imps.append(true_imp_row)
    opt_imps = np.array(opt_imps)
    if true_given:
        true_imps = np.array(true_imps)


    if true_given:
        imp_max = np.amax(np.maximum(np.abs(opt_imps), np.abs(true_imps)))
    else:
        imp_max = np.amax(np.abs(opt_imps))
        
    W_opt_imp = np.sum(opt_imps,2)
    if true_given:
        W_true_imp = np.sum(true_imps,2)
        W_imp_max = np.amax(np.maximum(W_opt_imp, W_true_imp))
    else:
        W_imp_max = np.amax(W_opt_imp)
    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            # Set background color based on weight of impulse
            color = cmap((W_opt_imp[n_pre,n_post] -(-W_imp_max))/(2*W_imp_max))
            # Set it slightly transparent
            tcolor = list(color)
            tcolor[3] = 0.75
            tcolor = tuple(tcolor)
            plt.subplot(N,N,n_pre*N+n_post + 1, axisbg=tcolor)

            # Plot the inferred impulse response
            plt.plot(np.squeeze(opt_imps[n_pre,n_post,:]),'-k')
            plt.hold(True)
            plt.plot(np.zeros_like(np.squeeze(opt_imps[n_pre,n_post,:])), 'k:')
            if true_given:
                plt.plot(np.squeeze(true_imps[n_pre,n_post,:]),'-b')
            
            plt.xlabel("")
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("")
            plt.ylim(-imp_max,imp_max)

    f.savefig(os.path.join(resdir,'imp_resp.pdf'))
    plt.close(f)
    
    # Plot the firing rates
    print "Plotting firing rates"
    for n in np.arange(N):
        opt_state_n = opt_state['glms'][n]
        if true_given:
            true_state_n = true_state['glms'][n]
        
        f = plt.figure()
        plt.plot(opt_state_n['lam'],'r')
        if true_given:
            plt.hold(True)
            plt.plot(true_state_n['lam'],'b')
            
        # Plot the spike times
        St = np.nonzero(network_glm.glm.S.get_value()[:,n])[0]
        plt.plot(St,0.1*np.ones_like(St),'kx')
        plt.title('Firing rate %d' % n)

        # Zoom in on a small fraction
        plt.xlim(10000,12000)
        f.savefig(os.path.join(resdir,'firing_rate_%d.pdf' % n))
        plt.close(f)

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    
    parser.add_option("-d", "--dataFile", dest="dataFile",
                      help="Data file to load")

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

    # Load data                                                              
    if options.dataFile.endswith('.mat'):
        print "Loading data from %s" % options.dataFile
        import scipy.io
        data = scipy.io.loadmat(options.dataFile, squeeze_me=True, mat_dtype=True)
        data['N'] = np.int(data['N'])
    elif options.dataFile.endswith('.pkl'):
        print "Loading data from %s" % options.dataFile
        import cPickle
        with open(options.dataFile,'r') as f:
            data = cPickle.load(f)
    else:
        raise Exception("Unrecognized file type: %s" % options.dataFile)

    # Load results
    if options.resultsFile.endswith('.pkl'):
        print "Loading results from %s" % options.resultsFile
        with open(options.resultsFile,'r') as f:
            x_inf = cPickle.load(f)
    else:
        raise Exception("Unrecognized file type: %s" % options.resultsFile)

    print "Initializing GLM"
    from glm_shared import NetworkGlm
    from models.rgc import Rgc
    model = Rgc
    glm = NetworkGlm(model)
    print "Conditioning on the data"
    glm.set_data(data)

    print "Plotting results"
    plot_results(glm, x_inf)
