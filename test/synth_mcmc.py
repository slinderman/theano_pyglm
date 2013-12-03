# Run as script using 'python -m test.synth'
from glm_shared import *
from models.model_factory import *
from inference.gibbs import gibbs_sample

import cPickle
import numpy as np
import scipy.io

def plot_results(network_glm, x_trues, x_opts):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    import matplotlib
    matplotlib.use('Agg')       # To enable saving remotely
    import matplotlib.pyplot as plt

    true_state = network_glm.get_state(x_trues)
    opt_state = network_glm.get_state(x_opts)

    N = network_glm.N
    
    # Plot the inferred connectivity matrix
    f = plt.figure()
    plt.subplot(1,2,1)
    W_true = true_state['net']
    W_inf = opt_state['net']
    W_max = np.amax(np.maximum(np.abs(W_true),np.abs(W_inf)))
    px_per_node = 10
    plt.imshow(np.kron(W_true,np.ones((px_per_node,px_per_node))),
               vmin=-W_max,vmax=W_max,
               extent=[0,1,0,1],
               interpolation='nearest')
    plt.colorbar()
    plt.title('True Network')
    plt.subplot(1,2,2)

    plt.imshow(np.kron(W_inf,np.ones((px_per_node,px_per_node))),
               vmin=-W_max,vmax=W_max,
               extent=[0,1,0,1],
               interpolation='nearest')
    plt.colorbar()
    plt.title('Inferred Network')

    f.savefig('conn.pdf')
    
    # Plot the stimulus tuning curve
    for n in np.arange(N):
        f = plt.figure()
        if 'stim_t' in true_state[n].keys() and \
            'stim_x' in true_state[n].keys():
            plt.subplot(1,2,1)
            plt.plot(true_state[n]['stim_x'],'b')
            plt.hold(True)
            plt.plot(opt_state[n]['stim_x'],'--r')
            plt.title('GLM[%d]: Spatial stimulus filter' % n)

            plt.subplot(1,2,2)
            plt.plot(true_state[n]['stim_t'],'b')
            plt.hold(True)
            plt.plot(opt_state[n]['stim_t'],'--r')
            plt.title('GLM[%d]: Temporal stimulus filter' % n)
        elif 'stim' in true_state[n].keys():
            plt.plot(true_state[n]['stim'],'b')
            plt.hold(True)
            plt.plot(opt_state[n]['stim'],'--r')
            plt.title('GLM[%d]: stimulus filter' % n)
        f.savefig('stim_resp_%d.pdf' % n)

    # Plot the impulse responses
    W_true = true_state['net']
    W_opt = opt_state['net']
    f = plt.figure()
    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            plt.subplot(N,N,n_pre*N+n_post + 1)
            plt.plot(W_true[n_pre,n_post]*true_state[n_post]['ir'][n_pre,:],'b')
            plt.hold(True)
            plt.plot(W_opt[n_pre,n_post]*opt_state[n_post]['ir'][n_pre,:],'r')
            #plt.title('Imp Response %d->%d' % (n_pre,n_post))
            plt.xlabel("")
            plt.ylabel("")

    f.savefig('imp_resp.pdf')

    # Infer the firing rates
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(true_state[n]['lam'],'b')
        plt.hold(True)
        plt.plot(opt_state[n]['lam'],'r')

        # TODO Plot the spike times
        plt.title('Firing rate %d' % n)
    f.savefig('firing_rate.pdf')

def generate_synth_data(glm,
                        resultsDir,
                        T_start=0, T_stop=1000,
                        dt=0.1,
                        dt_stim=10):
    """ Generate synthetic data from the given model.
    """

    D_stim = model['bkgd']['D_stim']
    N = model['N']

    # Sample random parameters from the model
    x_true = glm.sample()

    # Generate random stimulus
    stim = np.random.randn(T_stop/dt_stim,D_stim)

    # Initialize the GLMs with the stimulus
    temp_data = {"S": np.zeros((T_stop/dt,N)),
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim}
    glm.set_data(temp_data)

    # Simulate spikes
    S,X = glm.simulate(x_true, (T_start, T_stop), dt)

    # TODO Fix this hack
    # Put vars into a dict
    var_dict = {}
    var_dict['net'] = x_true[0]
    for (n,glm_var) in enumerate(x_true[1:]):
        var_dict['glm%d'%n] = glm_var


    # Package data into dict
    data = {"S": np.zeros((T_stop/dt,N)),
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim,
            'vars': var_dict}

    # Save the data so we don't have to continually simulate!
    import os
    scipy.io.savemat(os.path.join(resultsDir, 'data.mat'), data)

    # Pickle the data so we can open it more easily
    with open(os.path.join(resultsDir, 'data.pkl'),'w') as f:
        cPickle.dump(data,f)

    return data

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--dataFile", dest="dataFile", default=None,
                      help="Use this data file. If not specified, simulate from model.")

    parser.add_option("-s", "--sampleFile", dest="sampleFile", default=None,
                      help="Use this sample file, either as filename in the config directory, or as a path.")

    parser.add_option("-r", "--resultsDir", dest="resultsDir", default='.',
                      help="Save the results to this directory.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if not options.dataFile is None and not os.path.exists(options.dataFile):
        raise Exception("Invalid data file specified: %s" % options.dataFile)

    if not options.sampleFile is None and not os.path.exists(options.sampleFile):
        raise Exception("Invalid sample file specified: %s" % options.sampleFile)

    if not options.resultsDir is None and not os.path.exists(options.resultsDir):
        raise Exception("Invalid sample file specified: %s" % options.resultsDir)

    return (options, args)


if __name__ == "__main__":
    # Parse command line args
    (options, args) = parse_cmd_line_args()

    print "Initializing GLM"
    model = make_model('spatiotemporal_glm', N=2)
    glm = NetworkGlm(model)

    # Load data
    if not options.dataFile is None:
        if options.dataFile.endswith('.mat'):
            print "Loading data from %s" % options.dataFile
            #data = scipy.io.loadmat(options.dataFile)
            # Scipy's IO is weird -- we can save dicts as structs but its hard to reload them
            raise Exception('Loading from .mat file is not implemented!')
        elif options.dataFile.endswith('.pkl'):
            print "Loading data from %s" % options.dataFile
            with open(options.dataFile,'r') as f:
                data = cPickle.load(f)
        else:
            raise Exception("Unrecognized file type: %s" % options.dataFile)

    else:
        print "Generating synthetic data"
        data = generate_synth_data(glm, options.resultsDir)

    # TODO Fix this hack
    var_dict = data['vars']
    x_true = []
    x_true.append(var_dict['net'])
    for n in range(len(var_dict.keys())-1):
        x_true.append(var_dict['glm%d'%n])

    # Initialize the GLM with the data
    #x_true = data['vars']
    glm.set_data(data)

    ll_true = glm.compute_log_p(x_true)
    print "true LL: %f" % ll_true

    # Sample random initial state
    x0 = glm.sample()
    ll0 = glm.compute_log_p(x0)
    print "LL0: %f" % ll0

    x_smpls = gibbs_sample(glm, x0)
    ll_trace = np.zeros(len(x_smpls))
    for (i,x_smpl) in enumerate(x_smpls):
        ll_trace[i] = glm.compute_log_p(x_smpl)

    #print "LL_opt: %f" % ll_opt

    plot_results(glm, x_true, x_smpls)

    # Run a synthetic MCMC
