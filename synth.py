from glm import *
from models.simple_weighted_two_neuron import *
from models.simple_two_neuron import *
from inference.map import map_estimate

def plot_results(glm, x_trues, x_opts):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    import matplotlib
    matplotlib.use('Agg')       # To enable saving remotely
    import matplotlib.pyplot as plt

    true_state = glm.get_state(x_trues)
    opt_state = glm.get_state(x_opts)

    N = len(glm.glms)
    
    # Plot the inferred connectivity matrix
    f = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(true_state['net'], extent=[0,1,0,1])
    plt.colorbar()
    plt.title('True Network')
    plt.subplot(1,2,2)
    plt.imshow(opt_state['net'], extent=[0,1,0,1])
    plt.colorbar()
    plt.title('Inferred Network')

    f.savefig('conn.pdf')
    
    # Plot the stimulus tuning curve
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(true_state[n]['stim'],'b')
        plt.hold(True)
        plt.plot(opt_state[n]['stim'],'--r')
        plt.title('GLM[%d]: Stim Response' % n)

    f.savefig('stim_resp.pdf')

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
            plt.title('Imp Response %d->%d' % (n_pre,n_post))

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


if __name__ == "__main__":
    # Test
    print "Initializing GLM"
    T_start = 0
    T_stop = 10000
    dt = 1
    model = SimpleTwoNeuronModel

    dt_stim = 100
    D_stim = model['bkgd']['D_stim']
    N = model['N']

    glm = NetworkGlm(model)
    x_true = glm.sample()

    # Generate random stimulus
    print "Generating random data"
    stim = np.random.randn(T_stop/dt_stim,D_stim)

    # Initialize the GLMs with the stimulus
    data = {"S": np.zeros((T_stop/dt,N)),
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim}
    glm.set_data(data)
    
    # Simulate spikes
    S,X = glm.simulate(x_true, (T_start,T_stop), dt)
    
    # Put the spikes into a data dictionary
    data = {"S": S,
            "X": X,
            "N": N,
            "dt": dt,
            "T": np.float(T_stop),
            "stim": stim,
            'dt_stim': dt_stim}
    glm.set_data(data)

    ll_true = glm.compute_log_p(x_true)
    print "true LL: %f" % ll_true

    # Sample random initial state
    x0 = glm.sample()
    ll0 = glm.compute_log_p(x0)
    print "LL0: %f" % ll0

    x_opt = map_estimate(glm, x0)
    ll_opt = glm.compute_log_p(x_opt)
    
    print "LL_opt: %f" % ll_opt

    plot_results(glm, x_true, x_opt)
