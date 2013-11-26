from glm import *
from models import *

def plot_results(glms, x_trues, x_opts):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    import matplotlib
    matplotlib.use('Agg')       # To enable saving remotely
    import matplotlib.pyplot as plt

    N = len(glms)
    # Plot the stimulus tuning curve
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(glms[n].stim_model.f_stim_resp(x_trues[n]),'b')
        plt.hold(True)
        plt.plot(glms[n].stim_model.f_stim_resp(x_opts[n]),'r')
        plt.title('GLM[%d]: Stim Response' % n)

    f.savefig('stim_resp.pdf')

    # Plot the impulse responses
    f = plt.figure()
    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            plt.subplot(N,N,n_pre*N+n_post + 1)
            plt.plot(glms[n_post].network.imp_models[n_pre].f_impulse(x_trues[n_post]),'b')
            plt.hold(True)
            plt.plot(glms[n_post].network.imp_models[n_pre].f_impulse(x_opts[n_post]),'r')
            plt.title('Imp Response %d->%d' % (n_pre,n_post))

    f.savefig('imp_resp.pdf')

    # Infer the firing rates
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(glms[n].f_lam(x_trues[n]),'b')
        plt.hold(True)
        plt.plot(glms[n].f_lam(x_opts[n]),'r')
        plt.title('Firing rate %d' % n)
    f.savefig('firing_rate.pdf')


if __name__ == "__main__":
    # Test
    print "Initializing GLM"
    T_max = 10000
    dt_stim = 100
    dt = 1
    D_stim = SimpleModel['bkgd']['D_stim']
    N = SimpleModel['N']

    glm = NetworkGlm(SimpleModel)

    # Generate random stimulus
    print "Generating random data"
    stim = np.random.randn(T_max/dt_stim,D_stim)
    istim = np.zeros((T_max/dt,D_stim))
    for d in np.arange(D_stim):
        istim[:,d] = np.interp(np.arange(0,T_max,dt),
                               np.arange(0,T_max,dt_stim),
                               stim[:,d])

    # Initialize the GLMs with the stimulus
    data = {"S" : np.zeros((T_max/dt,N)),
            "N" : N,
            "dt" : dt,
            "T" : np.float(T_max),
            "stim" : istim}
    for n in np.arange(N):
        glms[n].set_data(data)

    # Simulate spikes
    S,X = simulate(glms,x_trues,istim, dt)

    # Put the spikes into a data dictionary
    data = {"S" : S,
            "X" : X,
            "N" : N,
            "dt" : dt,
            "T" : np.float(T_max),
            "stim" : istim}

    for n in np.arange(N):
        glms[n].set_data(data)

    lam_true = glms[0].f_lam(x_trues[0])

    # Save the data
    data["lam"] = lam_true
    import scipy.io
    scipy.io.savemat("data.mat", data)
    scipy.io.savemat('params.mat', glms[0].params())

    ll_true = glms[0].f_lp(x_trues[0])
    print "true LL: %f" % ll_true

    # Sample random initial state

    x_opts = []
    for n in np.arange(N):
        print "Fitting GLM %d" % n
        x0 = glms[n].sample()
        ll0 = glms[n].f_lp(x0)
        print "LL0: %f" %  ll0

        x_opt = fit_glm(glms[n],x0)

        print "x_true:\t%s" % str(x_trues[n])
        print "x_inf:\t%s" % str(x_opt)
        print "Opt LL: %f" % np.float(glms[n].f_lp(x_opt))

        x_opts.append(x_opt)

    plot_results(glms, x_trues, x_opts)