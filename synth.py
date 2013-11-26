from glm import *
from models import *

def plot_results(glm, x_trues, x_opts):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    import matplotlib
    matplotlib.use('Agg')       # To enable saving remotely
    import matplotlib.pyplot as plt

    N = len(glm.glms)
    # Plot the stimulus tuning curve
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(glm.glms[n].bkgd_model.f_stim_resp(x_trues[n]),'b')
        plt.hold(True)
        plt.plot(glm.glms[n].bkgd_model.f_stim_resp(x_opts[n]),'r')
        plt.title('GLM[%d]: Stim Response' % n)

    f.savefig('stim_resp.pdf')

    # Plot the impulse responses
    f = plt.figure()
    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            plt.subplot(N,N,n_pre*N+n_post + 1)
            plt.plot(glm.glms[n_post].imp_model.imp_models[n_pre].f_impulse(x_trues[n_post]),'b')
            plt.hold(True)
            plt.plot(glm.glms[n_post].imp_model.imp_models[n_pre].f_impulse(x_opts[n_post]),'r')
            plt.title('Imp Response %d->%d' % (n_pre,n_post))

    f.savefig('imp_resp.pdf')

    # Infer the firing rates
    f = plt.figure()
    for n in np.arange(N):
        plt.subplot(1,N,n+1)
        plt.plot(glm.glms[n].f_lam(x_trues[n]),'b')
        plt.hold(True)
        plt.plot(glm.glms[n].f_lam(x_opts[n]),'r')

        # TODO Plot the spike times
        plt.title('Firing rate %d' % n)
    f.savefig('firing_rate.pdf')


if __name__ == "__main__":
    # Test
    print "Initializing GLM"
    T_start = 0
    T_stop = 1000
    dt = 0.1
    model = SimpleModel

    dt_stim = 10
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

    lam_true = glm.glms[0].f_lam(x_true[0])

    # Save the data
    data["lam"] = lam_true
    import scipy.io
    scipy.io.savemat("data.mat", data)

    ll_true = glm.f_lp(x_true)
    print "true LL: %f" % ll_true

    # Sample random initial state
    x0 = glm.sample()
    ll0 = glm.f_lp(x0)
    print "LL0: %f" % ll0

    x_opt = glm.fit(x0)
    ll_opt = glm.f_lp(x_opt)
    print "LL_opt: %f" % ll_opt

    plot_results(glm, x_true, x_opt)