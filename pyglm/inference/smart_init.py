import numpy as np

from pyglm.components.bkgd import BasisStimulus, SpatiotemporalStimulus, SharedTuningCurveStimulus
from pyglm.utils.sta import sta
from pyglm.utils.basis import project_onto_basis

def initialize_with_data(population, data, x0, Ns=None):
    """ Initialize the parameters x0 with smart draws from the data
    """
    initialize_stim_with_sta(population, data, x0, Ns=Ns)
    initialize_with_dense_graph(population, data, x0)
    # initialize_with_no_coupling(population, data, x0)

def initialize_with_dense_graph(population, data, x0):
    """ Initialize with a dense network
    """
    if 'A' in x0['net']['graph']:
        x0['net']['graph']['A'] = np.ones_like(x0['net']['graph']['A'])

def initialize_with_no_coupling(population, data, x0):
    for glm in x0['glms']:
        if 'w_ir' in glm['imp']:
            glm['imp']['w_ir'] = np.zeros_like(glm['imp']['w_ir'])

        if 'bias' in glm['bias']:
            glm['bias']['bias'] = 0

def initialize_stim_with_sta(population, data, x0, Ns=None):
    """ Initialize the stimulus response parameters with the STA
        TODO: Move this to the bkgd model once we have decided upon the
        correct function signature
    """
    if Ns is None:
        Ns = np.arange(population.N)

    if isinstance(Ns,int):
        Ns = [Ns]

    temporal = isinstance(population.glm.bkgd_model, BasisStimulus)
    spatiotemporal = isinstance(population.glm.bkgd_model, SpatiotemporalStimulus)
    
    if not (temporal or spatiotemporal):
        return

    # Compute the STA
    print "Initializing with the STA"
    # TODO Fix these super hacky calls
    if temporal:
        s = sta(data['stim'],
                data,
                population.glm.bkgd_model.ibasis.get_value().shape[0],
                Ns=Ns)
    elif spatiotemporal:
        s = sta(data['stim'],
                data,
                population.glm.bkgd_model.ibasis_t.get_value().shape[0],
                Ns=Ns)
        
    else:
       # We're only initializing the basis function stim models now
       return

    # Compute the initial weights for each neuron
    for i,n in enumerate(Ns):
        sn = np.squeeze(s[i,:,:])
        if sn.ndim == 1:
            sn = np.reshape(sn, [sn.size, 1])

        if spatiotemporal:
           # Factorize the STA into a spatiotemporal filter using SVD
           # CAUTION! Numpy svd returns V transpose whereas Matlab svd returns V!
           U,Sig,V = np.linalg.svd(sn)
           f_t = U[:,0] * np.sqrt(Sig[0])
           f_x = V[0,:] * np.sqrt(Sig[0])

           # Project this onto the spatial and temporal bases
           w_t = project_onto_basis(f_t, population.glm.bkgd_model.ibasis_t.get_value())
           w_x = project_onto_basis(f_x, population.glm.bkgd_model.ibasis_x.get_value())

           # Flatten into 1D vectors
           w_t = np.ravel(w_t)
           w_x = np.ravel(w_x)
           
           x0['glms'][n]['bkgd']['w_x'] = w_x
           x0['glms'][n]['bkgd']['w_t'] = w_t
        elif temporal:
            # Only using a temporal filter
            D_stim = sn.shape[1]
            B = population.glm.bkgd_model.ibasis.get_value().shape[1]
            
            # Project this onto the spatial and temporal bases
            w_t = np.zeros((B*D_stim,1))
            for d in np.arange(D_stim):
                w_t[d*B:(d+1)*B] = project_onto_basis(sn[:,d], 
                                                      population.glm.bkgd_model.ibasis.get_value())
            # Flatten into a 1D vector 
            w_t = np.ravel(w_t)
            x0['glms'][n]['bkgd']['w_stim'] = w_t    


def initialize_locations_by_correlation(population, x0, maxlag=300):
    """
    Initialize the locations of a shared tuning curve background model
    by setting each neuron's location to the stimulus location where it is
    most correlated.
    """
    if not isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
        return

    location_model = population.glm.bkgd_model.locations
    assert len(population.data_sequences) > 0
    data = population.data_sequences[-1]
    stim = data['stim']
    spks = data['S']


    # Downsample the spikes to the resolution of the stimulus
    Tspks, N = spks.shape
    Tstim = stim.shape[0]
    # Flatten higher dimensional stimuli
    if stim.ndim == 3:
        stimf = stim.reshape((Tstim, -1))
    else:
        stimf = stim

    # Downsample spikes to bins of same size as stimulus
    # ds = Tspks // Tstim
    # spks_ds = spks.reshape((Tstim, ds, N)).sum(axis=1)
    # mean_ds = spks_ds.mean(axis=0)
    #
    # # Compute the correlation with each stimulus entry.
    # # Since the stimulus is 1D, this is just a dot product
    # # The result is a D x N matrix
    # stimc = stimf-meanf
    # spksc = spks_ds-mean_ds
    #
    # corr = np.dot(stimc.T, spksc)
    # for lag in range(1,maxlag):
    #     corr += np.dot(stimc[:-lag,:].T, spksc[lag:,:])
    #
    # locs = np.argmax(np.abs(corr), axis=0)
    # locs = locs.reshape((N,1))
    #
    s = sta(stimf,
            data,
            maxlag,
            Ns=np.arange(N))


    # Get the total power in each pixel by summing across time
    s_total = np.abs(s).sum(axis=1)

    locs = np.argmax(s_total, axis=1)
    locs = locs.reshape((N,1))

    L0 = None
    if stim.ndim == 2:
        L0 = locs
    elif stim.ndim == 3:
        L0 = np.zeros((N,2))
        locsi, locsj = np.unravel_index(locs, stim.shape[1:])
        L0[:,0], L0[:,1] = locsi.ravel(), locsj.ravel()
    x0['latent'][location_model.name]['L'] = L0.ravel().astype(np.int)


    # import matplotlib.pyplot as plt
    # plt.figure()
    #
    # # Get the limits by finding the max absolute value per neuron
    # s_max = np.amax(np.abs(s.reshape((N,-1))), axis=1)
    #
    # lags_to_plot = np.arange(maxlag, step=50)
    # for n in range(N):
    #     for j,l in enumerate(lags_to_plot):
    #         plt.subplot(N,len(lags_to_plot), n*len(lags_to_plot) + j + 1)
    #         plt.title('N: %d, Lag: %d' % (n, j))
    #         plt.imshow(np.kron(s[n,l,:].reshape((5,5)), np.ones((10,10))),
    #                    vmin=-s_max[n], vmax=s_max[n],
    #                    cmap='RdGy')
    #         if j == len(lags_to_plot) - 1:
    #             plt.colorbar()
    #
    # plt.savefig('sta.pdf')