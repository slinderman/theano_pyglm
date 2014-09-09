""" Compute the spike-triggered average as an initialization for spatiotemporal
    stimulus filters.
"""
import numpy as np

def sta(stim, data, L, Ns=None):
    """ Compute the spike-triggered average as an initialization for spatiotemporal
    stimulus filters.
    data : dictionary containing the following keys
           'S'  : TxN matrix of spike counts for T time bins and N neurons
           'dt' : bin size for spike matrix
           'dt_stim' : time between stimulus frames
    stim : stimulus at sampling interval data['dt_stim']
    L    : length of the STA in bins of size data['dt']
    """

    # import pdb; pdb.set_trace()
    D_stim = stim.shape[1]
    # Compute the STA in chunks
    maxchunk = 1e8
    chunksz = maxchunk//D_stim//L

    # Interpolate stimulus at the resolution of the data
    S = data['S']
    (nt,N) = S.shape
    dt = data['dt']
    dt_stim = data['dt_stim']
    t = dt * np.arange(nt)
    t_stim = dt_stim * np.arange(stim.shape[0])
    istim = np.zeros((nt, D_stim))
    for d in np.arange(D_stim):
        istim[:, d] = np.interp(t,
                                t_stim,
                                stim[:, d])

        # Correct for bias introduced by interpolating the stimulus
        # Assuming the original stimulus was Gaussian white noise!
        # Each stimulus frame is repeated (dt_stim/dt) times, so 
        # divide by this to keep the stimulus * weight dot product the same
        istim[:,d] /= (dt_stim/dt)

    # Pad istim with L zeros
    istim = np.vstack((np.zeros((L,D_stim)),
                       istim))


    A = np.zeros((len(Ns),L,D_stim))

    # Only compute STA for the specified neurons Ns
    if Ns is None:
        Ns = range(N)
    # Make sure Ns is a list
    if isinstance(Ns, int):
        Ns = [Ns]

    for chunk in np.arange(np.ceil(float(nt)/chunksz)):
        print "Chunk %d" % chunk
        start = chunk*chunksz
        end = min((chunk+1)*chunksz, nt)

        # Make lagged stim matrix to compute STA
        stim_lag = np.zeros((end-start,L*D_stim))
        # stim_lag[:,:D_stim] = istim[start:end,:]
        for l in np.arange(L):
            stim_lag[:,(l*D_stim):((l+1)*D_stim)] = \
                istim[L+start-l:L+end-l, :]
            # np.vstack((np.zeros((l,D_stim)),istim[start:end-l,:]))

        for i,n in enumerate(Ns):
            Sn = S[start:end,n]
            # Aflat += np.sum(stim_lag[Sn,:],axis=0)
            Aflat = np.dot(Sn, stim_lag)

            # Reshape into L x D_stim array
            # (this is more readable and less error prone than using Numpy reshape)
            for l in np.arange(L):
                A[i,l,:] += Aflat[(l*D_stim):((l+1)*D_stim)]

    # Normalize
    for i,n in enumerate(Ns):
        A[i,:,:] /= np.sum(S[:,n])

    # Flip the result so that the first column is the most recent stimulus frame
    #A = A[:,::-1,:]
    return A
