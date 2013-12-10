""" Compute the spike-triggered average as an initialization for spatiotemporal
    stimulus filters.
"""
import numpy as np

def sta(stim, data, L):
    """ Compute the spike-triggered average as an initialization for spatiotemporal
    stimulus filters.
    data : dictionary containing the following keys
           'S'  : TxN matrix of spike counts for T time bins and N neurons
           'dt' : bin size for spike matrix
           'dt_stim' : time between stimulus frames
    stim : stimulus at sampling interval data['dt_stim']
    L    : length of the STA in bins of size data['dt']
    """
    # Interpolate stimulus at the resolution of the data
    D_stim = stim.shape[1]
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
        
    # TODO Make lagged stim matrix to compute STA
    # Initialize STA
    A = np.zeros((N,L,D_stim))
    for ti in np.arange(nt):
        # if ti % 1000 == 0:
            # print "STA iter %d:" % ti
        if ti<L-1:
            stim_pad = np.concatenate((np.zeros((L-ti-1,D_stim)),istim[:ti+1,:]))
        else:
            stim_pad = istim[ti-L+1:ti+1,:]

        assert stim_pad.shape == (L,D_stim)

        A += np.tensordot(np.reshape(S[ti,:],(N,1)), 
                          np.reshape(stim_pad, (1,L,D_stim)),
                          axes=[1,0])

    # Normalize
    for n in np.arange(N):
        A[n,:,:] /= np.sum(S[:,n])

    # Flip the result so that the first column is the most recent stimulus frame
    A = A[:,::-1,:]
    return A
