import numpy as np
import scipy
import os

def create_basis(prms):
    """ Create a basis for impulse response functions
    """
    type = prms['type'].lower()
    if type == 'exp':
        basis = create_exp_basis(prms)
    elif type == 'cosine':
        basis = create_cosine_basis(prms)
    elif type == 'file':
        if os.path.exists(prms["fname"]):
            basis = load_basis_from_file(prms['fname'])
    else:
        raise Exception("Unrecognized basis type: %s", type)
    return basis

def load_basis_from_file(prms):
    """
    Load a basis from a file
    """
    fname = prms["fname"]
    if not os.path.exists(prms["fname"]):
        raise Exception("Invalid basis file: %s", fname)
    bas_dict = scipy.io.loadmat(fname)
    if "basis" not in bas_dict.keys():
        raise Exception("Invalid basis file: %s", fname)
    
    basis = bas_dict["basis"]
    
    #if T_max is not None:
    #    # Interpolate the basis  at T_max evenly spaced points
    #    (t_bas,n_bas) = basis.shape
    #    cur_tt = np.linspace(0,1,t_bas)
    #    new_tt = np.linspace(0,1,T_max)
    #
    #    new_basis = np.zeros((T_max,n_bas))
    #    for b in np.arange(n_bas):
    #        new_basis[:,b] = np.interp(new_tt,
    #                                   cur_tt,
    #                                   basis[:,b])
    #
    #    basis = new_basis
    return basis
   
def create_cosine_basis(prms):
    """
    Create a basis of raised cosine tuning curves
    """
    # Set default parameters. These can be overriden by kwargs
    #prms = {'n_eye' : 0,
    #        'n_cos' : 3,
    #        'a': 1.0/120,
    #        'b': 0.5,
    #        'orth' : False,
    #        'norm' : True}
    #prms.update(kwargs)
    n_pts = 100             # Number of points at which to evaluate the basis
    n_cos = prms['n_cos']   # Number of cosine basis functions
    n_eye = prms['n_eye']   # Number of identity basis functions
    n_bas = n_eye + n_cos
    basis = np.zeros((n_pts,n_bas))
    
    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)
    
    # The remaining basis elements are raised cosine functions with peaks
    # logarithmically warped between [n_eye*dt:dt_max].
    
    a = prms['a']                       # Scaling in log time
    b = prms['b']                       # Offset in log time
    nlin = lambda t: np.log(a*t+b)      # Nonlinearity
    u_ir = nlin(np.arange(n_pts))       # Time in log time
    ctrs = u_ir[np.floor(np.linspace(n_eye+1,(n_pts/2.0),n_cos)).astype(np.int)]
    if len(ctrs) == 1:
        w = ctrs/2
    else:
        w = (ctrs[-1]-ctrs[0])/(n_cos-1)    # Width of the cosine tuning curves
    
    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda u,c,w: (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(u-c)*np.pi/w/2.0)))+1)/2.0
    for i in np.arange(n_cos):
        basis[:,n_eye+i] = basis_fn(u_ir,ctrs[i],w)
    
    
    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']: 
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)
    
    return basis

def create_exp_basis(prms):
    """
    Create a basis of exponentially decaying functions
    """
    # Set default parameters. These can be overriden by kwargs

    # Default to a raised cosine basis
    n_pts = 100             # Number of points at which to evaluate the basis
    n_exp = prms['n_exp']   # Number of exponential basis functions
    n_eye = prms['n_eye']   # Number of identity basis functions
    n_bas = n_eye + n_cos
    basis = np.zeros((n_pts,n_bas))
    
    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)
    
    # The remaining basis elements are exponential functions with logarithmically
    # spaced time constants
    taus = np.logspace(1, n_pts/2, n_exp)
    
    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda t,tau: np.exp(-t/tau)
    for i in np.arange(n_exp):
        basis[:,n_eye+i] = basis_fn(np.arange(n_pts),taus[i])
    
    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']: 
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)
    
    return basis

def create_gaussian_basis(prms):
    """
    Create a basis of Gaussian bumps.
    This is primarily for spatial filters.
    """
    # Set default parameters. These can be overriden by kwargs

    # Default to a raised cosine basis
    n_pts = 100             # Number of points at which to evaluate the basis
    n_gauss = prms['n_gauss']   # Number of exponential basis functions
    n_eye = prms['n_eye']   # Number of identity basis functions
    n_bas = n_eye + n_gauss
    basis = np.zeros((n_pts,n_bas))

    # The first n_eye basis elements are identity vectors in the first time bins
    basis[:n_eye,:n_eye] = np.eye(n_eye)

    # The remaining basis functions are Gaussian bumps at equally spaced points
    mus = np.arange(n_gauss)
    sigma = 1.0

    # Basis function is a raised cosine centered at c with width w
    basis_fn = lambda t,mu,sig: np.exp(-0.5/sig*(t-mu)**2)
    for i in np.arange(n_gauss):
        basis[:,i] = basis_fn(np.arange(n_pts),mus[i])

    # Orthonormalize basis (this may decrease the number of effective basis vectors)
    if prms['orth']:
        basis = scipy.linalg.orth(basis)
    if prms['norm']:
        # Normalize such that \int_0^1 b(t) dt = 1
        basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)
    basis = basis / np.tile(np.sqrt(np.sum(basis**2,axis=0)), [n_pts,1])

    return basis

def convolve_with_basis(stim, basis):
    """ Project stimulus onto a basis. 
    :param stim   TxD matrix of inputs. 
                  T is the number of time bins 
                  D is the number of stimulus dimensions.
    :param basis  RxB basis matrix
                  R is the length of the impulse response
                  B is the number of bases
    
    :rtype TxDxB tensor of stimuli convolved with bases
    """
    (T,D) = stim.shape
    (R,B) = basis.shape
    
    import scipy.signal as sig
    
    # First, by convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis = np.vstack((np.zeros((1,B)),basis))
    
    # Initialize array for filtered stimulus
    fstim = np.empty((T,D,B))
    
    # Compute convolutions
    for b in np.arange(B):
        assert np.all(np.isreal(stim))
        assert np.all(np.isreal(basis[:,b]))
        fstim[:,:,b] = sig.convolve2d(stim, 
                                      np.reshape(basis[:,b],[R+1,1]), 
                                      'full')[:T,:]
    
    return fstim

def convolve_with_2d_basis(stim, basis):
    """ Project stimulus onto a basis.
    :param stim   TxD matrix of inputs.
                  T is the number of time bins
                  D is the number of stimulus dimensions.
    :param basis  RxD basis matrix
                  R is the length of the impulse response
                  D is the number of stimulus dimensions.

    :rtype Tx1 vector of stimuli convolved with the 2D basis
    """
    (T,D) = stim.shape
    (R,Db) = basis.shape
    assert D==Db, "Spatial dimension of basis must match spatial dimension of stimulus."

    import scipy.signal as sig

    # First, by convention, the impulse responses are apply to times
    # (t-R:t-1). That means we need to prepend a row of zeros to make
    # sure the basis remains causal
    basis = np.vstack((np.zeros((1,D)),basis))

    # Flip the spatial dimension for convolution
    # TODO Check that temporal filter is flipped by convention
    basis = basis[:,::-1]

    # Compute convolution
    # TODO Performance can be improved for rank 2 filters
    assert np.all(np.isreal(stim))
    assert np.all(np.isreal(basis))
    fstim = sig.convolve2d(stim,basis,'full')

    # Only keep the first T time bins and the D-th spatial vector
    # This is the only vector for which the filter and stimulus completely overlap
    return fstim[:T,D]

def project_onto_basis(f, basis):
        """
        Project the function f onto the basis.
        :param f     Rx1 function
        :param basis RxB basis
        :rtype Bx1 vector of basis coefficients 
        """
        (R,B) = basis.shape
        assert np.size(f)==R, "Function is not the same size as the basis!"
        
        f = np.reshape(f,[R,1])
        
        # Regularize the projection
        Q = 1*np.eye(B)
        
        beta = np.dot(np.dot(scipy.linalg.inv(np.dot(obasis.T,obasis)+Q), obasis.T),f)
               
        return beta