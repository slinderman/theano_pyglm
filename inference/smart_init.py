import numpy as np

from components.bkgd import BasisStimulus, SpatiotemporalStimulus

from utils.sta import sta
from utils.basis import project_onto_basis

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
