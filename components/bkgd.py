import theano
import theano.tensor as T
from utils.basis import *

def create_bkgd_component(model, vars, offset):
    type = model['bkgd']['type'].lower()
    if type == 'basis':
        bkgd = BasisStimulus(model, vars, offset)
    elif type == 'spatiotemporal':
        bkgd = SpatiotemporalStimulus(model, vars, offset)
    else:
        raise Exception("Unrecognized backgound model: %s" % type)
    return bkgd


class BasisStimulus:
    """ Filter the stimulus and expose the filtered stimulus
    """

    def __init__(self, model, vars, v_offset):
        """ Initialize the filtered stim model
        """

        self.prms = model['bkgd']
        self.mu = self.prms['mu']
        self.sigma = self.prms['sigma']

        # Create a basis for the stimulus response
        self.basis = create_basis(self.prms['basis'])
        (_,B) = self.basis.shape

        # The basis is interpolated once the data is specified
        self.ibasis = theano.shared(value=np.zeros((2,B)))

        # Compute the number of parameters
        self.n_vars = B*self.prms['D_stim']

        # Store stimulus in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,self.n_vars)))


        self.w_stim = vars[v_offset:v_offset+self.n_vars]
        
        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_stim-self.mu)**2)
        
        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim,self.w_stim)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_stim  = theano.function([vars],self.I_stim)
        
        # A function handle for the stimulus response
        stim_resp = T.dot(self.ibasis,self.w_stim)
        self.f_stim_resp = theano.function([vars],stim_resp)
        
    def get_state(self, vars):
        """ Get the stimulus response
        """
        return {'stim' : self.f_stim_resp(*vars)}
        
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate stimulus at the resolution of the data
        dt = data['dt']
        dt_stim = data['dt_stim']
        t = np.arange(0, data['T'], dt)
        t_stim = np.arange(0, data['T'], dt_stim)
        stim = np.zeros((len(t), self.prms['D_stim']))
        for d in np.arange(self.prms['D_stim']):
            stim[:, d] = np.interp(t,
                                   t_stim,
                                   data['stim'][:, d])

        # Interpolate basis at the resolution of the data
        (L,B) = self.basis.shape
        t_int = np.linspace(0,1,self.prms['dt_max']/dt)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])
            # Keep the total L1 norm the same
            #ibasis[:,b] /= (np.sum(ibasis[:,b])/np.sum(self.basis[:,b]))

        # Normalize so that the interpolated basis has volume 1
        ibasis = ibasis / self.prms['dt_max']
        self.ibasis.set_value(ibasis)

        # Project the stimulus onto the basis
        cstim = convolve_with_basis(stim, ibasis)

        # Flatten this manually (there's surely a way to do this with numpy)
        (nT,D,B) = cstim.shape 
        fstim = np.empty((nT,D*B))
        for d in np.arange(D):
            for b in np.arange(B):
                fstim[:,d*B+b] = cstim[:,d,b]
        self.stim.set_value(fstim)

    def sample(self):
        """
        return a sample of the variables
        """
        w_stim = self.mu + self.sigma * np.random.randn(self.n_vars)
        return w_stim


class SpatiotemporalStimulus:
    """ Filter the stimulus with a 2D spatiotemporal filter. Approximate the
        spatiotemporal filter with a rank-2 approximation, namely f(t,x) = f(t)*f(x)'.

    """

    def __init__(self, model, vars, v_offset):
        """ Initialize the filtered stim model
        """

        self.prms = model['bkgd']
        self.mu = self.prms['mu']
        self.sigma = self.prms['sigma']

        # Create a basis for the stimulus response
        self.spatial_basis = create_basis(self.prms['spatial_basis'])
        (_,Bx) = self.spatial_basis.shape

        self.temporal_basis = create_basis(self.prms['temporal_basis'])
        (_,Bt) = self.temporal_basis.shape

        # The basis is interpolated once the data is specified
        #self.ibasisx = theano.shared(value=np.zeros((2,Bx)))
        self.ibasist = theano.shared(value=np.zeros((2,Bt)))

        # Compute the number of parameters
        self.n_vars = Bx+Bt

        # Store stimulus (after projection onto the spatiotemporal basis, in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,Bx*Bt)))

        # Get the two factors of the stimulus response
        self.w_x = vars[v_offset:v_offset+Bx]
        v_offset += Bx
        self.w_t = vars[v_offset:v_offset+Bt]
        # Compute the number of parameters
        self.n_vars = Bx+Bt

        # The weights are an outer product of the two factors
        # TODO Check that reshaping the weights matches the reshaping of the stimulus
        self.w_stim = T.reshape(T.dot(self.w_t, T.transpose(self.w_x)),
                                (Bx*Bt,))

        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_stim-self.mu)**2)

        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim,self.w_stim)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_stim  = theano.function([vars],self.I_stim)

        # A function handle for the stimulus response
        #stim_resp = T.dot(self.ibasis,self.w_stim)
        #self.f_stim_resp = theano.function([vars],stim_resp)

    def get_state(self, vars):
        """ Get the stimulus response
        """
        #return {'stim' : self.f_stim_resp(*vars)}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate stimulus at the resolution of the data
        dt = data['dt']
        dt_stim = data['dt_stim']
        t = np.arange(0, data['T'], dt)
        nt = len(t)
        t_stim = np.arange(0, data['T'], dt_stim)
        stim = np.zeros((nt, self.prms['D_stim']))
        for d in np.arange(self.prms['D_stim']):
            stim[:, d] = np.interp(t,
                                   t_stim,
                                   data['stim'][:, d])

        # TODO Interpolate in spatial dimension as well?

        # Interpolate basis at the resolution of the data
        (L,Bt) = self.temporal_basis.shape
        t_int = np.linspace(0,1,self.prms['dt_max']/dt)
        t_bas = np.linspace(0,1,L)
        ibasis_t = np.zeros((len(t_int), Bt))
        for b in np.arange(Bt):
            ibasis_t[:,b] = np.interp(t_int, t_bas, self.temporal_basis[:,b])
            # Keep the total L1 norm the same
            #ibasis[:,b] /= (np.sum(ibasis[:,b])/np.sum(self.basis[:,b]))

        # Normalize so that the interpolated basis has volume 1
        ibasis_t = ibasis_t / self.prms['dt_max']

        # Take all pairs of temporal and spatial basis vectors
        (_,Bt) = ibasis_t.shape
        (_,Bx) = self.spatial_basis.shape


        # Flatten this manually (there's surely a way to do this with numpy)
        fstim = np.empty((nt,Bx*Bt))
        for bx in np.arange(Bx):
            for bt in np.arange(Bt):
                bas = ibasis_t[:,bt] * self.spatial_basis[:,bx].T
                fstim[:,bx*Bt+bt] = convolve_with_2d_basis(stim, bas)
        self.stim.set_value(fstim)

    def sample(self):
        """
        return a sample of the variables
        """
        w_stim = self.mu + self.sigma * np.random.randn(self.n_vars)
        return w_stim
