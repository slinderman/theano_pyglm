import theano
import theano.tensor as T
from utils.basis import *

def create_bkgd_component(model):
    type = model['bkgd']['type'].lower()
    if type == 'basis':
        bkgd = BasisStimulus(model)
    elif type == 'spatiotemporal':
        bkgd = SpatiotemporalStimulus(model)
    else:
        raise Exception("Unrecognized backgound model: %s" % type)
    return bkgd


class BasisStimulus:
    """ Filter the stimulus and expose the filtered stimulus
    """

    def __init__(self, model):
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


        self.w_stim = T.dvector('w_stim')
        
        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_stim-self.mu)**2)
        
        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim,self.w_stim)

        # Create callable functions to compute firing rate, log likelihood, and gradients
#         self.f_I_stim  = theano.function([vars],self.I_stim)
        
        # A function handle for the stimulus response
        self.stim_resp = T.dot(self.ibasis,self.w_stim)
#         self.f_stim_resp = theano.function([vars],stim_resp)
    
    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_stim): self.w_stim}
    
    def get_state(self):
        """ Get the stimulus response
        """
        return {'stim_response' : self.stim_resp}
        
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
        Lt_int = self.prms['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])
            # Keep the total L1 norm the same
            #ibasis[:,b] /= (np.sum(ibasis[:,b])/np.sum(self.basis[:,b]))

        # Normalize so that the interpolated basis has volume 1
        if self.prms['basis']['norm']:
            ibasis = ibasis / np.tile(np.sum(ibasis,0),[Lt_int,1])
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
        w = self.mu + self.sigma * np.random.randn(self.n_vars)
        return {str(self.w_stim): w}


class SpatiotemporalStimulus:
    """ Filter the stimulus with a 2D spatiotemporal filter. Approximate the
        spatiotemporal filter with a rank-2 approximation, namely f(t,x) = f(t)*f(x)'.

    """

    def __init__(self, model):
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

        # Save the filter sizes
        self.Bx = Bx
        self.Bt = Bt

        # The basis is interpolated once the data is specified
        self.ibasis_x = theano.shared(value=np.zeros((2,Bx)))
        self.ibasis_t = theano.shared(value=np.zeros((2,Bt)))

        # Compute the number of parameters
        self.n_vars = Bx+Bt

        # Store stimulus (after projection onto the spatiotemporal basis, in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,Bx*Bt)))

        # Get the two factors of the stimulus response
        self.w_x = T.dvector('w_x')
        self.w_t = T.dvector('w_t')

        # The weights are an outer product of the two factors
        w_op = T.dot(T.shape_padright(self.w_t, n_ones=1),
                     T.shape_padleft(self.w_x, n_ones=1))
        # w_op = T.dot(self.w_x, T.transpose(self.w_t))

        # Flatten the outer product to get a combined weight for each
        # pair of spatial and temporal basis functions.
        self.w_stim = T.reshape(w_op, (Bt*Bx,))

        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_x-self.mu)**2) + \
                     -0.5/self.sigma**2 *T.sum((self.w_t-self.mu)**2)

        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim, self.w_stim)


        # Create function handles for the stimulus responses
        self.stim_resp_t = T.dot(self.ibasis_t,self.w_t)        
        self.stim_resp_x = T.dot(self.ibasis_x,self.w_x)

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_x): self.w_x,
                str(self.w_t): self.w_t}
    
    def sample(self, n=None):
        """
        return a sample of the variables
        """

        w_x = self.mu + self.sigma * np.random.randn(self.Bx)
        w_t = self.mu + self.sigma * np.random.randn(self.Bt)
        return {str(self.w_x) : w_x,
                str(self.w_t) : w_t}
    
    def get_state(self):
        """ Get the stimulus response
        """
        # The filters are non-identifiable as we can negate both the
        # temporal and the spatial filters and get the same net effect.
        # By convention, choose the sign that results in the most
        # positive temporal filter.
        sign = T.sgn(T.sum(self.stim_resp_t))
        return {'stim_response_x' : sign*self.stim_resp_x,
                'stim_response_t' : sign*self.stim_resp_t}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate stimulus at the resolution of the data
        dt = data['dt']
        dt_stim = data['dt_stim']
        t = np.arange(0, data['T'], dt)
        nt = len(t)
#         t_stim = np.arange(0, data['T'], dt_stim)
        t_stim = dt_stim * np.arange(data['stim'].shape[0])
        stim = np.zeros((nt, self.prms['D_stim']))
        for d in np.arange(self.prms['D_stim']):
            stim[:, d] = np.interp(t,
                                   t_stim,
                                   data['stim'][:, d])

        # TODO Interpolate in spatial dimension as well?

        # Interpolate basis at the resolution of the data
        (Lt,Bt) = self.temporal_basis.shape
        Lt_int = self.prms['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,Lt)
        ibasis_t = np.zeros((len(t_int), Bt))
        for b in np.arange(Bt):
            ibasis_t[:,b] = np.interp(t_int, t_bas, self.temporal_basis[:,b])
        
        (Lx,Bx) = self.spatial_basis.shape
        Lx_int = self.prms['D_stim']
        x_int = np.linspace(0,1,Lx_int)
        x_bas = np.linspace(0,1,Lx)
        ibasis_x = np.zeros((len(x_int), Bx))
        for b in np.arange(Bx):
            ibasis_x[:,b] = np.interp(x_int, x_bas, self.spatial_basis[:,b])

        # Normalize so that the interpolated basis has volume 1
#         if self.prms['temporal_basis']['norm']:
#             ibasis_t = ibasis_t / self.prms['dt_max']
        # Normalize so that the interpolated basis has unit L1 norm
        if self.prms['temporal_basis']['norm']:
            ibasis_t = ibasis_t / np.tile(np.sum(ibasis_t,0),[Lt_int,1])

        # Save the interpolated bases
        self.ibasis_t.set_value(ibasis_t)
        self.ibasis_x.set_value(ibasis_x)

        # Take all pairs of temporal and spatial basis vectors
        (_,Bt) = ibasis_t.shape
        (_,Bx) = ibasis_x.shape

        # Filter the stimulus with each spatiotemporal filter combo
        # fstim = np.empty((nt,Bt,Bx))
        # for bt in np.arange(Bt):
        #     for bx in np.arange(Bx):
        #         # atleast_2d gives row vectors
        #         bas = np.dot(np.atleast_2d(ibasis_t[:,bt]).T,
        #                      np.atleast_2d(ibasis_x[:,bx]))
        #         fstim[:,bt,bx] = convolve_with_2d_basis(stim, bas)

        # Leverage low rank to speed up convolutions
        print "Convolving the stimulus with the low rank filters"
        fstim = convolve_with_low_rank_2d_basis(stim, ibasis_x, ibasis_t)

        # Permute output to get shape(T,Bt,Bx)
        assert fstim.shape == (nt,Bx,Bt)
        fstim = np.transpose(fstim, axes=[0,2,1])
        
        # Flatten the filtered stimulus 
        fstim2 = np.reshape(fstim,(nt,Bt*Bx))

        self.stim.set_value(fstim2)

