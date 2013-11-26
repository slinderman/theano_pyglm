import theano
import theano.tensor as T
from utils.basis import *

def create_bkgd_component(model, vars, offset):
    type = model['bkgd']['type'].lower()
    if type == 'basis':
        bkgd = BasisStimulus(model, vars, offset)
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
        self.log_p = -0.5/self.sigma**2 *T.sum(T.pow(self.w_stim-self.mu,2))
        
        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim,self.w_stim)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_stim  = theano.function([vars],self.I_stim)
        
        # A function handle for the stimulus response
        stim_resp = T.dot(self.ibasis,self.w_stim)
        self.f_stim_resp = theano.function([vars],stim_resp)
        
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
