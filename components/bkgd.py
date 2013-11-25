import theano
import theano.tensor as T
from utils.basis import *

class FilteredStimulus:
    """ Filter the stimulus and expose the filtered stimulus
    """

    def __init__(self, vars, v_offset, D, **kwargs):
        """ Initialize the filtered stim model
        """
        prms = {'T_max' : 100,
                'basis_type' : 'cosine',
                'orth' : False,
                'norm' : True,
                'n_cos' : 3,
                'mu_w' : 0,
                'sig_w' : 0.1}
        prms.update(kwargs)

        self.mu_w = prms['mu_w']
        self.sig_w = prms['sig_w']

        # Create a basis for the stimulus response
        self.basis = create_impulse_basis(**prms)
        
        # Compute the number of parameters
        (_,B) = self.basis.shape
        self.n_vars = B*D 

        # Store stimulus in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,self.n_vars)))


        self.w_stim = vars[v_offset:v_offset+self.n_vars]
        
        # Log probability
        self.log_p = -0.5/self.sig_w**2 *T.sum(T.pow(self.w_stim-self.mu_w,2))
        
        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim,self.w_stim)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_stim  = theano.function([vars],self.I_stim)
        
        # A function handle for the stimulus response
        stim_resp = T.dot(self.basis,self.w_stim)
        self.f_stim_resp = theano.function([vars],stim_resp)
        
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Project the stimulus onto the basis
        cstim = convolve_with_basis(data["stim"],
                                    self.basis)

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
        w_stim = self.mu_w + self.sig_w * np.random.randn(self.n_vars)
        return w_stim

    def params(self):
        return {'basis' : self.basis}