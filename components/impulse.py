import theano
import theano.tensor as T
from utils.basis import *

class BasisImpulse:
    """ Impulse response function for a single connection. The impulse
        is decomposed into the sum of basis functions. 
        We impose no constraints on the basis functions,
        i.e. they are unnormalized and can be positive or negative.
    """
    
    def __init__(self, n_pre, n_post, vars, D, offset, **kwargs):
        """ Initialize the filtered stim model
        """
        self.n_pre = n_pre
        self.n_post = n_post

        prms = {'T_max' : 10,
                'basis' : None,
                'basis_type' : 'cosine',
                'mu_w' : 0,
                'sig_w' : 1}
        prms.update(**kwargs)
        
        self.mu_w = prms['mu_w']
        self.sig_w = prms['sig_w']
                
        # Create a basis for the stimulus response
        if prms['basis'] is None:
            self.basis = create_basis(**prms)
        else:
            self.basis = prms['basis']

        # Compute the number of parameters
        (_,B) = self.basis.shape
        self.n_vars = B*D

        # Store stimulus in shared memory
        self.ir = theano.shared(value=np.zeros((1,self.n_vars)))

        self.w_ir = vars[offset:offset+self.n_vars]

        # Log probability
        self.log_p = -0.5/self.sig_w**2 *T.sum((self.w_ir-self.mu_w)**2)

        # Expose outputs to the Glm class
        self.I_ir = T.dot(self.ir,self.w_ir)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_ir  = theano.function([vars],self.I_ir)

        # A function handle for the impulse response
        impulse = T.dot(self.basis,self.w_ir)
        self.f_impulse = theano.function([vars],impulse)

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Compute number of params
#         self.n_vars.set_value(data["stim"].shape[1]*self.basis.shape[1])

        # Project the presynaptic spiking onto the basis
        nT = data["S"].shape[0]
        cstim = convolve_with_basis(np.reshape(data["S"][:,self.n_pre], (nT,1)),
                                    self.basis)

        # Flatten this manually (there's surely a way to do this with numpy)
        (nT,D,B) = cstim.shape
        fstim = np.empty((nT,D*B))
        for d in np.arange(D):
            for b in np.arange(B):
                fstim[:,d*B+b] = cstim[:,d,b]
                
        self.ir.set_value(fstim)

    def sample(self):
        """
        return a sample of the variables
        """
        w_imp = self.mu_w + self.sig_w * np.random.randn(self.n_vars)
        return w_imp

    def params(self):
        return {'basis' : self.basis}