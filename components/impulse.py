import theano
import theano.tensor as T
from utils.basis import *
from component import Component

def create_impulse_component(model, vars, offset, n_post):
    return ConcatenatedImpulses(model, vars, offset, n_post)

class ConcatenatedImpulses(Component):
    """ Encapsulate all impulses into one object
    """
    def __init__(self, model, vars, v_offset, n):
        prms = model['impulse']
        #
        ## Create a basis for the stimulus response
        #self.basis = create_basis(**prms)
        ## Compute the number of parameters
        #(_,B) = self.basis.shape
        #
        # Keep track of the number of variables
        self.n_vars = 0

        # Define impulse models for each presynaptic neuron
        self.imp_models = []
        for n_pre in np.arange(model['N']):
            imp_model = create_single_impulse_response(model, vars, v_offset, n_pre, n)
            self.imp_models.append(imp_model)
            self.n_vars += imp_model.n_vars
            v_offset += imp_model.n_vars

        I_imps = map(lambda im: T.reshape(im.I_ir, (im.I_ir.size, 1)),
                     self.imp_models)
        lp_imps = map(lambda im: im.log_p, self.imp_models)

        # Concatenate impulse currents into a matrix
        self.I_imp = T.concatenate(I_imps, axis=1)
        self.log_p = T.sum(T.stack(lp_imps))

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Set data for impulse responses
        for imp_model in self.imp_models:
            imp_model.set_data(data)

    def sample(self):
        """
        return a sample of the variables
        """
        vars = []

        # Sample impulse responses
        for imp_model in self.imp_models:
            w_imp = imp_model.sample()
            vars = np.concatenate((vars,w_imp))
        return vars

    def params(self):
        return None

def create_single_impulse_response(model, vars, offset, n_pre, n_post):
    type = model['impulse']['type'].lower()
    if type == 'basis':
        imp = BasisImpulse(model,
                           n_pre, n_post,
                           vars,
                           offset
                           )
    else:
        raise Exception("Unrecognized impulse model: %s" % type)
    return imp

class BasisImpulse:
    """ Impulse response function for a single connection. The impulse
        is decomposed into the sum of basis functions. 
        We impose no constraints on the basis functions,
        i.e. they are unnormalized and can be positive or negative.
    """
    
    def __init__(self, model, n_pre, n_post, vars, offset):
        """ Initialize the filtered stim model
        """
        self.n_pre = n_pre
        self.n_post = n_post

        prms = model['impulse']
        self.mu_w = prms['mu']
        self.sig_w = prms['sigma']
                
        # Create a basis for the stimulus response
        self.basis = create_basis(**prms['basis'])

        # Compute the number of parameters
        (_,B) = self.basis.shape
        self.n_vars = B

        # Store stimulus in shared memory
        self.ir = theano.shared(value=np.zeros((2,self.n_vars)))

        self.w_ir = vars[offset:offset+self.n_vars]

        # Log probability
        self.log_p = -0.5/self.sig_w**2 *T.sum((self.w_ir-self.mu_w)**2)

        # Expose outputs to the Glm class
        self.I_ir = T.dot(self.ir, self.w_ir)

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