import theano
import theano.tensor as T
from utils.basis import *
from component import Component

def create_impulse_component(model):
    return LinearBasisImpulses(model)

class LinearBasisImpulses(Component):
    """ Linear impulse response functions. Here we make use of Theano's
        tensordot to sum up the currents from each presynaptic neuron.
    """
    def __init__(self, model):
        self.prms = model['impulse']

        # Number of presynaptic neurons
        self.N = model['N']

        # Get parameters of the prior
        self.mu = self.prms['mu']
        self.sigma = self.prms['sigma']

        # Create a basis for the impulse responses response
        self.basis = create_basis(self.prms['basis'])
        (_,self.B) = self.basis.shape
        # The basis is interpolated once the data is specified
        self.ibasis = theano.shared(value=np.zeros((2,self.B)))

        # Initialize memory for the filtered spike train
        self.ir = theano.shared(name='ir',
                                value=np.zeros((1,self.N,self.B)))

        # Define weights
        self.w_ir = T.dvector('w_ir')
        # Repeat them (in a differentiable manner) to create a 3-tensor
        w_ir2 = T.reshape(self.w_ir, [self.N,self.B])
        w_ir3 = T.reshape(self.w_ir, [1,self.N,self.B])

        # Make w_ir3 broadcastable in the 1st dim
        T.addbroadcast(w_ir3,0)

        #ov = T.ones((self.ir.shape[0],1))
        #w_ir_rep = T.tensordot(ov,w_ir3,axes=[1,0])

        # Take the elementwise product of the filtered stimulus and
        # the repeated weights to get the weighted impulse current along each
        # impulse basis dimension. Then sum over bases to get the
        # total coupling current from each presynaptic neurons at
        # all time points
        #self.I_imp = T.sum(self.ir*w_ir_rep, axis=2)
        self.I_imp = T.sum(self.ir*w_ir3, axis=2)
        self.log_p = T.sum(-0.5/self.sigma**2 * (self.w_ir-self.mu)**2)

        # Define a helper variable for the impulse response
        # after projecting onto the basis
        self.impulse = T.dot(w_ir2, T.transpose(self.ibasis))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_ir): self.w_ir}

    def sample(self):
        """
        return a sample of the variables
        """
        w = self.mu + self.sigma*np.random.randn(self.N*self.B)
        return {str(self.w_ir): w}

    def get_state(self):
        """ Get the impulse responses
        """
        return {'impulse' : self.impulse}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate basis at the resolution of the data
        dt = data['dt']
        (L,B) = self.basis.shape
        Lt_int = self.prms['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if self.prms['basis']['norm']:
            ibasis = ibasis / self.prms['dt_max']
        # Normalize so that the interpolated basis has unit L1 norm
#         if self.prms['basis']['norm']:
#             ibasis = ibasis / np.tile(np.sum(ibasis,0),[Lt_int,1])
        self.ibasis.set_value(ibasis)

        # Project the presynaptic spiking onto the basis
        nT,Ns = data["S"].shape
        assert Ns == self.N, "ERROR: Spike train must be (TxN) " \
                             "dimensional where N=%d" % self.N
        fS = convolve_with_basis(data["S"], ibasis)

        # Flatten this manually to be safe
        # (there's surely a way to do this with numpy)
        (nT,Nc,B) = fS.shape
        assert Nc == self.N, "ERROR: Convolution with spike train " \
                             "resulted in incorrect shape: %s" % str(fS.shape)
        self.ir.set_value(fS)

class ConcatenatedImpulses(Component):
    """ Encapsulate all impulses into one object
    """
    def __init__(self, model):
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
            imp_model = create_single_impulse_response(model, n_pre)
            self.imp_models.append(imp_model)
            self.n_vars += imp_model.n_vars

        I_imps = map(lambda im: T.shape_padright(im.I_ir, n_ones=1),
                     self.imp_models)
        lp_imps = map(lambda im: T.shape_padright(im.log_p, n_ones=1),
                      self.imp_models)

        # Concatenate impulse currents into a matrix
        self.I_imp = T.concatenate(I_imps, axis=1)
        self.log_p = T.sum(T.concatenate(lp_imps, axis=0))
        
#         self.f_I_imp = theano.function([vars], self.I_imp)

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        v = {}
        for (n,im) in enumerate(self.imp_models):
            v['ir%d'%n] = im.get_variables()
        return v

    def sample(self):
        """
        return a sample of the variables
        """
        v = {}
        for (n,im) in enumerate(self.imp_models):
            v['ir%d'%n] = im.sample()
        return v

    def get_state(self, vars):
        """ Get the impulse responses 
        """
        imps = map(lambda im: im.f_impulse(*vars), self.imp_models)
        imps = np.array(imps)
        return {'ir' : imps}
        

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Set data for impulse responses
        for imp_model in self.imp_models:
            imp_model.set_data(data)

    def params(self):
        return None

def create_single_impulse_response(model, n_pre):
    type = model['impulse']['type'].lower()
    if type == 'basis':
        imp = BasisImpulse(model,
                           n_pre)

    elif type == 'dirichlet':
        imp = DirichletImpulse(model,
                               n_pre)
    else:
        raise Exception("Unrecognized impulse model: %s" % type)
    return imp

class BasisImpulse:
    """ Impulse response function for a single connection. The impulse
        is decomposed into the sum of basis functions. 
        We impose no constraints on the basis functions,
        i.e. they are unnormalized and can be positive or negative.
    """
    
    def __init__(self, model, n_pre):
        """ Initialize the filtered stim model
        """
        self.n_pre = n_pre
        
        self.prms = model['impulse']
        self.mu = self.prms['mu']
        self.sigma = self.prms['sigma']
                
        # Create a basis for the stimulus response
        self.basis = create_basis(self.prms['basis'])
        (_,B) = self.basis.shape

        # The basis is interpolated once the data is specified
        self.ibasis = theano.shared(value=np.zeros((2,B)))

        # Compute the number of parameters
        self.n_vars = B

        # Store filtered spike train in shared memory
        self.ir = theano.shared(name='ir%d'%n_pre,
                                value=np.zeros((2,self.n_vars)))

        self.w_ir = T.dvector('w_ir%d'%n_pre)

        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_ir-self.mu)**2)

        # Expose outputs to the Glm class
        self.I_ir = T.dot(self.ir, self.w_ir)

        # Create callable functions to compute firing rate, log likelihood, and gradients
#         self.f_I_ir  = theano.function([vars],self.I_ir)

        # A function handle for the impulse response
        self.impulse = T.dot(self.ibasis,self.w_ir)
#         self.f_impulse = theano.function([vars],impulse)

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_ir): self.w_ir}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate basis at the resolution of the data
        dt = data['dt']
        (L,B) = self.basis.shape
        Lt_int = self.prms['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if self.prms['basis']['norm']:
            ibasis = ibasis / self.prms['dt_max']
        # Normalize so that the interpolated basis has unit L1 norm
#         if self.prms['basis']['norm']:
#             ibasis = ibasis / np.tile(np.sum(ibasis,0),[Lt_int,1])
        self.ibasis.set_value(ibasis)

        # Project the presynaptic spiking onto the basis
        nT = data["S"].shape[0]
        cstim = convolve_with_basis(np.reshape(data["S"][:,self.n_pre], (nT,1)),
                                    ibasis)

        # Flatten this manually to be safe
        # (there's surely a way to do this with numpy)
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
        w = self.mu + self.sigma * np.random.randn(self.n_vars)
        return {str(self.w_ir): w}


class DirichletImpulse:
    """ Impulse response function for a single connection. The impulse
        response is a convex combination of nonnegative basis functions,
        each of which integrates to one. Therefore, the entire impulse
        is normalized and has unit area under the curve.
    """

    def __init__(self, model, n_pre, n_post, vars, offset):
        """ Initialize the filtered stim model
        """
        self.n_pre = n_pre
        self.n_post = n_post

        self.prms = model['impulse']
        self.alpha = self.prms['alpha']

        # Create a basis for the stimulus response
        self.basis = create_basis(self.prms['basis'])
        (_,B) = self.basis.shape

        # The basis is interpolated once the data is specified
        self.ibasis = theano.shared(value=np.zeros((2,B)))

        # Compute the number of parameters
        self.n_vars = B

        # Store filtered spike train in shared memory
        self.ir = theano.shared(value=np.zeros((2,self.n_vars)))

        # The variables are log-gamma distributed
        self.lng = vars[offset:offset+self.n_vars]
        self.g = T.exp(self.lng)
        self.w = self.g / T.sum(self.g)

        # Log probability
        self.log_p = -B*scipy.special.gammaln(self.alpha) + \
                      T.sum((self.alpha-1.0)*self.g) + \
                      T.sum(-1.0*self.g)

        # Expose outputs to the Glm class
        self.I_ir = T.dot(self.ir, self.w)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_ir  = theano.function([vars],self.I_ir)

        # A function handle for the impulse response
        impulse = T.dot(self.ibasis,self.w)
        self.f_impulse = theano.function([vars], impulse)

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Interpolate basis at the resolution of the data
        dt = data['dt']
        (L,B) = self.basis.shape
        t_int = np.linspace(0,1,self.prms['dt_max']/dt)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])
            #Keep the total L1 norm the same
            #ibasis[:,b] /= (np.sum(ibasis[:,b])/np.sum(self.basis[:,b]))

        # Normalize so that the interpolated basis has volume 1
        ibasis = ibasis / self.prms['dt_max']
        self.ibasis.set_value(ibasis)

        # Project the presynaptic spiking onto the basis
        nT = data["S"].shape[0]
        cstim = convolve_with_basis(np.reshape(data["S"][:,self.n_pre], (nT,1)),
                                    ibasis)

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
        w_imp = np.random.dirichlet(self.alpha*np.ones(self.n_vars,))
        return w_imp

    def params(self):
        return {'impulse' : self.f_impulse(vars)}
