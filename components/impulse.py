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

        #I_imps = map(lambda im: T.reshape(im.I_ir, (im.I_ir.size, 1), ndim=2),
        #             self.imp_models)
        I_imps = map(lambda im: T.shape_padright(im.I_ir, n_ones=1),
                     self.imp_models)
        lp_imps = map(lambda im: im.log_p, self.imp_models)

        # Concatenate impulse currents into a matrix
        self.I_imp = T.concatenate(I_imps, axis=1)
        self.log_p = T.sum(T.stack(lp_imps))

        self.f_I_imp = theano.function([vars], self.I_imp)

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
                           offset)

    elif type == 'dirichlet':
        imp = DirichletImpulse(model,
                               n_pre, n_post,
                               vars,
                               offset)
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
        self.ir = theano.shared(value=np.zeros((2,self.n_vars)))

        self.w_ir = vars[offset:offset+self.n_vars]

        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_ir-self.mu)**2)

        # Expose outputs to the Glm class
        self.I_ir = T.dot(self.ir, self.w_ir)

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_I_ir  = theano.function([vars],self.I_ir)

        # A function handle for the impulse response
        impulse = T.dot(self.ibasis,self.w_ir)
        self.f_impulse = theano.function([vars],impulse)

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
        w_imp = self.mu + self.sigma * np.random.randn(self.n_vars)
        return w_imp

    def params(self):
        return {'basis' : self.basis}


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
                      T.sum((self.alpha-1)*self.g) + \
                      T.sum(-self.g)

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
        return {'basis' : self.basis}