import theano
import theano.tensor as T
from utils.basis import *
from component import Component

def create_impulse_component(model):
    typ = model['impulse']['type'].lower()
    if typ.lower() == 'basis': 
        return LinearBasisImpulses(model)
    elif typ.lower() == 'normalized' \
         or typ.lower() == 'dirichlet':
        return NormalizedBasisImpulses(model)

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
        return {'impulse' : self.impulse,
                'basis' : self.ibasis}

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

class NormalizedBasisImpulses(Component):
    """ Normalized impulse response functions. Here we make use of Theano's
        broadcasting to sum up the currents from each presynaptic neuron.
    """
    def __init__(self, model):
        self.prms = model['impulse']

        # Number of presynaptic neurons
        self.N = model['N']

        # Get parameters of the prior
        self.alpha = self.prms['alpha']

        # Create a basis for the impulse responses response
        self.basis = create_basis(self.prms['basis'])
        (_,self.B) = self.basis.shape
        # The basis is interpolated once the data is specified
        self.ibasis = theano.shared(value=np.zeros((2,self.B)))

        # Initialize memory for the filtered spike train
        self.ir = theano.shared(name='ir',
                                value=np.zeros((1,self.N,self.B)))

        # Define Dirichlet distributed weights by normalizing gammas
        # The variables are log-gamma distributed
        self.lng = T.dvector('w_lng')
        self.g = T.exp(self.lng)
        self.g2 = T.reshape(self.g, [self.N,self.B])
        self.g_sum = T.reshape(T.sum(self.g2, axis=1), [self.N,1])
        
        # Normalize the gammas to get a Dirichlet draw
        T.addbroadcast(self.g_sum, 1)
        self.w_ir2 = self.g2 / self.g_sum
        self.w_ir2.name = 'w_ir'

        # Repeat them (in a differentiable manner) to create a 3-tensor
        self.w_ir3 = T.reshape(self.w_ir2, [1,self.N,self.B])

        # Make w_ir3 broadcastable in the 1st dim
        T.addbroadcast(self.w_ir3,0)

        # Take the elementwise product of the filtered stimulus and
        # the repeated weights to get the weighted impulse current along each
        # impulse basis dimension. Then sum over bases to get the
        # total coupling current from each presynaptic neurons at
        # all time points
        self.I_imp = T.sum(self.ir*self.w_ir3, axis=2)

        # Log probability
        self.log_p = -self.B*self.N*scipy.special.gammaln(self.alpha) + \
                      T.sum((self.alpha-1.0)*self.g) + \
                      T.sum(-1.0*self.g)

        # Define a helper variable for the impulse response
        # after projecting onto the basis
        self.impulse = T.dot(self.w_ir2, T.transpose(self.ibasis))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.lng): self.lng}

    def sample(self):
        """
        return a sample of the variables
        """
        g = np.random.gamma(self.alpha,np.ones(self.N*self.B))
        lng = np.log(g)
        return {str(self.lng): lng}

    def get_state(self):
        """ Get the impulse responses
        """
        return {'impulse' : self.impulse,
                'basis' : self.ibasis}

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


