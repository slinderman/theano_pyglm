import theano
import theano.tensor as T
from utils.basis import *
from component import Component
from priors import create_prior

def create_impulse_component(model):
    typ = model['impulse']['type'].lower()
    if typ.lower() == 'basis': 
        return LinearBasisImpulses(model)
    elif typ.lower() == 'normalized':
        return NormalizedBasisImpulses(model)
    elif typ.lower() == 'dirichlet':
        return DirichletImpulses(model)
    elif typ.lower() == 'exponential':
        return ExponentialImpulses(model)

class LinearBasisImpulses(Component):
    """ Linear impulse response functions. Here we make use of Theano's
        tensordot to sum up the currents from each presynaptic neuron.
    """
    def __init__(self, model):
        self.prms = model['impulse']
        self.prior = create_prior(self.prms['prior'])

        # Number of presynaptic neurons
        self.N = model['N']

        # Get parameters of the prior
        # self.mu = self.prms['mu']
        # self.sigma = self.prms['sigma']

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

        # Take the elementwise product of the filtered stimulus and
        # the repeated weights to get the weighted impulse current along each
        # impulse basis dimension. Then sum over bases to get the
        # total coupling current from each presynaptic neurons at
        # all time points
        self.I_imp = T.sum(self.ir*w_ir3, axis=2)
        # self.log_p = T.sum(-0.5/self.sigma**2 * (self.w_ir-self.mu)**2)
        self.log_p = self.prior.log_p(w_ir2)


        # Define a helper variable for the impulse response
        # after projecting onto the basis
        self.impulse = T.dot(w_ir2, T.transpose(self.ibasis))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_ir): self.w_ir}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.prior.set_hyperparameters(model['prior'])

    def sample(self):
        """
        return a sample of the variables
        """
        # w = self.mu + self.sigma*np.random.randn(self.N*self.B)
        w = self.prior.sample(size=(self.N, self.B)).ravel()
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

        # Log probability of a set of independent log-gamma r.v.'s
        # This is log p(log(g)) under the prior. Since we are taking the
        # log, we multiply by a factor of g to ensure normalization and
        # thus the \alpha-1 in the exponent becomes \alpha
        self.log_p = -self.B*self.N*scipy.special.gammaln(self.alpha) \
                     + T.sum(self.alpha*self.lng) \
                     - T.sum(self.g)

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
        # t_int = np.linspace(0,1,Lt_int)
        t_int = np.arange(0.0, self.prms['dt_max'], step=dt)
        # t_bas = np.linspace(0,1,L)
        t_bas = np.linspace(0.0, self.prms['dt_max'], L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if self.prms['basis']['norm']:
            ibasis = ibasis / np.trapz(ibasis,t_int,axis=0)
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

class DirichletImpulses(Component):
    """ Normalized impulse response functions using a Dirichlet prior.
        Here we separate out the parameters of each impulse response
        so that they may be sampled separately.
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

        # Parameterize the Dirichlet distributed weight vectors using the
        # expanded mean parameterization. That is, the parameters are B
        # gamma distributed variables, and beta is the normalized gammas.
        # To handle boundary conditions, let beta actually be normalized
        # absolute value of the gammas.
        self.gs = []
        self.betas = []
        for n in range(self.N):
            g = T.dvector('g_%d'%n)
            gabs = abs(g)
            beta = gabs/T.sum(gabs)
            self.gs.append(g)
            self.betas.append(beta)

        # # Define Dirichlet distributed weight vectors
        # self.betas = []
        # for n in range(self.N):
        #     self.betas.append(T.dvector('beta_%d'%n))

        # Concatenate the betas into a 3-tensor for easy computation of the
        # impulse response current. The shape of Beta should be 1xNxB
        self.beta3 = T.concatenate(map(lambda beta: T.shape_padleft(beta, n_ones=2), self.betas),
                                  axis=1)

        # Take the elementwise product of the filtered stimulus and
        # the repeated weights to get the weighted impulse current along each
        # impulse basis dimension. Then sum over bases to get the
        # total coupling current from each presynaptic neurons at
        # all time points
        self.I_imp = T.sum(self.ir*self.beta3, axis=2)

        # I_imps = []
        # for n,beta in enumerate(self.betas):
        #     I = T.dot(self.ir[:,n,:], beta)
        #     I_imps.append(I)
        #
        # self.I_imp = T.concatenate(map(lambda I: T.shape_padright(I, n_ones=1), I_imps),
        #                            axis=1)

        # Log probability of a set of Dirichlet distributed vectors
        # self.log_p = (self.alpha-1) * T.sum(T.log(beta2))
        self.log_p = T.constant(0.0)
        for g in self.gs:
            self.log_p += (self.alpha-1) * T.sum(T.log(abs(g))) - T.sum(abs(g))

        # Define a helper variable for the impulse response
        # after projecting onto the basis
        beta2 = T.concatenate(map(lambda beta: T.shape_padleft(beta, n_ones=1), self.betas),
                                  axis=0)

        self.impulse = T.dot(beta2, T.transpose(self.ibasis))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        v = {}
        # for beta in self.betas:
        #     v[str(beta)] = beta
        for g in self.gs:
            v[str(g)] = g

        return v

    def sample(self):
        """
        return a sample of the variables
        """
        v = {}
        # for beta in self.betas:
        #     v[str(beta)] = np.random.dirichlet(self.alpha*np.ones(self.B))
        for g in self.gs:
            v[str(g)] = np.random.gamma(self.alpha, np.ones(self.B))
        return v

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
        # t_int = np.linspace(0,1,Lt_int)
        t_int = np.arange(0.0, self.prms['dt_max'], step=dt)
        # t_bas = np.linspace(0,1,L)
        t_bas = np.linspace(0.0, self.prms['dt_max'], L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if self.prms['basis']['norm']:
            ibasis = ibasis / np.trapz(ibasis,t_int,axis=0)
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


class ExponentialImpulses(Component):
    """ Exponential impulse response functions. Here we make use of Theano's
        broadcasting to sum up the currents from each presynaptic neuron.
    """
    def __init__(self, model):
        self.prms = model['impulse']

        # Number of presynaptic neurons
        self.N = model['N']

        # Get parameters of the prior
        self.tau0 = self.prms['tau0']
        self.sigma = self.prms['sigma']

        # Impulse responses are parameterized by a time constant tau
        self.taus = T.dvector('taus_ir')

        # Spike train is shared variable populated by the data
        self.S = theano.shared(name='S', value=np.zeros((1,self.N)))

        # Number of time bins is a shared variable set by the data
        self.T_bins = T.shape(self.S)[0]
        self.t_ir = theano.shared(name='t_ir', value=np.zeros((1,)))

        # The impulse response is exponentially decaying function of t_ir
        # self.impulse = T.exp(-self.t_ir/ self.tau)
        # Scan computes an exponentially decreasing impulse with a different
        # time constant for each impulse response function. The results
        # are stacked together intoa matrix of size 
        filt_fn = lambda tau: (self.t_ir>1e-14)*T.exp(-self.t_ir/tau)
        self.impulse,_ = theano.scan(fn=lambda tau: T.exp(-self.t_ir[1:]/tau),
                                     outputs_info=None,
                                     sequences=[self.taus],
                                     non_sequences=[])
                
        # The filtered stimulus is found by convolving the spike train with the
        # impulse response function and keeping the first T_bins 
        from theano.tensor.signal.conv import conv2d
        def filter_spike_train(n,S,taus):
            """ Helper function to filter the spike train
            """
            filt = T.shape_padright(filt_fn(taus[n]), n_ones=1)
            filtered_S = conv2d(T.shape_padright(S[:,n], n_ones=1), 
                                filt, 
                                border_mode='full')
            return filtered_S[0,:,0]
        
        self.ir,_ = theano.scan(fn=filter_spike_train,
                                outputs_info=None,
                                sequences=[T.arange(self.N)],
                                non_sequences=[self.S, self.taus])
        # Keep only the first T_bins and the central portion of the impulse
        # responses
        self.ir = self.ir[:, :self.T_bins]
        self.ir = T.transpose(self.ir)
        self.I_imp = T.reshape(self.ir, (self.T_bins, self.N))
        
        # TODO: Log probability of tau
        self.log_p = 0.0
                      
    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.taus): self.taus}

    def sample(self):
        """
        return a sample of the variables
        """
        #ln_taus = np.log(self.tau0) + self.sigma * np.random.randn(size=(self.N,))
        #taus = np.exp(ln_taus)
        taus = np.random.lognormal(np.log(self.tau0), self.sigma, size=(self.N,))
        print "Taus: %s" % str(taus)
        return {str(self.taus): taus}

    def get_state(self):
        """ Get the impulse responses
        """
        return {'impulse' : self.impulse,
                'I_imp' : self.I_imp}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Set data
        self.S.set_value(data['S'])

        # Set t_ir, the time delta for each impulse bin
        N_ir = self.prms['dt_max'] / data['dt']
        t_ir = data['dt']*np.arange(N_ir)
        #t_ir = np.reshape(t_ir, (N_ir, 1))
        self.t_ir.set_value(t_ir)

