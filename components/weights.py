"""
Weight models for the Network GLM
"""
import theano
import theano.tensor as T
import numpy as np
from component import Component

from inference.hmc import hmc
from inference.slicesample import slicesample

def create_weight_component(model):
        type = model['network']['weight']['type'].lower()
        if type == 'constant':
            weight = ConstantWeightModel(model)
        elif type == 'gaussian':
            weight = GaussianWeightModel(model)
        else:
            raise Exception("Unrecognized weight model: %s" % type)
        return weight

class WeightModel(Component):
    """ GraphModel extends component with graph specific functions assumed
        by the parent network.
    """
    def sample_W(self, state, network_glm, n_pre, n_post):
        """
        Sample a specific entry in W
        """
        raise Exception('sample_W has not been implemented!')

    def gibbs_sample_parameters(self, state):
        """ Gibbs sample any hyperparameters of the weight matrix
        """
        raise Exception('gibbs_sample_parameters has not been implemented!')

class ConstantWeightModel(WeightModel):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        N = model['N']

        prms = model['network']['weight']
        self.value = prms['value']
        
        # Define weight matrix
        self.W = self.value * T.ones((N,N))

        # Define log probability
        self.log_p = T.constant(0.0)

        self.vars = []

    def sample_W(self, state, network_glm, n_pre, n_post):
        """
        Sample a specific entry in W
        """
        return state[0]

    def gibbs_sample_parameters(self, state):
        """ Gibbs sample any hyperparameters of the weight matrix
        """
        return state[0]

class GaussianWeightModel(WeightModel):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        N = model['N']

        prms = model['network']['weight']
        self.mu = prms['mu']
        self.sigma = prms['sigma']

        # Define weight matrix
        self.W_flat = T.dvector(name='W')
        self.W = T.reshape(self.W_flat,(N,N))

        # Define log probability
        self.log_p = T.sum(-1.0/(2.0*self.sigma**2) * (self.W-self.mu)**2)
        self.f_lp = theano.function(self.W,self.log_p)

        # Define a getter for the variables of the model
        self.vars = [self.W_flat]
        
        # TODO remove this hack
        self.grads = {}
        
    def sample(self):
        """
        return a sample of the variables
        """
        N = self.model['N']
        W = self.mu + self.sigma * np.random.randn(N,N)
        W_flat = np.reshape(W,(N**2,))
        return [W_flat]

    def sample_W(self, state, network_glm, n_pre, n_post):
        """
        Sample a specific entry in W
        """
        # How to sample W?
        # We could use slice sampling, especially if we are using 
        # Gauss-Hermite quadrature. We'll already have a number of points
        # at which we know the likelihood...
        
        # For now let's just use HMC...
        x_net = state[0]
        [A,W] = x_net
        _,shapes = pack(x_net)
        def lp(w):
            W[n_pre,n_post] = w
            all_vars = [n_post] + unpack(x_net,shapes) + state[n_post] 
            return network_glm.glm.f_lp(*all_vars) + \
                   self.f_lp(W) 
        
        # Get the gradients
#        if 'g_lp' in self.grads.keys():
#            g_lp = self.grads['g_lp']
#        else:
#            g = T.grad(network_glm.glm.log_p, self.W)
#            g_lp = theano.function(network_glm.vars, g)
#            
#            
#            self.grads['g_lp'] = g_lp
#            
#        def g_nll(w):
#            W[n_pre,n_post] = w
#            all_vars = [n_post] + unpack(x_net,shapes) + state[n_post] 
#            return -1.0*g_lp(*all_vars)
#         
#        x_net = hmc(nll,)

        w_new = slicesample(lp)

    def gibbs_sample_parameters(self, state):
        """ Gibbs sample any hyperparameters of the weight matrix
        """
        return state
