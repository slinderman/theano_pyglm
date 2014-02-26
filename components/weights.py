"""
Weight models for the Network GLM
"""
import theano
import theano.tensor as T
import numpy as np
from component import Component
from priors import create_prior

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

class ConstantWeightModel(Component):
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

    def get_state(self):
        return {'W': self.W}
    
class GaussianWeightModel(Component):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        N = model['N']
        prms = model['network']['weight']

        self.prior = create_prior(prms['prior'])
	
        # Implement refractory period by having negative mean on self loops
        if 'refractory_prior' in prms:
            #self.mu[np.diag_indices(N)] = prms['mu_refractory']
            self.refractory_prior = create_prior(prms['refractory_prior'])

            # Get the upper and lower diagonal indices so that we can evaluate
            # the log prob of the refractory weights separately from the
            # log prob of the regular weights
            self.diags = np.ravel_multi_index(np.diag_indices(N), (N,N))
            lower = np.ravel_multi_index(np.tril_indices(N,k=-1), (N,N))
            upper = np.ravel_multi_index(np.triu_indices(N,k=1), (N,N))
            self.nondiags = np.concatenate((lower, upper))

        # Define weight matrix
        self.W_flat = T.dvector(name='W')
        self.W = T.reshape(self.W_flat,(N,N))

        if hasattr(self, 'refractory_prior'):
            self.log_p = self.prior.log_p(self.W.take(self.nondiags)) + \
                         self.refractory_prior.log_p(self.W.take(self.diags))
        else:
            self.log_p = self.prior.log_p(self.W)

    def sample(self):
        """
        return a sample of the variables
        """
        N = self.model['N']

        if hasattr(self, 'refractory_prior'):
            W = np.zeros((N,N))
            W_diags = np.array([self.refractory_prior.sample() for n in np.arange(N)])
            W_nondiags = np.array([self.refractory_prior.sample() for n in np.arange(N**2-N)])
            np.put(W, self.diags,  W_diags)
            np.put(W, self.nondiags, W_nondiags)
            W_flat = np.reshape(W,(N**2,))
        else:
            W_flat = np.array([self.refractory_prior.sample() for n in np.arange(N**2)])

        return {str(self.W_flat): W_flat}

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.W_flat): self.W_flat}
    
    def get_state(self):
        return {'W': self.W}
