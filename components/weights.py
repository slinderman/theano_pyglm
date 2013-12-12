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

class GaussianWeightModel(Component):
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
        
    def sample(self):
        """
        return a sample of the variables
        """
        N = self.model['N']
        W = self.mu + self.sigma * np.random.randn(N,N)
        W_flat = np.reshape(W,(N**2,))
        return [W_flat]

