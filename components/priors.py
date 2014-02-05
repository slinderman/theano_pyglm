import theano
import theano.tensor as T
import numpy as np

from components.component import Component

def create_prior(model, **kwargs):
    typ = model['type'].lower()
    if typ == 'normal' or \
       typ == 'gaussian':
       return Gaussian(model, **kwargs)
    elif typ == 'spherical_gaussian':
        return SphericalGaussian(model, **kwargs)
    else:
        raise Exception("Unrecognized prior type: %s" % typ)

class Gaussian(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model, name='gaussian'):
        import pdb; pdb.set_trace()
        self.prms = model
        self.value = T.dscalar(name=name)
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])

        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)

    def get_variables(self):
        return {str(self.value): self.value}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model 
        """ 
        self.mu.set_value(self.prms['mu'])
        self.sigma.set_value(self.prms['sigma'])
        
    def sample(self):
        """ Sample from the prior
        """
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn()
        return {str(self.value): v}

class SphericalGaussian(Component):
    """ Wrapper for a vector random variable with a spherical distribution
    """
    def __init__(self, model, name='spherical_gaussian', D=1):
        self.prms = model
        import pdb; pdb.set_trace()
        self.D = D
        self.value = T.dvector(name=name)
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])
        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)

    def get_variables(self):
        return {str(self.value): self.value}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model 
        """ 
        self.mu.set_value(self.prms['mu'])
        self.sigma.set_value(self.prms['sigma'])
    
    def sample(self):
        """ Sample from the prior
        """
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(self.D)
        return {str(self.value): v}

