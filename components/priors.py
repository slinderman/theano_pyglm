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
    elif typ == 'group_lasso' or \
         typ == 'grouplasso':
        return GroupLasso(model, **kwargs)
    else:
        raise Exception("Unrecognized prior type: %s" % typ)

class Gaussian(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model, name='gaussian'):
        self.prms = model
 #       self.value = T.dscalar(name=name)
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])

#        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)
    
    def log_p(self, value):
        """ Compute log prob of the given value under this prior
        """
        return -0.5/self.sigma**2  * T.sum((value-self.mu)**2)

    def get_variables(self):
#        return {str(self.mu): self.mu, 
#                str(self.sigma) : self.sigma}
        return {}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model 
        """ 
        self.mu.set_value(self.prms['mu'])
        self.sigma.set_value(self.prms['sigma'])
        
    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(*size)
        return v

class SphericalGaussian(Component):
    """ Wrapper for a vector random variable with a spherical distribution
    """
    def __init__(self, model, name='spherical_gaussian', D=1):
        self.prms = model
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

class GroupLasso(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model, name='gaussian'):
        self.prms = model
        self.lam = theano.shared(name='lam', value=self.prms['lam'])
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])
    
    def log_p(self, value):
        """ Compute log prob of the given value under this prior
            Value should be NxB where N is the number of groups and
            B is the number of parameters per group.
        """
        mu = self.mu.get_value()
        sigma = self.sigma.get_value()
        return -1.0*self.lam * T.sum(T.sqrt(T.sum(((value-mu)/sigma)**2, axis=1)))


    def get_variables(self):
        return {}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model 
        """
        self.mu.set_value(self.prms['mu'])
        self.sigma.set_value(self.prms['sigma'])
        
    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        N = size[0]
        norms = np.random.laplace(0, self.lam.get_value(), size=(N,1))
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(*size)
        v_norms = np.sqrt(np.sum(v**2,axis=1)).reshape(N,1)
        vf = (v*norms/v_norms)

        return vf
