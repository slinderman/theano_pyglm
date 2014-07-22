import theano
import theano.tensor as T
import numpy as np
from component import Component

def create_nlin_component(model):
    type = model['nonlinearity']['type'].lower()
    if type == 'exp':
        nlin = ExpNonlinearity(model)
    elif type == 'explinear':
        nlin = ExpLinearNonlinearity(model)
    else:
        raise Exception("Unrecognized nonlinearity model: %s" % type)
    return nlin

class ExpNonlinearity(Component):
    """ Standard exponential nonlinearity.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self.nlin = T.exp
        self.log_p = T.constant(0.)


        self.f_nlin = np.exp


class ExpLinearNonlinearity(Component):
    """ Exponential nonlinearity (\lambda=e^x) for x<0,
        Linear (\lambda=1+x) for x>0.
        This is nice because it satisfies a Lipschitz bound of 1.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        # self.nlin = lambda x: T.switch(T.lt(x,0), T.exp(x), 1.0+x)
        self.nlin = lambda x: T.log(1.0+T.exp(x))
        self.log_p = T.constant(0.)
        
        # self.f_nlin = lambda x: np.exp(x)*(x<0) + (1.0+x)*(x>=0)
        self.f_nlin = lambda x: np.log(1.0+np.exp(x))
