import theano
import theano.tensor as T
import numpy as np
from component import Component

def create_nlin_component(model, vars, offset):
    type = model['nonlinearity']['type'].lower()
    if type == 'exp':
        nlin = ExpNonlinearity(model, vars, offset)
    else:
        raise Exception("Unrecognized nonlinearity model: %s" % type)
    return nlin

class ExpNonlinearity(Component):
    """ Standard exponential nonlinearity.
    """

    def __init__(self, model, vars, offset):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self.nlin = T.exp
        self.log_p = T.constant(0.)


        self.f_nlin = np.exp