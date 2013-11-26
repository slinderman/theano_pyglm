import theano
import numpy as np
from component import Component

def create_bias_component(model, vars, offset):
    type = model['bias']['type'].lower()
    if type == 'constant':
        bias = ConstantBias(model, vars, offset)
    else:
        raise Exception("Unrecognized bias model: %s" % type)
    return bias

class ConstantBias(Component):
    """
    """
    
    def __init__(self, model, vars, offset):
        """ Initialize a simple scalara bias. This is only in a class
            for consistency with the other model components
        """

        prms = model['bias']
        self.mu_bias = prms['mu']
        self.sig_bias = prms['sigma']

        # Define a bias to the membrane potential
        bias = vars[offset]

        self.n_vars = 1
        self.I_bias = bias
        self.log_p = -0.5/self.sig_bias**2 * (bias - self.mu_bias)**2

        self.f_I_bias = theano.function([vars],bias)

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        pass

    def sample(self):
        """
        return a sample of the variables
        """
        bias = self.mu_bias + self.sig_bias * np.random.randn()
        return bias

    def params(self):
        pass