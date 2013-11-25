import theano
import numpy as np

class Bias:
    """
    Generic class for a bias. This is overkill, the bias is really just
    a constant, but we use this abstraction for consistency with other 
    components.
    """
    def __init__(self, model, vars, offset):
        """
        Create the specific bias implementation
        """
        type = model['bias']['type'].lower() 
        if type == 'constant':
            self.__bias = ConstantBias(model, vars, offset)
        
        else:
            raise Exception("Unrecognized bias model: %s" % type)
    
    def set_data(self, data):
        self.__bias.set_data(data)
        
    def sample(self):
        return self.__bias.sample()
    
    def params(self):
        return self.__bias.params()
    
class ConstantBias:
    """
    """
    
    def __init__(self, vars, offset, **kwargs):
        """ Initialize a simple scalara bias. This is only in a class
            for consistency with the other model components
        """

        prms = {'mu_bias' : -3,
                'sig_bias' : 0.1}
        prms.update(kwargs)

        self.mu_bias = prms['mu_bias']
        self.sig_bias = prms['sig_bias']

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