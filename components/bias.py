import theano
import numpy as np

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