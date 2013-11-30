import theano
import theano.tensor as T

from impulse import *

from component import Component
from graph import *
from weights import *

class Network(Component):
    """ The network component encapsulates the impulse responses, weights,
    and adjacency matrix of the spiking interactions.
    """

    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        prms = model['network']

        # Keep track of the number of variables
        self.n_vars = 0

        # Create the graph model for the adjacency matrix
        self.graph = create_graph_component(model)

        # Create the weight model for the weight matrix
        self.weights = create_weight_component(model)

        # Compute log probability
        self.log_p = self.graph.log_p + self.weights.log_p

        # Concatenate the variables into a single list of variables
        self.vars = self.graph.vars + self.weights.vars
        
        # Create function to evaluate W
        self.f_A = theano.function(self.vars, self.graph.A,
                                   on_unused_input='ignore')
        self.f_W = theano.function(self.vars, self.weights.W,
                                   on_unused_input='ignore')

        self.f_lp = theano.function(self.vars, self.log_p,
                                    on_unused_input='ignore')
        
    def get_state(self, net_vars):
        """ Get the effective weights (A*W)
        """
        W = self.f_W(net_vars)
        A = self.f_A(net_vars)
        
        return {'net' : A*W}
    
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        self.graph.set_data(data)
        self.weights.set_data(data)

    def sample(self):
        """
        return a sample of the variables
        """
        # Sample adjacency and weight matrices
        A_vars = self.graph.sample()
        W_vars = self.weights.sample()
        
        vars = []
        if A_vars is not None:
            if W_vars is not None:
                vars = np.concatenate((A_vars,W_vars))
            else:
                vars = A_vars
        elif W_vars is not None:
                vars = W_vars
        return vars
