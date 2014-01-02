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
        self.model = model
        self.prms = model['network']

        # Keep track of the number of variables
        self.n_vars = 0

        # Create the graph model for the adjacency matrix
        self.graph = create_graph_component(model)

        # Create the weight model for the weight matrix
        self.weights = create_weight_component(model)

        # Compute log probability
        self.log_p = self.graph.log_p + self.weights.log_p
        
    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {'graph' : self.graph.get_variables(),
                'weights' : self.weights.get_variables()}
        
    def get_state(self):
        """ Get the state of the graph and weights
        """
        state = {'graph' : self.graph.get_state(),
                 'weights' : self.weights.get_state()}
        return state
    
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        self.graph.set_data(data)
        self.weights.set_data(data)

    def sample(self):
        """
        return a sample of the variables
        """
        return {'graph' : self.graph.sample(),
                'weights' : self.weights.sample()}
    
