from pyglm.components.graph import *
from pyglm.components.weights import *

class _NetworkBase(Component):
    """ The network component encapsulates the impulse responses, weights,
    and adjacency matrix of the spiking interactions.
    """

    def __init__(self, model, latent):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.latent = latent
        self.prms = model['network']

    @property
    def graph(self):
        raise NotImplementedError()

    @property
    def weights(self):
        raise NotImplementedError()

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

    def sample(self, acc):
        """
        return a sample of the variables
                """
        return {'graph' : self.graph.sample(acc),
                'weights' : self.weights.sample(acc)}
    

class TheanoNetwork(_NetworkBase):
    def __init__(self, model, latent):
        super(TheanoNetwork, self).__init__(model, latent)

        # Create the graph model for the adjacency matrix
        self._graph = create_graph_component(model, latent)

        # Create the weight model for the weight matrix
        self._weights = create_weight_component(model, latent)

        # Compute log probability
        self._log_p = self.graph.log_p + self.weights.log_p

    @property
    def graph(self):
        return self._graph

    @property
    def weights(self):
        return self._weights

    @property
    def log_p(self):
        return self._log_p

class KayakNetwork(_NetworkBase):
    def __init__(self, model, latent):
        super(KayakNetwork, self).__init__(model, latent)

        # Create the graph model for the adjacency matrix
        self._graph = KayakCompleteGraphModel(model)

        # Create the weight model for the weight matrix
        self._weights = KayakConstantWeightModel(model)

        # Compute log probability
        self._log_p = self.graph.log_p + self.weights.log_p

    @property
    def graph(self):
        return self._graph

    @property
    def weights(self):
        return self._weights

    @property
    def log_p(self):
        return self._log_p