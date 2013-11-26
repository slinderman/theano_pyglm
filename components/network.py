import theano
import theano.tensor as T

from impulse import *

from component import Component

class Network(Component):
    """ The network component encapsulates the impulse responses, weights,
    and adjacency matrix of the spiking interactions.
    """

    def __init__(self, model, vars, v_offset):
        """ Initialize the filtered stim model
        """
        prms = model['network']

        # Create a basis for the stimulus response
        self.basis = create_basis(**prms)
        # Compute the number of parameters
        (_,B) = self.basis.shape
        
        # Keep track of the number of variables
        self.n_vars = 0
        
        # Initialize a potentially dynamic weight matrix
        self.W = theano.shared(np.ones((1,N)))


    def create_weight_component(self, model, vars, offset):
        type = model['network']['weight']['type'].lower()
        if type == 'constant':
            weight = ConstantWeights(model, vars, offset)
        else:
            raise Exception("Unrecognized weight model: %s" % type)
        return weight

    def create_graph_component(self, model, vars, offset):
        type = model['network']['graph']['type'].lower()
        if type == 'constant':
            graph = CompleteGraph(model, vars, offset)
        else:
            raise Exception("Unrecognized weight model: %s" % type)
        return graph
        
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Set data for impulse responses
        for imp_model in self.imp_models:
            imp_model.set_data(data)
            
            
        # TODO - SET THE WEIGHTS FROM THE OUTPUT OF A PARTICLE FILTER
        self.W.set_value(np.ones_like(data["S"]))

    def sample(self):
        """
        return a sample of the variables
        """
        vars = []
        
        # Sample impulse responses
        for imp_model in self.imp_models:
            w_imp = imp_model.sample()
            vars = np.concatenate((vars,w_imp))
        return vars

    def params(self):
        return {'basis' : self.basis}