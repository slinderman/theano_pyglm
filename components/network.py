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
        
        # Create function to evaluate W
#         self.f_A = theano.function(self.vars, self.graph.A,
#                                    on_unused_input='ignore')
#         self.f_W = theano.function(self.vars, self.weights.W,
#                                    on_unused_input='ignore')
# 
#         self.f_lp = theano.function(self.vars, self.log_p,
#                                     on_unused_input='ignore')
    
    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {'graph' : self.graph.get_variables(),
                'weights' : self.weights.get_variables()}
        
    def get_state(self, net_vars):
        """ Get the effective weights (A*W)
        """
        W = self.f_W(*net_vars)
        A = self.f_A(*net_vars)
        
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
        return {'graph' : self.graph.sample(),
                'weights' : self.weights.sample()}
    
    def gibbs_step(self, state, network_glm):
        """ Sample network given the GLM state variables.
            This can be done in parallel for each column of A and W.
            If W is Gaussian, we can use Gauss-Hermite quadrature to integrate out
            the W's and use a collapsed Gibbs sampler for A.
        """
        x_net = state[0]
        
        N = self.model['N']
        for n_post in np.arange(N):
            # Sample the entries in random order
            perm = np.random.permutation(N)
            for n_pre in perm:
                x_net = self.graph.sample_A(state,network_glm,n_pre,n_post)
                x_net = self.weights.sample_W(state,network_glm,n_pre,n_post)
                
        # Sample latent variables of the graph and weights
        x_net = self.graph.gibbs_sample_parameters(state)
        x_net = self.weights.gibbs_sample_parameters(state)
        
        return x_net