"""
Weight models for the Network GLM
"""
import theano
import theano.tensor as T
import numpy as np
from component import Component
from inference.log_sum_exp import log_sum_exp_sample

def create_graph_component(model):
        type = model['network']['graph']['type'].lower()
        if type == 'complete':
            graph = CompleteGraphModel(model)
        elif type == 'erdos_renyi' or \
             type == 'erdosrenyi':
            graph = ErdosRenyiGraphModel(model)
        else:
            raise Exception("Unrecognized graph model: %s" % type)
        return graph

class CompleteGraphModel(Component):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        N = model['N']

        # Define complete adjacency matrix
        self.A = T.ones((N,N))

        # Define log probability
        self.log_p = T.constant(0.0)
        
    def get_state(self):
        return {'A': self.A}
    
class ErdosRenyiGraphModel(Component):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.prms = model['network']['graph']
        N = model['N']
        self.rho = self.prms['rho'] * np.ones((N,N))

        if 'rho_refractory' in self.prms:
            self.rho[np.diag_indices(N)] = self.prms['rho_refractory']

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # Define log probability
        self.log_p = T.sum(self.A*np.log(np.minimum(0.99,self.rho)) + 
			   (1-self.A)*np.log(np.maximum(0.01,1.0-self.rho)))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A}


    def sample(self):
        N = self.model['N']
        A = np.random.rand(N,N) < self.rho
        A = A.astype(np.int8)
        return {str(self.A) : A}

    def get_state(self):
        return {'A': self.A}
