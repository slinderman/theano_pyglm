"""
Weight models for the Network GLM
"""
import theano
import theano.tensor as T
import numpy as np
from component import Component

def create_graph_component(model):
        type = model['network']['graph']['type'].lower()
        if type == 'complete':
            graph = CompleteGraphModel(model)
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
        self.n_vars = 0

        # Define log probability
        self.log_p = T.constant(0.0)
        self.vars = []




