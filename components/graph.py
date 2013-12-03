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
            graph = CompleteGraphModel(modeel)
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
        self.n_vars = 0

        # Define log probability
        self.log_p = T.constant(0.0)
        self.vars = []

class ErdosRenyiGraphModel(Component):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.prms = model['network']['graph']
        N = model['N']
        rho = self.prms['rho']
        
        # Define complete adjacency matrix
        self.A = T.dmatrix('A')
        self.n_vars = N**2

        # Define log probability
        self.log_p = T.sum(self.A*np.log(rho) + (1-self.A)*np.log(1.0-rho))
        self.vars = [self.A]

    def sample(self):
        N = model['N']
        rho = self.prms['rho']
        
        A = np.random.rand(N,N) < rho
        return A
    
    def sample_A(self, state, network_glm, n_pre, n_post):
        """
        Sample a specific entry in A
        """
        # Compute the log prob with and without this connection
        x_net = state[0]
        x_glm = state[n_post+1]
        
        x_net['A'][n_pre,n_post] = 1
        log_pr_A = network_glm.glm.f_lp(*([n_post]+x_net+x_glm))
        
        x_net['A'][n_pre,n_post] = 0
        log_pr_noA = network_glm.glm.f_lp(*([n_post]+x_net+x_glm))
        
        # Sample A[n_pre,n_post]
        x_net['A'][n_pre,n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])
        
        return x_net
    
    def gibbs_sample_parameters(self, state):
        x_net = state[0]
        return x_net