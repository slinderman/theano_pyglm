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
    elif type == 'sbm':
        graph = StochasticBlockGraphModel(model)
    elif type == 'distance':
        graph = LatentDistanceGraphModel(model)
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
        self.A = T.ones((N, N))

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
        self.rho = self.prms['rho'] * np.ones((N, N))

        if 'rho_refractory' in self.prms:
            self.rho[np.diag_indices(N)] = self.prms['rho_refractory']

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # Define log probability
        self.log_p = T.sum(self.A * np.log(np.minimum(0.999, self.rho)) +
                           (1 - self.A) * np.log(np.maximum(0.001, 1.0 - self.rho)))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A}


    def sample(self):
        N = self.model['N']
        A = np.random.rand(N, N) < self.rho
        A = A.astype(np.int8)
        return {str(self.A): A}

    def get_state(self):
        return {'A': self.A}


class StochasticBlockGraphModel(Component):
    def __init__(self, model):
        """ Initialize the stochastic block model for the adjacency matrix
        """
        self.model = model
        self.prms = model['network']['graph']
        self.N = model['N']

        # SBM has R latent clusters
        self.R = self.prms['R']
        # A RxR matrix of connection probabilities per pair of clusters
        self.B = T.dmatrix('B')
        # SBM has a latent block or cluster assignment for each node
        self.Y = T.lvector('Y')
        # For indexing, we also need Y as a column vector and tiled matrix
        self.Yv = T.reshape(self.Y, [self.N, 1])
        self.Ym = T.tile(self.Yv, [1, self.N])
        self.pA = self.B[self.Ym, T.transpose(self.Ym)]

        # A probability of each cluster
        self.alpha = T.dvector('alpha')

        # Hyperparameters governing B and alpha
        self.b0 = self.prms['b0']
        self.b1 = self.prms['b1']
        self.alpha0 = self.prms['alpha0']

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # Define log probability
        log_p_B = T.sum((self.b0 - 1) * T.log(self.B) + (self.b1 - 1) * T.log(1 - self.B))
        log_p_alpha = T.sum((self.alpha0 - 1) * T.log(self.alpha))
        log_p_A = T.sum(self.A * T.log(self.pA) + (1 - self.A) * T.log(1 - self.pA))

        self.log_p = log_p_B + log_p_alpha + log_p_A

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A,
                str(self.Y): self.Y,
                str(self.B): self.B,
                str(self.alpha): self.alpha}

    def sample(self):
        N = self.model['N']

        # Sample alpha from a Dirichlet prior
        alpha = np.random.dirichlet(self.alpha0)

        # Sample B from a Beta prior
        B = np.random.beta(self.b0, self.b1, (self.R, self.R))

        # Sample Y from categorical dist
        Y = np.random.choice(self.R, size=self.N, p=alpha)

        pA = self.pA.eval({self.B: B,
                           self.Y: Y})

        A = np.random.rand(N, N) < pA
        A = A.astype(np.int8)
        return {str(self.A): A,
                str(self.Y): Y,
                str(self.B): B,
                str(self.alpha): alpha}

    def get_state(self):
        return {str(self.A): self.A,
                str(self.Y): self.Y,
                str(self.B): self.B,
                str(self.alpha): self.alpha}

class LatentDistanceGraphModel(Component):
    def __init__(self, model):
        """ Initialize the stochastic block model for the adjacency matrix
        """
        self.model = model
        self.prms = model['network']['graph']
        self.N = model['N']

        # Latent distance model has NxR matrix of locations L
        self.L = T.dmatrix('L')

        # Compute the distance between each pair of locations
        # Reshape L into a Nx1xD matrix and a 1xNxD matrix, then add the requisite
        # broadcasting in order to subtract the two matrices
        L1 = self.L.dimshuffle(0,'x',1)     # Nx1xD
        L2 = self.L.dimshuffle('x',0,1)     # 1xNxD
        T.addbroadcast(L1,1)
        T.addbroadcast(L2,0)
        self.D = T.sum((L1-L2)**2, axis=2)

        # There is a distance scale, \delta
        self.delta = T.dscalar(name='delta')

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # The probability of A is exponentially decreasing in delta
        self.pA = T.exp(-1.0*self.D/self.delta)

        # Define log probability
        self.log_p = T.sum(self.A * T.log(self.pA) + (1 - self.A) * T.log(1 - self.pA))

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A,
                str(self.L): self.L,
                str(self.delta): self.delta}

    def sample(self):
        N = self.model['N']

        # TODO: Sample locations from prior
        L = np.arange(N).reshape((N, 1)).astype(np.float)

        # TODO: Sample delta from prior
        delta = 2.0

        # Sample A from pA
        pA = self.pA.eval({self.L: L,
                           self.delta: delta})

        A = np.random.rand(N, N) < pA
        A = A.astype(np.int8)
        return {str(self.A): A,
                str(self.L): L,
                str(self.delta): delta}

    def get_state(self):
        return {str(self.A): self.A,
                str(self.L): self.L,
                str(self.delta): self.delta}