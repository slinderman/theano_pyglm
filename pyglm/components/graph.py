"""
Weight models for the Network GLM
"""
import numpy as np

import theano
import theano.tensor as T
from pyglm.components.component import Component


def create_graph_component(model, latent):
    type = model['network']['graph']['type'].lower()
    if type == 'complete':
        graph = TheanoCompleteGraphModel(model)
    elif type == 'erdos_renyi' or \
                    type == 'erdosrenyi':
        graph = ErdosRenyiGraphModel(model)
    elif type == 'sbm':
        graph = StochasticBlockGraphModel(model, latent)
    elif type == 'distance':
        graph = LatentDistanceGraphModel(model, latent)
    else:
        raise Exception("Unrecognized graph model: %s" % type)
    return graph


class _GraphModelBase(Component):
    @property
    def A(self):
        raise NotImplementedError()

    def get_state(self):
        return {'A': self.A}

class TheanoCompleteGraphModel(_GraphModelBase):
    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        N = model['N']

        # Define complete adjacency matrix
        self._A = T.ones((N, N))

        self._log_p = T.constant(0.0)

    @property
    def A(self):
        return self._A

    @property
    def log_p(self):
        # Define log probability
        return self._log_p

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

        self.pA = theano.shared(value=self.rho, name='pA')

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # Allow for scaling the log likelihood of the graph so that we can do
        # Annealed importance sampling
        self.lkhd_scale = theano.shared(value=1.0, name='lkhd_scale')


        # Define log probability
        self.lkhd = T.sum(self.A * np.log(np.minimum(1.0-1e-8, self.rho)) +
                           (1 - self.A) * np.log(np.maximum(1e-8, 1.0 - self.rho)))

        self.log_p = self.lkhd_scale * self.lkhd

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A}


    def sample(self, acc):
        N = self.model['N']
        A = np.random.rand(N, N) < self.rho
        A = A.astype(np.int8)
        return {str(self.A): A}

    def get_state(self):
        return {'A': self.A}


class StochasticBlockGraphModel(Component):
    def __init__(self, model, latent):
        """ Initialize the stochastic block model for the adjacency matrix
        """
        self.model = model
        self.latent = latent
        self.prms = model['network']['graph']
        self.N = model['N']

        # Get the number of latent types (R) and the latent type vector (Y)
        self.type_name = self.prms['types']
        self.R = self.latent[self.type_name].R
        self.Y = self.latent[self.type_name].Y

        # A RxR matrix of connection probabilities per pair of clusters
        self.B = T.dmatrix('B')

        # For indexing, we also need Y as a column vector and tiled matrix
        self.Yv = T.reshape(self.Y, [self.N, 1])
        self.Ym = T.tile(self.Yv, [1, self.N])
        self.pA = self.B[self.Ym, T.transpose(self.Ym)]

        # Hyperparameters governing B and alpha
        self.b0 = self.prms['b0']
        self.b1 = self.prms['b1']

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # Define log probability
        log_p_B = T.sum((self.b0 - 1) * T.log(self.B) + (self.b1 - 1) * T.log(1 - self.B))
        log_p_A = T.sum(self.A * T.log(self.pA) + (1 - self.A) * T.log(1 - self.pA))

        self.log_p = log_p_B + log_p_A

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A,
                str(self.B): self.B}

    def sample(self, acc):
        N = self.model['N']

        # Sample B from a Beta prior
        B = np.random.beta(self.b0, self.b1, (self.R, self.R))

        # We need a sample of Y in order to evaluate pA!
        Y = acc['latent'][self.type_name]['Y']
        pA = self.pA.eval({self.B: B})

        A = np.random.rand(N, N) < pA
        A = A.astype(np.int8)
        return {str(self.A): A,
                str(self.B): B}

    def get_state(self):
        return {str(self.A): self.A,
                str(self.B): self.B}

class LatentDistanceGraphModel(Component):
    def __init__(self, model, latent):
        """ Initialize the stochastic block model for the adjacency matrix
        """
        self.model = model
        self.prms = model['network']['graph']
        self.N = model['N']
        self.N_dims = self.prms['N_dims']

        # Get the latent location
        self.location = latent[self.prms['locations']]
        self.Lm = self.location.Lm
        # self.location_prior = create_prior(self.prms['location_prior'])
        #
        # # Latent distance model has NxR matrix of locations L
        # self.L = T.dvector('L')
        # self.Lm = T.reshape(self.L, (self.N, self.N_dims))

        # Compute the distance between each pair of locations
        # Reshape L into a Nx1xD matrix and a 1xNxD matrix, then add the requisite
        # broadcasting in order to subtract the two matrices
        L1 = self.Lm.dimshuffle(0,'x',1)     # Nx1xD
        L2 = self.Lm.dimshuffle('x',0,1)     # 1xNxD
        T.addbroadcast(L1,1)
        T.addbroadcast(L2,0)
        #self.D = T.sqrt(T.sum((L1-L2)**2, axis=2))
        #self.D = T.sum((L1-L2)**2, axis=2)

        # It seems we need to use L1 norm for now because
        # Theano doesn't properly compute the gradients of the L2
        # norm. (It gives NaNs because it doesn't realize that some
        # terms will cancel out)
        # self.D = (L1-L2).norm(1, axis=2)
        self.D = T.pow(L1-L2,2).sum(axis=2)

        # There is a distance scale, \delta
        self.delta = T.dscalar(name='delta')

        # Define complete adjacency matrix
        self.A = T.bmatrix('A')

        # The probability of A is exponentially decreasing in delta
        # self.pA = T.exp(-1.0*self.D/self.delta)
        self.pA = T.exp(-0.5*self.D/self.delta**2)

        if 'rho_refractory' in self.prms:
            self.pA += T.eye(self.N) * (self.prms['rho_refractory']-self.pA)
            # self.pA[np.diag_indices(self.N)] = self.prms['rho_refractory']

        # Allow for scaling the log likelihood of the graph so that we can do
        # Annealed importance sampling
        self.lkhd_scale = theano.shared(value=1.0, name='lkhd_scale')

        # Define log probability
        self.lkhd = T.sum(self.A * T.log(self.pA) + (1 - self.A) * T.log(1 - self.pA))
        # self.log_p = self.lkhd_scale * self.lkhd + self.location_prior.log_p(self.Lm)
        self.log_p = self.lkhd_scale * self.lkhd

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.A): self.A,
                # str(self.L): self.L,
                str(self.delta): self.delta}

    def sample(self, acc):
        N = self.model['N']

        # #  Sample locations from prior
        # L = self.location_prior.sample(None)
        #
        # # DEBUG!  Permute the neurons such that they are sorted along the first dimension
        # # This is only for data generation
        # if self.prms['sorted']:
        #     print "Warning: sorting the neurons by latent location. " \
        #           "Do NOT do this during inference!"
        #     perm = np.argsort(L[:,0])
        #     L = L[perm, :]
        # L = L.ravel()

        # TODO: Sample delta from prior
        delta = self.prms['delta']

        # Sample A from pA
        pA = self.pA.eval({ #self.L: L,
                           self.delta: delta})

        A = np.random.rand(N, N) < pA
        A = A.astype(np.int8)
        return {str(self.A): A,
                # str(self.L): L,
                str(self.delta): delta}

    def get_state(self):
        return {str(self.A): self.A,
                # str(self.L): self.Lm,
                str(self.delta): self.delta}