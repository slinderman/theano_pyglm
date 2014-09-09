"""
Provide latent types that can be used by the graph/bkgd/impulse models
"""
import numpy as np

import theano.tensor as T
from component import Component
from pyglm.utils.basis import create_basis
from pyglm.components.priors import create_prior


def create_latent_component(model, **kwargs):
    typ = model['type'].lower()
    if typ == 'latent_type' or \
       typ == 'latenttype':
       return LatentType(model, **kwargs)
    elif typ == 'latent_type_with_tuning_curves' or \
       typ == 'latenttypewithtuningcurves':
       return LatentTypeWithTuningCurve(model, **kwargs)
    elif typ == 'latent_location' or \
         typ == 'latentlocation':
       return LatentLocation(model, **kwargs)
    else:
        raise Exception("Unrecognized latent component type: %s" % typ)

class LatentVariables(Component):
    """
    Container for a set of latent variables, e.g. neuron types/locations
    """
    def __init__(self, model):
        """
        Go through the items in the model, each of which specifies a latent variable component
        """
        if 'latent' in model.keys():
            self.prms = model['latent']
        else:
            self.prms = {}

        self.log_p = T.constant(0.0)

        # Enumerate and create latent variable component
        self.latentlist = []
        self.latentdict = {}

        for (k,v) in self.prms.items():
            # Create the latent component
            latent_component = create_latent_component(v)
            self.log_p += latent_component.log_p

            # Add to the list of latent variable components
            self.latentlist.append(latent_component)
            self.latentdict[latent_component.name] = latent_component

    def get_variables(self):
        v = {}
        for (name, comp) in self.latentdict.items():
            v[name] = comp.get_variables()

        return v

    def get_state(self):
        st = {}
        for (name, comp) in self.latentdict.items():
            st[name] = comp.get_state()

        return st

    def sample(self, acc):
        s = {}
        for (name, comp) in self.latentdict.items():
            s[name] = comp.sample(acc)
        return s

    def set_data(self, data):
        for comp in self.latentlist:
            comp.set_data(data)

    # Allow consumers to access this container as a dict
    def __getitem__(self, item):
        return self.latentdict[item]

class LatentType(Component):
    def __init__(self, model):
        self.prms = model
        self.name = self.prms['name']

        # There are N neurons to assign types to
        self.N = model['N']

        # There are has R latent types
        self.R = self.prms['R']
        # Each neuron has a latent type Y
        self.Y = T.lvector('Y')

        # A probability of each type with a symmetric Dirichlet prior
        self.alpha = T.dvector('alpha')
        self.alpha_prior = create_prior(self.prms['alpha_prior'])
        # self.alpha0 = self.prms['alpha0']

        # Define log probability
        # log_p_alpha = T.sum((self.alpha0 - 1) * T.log(self.alpha))
        log_p_alpha = self.alpha_prior.log_p(self.alpha)
        log_p_Y = T.sum(T.log(self.alpha[self.Y]))

        self.log_p = log_p_alpha + log_p_Y

    def get_variables(self):
        return {str(self.alpha) : self.alpha,
                str(self.Y) : self.Y}

    def get_state(self):
        return {str(self.alpha) : self.alpha,
                str(self.Y) : self.Y}

    def sample(self, acc):
        """
        return a sample of the variables
        """
        # Sample alpha from a Dirichlet prior
        # alpha = np.random.dirichlet(self.alpha0*np.ones(self.R))
        alpha = self.alpha_prior.sample(acc)

        # Sample Y from categorical dist
        Y = np.random.choice(self.R, size=self.N, p=alpha)
        return {str(self.alpha) : alpha,
                str(self.Y) : Y}

class LatentTypeWithTuningCurve(LatentType):
    """
    Extent the basic latent type component to also include tuning curves
    """
    def __init__(self, model):
        super(LatentTypeWithTuningCurve, self).__init__(model)

        # Also initialize the tuning curves
        self.mu = self.prms['mu']
        self.sigma = self.prms['sigma']

        # Create a basis for the stimulus response
        self.spatial_basis = create_basis(self.prms['spatial_basis'])
        self.spatial_shape = self.prms['spatial_shape']
        self.spatial_ndim = len(self.spatial_shape)
        (_,Bx) = self.spatial_basis.shape

        self.temporal_basis = create_basis(self.prms['temporal_basis'])
        (_,Bt) = self.temporal_basis.shape

        # Save the filter sizes
        self.Bx = Bx
        self.Bt = Bt

        # Initialize RxBx and RxBt matrices for the per-type tuning curves
        self.w_x = T.dmatrix('w_x')
        self.w_t = T.dmatrix('w_t')

        # Create function handles for the stimulus responses
        self.stim_resp_t = T.dot(self.temporal_basis, self.w_t)
        self.stim_resp_x = T.dot(self.spatial_basis, self.w_x)

        # Add the probability of these tuning curves to the log probability
        self.log_p += -0.5/self.sigma**2 *T.sum((self.w_x-self.mu)**2) + \
                      -0.5/self.sigma**2 *T.sum((self.w_t-self.mu)**2)

    def get_variables(self):
        v = super(LatentTypeWithTuningCurve, self).get_variables()

        v.update({str(self.w_x): self.w_x,
                  str(self.w_t): self.w_t})
        return v

    def get_state(self):
        st = super(LatentTypeWithTuningCurve, self).get_state()

        # The filters are non-identifiable as we can negate both the
        # temporal and the spatial filters and get the same net effect.
        # By convention, choose the sign that results in the most
        # positive temporal filter.
        sign = T.sgn(T.sum(self.stim_resp_t, axis=0))
        T.addbroadcast(sign, 0)

        # Similarly, we can trade a constant between the spatial and temporal
        # pieces. By convention, set the temporal filter to norm 1.
        Z = T.sqrt(T.sum(self.stim_resp_t**2, axis=0))
        T.addbroadcast(Z, 0)

        # Compute the normalized temporal response
        stim_resp_t = sign*(1.0/Z)*self.stim_resp_t

        # Finally, reshape the spatial component as necessary
        if self.spatial_ndim == 2:
            stim_resp_x = sign*Z*self.stim_resp_x
            stim_resp_x = T.reshape(stim_resp_x,
                                    self.spatial_shape + (self.R,))
        else:
            stim_resp_x = sign*Z*self.stim_resp_x

        st.update({'stim_response_x' : stim_resp_x,
                   'stim_response_t' : stim_resp_t})

        return st

    def sample(self, acc):
        """
        Return a sample of the types and tuning curves
        """
        s = super(LatentTypeWithTuningCurve, self).sample(acc)
        w_x = self.mu + self.sigma * np.random.randn(self.Bx, self.R)
        w_t = self.mu + self.sigma * np.random.randn(self.Bt, self.R)
        # w_x = np.zeros((self.Bx, self.R))
        # w_x[4,:] = 10

        # for r in range(self.R):
        #     w_x[:,r] = (r+1)*np.ones((self.Bx,))
        # w_x[0,:] = 1.0
        # w_t = np.ones((self.Bt, self.R))

        s.update({str(self.w_x) : w_x,
                  str(self.w_t) : w_t})
        return s

    def set_data(self, data):
        # Interpolate stimulus at the resolution of the data
        dt = data['dt']

        # Interpolate the temporal basis at the resolution of the data
        (Lt,Bt) = self.temporal_basis.shape
        Lt_int = self.prms['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,Lt)
        ibasis_t = np.zeros((len(t_int), Bt))
        for b in np.arange(Bt):
            ibasis_t[:,b] = np.interp(t_int, t_bas, self.temporal_basis[:,b])

        # Normalize so that the interpolated basis has unit L1 norm
        if self.prms['temporal_basis']['norm']:
            ibasis_t = ibasis_t / np.tile(np.sum(ibasis_t,0),[Lt_int,1])

        # Save the interpolated bases
        self.interpolated_temporal_basis = ibasis_t
        self.interpolated_spatial_basis = self.spatial_basis


class LatentLocation(Component):
    """
    The latent location of a neuron in either real or abstract space
    """
    def __init__(self, model):
        self.prms = model
        self.name = self.prms['name']
        self.N_dims = self.prms['N_dims']
        self.N = model['N']

        # Create a location prior
        self.location_prior = create_prior(self.prms['location_prior'])

        # Make sure the sample is of the correct type
        from pyglm.components.priors import Categorical, JointCategorical
        if isinstance(self.location_prior, (Categorical,JointCategorical)):
            self.Lflat = T.lvector('L')
            self.dtype = np.int
        else:
            self.Lflat = T.dvector('L')
            self.dtype = np.float

        # Latent distance model has NxN_dims matrix of locations Lm
        self.Lmatrix = T.reshape(self.Lflat, (self.N, self.N_dims))
        self.Lmatrix.name = 'L'

        # Define log probability
        self.log_p = self.location_prior.log_p(self.Lmatrix)

    def get_variables(self):
        return {str(self.Lflat) : self.Lflat}

    def get_state(self):
        return {str(self.Lmatrix) : self.Lmatrix}

    def sample(self, acc):
        """
        return a sample of the variables
        """
        #  Sample locations from prior
        L = self.location_prior.sample(None, self.N)

        # print Warning("Debugging locations!")
        # L[0,0] = 2
        # L[0,1] = 2

        # DEBUG!  Permute the neurons such that they are sorted along the first dimension
        # This is only for data generation
        # if self.prms['sorted']:
        #     print "Warning: sorting the neurons by latent location. " \
        #           "Do NOT do this during inference!"
        #     perm = np.argsort(L[:,0])
        #     L = L[perm, :]
        L = L.ravel()

        return {str(self.Lflat) : L}
