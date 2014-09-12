import sys
import hashlib
import cPickle
from os.path import expanduser

import theano
import theano.tensor as T

from pyglm.components.component import Component
from pyglm.utils.basis import *


def create_bkgd_component(model, glm, latent):
    type = model['bkgd']['type'].lower()
    if type == 'no_stimulus' or \
       type == 'none' or \
       type == 'nostimulus':
       bkgd = NoStimulus(model)
    elif type == 'basis':
        bkgd = BasisStimulus(model)
    elif type == 'spatiotemporal':
        bkgd = SpatiotemporalStimulus(model)
    elif type == 'sharedtuningcurve':
        bkgd = SharedTuningCurveStimulus(model, glm, latent)
    else:
        raise Exception("Unrecognized backgound model: %s" % type)
    return bkgd

class NoStimulus(Component):
    """ No stimulus dependence. Constant biases are handled by the 
        bias component. 
    """

    def __init__(self, model):
        """ No stimulus.
        """        
        # Log probability
        self.log_p = T.constant(0.0)

        # Due a theano quirk, I_stim cannot directly be a constant
        self.stim = T.constant(0.0)
        # Expose outputs to the Glm class
        self.I_stim = T.dot(T.constant(1.0), self.stim)
    
class BasisStimulus(Component):
    """ Filter the stimulus and expose the filtered stimulus
    """

    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.bkgd_model = model['bkgd']
        # Create a basis for the stimulus response
        self.basis = create_basis(self.bkgd_model['basis'])
        (_,B) = self.basis.shape

        # The basis is interpolated once the data is specified
        # self.ibasis = theano.shared(value=np.zeros((2,B)))
        self.initialize_basis()

        # Compute the number of parameters
        self.n_vars = B*self.bkgd_model['D_stim']
        
        # Create a spherical Gaussian prior on the weights
#        self.prior = create_prior(self.prms['prior'], name='w_stim', D=self.n_vars)
        self.w_stim = T.dvector('w_stim')

        # Store stimulus in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,self.n_vars)))

        # Log probability
#        self.log_p = self.prior.log_p
#         self.log_p = -0.5/0.1**2 *T.sum(T.pow(self.w_stim-0,2))
        self.log_p = T.sum(-0.5/(0.01**2) * (self.w_stim-0.0)**2)
#         self.log_p = 0.0

        # Expose outputs to the Glm class
#        self.I_stim = T.dot(self.stim,self.prior.value)
        self.I_stim = T.dot(self.stim,self.w_stim)

        # A symbolic variable for the for the stimulus response
#        self.stim_resp = T.dot(self.ibasis,self.prior.value)
        self.stim_resp = T.dot(self.ibasis,self.w_stim)
   
    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_stim): self.w_stim}
#        vs = {}
#        vs.update(self.prior.get_variables())
#        return vs
    
    def get_state(self):
        """ Get the stimulus response
        """
        return {'stim_response' : self.stim_resp,
                'basis' : self.ibasis}

    def initialize_basis(self):
        (_,B) = self.basis.shape

        # Interpolate stimulus at the resolution of the data
        dt = self.model['dt']

        # Interpolate basis at the resolution of the data
        (L,B) = self.basis.shape
        Lt_int = self.bkgd_model['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, self.basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if self.bkgd_model['basis']['norm']:
            ibasis = ibasis / np.tile(np.sum(ibasis,0),[Lt_int,1])

        self.ibasis = theano.shared(value=ibasis)

    def preprocess_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Check data shape
        if not abs(data['stim'].shape[0] * data['dt_stim'] - data['T']) < data['dt_stim']:
            raise Exception('Stimulus length is not the same as data time length!')

        if not self.bkgd_model['D_stim'] == data['stim'].shape[1]:
            raise Exception("Stim dimension (%d) is not equal to that specified by model (%d)" % (data['stim'].shape[1], self.bkgd_model['D_stim']))

        # Interpolate stimulus at the resolution of the data
        D_stim = self.bkgd_model['D_stim']
        dt = self.model['dt']
        dt_stim = self.bkgd_model['dt_stim']
        t = dt * np.arange(data['S'].shape[0])
        t_stim = dt_stim * np.arange(data['stim'].shape[0])
        stim = np.zeros((len(t), D_stim))
        for d in np.arange(D_stim):
            stim[:, d] = np.interp(t,
                                   t_stim,
                                   data['stim'][:, d])

        # Project the stimulus onto the basis
        cstim = convolve_with_basis(stim, self.ibasis.get_value())

        # Flatten this manually (there's surely a way to do this with numpy)
        (nT,D,B) = cstim.shape 
        fstim = np.empty((nT,D*B))
        for d in np.arange(D):
            for b in np.arange(B):
                fstim[:,d*B+b] = cstim[:,d,b]

        data['fstim'] = fstim

    def set_data(self, data):
        self.stim.set_value(data['fstim'])

    def set_hyperparameters(self, model):
        """ Set hyperparameters of the model
        """
        pass

    def sample(self, acc):
        """
        return a sample of the variables
                """
        smpl = {str(self.w_stim) : 0.01*np.random.randn(self.n_vars)}
        return smpl


class SpatiotemporalStimulus(Component):
    """ Filter the stimulus with a 2D spatiotemporal filter. Approximate the
        spatiotemporal filter with a rank-2 approximation, namely f(t,x) = f(t)*f(x)'.

    """

    def __init__(self, model):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.bkgd_model = model['bkgd']
        self.mu = self.bkgd_model['mu']
        self.sigma = self.bkgd_model['sigma']

        # Create a basis for the stimulus response
        self.spatial_basis = create_basis(self.bkgd_model['spatial_basis'])
        (_,Bx) = self.spatial_basis.shape

        self.temporal_basis = create_basis(self.bkgd_model['temporal_basis'])
        (_,Bt) = self.temporal_basis.shape

        # Save the filter sizes
        self.Bx = Bx
        self.Bt = Bt

        # The basis is interpolated once the data is specified
        # self.ibasis_x = theano.shared(value=np.zeros((2,Bx)))
        # self.ibasis_t = theano.shared(value=np.zeros((2,Bt)))
        self.initialize_basis()

        # Compute the number of parameters
        self.n_vars = Bx+Bt

        # Store stimulus (after projection onto the spatiotemporal basis, in shared memory
        self.stim = theano.shared(name='stim',
                                  value=np.zeros((1,Bx*Bt)))

        # Get the two factors of the stimulus response
        self.w_x = T.dvector('w_x')
        self.w_t = T.dvector('w_t')

        # The weights are an outer product of the two factors
        w_op = T.dot(T.shape_padright(self.w_t, n_ones=1),
                     T.shape_padleft(self.w_x, n_ones=1))
        # w_op = T.dot(self.w_x, T.transpose(self.w_t))

        # Flatten the outer product to get a combined weight for each
        # pair of spatial and temporal basis functions.
        self.w_stim = T.reshape(w_op, (Bt*Bx,))

        # Log probability
        self.log_p = -0.5/self.sigma**2 *T.sum((self.w_x-self.mu)**2) + \
                     -0.5/self.sigma**2 *T.sum((self.w_t-self.mu)**2)

        # Expose outputs to the Glm class
        self.I_stim = T.dot(self.stim, self.w_stim)


        # Create function handles for the stimulus responses
        self.stim_resp_t = T.dot(self.ibasis_t,self.w_t)
        self.stim_resp_x = T.dot(self.ibasis_x,self.w_x)

    def get_variables(self):
        """ Get the theano variables associated with this model.
        """
        return {str(self.w_x): self.w_x,
                str(self.w_t): self.w_t}

    def sample(self, acc):
        """
        return a sample of the variables
                """

        w_x = self.mu + self.sigma * np.random.randn(self.Bx)
        w_t = self.mu + self.sigma * np.random.randn(self.Bt)
        return {str(self.w_x) : w_x,
                str(self.w_t) : w_t}

    def get_state(self):
        """ Get the stimulus response
        """
        # The filters are non-identifiable as we can negate both the
        # temporal and the spatial filters and get the same net effect.
        # By convention, choose the sign that results in the most
        # positive temporal filter.
        sign = T.sgn(T.sum(self.stim_resp_t))

        # Similarly, we can trade a constant between the spatial and temporal
        # pieces. By convention, set the temporal filter to norm 1.
        Z = self.stim_resp_t.norm(2)
        stim_resp_t = sign*(1.0/Z)*self.stim_resp_t

        # Finally, reshape the spatial component as necessary
        if 'shape' in self.bkgd_model:
            stim_resp_x = sign*Z*T.reshape(self.stim_resp_x, self.bkgd_model['shape'])
        else:
            stim_resp_x = sign*Z*self.stim_resp_x

        return {'stim_response_x' : stim_resp_x,
                'stim_response_t' : stim_resp_t,
                'basis_t' : self.ibasis_t}

    def initialize_basis(self):
        # Interpolate stimulus at the resolution of the data
        dt = self.model['dt']

        # Interpolate basis at the resolution of the data
        (Lt,Bt) = self.temporal_basis.shape
        Lt_int = self.bkgd_model['dt_max']/dt
        t_int = np.linspace(0,1,Lt_int)
        t_bas = np.linspace(0,1,Lt)
        ibasis_t = np.zeros((len(t_int), Bt))
        for b in np.arange(Bt):
            ibasis_t[:,b] = np.interp(t_int, t_bas, self.temporal_basis[:,b])

        (Lx,Bx) = self.spatial_basis.shape
        Lx_int = self.bkgd_model['D_stim']
        x_int = np.linspace(0,1,Lx_int)
        x_bas = np.linspace(0,1,Lx)
        ibasis_x = np.zeros((len(x_int), Bx))
        for b in np.arange(Bx):
            ibasis_x[:,b] = np.interp(x_int, x_bas, self.spatial_basis[:,b])

        # Normalize so that the interpolated basis has unit L1 norm
        if self.bkgd_model['temporal_basis']['norm']:
            ibasis_t = ibasis_t / np.tile(np.sum(ibasis_t,0),[Lt_int,1])

        # Save the interpolated bases
        self.ibasis_t = theano.shared(value=ibasis_t)
        self.ibasis_x = theano.shared(value=ibasis_x)

    def preprocess_data(self, data):
        dt = self.model['dt']
        dt_stim = data['dt_stim']
        t = np.arange(0, data['T'], dt)
        nt = len(t)
        t_stim = dt_stim * np.arange(data['stim'].shape[0])
        stim = np.zeros((nt, self.bkgd_model['D_stim']))
        for d in np.arange(self.bkgd_model['D_stim']):
            stim[:, d] = np.interp(t,
                                   t_stim,
                                   data['stim'][:, d])


        # Take all pairs of temporal and spatial basis vectors
        ibasis_t = self.ibasis_t.get_value()
        ibasis_x = self.ibasis_x.get_value()
        (_,Bt) = ibasis_t.shape
        (_,Bx) = ibasis_x.shape

        # Filter the stimulus with each spatiotemporal filter combo
        # fstim = np.empty((nt,Bt,Bx))
        # for bt in np.arange(Bt):
        #     for bx in np.arange(Bx):
        #         # atleast_2d gives row vectors
        #         bas = np.dot(np.atleast_2d(ibasis_t[:,bt]).T,
        #                      np.atleast_2d(ibasis_x[:,bx]))
        #         fstim[:,bt,bx] = convolve_with_2d_basis(stim, bas)

        # Leverage low rank to speed up convolutions
        print "Convolving the stimulus with the low rank filters"
        fstim = convolve_with_low_rank_2d_basis(stim, ibasis_x, ibasis_t)

        # Permute output to get shape(T,Bt,Bx)
        assert fstim.shape == (nt,Bx,Bt)
        fstim = np.transpose(fstim, axes=[0,2,1])

        # Flatten the filtered stimulus
        data['fstim'] = np.reshape(fstim,(nt,Bt*Bx))

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        self.stim.set_value(data['fstim'])


class SharedTuningCurveStimulus(Component):
    """
    Filter the stimulus with a set of shared tuning curves
    """
    def __init__(self, model, glm, latent):
        """ Initialize the filtered stim model
        """
        self.model = model
        self.bkgd_model = model['bkgd']
        self.n = glm.n
        self.tuningcurves = latent[self.bkgd_model['tuningcurves']]
        self.spatial_basis = self.tuningcurves.spatial_basis
        self.tc_spatial_shape = self.tuningcurves.spatial_shape
        self.tc_spatial_ndim = self.tuningcurves.spatial_ndim
        self.temporal_basis = self.tuningcurves.temporal_basis
        self.Bx = self.tuningcurves.Bx
        self.Bt = self.tuningcurves.Bt
        self.w_x = self.tuningcurves.w_x[:,self.tuningcurves.Y[self.n]]
        self.w_t = self.tuningcurves.w_t[:,self.tuningcurves.Y[self.n]]

        # Create a shared variable for the filtered stimulus. This is a 4D
        # tensor with dimensions:
        #   - time
        #   - location (pixel)
        #   - spatial basis
        #   - temporal basis
        # To get a stimulus current we need to select a location and take a
        # weighted sum along both the spatial and temporal axes.
        self.filtered_stim = theano.shared(name='stim',
                                           value=np.ones((1,1,1,1)))

        self.locations = latent[self.bkgd_model['locations']]
        self.L = self.locations.Lmatrix[self.n,:]
        self.loc_index = self.locations.location_prior.ravel_index(self.L)


        # Expose outputs to the Glm class

        # It matters that we do the dot products in order of outermost
        # to innermost dimension. This improves memory efficiency.
        # Compute the spatially filtered stimulus
        # Result is T x L x B_t
        self.I_stim_t = T.tensordot(self.filtered_stim,
                          self.w_t,
                          axes=[[3],[0]])
        self.I_stim_t.name = 'I_stim_t'

        # Take dot product with temporal basis coefficients
        # Result is T x L (where L is number of locations)
        self.I_stim_xt = T.tensordot(self.I_stim_t,
                          self.w_x,
                          axes=[[2],[0]])
        self.I_stim_xt.name = 'I_stim_xt'

        self.I_stim = self.I_stim_xt[:, self.loc_index]
        self.I_stim.name = 'I_stim'

        # There are no latent variables in this class. They all belong
        # to global latent variables.
        self.log_p = T.constant(0.0)


    def get_state(self):
        """ Get the theano variables associated with this model.
        """
        return {'I_stim_xt' : self.I_stim_xt}

    def preprocess_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        assert data['stim'].ndim == 1+self.tc_spatial_ndim
        dt = self.model['dt']
        dt_stim = data['dt_stim']
        t = np.arange(0, data['T'], dt)
        nt = len(t)
        temporal_shape = data['stim'].shape[0]
        spatial_shape = data['stim'].shape[1:]
        spatial_ndim = len(spatial_shape)
        D_stim = np.prod(spatial_shape)

        # Interpolate the stimulus so that it is defined on the same
        # time grid as t
        t_stim = dt_stim * np.arange(temporal_shape)
        stim = np.zeros((nt,) + spatial_shape)
        if spatial_ndim == 1:
            for d in np.arange(spatial_shape[0]):
                stim[:, d] = np.interp(t,
                                       t_stim,
                                       data['stim'][:, d])
        elif spatial_ndim ==2:
            for d1 in np.arange(spatial_shape[0]):
                for d2 in np.arange(spatial_shape[1]):
                    stim[:, d1, d2] = np.interp(t,
                                                t_stim,
                                                data['stim'][:, d1, d2])

        # Filter the stimulus with each spatiotemporal filter combo
        # Look for cached version of filtered stimulus
        fstim_cached = False
        pyglm_home = os.path.join(expanduser("~"), '.pyglm')
        hash = hashlib.sha1(stim)
        hash.update(self.spatial_basis)
        hash.update(self.temporal_basis)
        keystr = hash.hexdigest()
        cachefile = os.path.join(pyglm_home, "%s.pkl" % keystr)


        if os.path.exists(pyglm_home):
            if os.path.exists(cachefile):
                print "Found filtered stim in cache"
                with open(cachefile, 'r') as f:
                    fstim = cPickle.load(f)
                    fstim_cached = True

        if not fstim_cached:
            fstim = np.empty((nt, D_stim, self.Bx, self.Bt))
            print "Convolving stimulus with 3D basis"

            for bx in np.arange(self.Bx):
                for bt in np.arange(self.Bt):
                    # Print progress
                    sys.stdout.write('.')
                    sys.stdout.flush()

                    if self.tc_spatial_ndim == 1:
                        # atleast_2d gives row vectors
                        bas = np.dot(np.atleast_2d(self.tuningcurves.interpolated_temporal_basis[:,bt]).T,
                                     np.atleast_2d(self.tuningcurves.interpolated_spatial_basis[:,bx]))
                        fstim[:,:,bx,bt] = convolve_with_2d_basis(stim, bas, ['first', 'central'])
                    elif self.tc_spatial_ndim == 2:
                        # Take the outerproduct to make a 3D tensor
                        bas_xy = self.tuningcurves.interpolated_spatial_basis[:,bx].reshape((1,) + self.tc_spatial_shape )
                        bas_t = self.tuningcurves.interpolated_temporal_basis[:,bt].reshape((-1,1))
                        bas = np.tensordot(bas_t, bas_xy, [1,0])
                        fconv3d = convolve_with_3d_basis(stim, bas, ['first', 'central', 'central'])
                        fstim[:,:,bx,bt] = fconv3d.reshape((nt, D_stim))

                    else:
                        raise Exception('spatial dimension must be <= 2D')
            sys.stdout.write(" Done\n")

            # Save the file to cache
            if os.path.exists(pyglm_home):
                print "Saving filtered stim to cache file at: ", cachefile
                print "Expected file size: %.2f Gb" % (float(fstim.size*8)/2**30)
                with open(cachefile, 'w') as f:
                    cPickle.dump(fstim, f, protocol=-1)


        assert fstim.shape == (nt, D_stim, self.Bx, self.Bt)
        data['fstim'] = fstim

    def set_data(self, data):
        self.filtered_stim.set_value(data['fstim'])

