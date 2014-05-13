import theano
import theano.tensor as T
import numpy as np

from components.component import Component

def create_prior(model, **kwargs):
    typ = model['type'].lower()
    if typ == 'normal' or \
       typ == 'gaussian':
       return Gaussian(model, **kwargs)
    elif typ == 'spherical_gaussian':
        return SphericalGaussian(model, **kwargs)
    elif typ == 'group_lasso' or \
         typ == 'grouplasso':
        return GroupLasso(model, **kwargs)
    elif typ == 'dpp':
        return DeterminenalPointProcess(model, **kwargs)
    else:
        raise Exception("Unrecognized prior type: %s" % typ)

class Gaussian(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model, name='gaussian'):
        self.prms = model
 #       self.value = T.dscalar(name=name)
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])

#        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)
    
    def log_p(self, value):
        """ Compute log prob of the given value under this prior
        """
        return -0.5/self.sigma**2  * T.sum((value-self.mu)**2)

    def get_variables(self):
#        return {str(self.mu): self.mu, 
#                str(self.sigma) : self.sigma}
        return {}


    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.mu.set_value(model['mu'])
        self.sigma.set_value(model['sigma'])

    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(*size)
        return v

class SphericalGaussian(Component):
    """ Wrapper for a vector random variable with a spherical distribution
    """
    def __init__(self, model, name='spherical_gaussian', D=1):
        self.prms = model
        self.D = D
        self.value = T.dvector(name=name)
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])
        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)

    def get_variables(self):
        return {str(self.value): self.value}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        # TODO Fix me
        self.mu.set_value(model['mu'])
        self.sigma.set_value(model['sigma'])

    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        # TODO Use size
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(self.D)
        return {str(self.value): v}

class GroupLasso(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model, name='gaussian'):
        self.prms = model
        self.lam = theano.shared(name='lam', value=self.prms['lam'])
        self.mu = theano.shared(name='mu', value=self.prms['mu'])
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])
    
    def log_p(self, value):
        """ Compute log prob of the given value under this prior
            Value should be NxB where N is the number of groups and
            B is the number of parameters per group.
        """
        return -1.0*self.lam * T.sum(T.sqrt(T.sum(((value-self.mu)/self.sigma)**2, axis=1)))


    def get_variables(self):
        return {}

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model 
        """
        self.mu.set_value(model['mu'])
        self.sigma.set_value(model['sigma'])
        self.lam.set_value(model['lam'])
        
    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        N = size[0]
        norms = np.random.laplace(0, self.lam.get_value(), size=(N,1))
        v = self.mu.get_value() + self.sigma.get_value() * np.random.randn(*size)
        v_norms = np.sqrt(np.sum(v**2,axis=1)).reshape(N,1)
        vf = (v*norms/v_norms)

        return vf

class DeterminenalPointProcess(Component):
    """ Wrapper for a scalar random variable with a Normal distribution
    """
    def __init__(self, model):
        self.prms = model
        self.sigma = theano.shared(name='sigma', value=self.prms['sigma'])
        self.bound = theano.shared(name='bound', value=self.prms['bound'])

#        self.log_p = -0.5/self.sigma**2 *T.sum((self.value-self.mu)**2)
        from theano.sandbox.linalg.ops import Det
        self.d = Det()

        # Prior is multiplied by a spherical Gaussian with standard deviation
        # of 'bound' to prevent points from diverging to infinity
        self.gaussian = Gaussian({'mu' : 0, 'sigma' : self.prms['bound']})

    def log_p(self, L):
        """ Compute log prob of the given value under this prior
            Input: L ~ NxD
        """
        assert L.ndim == 2, "L must be 2d!"
        # Compute pairwise L2 norm
        L1 = L.dimshuffle(0,'x',1)     # Nx1xD
        L2 = L.dimshuffle('x',0,1)     # 1xNxD
        T.addbroadcast(L1,1)
        T.addbroadcast(L2,0)

        # Compute pairwise distances
        D = ((L1-L2)**2).sum(axis=2)

        # Compute the kernel
        K = T.exp(-D / self.sigma**2)

        # Log prob is the log determinant of the pairwise distances
        lp_det = T.log(self.d(K))

        # Also multiply by a spherical Gaussian with standard deviation of 'bound'
        # to prevent points from diverging to infinity
        lp_gauss = self.gaussian.log_p(L)

        return lp_det + lp_gauss

    def get_variables(self):
#        return {str(self.mu): self.mu,
#                str(self.sigma) : self.sigma}
        return {}


    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.sigma.set_value(model['sigma'])

    def sample(self, size=(1,)):
        """ Sample from the prior
        """
        N,D = size

        # TODO: Actually sample a DPP
        # TODO: For now we just run a Markov chain to sample
        L = T.dmatrix('L')
        lp = self.log_p(L)
        glp = T.grad(lp, L)

        f_lp = lambda x: -1.0*lp.eval({L : x.reshape((N,D))})
        f_glp = lambda x: -1.0*glp.eval({L : x.reshape((N,D))}).reshape(N*D)

        # x0 = L1.reshape(N*D)
        from inference.hmc import hmc
        x = self.bound.get_value() * np.random.randn(N*D)
        N_smpls = 1000
        for s in np.arange(N_smpls):
            x = hmc(f_lp, f_glp, 0.25, 1, x)

        assert np.all(np.isfinite(x))

        return x

def test_determinant():
    sigma = 5.0
    bound = 5.0
    det_prior = DeterminenalPointProcess({'sigma' : sigma, 'bound' : bound})
    L = T.dmatrix('L')
    lp = det_prior.log_p(L)
    glp = T.grad(lp, L)

    L1 = np.arange(6,step=1).reshape((3,2)).astype(np.float)
    print "theano lp: %f" % lp.eval({L : L1})

    # D1 = (L1 - L1.T)**2
    # lp_L1_test = np.log(np.linalg.det(np.exp(-D1))) + -0.5/10**2 * np.sum(L1**2)
    # print "test lp: %f" % lp_L1_test
    #
    # L2 = np.random.rand(3).reshape((3,1))
    # print "theano lp: %f" % lp.eval({L : L2})
    #
    # D2 = (L2 - L2.T)**2
    # lp_L2_test = np.log(np.linalg.det(np.exp(-D2))) + -0.5/10**2 * np.sum(L2**2)
    # print "test lp: %f" % lp_L2_test
    #
    # # TODO: Test 2d L
    #
    # # Compute gradients
    # print "theano glp: ",  glp.eval({L : L1})
    # print "theano glp: ",  glp.eval({L : L2})
    from inference.hmc import hmc
    N = 3
    D = 2
    f_lp = lambda x: -1.0*lp.eval({L : x.reshape((N,D))})
    f_glp = lambda x: -1.0*glp.eval({L : x.reshape((N,D))}).reshape(N*D)

    # x0 = L1.reshape(N*D)
    x0 = bound * np.random.randn(N*D)
    N_smpls = 1000
    smpls = [x0]
    for s in np.arange(N_smpls):
        x_next = hmc(f_lp, f_glp, 0.25, 1, smpls[-1])
        # print "Iteration %d:" % s
        # print x_next
        smpls.append(x_next)

    # Make a movie of the samples

    import pdb; pdb.set_trace()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    x0 = smpls[0].reshape((N,D))
    l = plt.plot(x0[:,0], x0[:,1], 'ko')
    plt.xlim(-4*bound, 4*bound)
    plt.ylim(-4*bound, 4*bound)

    with writer.saving(fig, "dpp_hmc_smpl.mp4", 100):
        for i in range(N_smpls):
            xi = smpls[i].reshape((N,D))
            l[0].set_data(xi[:,0], xi[:,1])
            writer.grab_frame()

if __name__ == "__main__":
    test_determinant()