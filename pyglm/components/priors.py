import numpy as np

import theano
import theano.tensor as T
from pyglm.components.component import Component
from pyglm.inference import hmc


def create_prior(model, **kwargs):
    typ = model['type'].lower()
    if typ == 'normal' or \
       typ == 'gaussian':
       return Gaussian(model, **kwargs)
    elif typ == 'categorical':
       return Categorical(model, **kwargs)
    elif typ == 'jointcategorical' or \
         typ == 'joint_categorical':
       return JointCategorical(model, **kwargs)
    elif typ == 'spherical_gaussian':
        return SphericalGaussian(model, **kwargs)
    elif typ == 'group_lasso' or \
         typ == 'grouplasso':
        return GroupLasso(model, **kwargs)
    elif typ == 'dpp':
        return DeterminenalPointProcess(model, **kwargs)
    elif typ == 'dirichlet':
        return Dirichlet(model)
    else:
        raise Exception("Unrecognized prior type: %s" % typ)



class Categorical(Component):
    """ Wrapper for a discrete random variable from a Categorical distribution
    """
    def __init__(self, model):
        self.prms = model
        self.p = theano.shared(name='p', value=self.prms['p'])
        self.min = self.prms['min']
        self.max = self.prms['max']

    def ravel_index(self,value):
        return value - self.min

    def log_p(self, value):
        """ Compute log prob of the given value under this prior
        """
        lp = T.constant(0.)
        lp += -np.Inf * T.lt(value, self.min)
        lp += -np.Inf * T.gt(value, self.max)
        lp += self.p[value]
        return lp

    def get_variables(self):
        return {}

    def sample(self, acc, size=(1,)):
        """ Sample from the prior
        """
        v = np.random.choice(self.p, size=size)
        return v


class JointCategorical(Component):
    """
    Wrapper for a discrete random variable from a product distribution of
    two categorical distributions.
    """
    def __init__(self, model):
        self.prms = model
        self.p0 = theano.shared(name='p0', value=self.prms['p0'])
        self.p1 = theano.shared(name='p1', value=self.prms['p1'])
        self.min0 = self.prms['min0']
        self.max0 = self.prms['max0']
        self.min1 = self.prms['min1']
        self.max1 = self.prms['max1']

    def ravel_index(self, value):
        return (self.max1-self.min1+1)*(value[0]-self.min0) + (value[1]-self.min1)

    def log_p(self, value):
        """ Compute log prob of the given value under this prior
        """
        # Add up the log probs from each value

        # Check for any values outside the given range
        # oob = (T.any(T.lt(value[:,0], self.min1))) or \
        #       (T.any(T.gt(value[:,0], self.max1))) or \
        #       (T.any(T.lt(value[:,1], self.min2))) or \
        #       (T.any(T.gt(value[:,1], self.max2)))
        # lp += -np.Inf * T.sum(T.lt(value[:,0], self.min1))
        # lp += -np.Inf * T.sum(T.gt(value[:,0], self.max1))
        # lp += -np.Inf * T.sum(T.lt(value[:,1], self.min2))
        # lp += -np.Inf * T.sum(T.gt(value[:,1], self.max2))

        lp = T.sum(T.log(self.p0[value[:,0]]))
        lp += T.sum(T.log(self.p1[value[:,1]]))

        # from theano.ifelse import ifelse
        # lp = ifelse(oob, T.constant(-np.Inf, dtype=np.float64), lp)


        # oob1 = T.or_(T.lt(value[:,0], self.min1),
        #             T.gt(value[:,0], self.max1))
        # oob2 = T.or_(T.lt(value[:,1], self.min2),
        #              T.gt(value[:,1], self.max2))
        #
        # T.where(oob1, -np.Inf, )

        return lp

    def get_variables(self):
        return {}

    def sample(self, acc, size=(1,)):
        """ Sample from the prior
        """
        v1 = np.random.choice(np.arange(self.min0, self.max0+1),
                              p=self.p0.get_value(), size=size).astype(np.int)
        v2 = np.random.choice(np.arange(self.min1, self.max1+1),
                              p=self.p1.get_value(), size=size).astype(np.int)

        v = np.concatenate((v1[:,None], v2[:,None]), axis=1)
        return v


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

    def sample(self, acc, size=(1,)):
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

    def sample(self, acc):
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
        
    def sample(self, acc, size=(1,)):
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

    def sample(self, acc, size=(1,)):
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
        x = self.bound.get_value() * np.random.randn(N*D)
        N_smpls = 1000
        for s in np.arange(N_smpls):
            x = hmc(f_lp, f_glp, 0.25, 1, x)

        assert np.all(np.isfinite(x))

        return x.reshape((N,D))

class Dirichlet(Component):
    """ Wrapper for a random vector from a Dirichlet distribution
    """
    def __init__(self, model):
        self.prms = model

        a = self.prms['alpha0']
        if np.isscalar(a):
            # Specified a symmetric Dirichlet prior
            R = self.prms['R']
            a = a * np.ones(R)
        else:
            assert a.ndim == 1

        self.alpha0 = theano.shared(name='alpha0', value=a)

    def log_p(self, value):
        """ Compute log prob of the given value under this prior
        """
        lp = T.sum((self.alpha0 - 1) * T.log(value))
        return lp

    def get_variables(self):
        return {}

    def sample(self, acc, size=None):
        """ Sample from the prior
        """
        alpha = np.random.dirichlet(self.alpha0.get_value(), size=size)
        return alpha


class DirichletMultinomial(Dirichlet):
    """
    Compound of Multinomial distribution with Dirichlet prior
    """
    def __init__(self, model):
        super(DirichletMultinomial, self).__init__(model)

        # Vector of multinomial observations
        self.N = model['N']
        self.Y = T.lvector('Y')

    def log_p(self, value):
        lp = super(DirichletMultinomial, self).log_p(value)
        lp += T.sum(T.log(self.alpha[self.Y]))
        return lp

    def sample(self, acc, size=None):
        alpha = super(DirichletMultinomial, self).sample(acc)
        Y = np.random.multinomial(self.N, alpha, size=size)

        return alpha, Y

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
    from pyglm.inference.hmc import hmc
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