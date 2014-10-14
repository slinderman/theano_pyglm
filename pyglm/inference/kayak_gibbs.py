""" Fit a Network GLM with MAP estimation. For some models, the log posterior
    is concave and has a unique maximum.
"""

import copy

from scipy.misc import logsumexp
from scipy.integrate import cumtrapz

from pyglm.utils.theano_func_wrapper import seval, _flatten
from pyglm.utils.packvec import *
from pyglm.utils.grads import *

from hips.inference.ars2 import AdaptiveRejectionSampler
from hips.inference.hmc import hmc
from pyglm.inference.log_sum_exp import log_sum_exp_sample
from pyglm.inference.coord_descent import coord_descent


class MetropolisHastingsUpdate(object):
    """
    Base class for MH updates. Each update targets a specific model component
    and requires certain configuration. For example, an update for the standard GLM
    might require differentiable parameters. Typical updates include:
        - Gibbs updates (sample from conditional distribution)
        - Hamiltonian Monte Carlo (uses gradient info to sample unconstrained cont. vars)
        - Slice sampling (good for correlaed multivariate Gaussians)
    """
    def __init__(self):
        self._target_components = []

    @property
    def target_components(self):
        # Return a list of components that this update applies to
        return self._target_components

    @property
    def target_variables(self):
        # Return a list of variables that this update applies to
        return []


    def preprocess(self, population):
        """ Do any req'd preprocessing
        """
        pass

    def update(self, x_curr):
        """ Take a MH step
        """
        return x_curr

class ParallelMetropolisHastingsUpdate(MetropolisHastingsUpdate):
    """ Extending this class indicates that the updates can be
        performed in parallel over n, the index of the neuron.
    """
    def update(self, x_curr, n):
        """ Take a MH step for the n-th neuron. This can be performed in parallel 
            over other n' \in [N]
        """
        pass


class HmcBiasUpdate(ParallelMetropolisHastingsUpdate):
    """
    Update the continuous and unconstrained bias parameters using Hamiltonian
    Monte Carlo. Stochastically follow the gradient of the parameters using
    Hamiltonian dynamics.
    """
    def __init__(self):
        super(HmcBiasUpdate, self).__init__()

        self.n_steps = 5
        self.avg_accept_rate = 0.9
        self.step_sz = 0.1

    def preprocess(self, population):
        """ Initialize functions that compute the gradient and Hessian of
            the log probability with respect to the differentiable GLM
            parameters, e.g. the weight matrix if it exists.
        """
        self.population = population

    def update(self, x, n):
        """ Gibbs sample the bias parameter.
        """
        curr_bias = x['glm_%d' % n]['bias']['I_bias']

        def lp(bias):
            self.population.glms[n].bias_model.I_bias.value = bias
            return self.population.glms[n].log_p.value

        def grad_lp(bias):
            self.population.glms[n].bias_model.I_bias.value = bias
            # import pdb; pdb.set_trace()
            return self.population.glms[n].log_p.grad(self.population.glms[n].bias_model.I_bias)

        # HMC with automatic parameter tuning
        new_bias, new_step_sz, new_accept_rate = hmc(lp,
                                                     grad_lp,
                                                     self.step_sz,
                                                     self.n_steps,
                                                     curr_bias,
                                                     adaptive_step_sz=True,
                                                     avg_accept_rate=self.avg_accept_rate,
                                                     negative_log_prob=False)

        # Update step size and accept rate
        self.step_sz = new_step_sz
        self.avg_accept_rate = new_accept_rate

        # Copy new bias parameter into x
        x['glm_%d' % n]['bias']['I_bias'] = new_bias
        self.population.glms[n].bias_model.I_bias.value = new_bias

        return x


class HmcBkgdUpdate(ParallelMetropolisHastingsUpdate):
    """
    Update the continuous and unconstrained bkgd parameters using Hamiltonian
    Monte Carlo. Stochastically follow the gradient of the parameters using
    Hamiltonian dynamics.
    """
    def __init__(self):
        super(HmcBkgdUpdate, self).__init__()

        self.n_steps = 2
        self.avg_accept_rate = 0.9
        self.step_sz = 0.1

    def preprocess(self, population):
        """ Initialize functions that compute the gradient and Hessian of
            the log probability with respect to the differentiable GLM
            parameters, e.g. the weight matrix if it exists.
        """
        self.population = population

    def update(self, x, n):
        """ Gibbs sample the GLM parameters. These are mostly differentiable
            so we use HMC wherever possible.
        """
        curr_bkgd = x['glm_%d' % n]['bkgd']['w_stim']

        def lp(bkgd):
            self.population.glms[n].bkgd_model.w_bkgd.value = bkgd
            return self.population.glms[n].log_p.value

        def grad_lp(bkgd):
            self.population.glms[n].bkgd_model.w_bkgd.value = bkgd
            return self.population.glms[n].log_p.grad(self.population.glms[n].bkgd_model.w_bkgd)

        # HMC with automatic parameter tuning
        new_bkgd, new_step_sz, new_accept_rate = hmc(lp,
                                                     grad_lp,
                                                     self.step_sz,
                                                     self.n_steps,
                                                     curr_bkgd,
                                                     adaptive_step_sz=True,
                                                     avg_accept_rate=self.avg_accept_rate,
                                                     negative_log_prob=False)

        # Update step size and accept rate
        self.step_sz = new_step_sz
        self.avg_accept_rate = new_accept_rate

        # Copy new bias parameter into x
        x['glm_%d' % n]['bkgd']['w_stim'] = new_bkgd
        self.population.glms[n].bkgd_model.w_bkgd.value = new_bkgd

        return x


class HmcDirichletImpulseUpdate(ParallelMetropolisHastingsUpdate):
    """
    Update the Dirichlet impulse response parameters using Hamiltonian
    Monte Carlo. Stochastically follow the gradient of the parameters using
    Hamiltonian dynamics.
    """
    def __init__(self):
        super(HmcDirichletImpulseUpdate, self).__init__()

        self.avg_accept_rate = 0.9
        self.n_steps = 2
        self.step_sz = 0.1

    def preprocess(self, population):
        """ Initialize functions that compute the gradient and Hessian of
            the log probability with respect to the differentiable GLM
            parameters, e.g. the weight matrix if it exists.
        """
        self.population = population

    def update(self, x, n_post):
        """ Gibbs sample the GLM parameters. These are mostly differentiable
            so we use HMC wherever possible.
        """
        A = x['net']['graph']['A']

        for n_pre in range(self.population.N):
            # Only sample if there is a connection from n_pre to n_post
            if A[n_pre, n_post]:
                curr_g = self.population.glms[n_post].imp_model.gs[n_pre].value

                def lp(g):
                    self.population.glms[n_post].imp_model.gs[n_pre].value = g
                    return self.population.glms[n_post].log_p.value

                def grad_lp(g):
                    self.population.glms[n_post].imp_model.gs[n_pre].value = g
                    return self.population.glms[n_post].log_p.grad(self.population.glms[n_post].imp_model.gs[n_pre])

                # HMC with automatic parameter tuning
                new_g, new_step_sz, new_accept_rate = hmc(lp,
                                                          grad_lp,
                                                          self.step_sz,
                                                          self.n_steps,
                                                          curr_g,
                                                          adaptive_step_sz=True,
                                                          avg_accept_rate=self.avg_accept_rate,
                                                          negative_log_prob=False)

                # Update step size and accept rate
                self.step_sz = new_step_sz
                self.avg_accept_rate = new_accept_rate

            else:
                # No edge: Sample g from the prior
                new_g = np.random.gamma(self.population.glms[n_post].imp_model.alpha,
                                        np.ones((1,1,self.population.glms[n_post].imp_model.B)))

            # Copy new impulse response parameter into x
            x['glm_%d' % n_post]['imp']['g_%d' % n_pre] = new_g
            self.population.glms[n_post].imp_model.gs[n_pre].value = new_g

        return x


class CollapsedGibbsNetworkColumnUpdate(ParallelMetropolisHastingsUpdate):

    def __init__(self):
        super(CollapsedGibbsNetworkColumnUpdate, self).__init__()

        # Define constants for Sampling
        self.DEG_GAUSS_HERMITE = 15
        self.GAUSS_HERMITE_ABSCISSAE, self.GAUSS_HERMITE_WEIGHTS = \
            np.polynomial.hermite.hermgauss(self.DEG_GAUSS_HERMITE)

    def preprocess(self, population):
        """ Initialize functions that compute the gradient and Hessian of
            the log probability with respect to the differentiable network
            parameters, e.g. the weight matrix if it exists.
        """
        self.population = population

        # Get the weight model
        self.mu_w = self.population.network.weights.mu
        self.sigma_w = self.population.network.weights.sigma

    def _glm_ll(self, n_pre, n_post, w):
        """ Compute the log likelihood of the GLM with A=True and given W
        """
        A_curr = self.population.network.graph.A.value
        A_curr[n_pre, n_post] = 1
        self.population.network.graph.A.value = A_curr

        W_curr = self.population.network.weights.W.value
        W_curr[n_pre, n_post] = w
        self.population.network.weights.W.value = W_curr

        # Compute the log likelihood for each data sequence
        ll = 0
        for data in self.population.data_sequences:
            self.population.set_data(data)
            ll += self.population.glms[n_post].ll.value

        return ll

    def _glm_ll_noA(self, n_pre, n_post):
        """ Compute the log likelihood of the GLM with A=True and given W
        """
        return self._glm_ll(n_pre, n_post, 0.)

    def _collapsed_sample_AW(self, n_pre, n_post, x):
        """
        Do collapsed Gibbs sampling for an entry A_{n,n'} and W_{n,n'} where
        n = n_pre and n' = n_post.
        """
        # Set sigma_w and mu_w
        mu_w = self.mu_w[n_pre, n_post]
        sigma_w = self.sigma_w[n_pre, n_post]

        A = x['net']['graph']['A']
        p_A = self.population.network.graph.rho
        W = x['net']['weights']['W']

        # Propose from the prior and see if A would change.
        prior_lp_A = np.log(p_A[n_pre, n_post])
        prior_lp_noA = np.log(1.0-p_A[n_pre, n_post])

        # Approximate G = \int_0^\infty p({s,c} | A, W) p(W_{n,n'}) dW_{n,n'}
        log_L = np.zeros(self.DEG_GAUSS_HERMITE)
        weighted_log_L = np.zeros(self.DEG_GAUSS_HERMITE)
        W_nns = np.sqrt(2) * sigma_w * self.GAUSS_HERMITE_ABSCISSAE + mu_w
        for i in np.arange(self.DEG_GAUSS_HERMITE):
            w = self.GAUSS_HERMITE_WEIGHTS[i]
            W_nn = W_nns[i]
            log_L[i] = self._glm_ll(n_pre, n_post, W_nn)

            # Handle NaNs in the GLM log likelihood
            if np.isnan(log_L[i]):
                log_L[i] = -np.Inf

            weighted_log_L[i] = log_L[i] + np.log(w/np.sqrt(np.pi))

            # Handle NaNs in the GLM log likelihood
            if np.isnan(weighted_log_L[i]):
                weighted_log_L[i] = -np.Inf

        # compute log pr(A_nn) and log pr(\neg A_nn) via log G
        log_G = logsumexp(weighted_log_L)
        if not np.isfinite(log_G):
            print weighted_log_L
            raise Exception("log_G not finie")

        # Compute log Pr(A_nn=1) given prior and estimate of log lkhd after integrating out W
        log_pr_A = prior_lp_A + log_G
        # Compute log Pr(A_nn = 0 | {s,c}) = log Pr({s,c} | A_nn = 0) + log Pr(A_nn = 0)
        log_pr_noA = prior_lp_noA + self._glm_ll_noA(n_pre, n_post)

        if np.isnan(log_pr_noA):
            log_pr_noA = -np.Inf

        # Sample A
        try:
            A[n_pre, n_post] = log_sum_exp_sample([log_pr_noA, log_pr_A])
            if np.allclose(p_A[n_pre, n_post], 1.0) and not A[n_pre, n_post]:
                print "Sampled no self edge"
                print log_pr_noA
                print log_pr_A
                # raise Exception("Sampled no self edge")
        except Exception as e:

            raise e

        x['net']['graph']['A'] = A
        self.population.network.graph.A.value = A

        # Sample W from its posterior, i.e. log_L with denominator log_G
        # If A_nn = 0, we don't actually need to resample W since it has no effect
        if A[n_pre,n_post] == 1:
            W[n_pre, n_post] = self._adaptive_rejection_sample_w(n_pre, n_post, mu_w, sigma_w, W_nns, log_L)
        else:
            # Sample W from the prior
            W[n_pre, n_post] = mu_w + sigma_w * np.random.randn()

        # Set W in state dict x
        x['net']['weights']['W'] = W
        self.population.network.weights.W.value = W

        self.population.network.graph.A.value = A

    def _adaptive_rejection_sample_w(self, n_pre, n_post, mu_w, sigma_w, ws_init, log_L):
        """
        Sample weights using adaptive rejection sampling.
        This only works for log-concave distributions, which will
        be the case if the nonlinearity is convex and log concave, and
        when the prior on w is log concave (as it is when w~Gaussian).
        """
        log_prior_W = -0.5/sigma_w**2 * (ws_init-mu_w)**2
        log_posterior_W = log_prior_W + log_L

        #  Define a function to evaluate the log posterior
        # For numerical stability, try to normalize 
        Z = np.amax(log_posterior_W)
        def _log_posterior(ws_in):
            ws = np.asarray(ws_in)
            shape = ws.shape
            ws = np.atleast_1d(ws)
            lp = np.zeros_like(ws)
            for (i,w) in enumerate(ws):
                lp[i] = -0.5/sigma_w**2 * (w-mu_w)**2 + \
                        self._glm_ll(n_pre, n_post, w) - Z

            if isinstance(ws_in, np.ndarray):
                return lp.reshape(shape)
            elif isinstance(ws_in, float) or isinstance(ws_in, np.float):
                return np.float(lp)

        # Only use the valid ws
        valid_ws = np.bitwise_and(np.isfinite(log_posterior_W),
                                  log_posterior_W > -1e8,
                                  log_posterior_W < 1e8)

        ars = AdaptiveRejectionSampler(_log_posterior, -np.inf, np.inf, ws_init[valid_ws], log_posterior_W[valid_ws] - Z)
        # ars = AdaptiveRejectionSampler(_log_posterior, np.amin(ws_init[valid_ws]), np.amax(ws_init[valid_ws]), ws_init[valid_ws], log_posterior_W[valid_ws] - Z)
        # return adaptive_rejection_sample(_log_posterior,
        #                                  ws[valid_ws], log_posterior_W[valid_ws] - Z,
        #                                  (-np.Inf, np.Inf),
        #                                  stepsz=sigma_w/2.0,
        #                                  debug=False)
        return ars.sample()

    def update(self, x, n):
        """ Collapsed Gibbs sample a column of A and W
        """
        N = self.population.N
        order = np.arange(N)
        np.random.shuffle(order)
        for n_pre in order:
            self._collapsed_sample_AW(n_pre, n, x)

        return x


class LatentLocationUpdate(MetropolisHastingsUpdate):
    """
    Gibbs sample the parameters of a latent distance model, namely the
    latent locations (if they are not given) and the distance scale.
    """
    def __init__(self):
        super(LatentLocationUpdate, self).__init__()

        # Use HMC if the locations are continuous
        # Otherwise, use a Metropolis-Hastings update
        self.avg_accept_rate = 0.9
        self.step_sz = 0.001

    def preprocess(self, population):
        self.N = population.model['N']

        # Get the location model(s)
        from pyglm.components.latent import LatentLocation
        self.location_models = []
        self.location_updates = []
        for latent_component in population.latent.latentlist:
            if isinstance(latent_component, LatentLocation):
                self.location_models.append(latent_component)

                # Make an update for this model
                if latent_component.dtype == np.int:
                    # update = _DiscreteLatentLocationUpdate(latent_component)
                    # update = _DiscreteGibbsLatentLocationUpdate(latent_component)
                    update = _DiscreteLocalGibbsLatentLocationUpdate(latent_component)
                else:
                    update = _ContinuousLatentLocationUpdate(latent_component)
                update.preprocess(population)
                self.location_updates.append(update)

    def update(self, x):
        """
        Update each location update in turn
        """
        for update in self.location_updates:
            x = update.update(x)

        return x

class _ContinuousLatentLocationUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample continuous latent locations
    """
    def __init__(self, latent_location_component):
        self.location = latent_location_component

    def preprocess(self, population):
        self.syms = population.get_variables()

        # Get the shape of L
        # TODO: Fix this hack!
        self.L = self.location.L
        self.L_shape = population.sample()['latent'][self.location.name]['L'].shape

        # Compute the log probability and its gradients, taking into
        # account the prior and the likelihood of any consumers of the
        # location.
        self.log_p = T.constant(0.)
        self.log_p += self.location.log_p

        self.g_log_p = T.constant(0.)
        self.g_log_p += T.grad(self.location.log_p, self.L)

        from pyglm.components.graph import LatentDistanceGraphModel
        if isinstance(population.network.graph, LatentDistanceGraphModel):
            self.log_p += population.network.graph.log_p
            self.g_log_p +=T.grad(population.network.graph.log_p, self.L)

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.log_p += population.glm.bkgd.log_p
            self.g_log_p +=T.grad(population.glm.bkgd.log_p, self.L)

    def _lp_L(self, L, x):
        # Set L in state dict x
        set_vars('L', x['latent'][self.location.name], L)
        lp = seval(self.log_p, self.syms, x)
        assert np.all(np.isfinite(lp))
        return lp


    def _grad_lp_wrt_L(self, L, x):
        # Set L in state dict x
        set_vars('L', x['latent'][self.location.name], L)
        g_lp = seval(self.g_log_p, self.syms, x)
        # if not np.all(np.isfinite(g_lp)):
        #     import pdb; pdb.set_trace()
        return g_lp

    def update(self, x):
        """
        Sample L using HMC given A and delta (distance scale)
        """
        nll = lambda L: -1.0 * self._lp_L(L.reshape(self.L_shape), x)
        grad_nll = lambda L: -1.0 * self._grad_lp_wrt_L(L.reshape(self.L_shape), x).ravel()

        # Automatically tune these paramseters
        n_steps = 10
        (L, new_step_sz, new_accept_rate) = hmc(nll,
                                                grad_nll,
                                                self.step_sz,
                                                n_steps,
                                                x['latent'][self.location.name]['L'].ravel(),
                                                adaptive_step_sz=True,
                                                avg_accept_rate=self.avg_accept_rate)

        # Update step size and accept rate
        self.step_sz = new_step_sz
        # print "Step: ", self.step_sz
        self.avg_accept_rate = new_accept_rate
        # print "Accept: ", self.avg_accept_rate

        # Update current L
        x['latent'][self.location.name]['L'] = L.reshape(self.L_shape)

        return x


class _DiscreteLatentLocationUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample discrete latent locations on a grid
    """
    def __init__(self, latent_location_component):
        self.location = latent_location_component

    def preprocess(self, population):
        self.N = population.N
        self.population = population
        self.syms = population.get_variables()
        self.L = self.location.Lmatrix

        # Compute the log probability and its gradients, taking into
        # account the prior and the likelihood of any consumers of the
        # location.
        self.log_p = self.location.log_p
        self.log_lkhd = T.constant(0.)

        from pyglm.components.graph import LatentDistanceGraphModel
        if isinstance(population.network.graph, LatentDistanceGraphModel):
            self.log_lkhd += population.network.graph.log_p

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.log_lkhd += population.glm.log_p

    def _lp_L(self, L, x, n):
        if not self._check_bounds(L):
            return -np.Inf

        # Set L in state dict x
        xn = self.population.extract_vars(x, n)
        set_vars('L', xn['latent'][self.location.name], L.ravel())
        lp = seval(self.log_p, self.syms, xn)
        lp += seval(self.log_lkhd, self.syms, xn)
        return lp

    def _check_bounds(self, L):
        """
        Return true if locations are within the allowable range
        """
        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        if isinstance(prior, Categorical):
            if np.any(L < prior.min) or np.any(L > prior.max):
                return False
        if isinstance(prior, JointCategorical):
            if np.any(L[:,0] < prior.min0) or np.any(L[:,0] > prior.max0) or \
               np.any(L[:,1] < prior.min1) or np.any(L[:,1] > prior.max1):
                return False
        return True

    def update(self, x):
        """
        Sample each entry in L using Metropolis Hastings
        """
        L = seval(self.location.Lmatrix, self.syms['latent'], x['latent'])
        # print "L: ", L
        for n in range(self.N):
            L_curr = L[n,:].copy()
            lp_curr = self._lp_L(L, x, n)

            # Make a symmetric proposal of \pm 1 step along each dimension independently
            L_prop = L_curr + np.random.randint(-1,2,L_curr.shape)
            L[n,:] = L_prop
            lp_prop = self._lp_L(L, x, n)

            # Accept or reject (ignoring proposal since it's symmetric)
            if np.log(np.random.rand()) < lp_prop - lp_curr:
                L[n,:] = L_prop
                # print "%d: [%d,%d]->[%d,%d]" % (n, L_curr[0], L_curr[1], L_prop[0],L_prop[1])
            else:
                L[n,:] = L_curr
                # print "%d: [%d,%d]->[%d,%d]" % (n, L_curr[0], L_curr[1], L_curr[0],L_curr[1])

        # Update current L
        if not self._check_bounds(L):
            import pdb; pdb.set_trace()
        x['latent'][self.location.name]['L'] = L.ravel()

        return x


class _DiscreteGibbsLatentLocationUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample discrete latent locations on a grid
    """
    def __init__(self, latent_location_component):
        self.location = latent_location_component

    def preprocess(self, population):
        self.N = population.N
        self.population = population
        self.syms = population.get_variables()
        self.L = self.location.Lmatrix

        # Compute the log probability and its gradients, taking into
        # account the prior and the likelihood of any consumers of the
        # location.
        self.log_p = self.location.log_p
        self.log_lkhd = T.constant(0.)

        from pyglm.components.graph import LatentDistanceGraphModel
        if isinstance(population.network.graph, LatentDistanceGraphModel):
            self.log_lkhd += population.network.graph.log_p

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.log_lkhd += population.glm.log_p

    def _lp_L(self, L, x, n):
        if not self._check_bounds(L):
            return -np.Inf

        # Set L in state dict x
        xn = self.population.extract_vars(x, n)
        set_vars('L', xn['latent'][self.location.name], L.ravel())
        lp = seval(self.log_p, self.syms, xn)
        lp += seval(self.log_lkhd, self.syms, xn)
        return lp

    def _check_bounds(self, L):
        """
        Return true if locations are within the allowable range
        """
        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        if isinstance(prior, Categorical):
            if np.any(L < prior.min) or np.any(L > prior.max):
                return False
        if isinstance(prior, JointCategorical):
            if np.any(L[:,0] < prior.min0) or np.any(L[:,0] > prior.max0) or \
               np.any(L[:,1] < prior.min1) or np.any(L[:,1] > prior.max1):
                return False
        return True

    def update(self, x):
        """
        Sample each entry in L using Metropolis Hastings
        """
        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        L = seval(self.location.Lmatrix, self.syms['latent'], x['latent'])
        # print "L: ", L
        for n in range(self.N):
            # Compute the probability of each possible location
            if isinstance(prior, Categorical):
                lnp = np.zeros(prior.max-prior.min + 1)
                for i,l in enumerate(range(prior.min, prior.max+1)):
                    L[n,0] = l
                    lnp[i] = self._lp_L(L, x, n)

                L[n] = prior.min + log_sum_exp_sample(lnp)

            elif isinstance(prior, JointCategorical):
                d1 = prior.max0-prior.min0+1
                d2 = prior.max1-prior.min1+1
                lnp = np.zeros((d1,d2))
                for i,l1 in enumerate(range(prior.min0, prior.max0+1)):
                    for j,l2 in enumerate(range(prior.min1, prior.max1+1)):
                        L[n,0] = l1
                        L[n,1] = l2
                        lnp[i,j] = self._lp_L(L, x, n)

                # import pdb; pdb.set_trace()
                # Gibbs sample from the 2d distribution
                ij = log_sum_exp_sample(lnp.ravel(order='C'))
                i,j = np.unravel_index(ij, (d1,d2), order='C')
                L[n,0] = prior.min0 + i
                L[n,1] = prior.min1 + j

            else:
                raise Exception('Only supporting Categorical and JointCategorical location priors')
        # Update current L
        if not self._check_bounds(L):
            import pdb; pdb.set_trace()
        x['latent'][self.location.name]['L'] = L.ravel()

        return x

class _DiscreteLocalGibbsLatentLocationUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample discrete latent locations on a grid
    This is a Metropolis-Hastings update that takes local steps proportional
    to their relative probability.
    """
    def __init__(self, latent_location_component):
        self.location = latent_location_component

    def preprocess(self, population):
        self.N = population.N
        self.population = population
        self.glm = self.population.glm
        self.syms = population.get_variables()
        self.L = self.location.Lmatrix
        self.Lflat = self.location.Lflat

        # Compute the log probability and its gradients, taking into
        # account the prior and the likelihood of any consumers of the
        # location.
        self.log_p = self.location.log_p
        self.log_lkhd = T.constant(0.)

        from pyglm.components.graph import LatentDistanceGraphModel
        if isinstance(population.network.graph, LatentDistanceGraphModel):
            self.log_lkhd += population.network.graph.log_p

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.log_lkhd += population.glm.ll

    def _precompute_vars(self, x, n):
        """ Precompute currents for sampling A and W
        """
        nvars = self.population.extract_vars(x, n)

        I_bias = seval(self.glm.bias_model.I_bias,
                       self.syms,
                       nvars)

        I_stim_xt = seval(self.glm.bkgd_model.I_stim_xt,
                       self.syms,
                       nvars)

        I_net = seval(self.glm.I_net,
                       self.syms,
                       nvars)

        return I_bias, I_stim_xt, I_net


    def _lp_L(self, L, x, n, I_bias, I_stim_xt, I_net):
        if not self._check_bounds(L):
            return -np.Inf

        # Extract the glm parameters
        s = \
        {
            'L' : self.Lflat,
            'I_stim_xt' : self.glm.bkgd_model.I_stim_xt,
            'I_bias' : self.glm.bias_model.I_bias,
            'I_net' : self.glm.I_net,
            'A' : self.population.network.graph.A,
            'n' : self.glm.n
        }

        xv = \
        {
            'L' : L.ravel(),
            'I_stim_xt' : I_stim_xt,
            'I_bias' : I_bias,
            'I_net' : I_net,
            'A' : x['net']['graph']['A'],
            'n' : n
        }

        lp = seval(self.log_p, s, xv)
        lp += seval(self.log_lkhd, s, xv)


        # # Set L in state dict x
        # xn = self.population.extract_vars(x, n)
        # set_vars('L', xn['latent'][self.location.name], L.ravel())
        # lp = seval(self.log_p, self.syms, xn)
        # lp += seval(self.log_lkhd, self.syms, xn)
        return lp

    def _check_bounds(self, L):
        """
        Return true if locations are within the allowable range
        """
        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        if isinstance(prior, Categorical):
            if np.any(L < prior.min) or np.any(L > prior.max):
                return False
        if isinstance(prior, JointCategorical):
            if np.any(L[:,0] < prior.min0) or np.any(L[:,0] > prior.max0) or \
               np.any(L[:,1] < prior.min1) or np.any(L[:,1] > prior.max1):
                return False
        return True

    def _get_neighbors(self, L):
        """
        Get valid neighbors of 2D location (l0,l1)
        """
        ne = []
        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        if isinstance(prior, Categorical):
            for ne0 in range(L[0]-1,L[0]+2):
                if ne0 >= prior.min and ne0 <= prior.max1:
                    ne.append((ne0))

        elif isinstance(prior, JointCategorical):
            for ne0 in range(L[0]-1,L[0]+2):
                for ne1 in range(L[1]-1,L[1]+2):
                    if ne0 >= prior.min0 and ne0 <= prior.max0:
                        if ne1 >= prior.min1 and ne1 <= prior.max1:
                            ne.append((ne0,ne1))

        return ne

    def update(self, x):
        """
        Sample each entry in L using Metropolis Hastings
        """

        prior = self.location.location_prior
        L = seval(self.location.Lmatrix, self.syms['latent'], x['latent'])

        # Update each of the N neuron locations serially
        # import pdb; pdb.set_trace()
        for n in range(self.N):
            print "Sampling location of neuron ", n
            # Precompute currents
            I_bias, I_stim_xt, I_net = self._precompute_vars(x, n)

            # Compute the probability of each neighboring location
            lnp_cache = {}
            curr_loc = L[n,:]
            curr_neighbors = self._get_neighbors(L[n,:])
            curr_lnps = []
            for ne in curr_neighbors:
                L[n,:] = np.array(ne)
                lnp_ne = self._lp_L(L, x, n, I_bias, I_stim_xt, I_net)
                lnp_cache[ne] = lnp_ne
                curr_lnps.append(lnp_ne)

            # Propose a neighbor according to its relative probability
            prop_loc = curr_neighbors[log_sum_exp_sample(curr_lnps)]

            # Compute acceptance probability
            prop_neighbors = self._get_neighbors(prop_loc)
            prop_lnps = []
            for ne in prop_neighbors:
                if ne in lnp_cache:
                    prop_lnps.append(lnp_cache[ne])
                else:
                    L[n,:] = np.array(ne)
                    lnp_ne = self._lp_L(L, x, n, I_bias, I_stim_xt, I_net)
                    lnp_cache[ne] = lnp_ne
                    prop_lnps.append(lnp_ne)

            # Acceptance probability is the ratio of normalizing constants
            lnp_accept = logsumexp(curr_lnps) - logsumexp(prop_lnps)

            if np.log(np.random.rand()) < lnp_accept:
                L[n,:] = np.array(prop_loc)
            else:
                # Reject and stay in current loc
                L[n,:] = np.array(curr_loc)

        # Update current L
        if not self._check_bounds(L):
            import pdb; pdb.set_trace()
        x['latent'][self.location.name]['L'] = L.ravel()

        return x


class LatentTypeUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample discrete latent locations on a grid
    """
    def __init__(self):
        pass

    def preprocess(self, population):
        self.N = population.N
        self.population = population
        self.syms = population.get_variables()

        # Get the shared tuning curve component
        from pyglm.components.latent import LatentType
        self.latent_types = []
        for latent_component in population.latent.latentlist:
            if isinstance(latent_component, LatentType):
                self.latent_types.append(latent_component)

        # # Compute the log probability and its gradients, taking into
        # # account the prior and the likelihood of any consumers of the
        # # location.

        from pyglm.components.graph import StochasticBlockGraphModel
        if isinstance(population.network.graph, StochasticBlockGraphModel):
            self.net_log_lkhd = population.network.graph.log_p
        else:
            self.net_log_lkhd = T.constant(0.)

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.glm_log_lkhd = population.glm.ll
        else:
            self.glm_log_lkhd = T.constant(0.)

    def _lp_L(self, latent_type, Y, x, n):
        # Set Yin state dict x
        xn = self.population.extract_vars(x, n)
        set_vars('Y', xn['latent'][latent_type.name], Y.ravel())
        lp = seval(latent_type.log_p, self.syms, xn)
        lp += seval(self.net_log_lkhd, self.syms, xn)

        # Compute the log likelihood for each data sequence
        for data in self.population.data_sequences:
            self.population.set_data(data)
            lp += seval(self.glm_log_lkhd, self.syms, xn)

        return lp

    def update(self, x):
        """
        Sample each entry in L using Metropolis Hastings
        """
        from pyglm.inference.log_sum_exp import log_sum_exp_sample
        for latent_type in self.latent_types:
            # Update the latent types
            R = latent_type.R
            Y = x['latent'][latent_type.name]['Y']
            print "Y: ", Y

            for n in range(self.N):
                print "Sampling latent type of neuron ", n
                lpr = np.zeros(R)
                for r in range(R):
                    Y[n] = r
                    lpr[r] = self._lp_L(latent_type, Y, x, n)

                Y[n] = log_sum_exp_sample(lpr)

            x['latent'][latent_type.name]['Y'] = Y

            # Update alpha with the conjugate dirichlet prior
            from pyglm.components.priors import Dirichlet
            if isinstance(latent_type.alpha_prior, Dirichlet):
                suffstats = latent_type.alpha_prior.alpha0.get_value()
                suffstats += np.bincount(Y, minlength=R)
                alpha = np.random.dirichlet(suffstats)
                x['latent'][latent_type.name]['alpha'] = alpha
            else:
                raise Warning('Cannot update alpha prior!')

        return x

class LatentLocationAndTypeUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample discrete latent locations on a grid
    along with the type of the neuron
    """
    def __init__(self):
        raise NotImplementedError('Joint update of location and type has not yet been implemented!')

    def preprocess(self, population):
        self.N = population.N
        self.population = population
        self.syms = population.get_variables()

        # Get the shared tuning curve component
        from pyglm.components.latent import LatentType
        self.latent_types = []
        for latent_component in population.latent.latentlist:
            if isinstance(latent_component, LatentType):
                self.latent_types.append(latent_component)

        # # Compute the log probability and its gradients, taking into
        # # account the prior and the likelihood of any consumers of the
        # # location.
        # self.log_p = self.location.log_p

        from pyglm.components.graph import StochasticBlockGraphModel
        if isinstance(population.network.graph, StochasticBlockGraphModel):
            self.net_log_lkhd = population.network.graph.log_p
        else:
            self.net_log_lkhd = T.constant(0.)

        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.glm_log_lkhd = population.glm.ll
        else:
            self.glm_log_lkhd = T.constant(0.)

    def _lp_L(self, latent_type, Y, x, n):
        # Set Yin state dict x
        xn = self.population.extract_vars(x, n)
        set_vars('Y', xn['latent'][latent_type.name], Y.ravel())
        lp = seval(latent_type.log_p, self.syms, xn)
        lp += seval(latent_type.net_log_lkhd, self.syms, xn)

        # Compute the log likelihood for each data sequence
        for data in self.population.data_sequences:
            self.population.set_data(data)
            lp += seval(self.glm_log_lkhd, self.syms, xn)

        return lp

    def update(self, x):
        """
        Sample each entry in L using Metropolis Hastings
        """
        from pyglm.inference.log_sum_exp import log_sum_exp_sample
        for latent_type in self.latent_types:
            # Update the latent types
            R = latent_type.R
            Y = x['latent'][latent_type.name]['Y']
            print "Y: ", Y

            for n in range(self.N):
                lpr = np.zeros(R)
                for r in range(R):
                    Y[n] = r
                    lpr[r] = self._lp_L(latent_type, Y, x, n)

                Y[n] = log_sum_exp_sample(lpr)

            x['latent'][latent_type.name]['Y'] = Y

            # Update alpha with the conjugate dirichlet prior
            from pyglm.components.priors import Dirichlet
            if isinstance(latent_type.alpha_prior, Dirichlet):
                suffstats = latent_type.alpha_prior.alpha0.get_value()
                suffstats += np.bincount(Y, minlength=R)
                alpha = np.random.dirichlet(suffstats)
                x['latent'][latent_type.name]['alpha'] = alpha
            else:
                raise Warning('Cannot update alpha prior!')

        from pyglm.components.priors import Categorical, JointCategorical
        prior = self.location.location_prior
        L = seval(self.location.Lmatrix, self.syms['latent'], x['latent'])
        # print "L: ", L
        for n in range(self.N):
            # Compute the probability of each possible location
            if isinstance(prior, Categorical):
                lnp = np.zeros(prior.max-prior.min + 1)
                for i,l in enumerate(range(prior.min, prior.max+1)):
                    L[n,0] = l
                    lnp[i] = self._lp_L(L, x, n)

                L[n] = prior.min + log_sum_exp_sample(lnp)

            elif isinstance(prior, JointCategorical):
                d1 = prior.max0-prior.min0+1
                d2 = prior.max1-prior.min1+1
                lnp = np.zeros((d1,d2))
                for i,l1 in enumerate(range(prior.min0, prior.max0+1)):
                    for j,l2 in enumerate(range(prior.min1, prior.max1+1)):
                        L[n,0] = l1
                        L[n,1] = l2
                        lnp[i,j] = self._lp_L(L, x, n)

                # import pdb; pdb.set_trace()
                # Gibbs sample from the 2d distribution
                ij = log_sum_exp_sample(lnp.ravel(order='C'))
                i,j = np.unravel_index(ij, (d1,d2), order='C')
                L[n,0] = prior.min0 + i
                L[n,1] = prior.min1 + j

            else:
                raise Exception('Only supporting Categorical and JointCategorical location priors')
        # Update current L
        if not self._check_bounds(L):
            import pdb; pdb.set_trace()
        x['latent'][self.location.name]['L'] = L.ravel()


        return x


class SharedTuningCurveUpdate(MetropolisHastingsUpdate):
    """
    A special subclass to sample continuous latent locations
    """
    def __init__(self):
        self.n_steps = 2
        self.avg_accept_rate = 0.9
        self.step_sz = 0.1

    def preprocess(self, population):
        self.population = population
        self.glm = self.population.glm
        self.N = population.N

        # Get the shared tuning curve component
        from pyglm.components.latent import LatentTypeWithTuningCurve
        self.tc_model = None
        for latent_component in population.latent.latentlist:
            if isinstance(latent_component, LatentTypeWithTuningCurve):
                self.tc_model = latent_component
                break

        if self.tc_model is None:
            return

        self.syms = population.get_variables()

        # Get the shape of w_x and w_t
        self.w_x = self.tc_model.w_x
        self.w_x_shape = (self.tc_model.Bx, self.tc_model.R)
        self.w_t = self.tc_model.w_t
        self.w_t_shape = (self.tc_model.Bt, self.tc_model.R)

        # Compute the log probability and its gradients, taking into
        # account the prior and the likelihood of any consumers of the
        # location.
        self.log_p = self.tc_model.log_p

        self.g_log_p_wrt_wx = T.constant(0.)
        self.g_log_p_wrt_wt = T.constant(0.)
        self.g_log_p_wrt_wx += T.grad(self.tc_model.log_p, self.w_x)
        self.g_log_p_wrt_wt += T.grad(self.tc_model.log_p, self.w_t)

        self.log_lkhd = T.constant(0.0)
        self.g_log_lkhd_wrt_wx = T.constant(0.)
        self.g_log_lkhd_wrt_wt = T.constant(0.)


        from pyglm.components.bkgd import SharedTuningCurveStimulus
        if isinstance(population.glm.bkgd_model, SharedTuningCurveStimulus):
            self.log_lkhd += population.glm.ll
            self.g_log_lkhd_wrt_wx += T.grad(population.glm.ll, self.w_x)
            self.g_log_lkhd_wrt_wt += T.grad(population.glm.ll, self.w_t)

    def _precompute_vars(self, x):
        """ Precompute currents for sampling the stimulus filters
        """
        I_biases = []
        I_nets = []

        for n in range(self.population.N):
            nvars = self.population.extract_vars(x, n)

            I_biases.append(seval(self.glm.bias_model.I_bias,
                           self.syms,
                           nvars))

            I_nets.append(seval(self.glm.I_net,
                           self.syms,
                           nvars))

        return I_biases, I_nets

    def _lp(self, x, I_biases, I_nets):
        """
        Compute the log posterior of x (across all GLMs)
        """
        # Set w_x in state dict x
        lp = seval(self.log_p, self.syms['latent'], x['latent'])

        for n in range(self.N):
            # Extract the glm parameters
            xn = self.population.extract_vars(x, n)

            s = \
            {
                'I_net' : self.glm.I_net,
                'I_bias' : self.glm.bias_model.I_bias,
                'n' : self.glm.n,
            }
            s.update(self.syms['latent'])

            xv = \
            {
                'I_net' : I_nets[n],
                'I_bias' : I_biases[n],
                'n' : n,
            }
            xv.update(xn['latent'])

            # Compute the GLM log likelihood for each data sequence
            for data in self.population.data_sequences:
                self.population.set_data(data)
                lp += seval(self.log_lkhd, s, xv)

        return lp

    def _lp_wx(self, w_x, x, I_biases, I_nets):
        """
        Compute the log posterior of x (across all GLMs)
        """
        # Set w_x in state dict x
        set_vars('w_x', x['latent'][self.tc_model.name], w_x)
        return self._lp(x, I_biases, I_nets)


    def _lp_wt(self, w_t, x, I_biases, I_nets):
        # Set w_t in state dict x
        set_vars('w_t', x['latent'][self.tc_model.name], w_t)
        return self._lp(x, I_biases, I_nets)

    def _grad_lp_wrt_wx(self, w_x, x, I_biases, I_nets):
        # Set L in state dict x
        set_vars('w_x', x['latent'][self.tc_model.name], w_x)
        g_lp = seval(self.g_log_p_wrt_wx, self.syms['latent'], x['latent'])

        for n in range(self.N):
            # print "Computing grad_lp_wrt_wx for neuron ", n
            xn = self.population.extract_vars(x, n)
            s = \
            {
                'I_net' : self.glm.I_net,
                'I_bias' : self.glm.bias_model.I_bias,
                'n' : self.glm.n,
            }
            s.update(self.syms['latent'])

            xv = \
            {
                'I_net' : I_nets[n],
                'I_bias' : I_biases[n],
                'n' : n,
            }
            xv.update(xn['latent'])

            # Compute the GLM log likelihood for each data sequence
            for data in self.population.data_sequences:
                self.population.set_data(data)
                g_lp += seval(self.g_log_lkhd_wrt_wx, s, xv)

        return g_lp


    # def _grad_lp_wrt_wx(self, w_x, x):
    #     # Set L in state dict x
    #     set_vars('w_x', x['latent'][self.tc_model.name], w_x)
    #     g_lp = seval(self.g_log_p_wrt_wx, self.syms['latent'], x['latent'])
    #
    #     for n in range(self.N):
    #         xn = self.population.extract_vars(x, n)
    #         g_lp += seval(self.g_log_lkhd_wrt_wx, self.syms, xn)
    #
    #     return g_lp

    def _grad_lp_wrt_wt(self, w_t, x, I_biases, I_nets):
        # Set L in state dict x
        set_vars('w_t', x['latent'][self.tc_model.name], w_t)
        g_lp = seval(self.g_log_p_wrt_wt, self.syms['latent'], x['latent'])

        for n in range(self.N):
            # print "Computing grad_lp_wrt_wt for neuron ", n
            xn = self.population.extract_vars(x, n)
            s = \
            {
                'I_net' : self.glm.I_net,
                'I_bias' : self.glm.bias_model.I_bias,
                'n' : self.glm.n,
            }
            s.update(self.syms['latent'])

            xv = \
            {
                'I_net' : I_nets[n],
                'I_bias' : I_biases[n],
                'n' : n,
            }
            xv.update(xn['latent'])

            # Compute the GLM log likelihood for each data sequence
            for data in self.population.data_sequences:
                self.population.set_data(data)
                g_lp += seval(self.g_log_lkhd_wrt_wt, s, xv)

        return g_lp

    def update(self, x):
        """
        Sample L using HMC given A and delta (distance scale)
        """
        if self.tc_model is None:
            return

        # Precompute other currents
        I_biases, I_nets = self._precompute_vars(x)

        # Update w_x
        nll_wx = lambda w_x: -1.0 * self._lp_wx(w_x.reshape(self.w_x_shape), x, I_biases, I_nets)
        grad_nll_wx = lambda w_x: -1.0 * self._grad_lp_wrt_wx(w_x.reshape(self.w_x_shape), x, I_biases, I_nets).ravel()

        # Automatically tune these parameters
        (w_x, new_step_sz, new_accept_rate) = hmc(nll_wx,
                                                  grad_nll_wx,
                                                  self.step_sz,
                                                  self.n_steps,
                                                  x['latent'][self.tc_model.name]['w_x'].ravel(),
                                                  adaptive_step_sz=True,
                                                  avg_accept_rate=self.avg_accept_rate)

        # Update step size and accept rate
        self.step_sz = new_step_sz
        # print "Step: ", self.step_sz
        self.avg_accept_rate = new_accept_rate
        # print "Accept: ", self.avg_accept_rate

        # Update current w_x
        x['latent'][self.tc_model.name]['w_x'] = w_x.reshape(self.w_x_shape)

        # Do the same for w_t
        nll_wt = lambda w_t: -1.0 * self._lp_wt(w_t.reshape(self.w_t_shape), x, I_biases, I_nets)
        grad_nll_wt = lambda w_t: -1.0 * self._grad_lp_wrt_wt(w_t.reshape(self.w_t_shape), x, I_biases, I_nets).ravel()

        # Automatically tune these paramseters
        (w_t, new_step_sz, new_accept_rate) = hmc(nll_wt,
                                                  grad_nll_wt,
                                                  self.step_sz,
                                                  self.n_steps,
                                                  x['latent'][self.tc_model.name]['w_t'].ravel(),
                                                  adaptive_step_sz=True,
                                                  avg_accept_rate=self.avg_accept_rate)

        # Update step size and accept rate
        self.step_sz = new_step_sz
        # print "Step: ", self.step_sz
        self.avg_accept_rate = new_accept_rate
        # print "Accept: ", self.avg_accept_rate

        # Update current w_t
        x['latent'][self.tc_model.name]['w_t'] = w_t.reshape(self.w_t_shape)

        return x


def initialize_updates(population):
    """ Compute the set of updates required for the given population.
        TODO: Figure out how to do this in a really principled way.
    """
    serial_updates = []
    parallel_updates = []

    # All populations have a parallel GLM sampler
    print "Initializing GLM samplers"
    bias_sampler = HmcBiasUpdate()
    bias_sampler.preprocess(population)
    parallel_updates.append(bias_sampler)

    # bkgd_sampler = HmcBkgdUpdate()
    # bkgd_sampler.preprocess(population)
    # parallel_updates.append(bkgd_sampler)

    imp_sampler = HmcDirichletImpulseUpdate()
    imp_sampler.preprocess(population)
    parallel_updates.append(imp_sampler)

    # All populations have a network sampler
    print "Initializing network sampler"
    net_sampler = CollapsedGibbsNetworkColumnUpdate()
    net_sampler.preprocess(population)
    parallel_updates.append(net_sampler)

    return serial_updates, parallel_updates

def gibbs_sample(population,
                 N_samples=1000,
                 x0=None, 
                 init_from_mle=True,
                 callback=None):
    """
    Sample the posterior distribution over parameters using MCMC.
    """
    N = population.model['N']
    dt = population.model['dt']

    # Draw initial state from prior if not given
    if x0 is None:
        x0 = population.sample()
        # DEBUG For stability
        x0['net']['weights']['W'] *= 0

    # Initialize the population
    population.set_parameters(x0)

    # Create updates for this population
    serial_updates, parallel_updates = initialize_updates(population)

    # DEBUG Profile the Gibbs sampling loop
    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    # Alternate fitting the network and fitting the GLMs
    x_smpls = [x0]
    x = x0

    import time
    start_time = time.time()

    for smpl in np.arange(N_samples):
        # Call the callback
        if callback is not None:
            callback(x)

        # Print the current log likelihood
        lp = population.compute_log_p(x)

        # Compute iters per second
        stop_time = time.time()
        if stop_time - start_time == 0:
            print "Gibbs iteration %d. Iter/s exceeds time resolution. Log prob: %.3f" % (smpl, lp)
        else:
            print "Gibbs iteration %d. Iter/s = %f. Log prob: %.3f" % (smpl,
                                                                       1.0/(stop_time-start_time),
                                                       lp)
        start_time = stop_time

        # Go through each parallel MH update
        for parallel_update in parallel_updates:
            for n in np.arange(N):
                # print "Parallel update: %s for neuron %d" % (str(type(parallel_update)), n)
                parallel_update.update(x, n)

        # Sample the serial updates
        for serial_update in serial_updates:
            print "Serial update: ", type(serial_update)
            serial_update.update(x)

        x_smpls.append(copy.deepcopy(x))

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open('mcmc.prof.txt', 'w') as f:
        f.write(s.getvalue())
        f.close()

    return x_smpls
