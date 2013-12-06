import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import scipy.linalg as linalg

from components.network import *
from components.bkgd import *
from components.bias import *
from components.impulse import *
from components.nlin import *


# TODO: Should the model contain inference code as well?
from inference.hmc import hmc

from utils.theano_func_wrapper import seval
from utils.packvec import *
from utils.grads import *

class NetworkGlm:
    """
    Network of connected GLMs.
    """
    def __init__(self, model):
        """
        Initialize the network GLM with given model. What needs to be set?
        """
        self.model = model
        self.N = model['N']

        # Create a network model to connect the GLMs
        self.network = Network(model)

        # Create a single GLM that is shared across neurons
        self.glm = Glm(model, self.network)

        self.f_lp = lambda vars: reduce(lambda lp_acc,n: lp_acc + self.glm.f_lp(n,vars[0],vars[n]),
                                        self.network.f_lp(vars['net']))

    def compute_log_p(self, vars):
        """ Compute the log joint probability under a given set of variables
        """
        lp = 0.0

        # Get set of symbolic variables
        syms = self.get_variables()

        lp += seval(self.network.log_p,
                    syms['net'],
                    vars['net'])
        for n in np.arange(self.N):
            nvars = self.extract_vars(vars, n)
            lp += seval(self.glm.log_p,
                        syms,
                        nvars)

        return lp

    def get_variables(self):
        """ Get a list of all variables
        """
        v = {}
        v['net'] = self.network.get_variables()
        v['glm'] = self.glm.get_variables()
        return v

    def sample(self):
        """
        Sample parameters of the GLM from the prior
        """
        v = {}
        v['net'] = self.network.sample()
        v['glms'] =[]
        for n in range(self.N):
            xn = self.glm.sample()
            xn['n'] = n
            v['glms'].append(xn)

        return v

    def extract_vars(self, vals, n):
        """ Hacky helper function to extract the variables for only the
         n-th GLM.
        """

        newvals = {}
        for (k,v) in vals.items():
            if k=='glms':
                newvals['glm'] = v[n]
            else:
                newvals[k] = v
        return newvals

    def get_state(self, vars):
        """ Get the 'state' of the system
        """
        net_vars = vars[0]
        glm_vars = vars[1:]
        state = {}
        state.update(self.network.get_state(net_vars))

        for n in np.arange(self.N):
            state.update(self.glm.get_state(n, net_vars, glm_vars[n]))

        return state

    def set_data(self, data):
        """
        Condition on the data
        """
        self.network.set_data(data)
        self.glm.set_data(data)

    def simulate(self, vars,  (T_start,T_stop), dt):
        """ Simulate spikes from a network of coupled GLMs
        :param glms - the GLMs to sample, one for each neuron
        :type glms    list of N GLMs
        :param vars - the variables corresponding to each GLM
        :type vars    list of N variable vectors
        :param dt    - time steps to simulate

        :rtype TxN matrix of spike counts in each bin
        """
        # Initialize the background rates
        N = self.model['N']
        assert np.mod(T_start,dt) < 1**-16, "T_start must be divisble by dt"
        assert np.mod(T_stop,dt) < 1**-16, "T_stop must be divisble by dt"
        t = np.arange(T_start, T_stop, dt)
        t_ind = np.arange(int(T_start/dt), int(T_stop/dt))
        assert len(t) == len(t_ind)
        nT = len(t)

        # Get set of symbolic variables
        syms = self.get_variables()\

        # Initialize the background rate
        X = np.zeros((nT,N))
        for n in np.arange(N):
#             X[:,n] = self.glm.bias_model.f_I_bias(*vars[n+1])
            nvars = self.extract_vars(vars, n)
            X[:,n] = seval(self.glm.bias_model.I_bias,
                           syms,
                           nvars)

        # Add stimulus induced currents if given
        for n in np.arange(N):
            #X[:,n] += self.glm.bkgd_model.f_I_stim(*vars[n+1])[t_ind]
            nvars = self.extract_vars(vars, n)
            X[:,n] += seval(self.glm.bkgd_model.I_stim,
                            syms,
                            nvars)

        print "Max background rate: %f" % np.exp(np.max(X))

        # Get the impulse response functions
        imps = []
        for n_post in np.arange(N):
            nvars = self.extract_vars(vars, n_post)
            imps.append(seval(self.glm.imp_model.impulse,
                                  syms,
                                  nvars))
        imps = np.transpose(np.array(imps), axes=[1,0,2])
        T_imp = imps.shape[2]

        # Debug: compute effective weights
        tt_imp = dt*np.arange(T_imp)
        Weff = np.trapz(imps, tt_imp, axis=2)
        print "Effective impulse weights: "
        print Weff


        # Iterate over each time step and generate spikes
        S = np.zeros((nT,N))
        acc = np.zeros(N)
        thr = -np.log(np.random.rand(N))

        for t in np.arange(nT):
            # Update accumulator
            if np.mod(t,10000)==0:
                print "Iteration %d" % t
            # TODO Handle nonlinearities with variables
            lam = np.array(map(lambda n: self.glm.nlin_model.f_nlin(X[t,n]),
                           np.arange(N)))
            acc = acc + lam*dt

            # Spike if accumulator exceeds threshold
            i_spk = acc > thr
            S[t,i_spk] += 1
            n_spk = np.sum(i_spk)

            # Compute the length of the impulse response
            t_imp = np.minimum(nT-t-1,T_imp)

            # Get the instantaneous connectivity
            # TODO Handle time varying weights
            At = np.tile(np.reshape(seval(self.network.graph.A,
                                          syms['net'],
                                          vars['net']),
                                    [N,N,1]),
                         [1,1,t_imp])
            Wt = np.tile(np.reshape(seval(self.network.weights.W,
                                          syms['net'],
                                          vars['net']),
                                    [N,N,1]),
                         [1,1,t_imp])

            # Iterate until no more spikes
            while n_spk > 0:
                # Add weighted impulse response to activation of other neurons)
                X[t+1:t+t_imp+1,:] += np.sum(At[i_spk,:,:t_imp] *
                                             Wt[i_spk,:,:t_imp] *
                                             imps[i_spk,:,:t_imp],0).T

                # Subtract threshold from the accumulator
                acc -= thr*i_spk
                acc[acc<0] = 0

                # Set new threshold after spike
                thr[i_spk] = -np.log(np.random.rand(n_spk))

                i_spk = acc > thr
                S[t,i_spk] += 1
                n_spk = np.sum(i_spk)

                if np.any(S[t,:]>10):
                    import pdb
                    pdb.set_trace()
                    raise Exception("More than 10 spikes in a bin!")
        # DEBUG:
        tt = dt * np.arange(nT)
        lam = np.zeros_like(X)
        for n in np.arange(N):
            lam[:,n] = self.glm.nlin_model.f_nlin(X[:,n])

        print "Max firing rate (post sim): %f" % np.max(lam)
        E_nS = np.trapz(lam,tt,axis=0)
        nS = np.sum(S,0)

        print "Sampled %s spikes." % str(nS)
        print "Expected %s spikes." % str(E_nS)

        if np.any(np.abs(nS-E_nS) > 3*np.sqrt(E_nS)):
            print "ERROR: Actual num spikes (%d) differs from expected (%d) by >3 std." % (E_nS,nS)

        return S,X


class Glm:
    def __init__(self, model, network):
        """
        Create a GLM for the spikes on the n-th neuron out of N
        This corresponds to the spikes in the n-th column of data["S"]
        """
        # Define the Poisson regression model
        self.n = T.iscalar('n')
        self.dt = theano.shared(name='dt',
                                value=1.0)
        self.S = theano.shared(name='S',
                               value=np.zeros((1, model['N'])))

        # Concatenate the variables into one long vector
#         self.vars = T.dvector(name='vars')
#         v_offset = 0

        # Define a bias to the membrane potential
        #self.bias_model = ConstantBias(vars, v_offset)
        self.bias_model = create_bias_component(model)
#         v_offset += self.bias_model.n_vars

        # Define stimulus and stimulus filter
        self.bkgd_model = create_bkgd_component(model)
#         v_offset += self.bkgd_model.n_vars

        # Create a list of impulse responses for each incoming connections
        self.imp_model = create_impulse_component(model)
#         v_offset += self.imp_model.n_vars

        # If a network is given, weight the impulse response currents and sum them up
        if network is not None:
            # Compute the effective incoming weights
#            W_eff = network.graph.A[:,n] * network.weights.W[:,n]
            An = network.graph.A[:,self.n]
            Wn = network.weights.W[:,self.n]
            W_eff = An * Wn
        else:
            W_eff = np.ones((model['N'],))

        self.I_net = T.dot(self.imp_model.I_imp, W_eff)

        # Rectify the currents to get a firing rate
        self.nlin_model = create_nlin_component(model)
        self.lam = self.nlin_model.nlin(self.bias_model.I_bias +
                                   self.bkgd_model.I_stim +
                                   self.I_net)

        # Compute the log likelihood under the Poisson process
        self.ll = T.sum(-self.dt*self.lam + T.log(self.lam)*self.S[:,self.n])

        # Compute the log prior
        lp_bias = self.bias_model.log_p
        lp_bkgd = self.bkgd_model.log_p
        lp_imp = self.imp_model.log_p
        lp_nlin = self.nlin_model.log_p
        self.log_p = self.ll + lp_bias + lp_bkgd + lp_imp + lp_nlin

#         # Compute the gradient of the log likelihood wrt vars
#         self.g = T.grad(self.log_p, self.vars)
#
#         # Finally, compute the Hessian
#         self.H,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
#                                sequences=T.arange(self.g.shape[0]),
#                                non_sequences=[self.g, self.vars])
#
#         # Create callable functions to compute firing rate, log likelihood, and gradients
#         #theano.config.on_unused_input = 'ignore'
#         self.f_lam = theano.function([self.n] + network.vars + [self.vars], lam)
#         self.f_lp = theano.function([self.n] + network.vars + [self.vars], self.log_p)
#         self.g_lp = theano.function([self.n] + network.vars + [self.vars], self.g)
#         self.H_lp = theano.function([self.n] + network.vars + [self.vars], self.H)

    def get_variables(self):
        """ Get a list of all variables
        """
        v = {str(self.n) : self.n}
        v['bias'] = self.bias_model.get_variables()
        v['bkgd'] = self.bkgd_model.get_variables()
        v['imp']  = self.imp_model.get_variables()
        v['nlin'] = self.nlin_model.get_variables()
        return v

    def get_state(self, n, net_vars, glm_vars):
        """ Get the state of this GLM
        """
        state = {}

        # Save the firing rate
#         state['lam'] = self.f_lam(*([n] + net_vars + glm_vars))
        # Get state from each component
        state.update(self.bias_model.get_state(glm_vars))
        state.update(self.bkgd_model.get_state(glm_vars))
        state.update(self.imp_model.get_state(glm_vars))
        return {n : state}

    #def compute_gradients(self):
    #    """ Compute gradients of this GLM's log prob wrt its variables.
    #    """
    #    # TODO Rather than using a single vector of parameters, let each component
    #    #      own its own variables and take the gradient with respect to each
    #    #      set of variables separately.
    #    var_list = [self.vars]
    #    g_list = T.grad(self.log_p, var_list)
    #    self.g = T.concatenate(g_list)
    #
    #    # Compute the hessian
    #    H_rows = []
    #    for gv1 in g_list:
    #        H_v1 = []
    #        for v2 in var_list:
    #            # Compute dgv1/dv2
    #            H_v1v2,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
    #                                   sequences=T.arange(gv1.shape[0]),
    #                                   non_sequences=[gv1, v2])
    #            H_v1.append(H_v1v2)
    #        H_rows.append(T.concatenate(H_v1, axis=1))
    #
    #    # Concatenate the Hessian blocks into a matrix
    #    self.H = T.concatenate(H_rows, axis=0)

    def set_data(self, data):
        """ Update the shared memory where the data is stored
        """
        if "S" in data.keys():
            self.S.set_value(data["S"])
        else:
            self.S.set_value(np.zeros_like(data["stim"]))

        self.dt.set_value(data["dt"])

        self.bkgd_model.set_data(data)
        self.imp_model.set_data(data)

    def sample(self):
        """ Sample a random set of parameters
        """
        v = {str(self.n) : -1}  # Doesn't make sense to sample n

        v['bias'] = self.bias_model.sample()
        v['bkgd'] = self.bkgd_model.sample()
        v['imp'] = self.imp_model.sample()
        v['nlin'] = self.nlin_model.sample()

        return v

    def gibbs_step(self, state, network_glm, n):
        """ Perform an HMC step to update the GLM parameters
        """
        x_glm_0, shapes = pack(state[n+1])
        nll = lambda x_glm: -1.0 * self.f_lp(*([n] + state[0] + unpack(x_glm, shapes)))
        g_nll = lambda x_glm: -1.0 * self.g_lp(*([n] + state[0] + unpack(x_glm, shapes)))

        # Call HMC
        # TODO Automatically tune these parameters
        epsilon = 0.001
        L = 10
        x_glm = hmc(nll, g_nll, epsilon, L, x_glm_0)

        return unpack(x_glm, shapes)