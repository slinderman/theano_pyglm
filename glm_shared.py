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

    def get_state(self):
        """ Get the state of this GLM
        """
        state = {}

        # Save the firing rate
        state['lam'] = self.lam
        state['bias'] = self.bias_model.get_state()
        state['bkgd'] = self.bkgd_model.get_state()
        state['imp']  = self.imp_model.get_state()
        state['nlin'] = self.nlin_model.get_state()
        return state
    
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
