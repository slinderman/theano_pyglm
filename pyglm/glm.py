import theano

from pyglm.components.bkgd import *
from pyglm.components.bias import *
from pyglm.components.impulse import *
from pyglm.components.nlin import *

class Glm(Component):
    def __init__(self, model, network, latent):
        """
        Create a GLM for the spikes on the n-th neuron out of N
        This corresponds to the spikes in the n-th column of data["S"]
        """
        # Define the Poisson regression model
        self.n = T.lscalar('n')
        self.dt = theano.shared(name='dt', value=model['dt'])
        self.S = theano.shared(name='S',
                               value=np.zeros((1, model['N'])))


        # Define a bias to the membrane potential
        self.bias_model = create_bias_component(model, self, latent)

        # Define stimulus and stimulus filter
        self.bkgd_model = create_bkgd_component(model, self, latent)

        # Create a list of impulse responses for each incoming connections
        self.imp_model = create_impulse_component(model, self, latent)

        # If a network is given, weight the impulse response currents and sum them up
        if network is not None:
            # Compute the effective incoming weights
            An = network.graph.A[:,self.n]
            Wn = network.weights.W[:,self.n]
            self.W_eff = An * Wn
        else:
            self.W_eff = np.ones((model['N'],))

        self.I_net = T.dot(self.imp_model.I_imp, self.W_eff)

        # Rectify the currents to get a firing rate
        self.nlin_model = create_nlin_component(model)
        self.lam = self.nlin_model.nlin(self.bias_model.I_bias +
                                   self.bkgd_model.I_stim +
                                   self.I_net)

        # Clip the rate to a reasonable range
        # self.lam = T.clip(self.lam, 1e-128, 1e128)
        self.lam.name = 'lambda'

        # Compute the log likelihood under the Poisson process
        self.ll = T.sum(-self.dt*self.lam + T.log(self.lam)*self.S[:,self.n])

        # Compute the log prior
        lp_bias = self.bias_model.log_p
        lp_bkgd = self.bkgd_model.log_p
        lp_imp = self.imp_model.log_p
        lp_nlin = self.nlin_model.log_p
        self.log_prior = lp_bias + lp_bkgd + lp_imp + lp_nlin

        # Allow for a scaling of the likelihood to implement AIS
        self.lkhd_scale = theano.shared(name='lkhd_scale', value=1.0)
        self.log_p = self.lkhd_scale * self.ll + self.log_prior

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

        # Save the firing rate and currents
        state['lam'] = self.lam
        state['I_bias'] = self.bias_model.I_bias
        state['I_bkgd'] = self.bkgd_model.I_stim
        state['I_net'] = self.I_net

        # Recursively save state of components
        state['bias'] = self.bias_model.get_state()
        state['bkgd'] = self.bkgd_model.get_state()
        state['imp']  = self.imp_model.get_state()
        state['nlin'] = self.nlin_model.get_state()
        return state

    def preprocess_data(self, data):
        self.bias_model.preprocess_data(data)
        self.bkgd_model.preprocess_data(data)
        self.imp_model.preprocess_data(data)
        self.nlin_model.preprocess_data(data)

    def set_data(self, data):
        """ Update the shared memory where the data is stored
        """
        if "S" in data.keys():
            self.S.set_value(data["S"])
        else:
            self.S.set_value(np.zeros_like(data["stim"]))

        # self.dt.set_value(data["dt"])

        self.bkgd_model.set_data(data)
        self.imp_model.set_data(data)

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.bkgd_model.set_hyperparameters(model['bkgd'])
        self.imp_model.set_hyperparameters(model['impulse'])
        self.bias_model.set_hyperparameters(model['bias'])

    def sample(self, acc):
        """ Sample a random set of parameters
        """
        v = {str(self.n) : -1}  # Doesn't make sense to sample n

        v['bias'] = self.bias_model.sample(acc)
        v['bkgd'] = self.bkgd_model.sample(acc)
        v['imp'] = self.imp_model.sample(acc)
        v['nlin'] = self.nlin_model.sample(acc)

        return v
