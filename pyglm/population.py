import numpy as np

from pyglm.components.latent import TheanoLatentVariables, KayakLatentVariables
from pyglm.components.network import TheanoNetwork, KayakNetwork
from glm import TheanoGlm, KayakGlm
from utils.theano_func_wrapper import seval

class _PopulationBase(object):
    """
    Population connected GLMs.
    """
    def __init__(self, model):
        """
        Initialize the population of GLMs connected by a network.
        """
        self.model = model
        self.N = model['N']

        # Initialize a list of data sequences
        self.data_sequences = []

    @property
    def latent(self):
        raise NotImplementedError()

    @property
    def network(self):
        raise NotImplementedError()

    def get_variables(self):
        """ Get a list of all variables
        """
        raise NotImplementedError()

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        raise NotImplementedError()

    def compute_log_prior(self, vars):
        raise NotImplementedError()

    def compute_ll(self, vars):
        raise NotImplementedError()

    def compute_log_p(self, vars):
        """ Compute the log joint probability under a given set of variables
        """
        lp = 0.0
        lp += self.compute_log_prior(vars)

        # Add the likelihood of each data sequence
        for data in self.data_sequences:
            self.set_data(data)
            lp += self.compute_ll(vars)

        return lp

    def preprocess_data(self, data):
        raise NotImplementedError()

    def add_data(self, data, set_as_current_data=True):
        """
        Add another data sequence to the population. Recursively call components
        to prepare the new data sequence. E.g. the background model may preprocess
        the stimulus with a set of basis filters.
        """
        # TODO: Figure out how to handle time varying weights with multiple
        # data sequences. Maybe we only allow one sequence.

        assert isinstance(data, dict), 'Data must be a dictionary'

        # Check for spike times in the data array
        assert 'S' in data, 'Data must contain an array of spike times'
        assert isinstance(data['S'], np.ndarray), 'Spike times must be a numpy array'

        if 'preprocessed' not in data or  not data['preprocessed']:
            data = self.preprocess_data(data)

        # Add the data to the list
        self.data_sequences.append(data)

        # By default, we set this as the current dataset
        if set_as_current_data:
            self.set_data(data)

    def set_data(self, data):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


    def simulate(self, vars, (T_start,T_stop), dt, stim, dt_stim):
        raise NotImplementedError()


class TheanoPopulation(_PopulationBase):

    def __init__(self, model):
        super(TheanoPopulation, self).__init__(model)
        # Initialize latent variables of the population
        self._latent = TheanoLatentVariables(model)

        # Create a network model to connect the GLMs
        self._network = TheanoNetwork(model, self.latent)

        # Create a single GLM that is shared across neurons
        # This is to simplify the model and reuse parameters.
        # Basically it speeds up the gradient calculations since we
        # can manually leverage conditional independencies among GLMs
        self._glm = TheanoGlm(model, self.network, self.latent)

    @property
    def latent(self):
        return self._latent

    @property
    def network(self):
        return self._network

    @property
    def glm(self):
        return self._glm

    def get_variables(self):
        """ Get a list of all variables
        """
        v = {}
        v['latent'] = self.latent.get_variables()
        v['net'] = self.network.get_variables()
        v['glm'] = self.glm.get_variables()
        return v

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.latent.set_hyperparameters(model)
        self.network.set_hyperparameters(model)
        self.glm.set_hyperparameters(model)

    def sample(self):
        """
        Sample parameters of the GLM from the prior
        """
        v = {}
        v['latent'] = self.latent.sample(v)
        v['net'] = self.network.sample(v)
        v['glms'] =[]
        for n in range(self.N):
            xn = self.glm.sample(v)
            xn['n'] = n
            v['glms'].append(xn)

        return v

    def extract_vars(self, vals, n):
        """ Hacky helper function to extract the variables for only the
         n-th GLM.s
        """

        newvals = {}
        for (k,v) in vals.items():
            if k=='glms':
                newvals['glm'] = v[n]
            else:
                newvals[k] = v
        return newvals

    def get_state(self):
        """ Get the 'state' of the system in symbolic Theano variables
        """
        state = {}
        state['latent'] = self.latent.get_state()
        state['net'] = self.network.get_state()
        state['glm'] = self.glm.get_state()

        return state

    def preprocess_data(self, data):
        """
        Preprocess the data to compute filtered stimuli, spike trains, etc.
        """
        assert isinstance(data, dict), 'Data must be a dictionary'
        self.latent.preprocess_data(data)
        self.network.preprocess_data(data)
        self.glm.preprocess_data(data)
        data['preprocessed'] = True
        return data

    def set_data(self, data):
        """
        Condition on the data
        """
        assert 'preprocessed' in data and data['preprocessed'] == True, \
            'Data must be preprocessed before it can be set'
        self.latent.set_data(data)
        self.network.set_data(data)
        self.glm.set_data(data)

    def compute_log_prior(self, vars):
        """ Compute the log joint probability under a given set of variables
        """
        lp = 0.0

        # Get set of symbolic variables
        syms = self.get_variables()

        lp += seval(self.latent.log_p,
                    syms['latent'],
                    vars['latent'])

        lp += seval(self.network.log_p,
                    syms['net'],
                    vars['net'])

        for n in range(self.N):
            nvars = self.extract_vars(vars, n)
            lp += seval(self.glm.log_prior,
                        syms,
                        nvars)

        return lp

    def compute_ll(self, vars):
        """ Compute the log likelihood under a given set of variables
        """
        ll = 0.0

        # Get set of symbolic variables
        syms = self.get_variables()

        # Add the likelihood from each GLM
        for n in range(self.N):
            nvars = self.extract_vars(vars, n)
            ll += seval(self.glm.ll,
                        syms,
                        nvars)

        return ll

    def eval_state(self, vars):
        """ Evaluate the population state expressions given the parameters,
            e.g. the stimulus response curves from the basis function weights.
        """
        # Get set of symbolic variables
        syms = self.get_variables()

        # Get the symbolic state expression to evaluate
        state_vars = self.get_state()
        state = {}

        state['latent'] = self._eval_state_helper(syms['latent'],
                                                  state_vars['latent'],
                                                  vars['latent'])

        state['net'] = self._eval_state_helper(syms['net'],
                                                  state_vars['net'],
                                                  vars['net'])

        glm_states = []
        for n in np.arange(self.N):
            nvars = self.extract_vars(vars, n)
            glm_states.append(self._eval_state_helper(syms,
                                                      state_vars['glm'],
                                                      nvars))
        state['glms'] = glm_states

        # Finally, evaluate the log probability and the log likelihood
        # state['logp'] = self.compute_log_p(vars)
        state['logprior'] = self.compute_log_prior(vars)
        state['ll'] = self.compute_ll(vars)
        state['logp'] = state['ll'] + state['logprior']
        return state

    def _eval_state_helper(self, syms, d, vars):
        """ Helper function to recursively evaluate state variables
        """
        state = {}
        for (k,v) in d.items():
            if isinstance(v,dict):
                state[k] = self._eval_state_helper(syms, v, vars)
            else:
                state[k] = seval(v, syms, vars)
        return state
        

    def simulate(self, vars, (T_start,T_stop), dt, stim, dt_stim):
        """ Simulate spikes from a network of coupled GLMs
        :param vars - the variables corresponding to each GLM
        :type vars    list of N variable vectors
        :param dt    - time steps to simulate

        :rtype TxN matrix of spike counts in each bin
        """
        # Initialize the background rates
        N = self.model['N']
        t = np.arange(T_start, T_stop, dt)
        t_ind = np.arange(int(T_start/dt), int(T_stop/dt))
        assert len(t) == len(t_ind)
        nT = len(t)

        # Get set of symbolic variables
        syms = self.get_variables()

        # Initialize the background rate
        X = np.zeros((nT,N))
        for n in np.arange(N):
            nvars = self.extract_vars(vars, n)
            X[:,n] = seval(self.glm.bias_model.I_bias,
                           syms,
                           nvars)

        # Add stimulus induced currents if given
        temp_data = {'S' : np.zeros((nT, N)),
                     'stim' : stim,
                     'dt_stim': dt_stim}
        self.add_data(temp_data)
        for n in np.arange(N):
            nvars = self.extract_vars(vars, n)
            X[:,n] += seval(self.glm.bkgd_model.I_stim,
                            syms,
                            nvars)
        print "Max background rate: %s" % str(self.glm.nlin_model.f_nlin(np.amax(X)))

        # Remove the temp data from the population data sequences
        self.data_sequences.pop()

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
        # tt_imp = dt*np.arange(T_imp)
        # Weff = np.trapz(imps, tt_imp, axis=2)
        # print "Effective impulse weights: "
        # print Weff

        # Iterate over each time step and generate spikes
        S = np.zeros((nT,N))
        acc = np.zeros(N)
        thr = -np.log(np.random.rand(N))

        # TODO: Handle time-varying weights appropriately
        time_varying_weights = False
        if not time_varying_weights:
            At = np.tile(np.reshape(seval(self.network.graph.A,
                                          syms['net'],
                                          vars['net']),
                                    [N,N,1]),
                         [1,1,T_imp])

            Wt = np.tile(np.reshape(seval(self.network.weights.W,
                                          syms['net'],
                                          vars['net']),
                                    [N,N,1]),
                         [1,1,T_imp])

        # Count the number of exceptions arising from more spikes per bin than allowable
        n_exceptions = 0
        for t in np.arange(nT):
            # Update accumulator
            if np.mod(t,10000)==0:
                print "Iteration %d" % t
            # TODO Handle nonlinearities with variables
            lam = self.glm.nlin_model.f_nlin(X[t,:])
            acc = acc + lam*dt

            # Spike if accumulator exceeds threshold
            i_spk = acc > thr
            S[t,i_spk] += 1
            n_spk = np.sum(i_spk)

            # Compute the length of the impulse response
            t_imp = np.minimum(nT-t-1,T_imp)

            # Get the instantaneous connectivity
            if time_varying_weights:
                # TODO: Really get the time-varying weights
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
            # Cap the number of spikes in a time bin
            max_spks_per_bin = 10
            while n_spk > 0:
                if np.any(S[t,:] >= max_spks_per_bin):
                    n_exceptions += 1
                    break
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

                #if np.any(S[t,:]>10):
                #    import pdb
                #    pdb.set_trace()
                #    raise Exception("More than 10 spikes in a bin! Decrease variance on impulse weights or decrease simulation bin width.")
                
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
            print "ERROR: Actual num spikes (%s) differs from expected (%s) by >3 std." % (str(nS),str(E_nS))

        print "Number of exceptions arising from multiple spikes per bin: %d" % n_exceptions

        return S,X


class KayakPopulation(_PopulationBase):

    def __init__(self, model):
        super(KayakPopulation, self).__init__(model)
        # Initialize latent variables of the population
        self._latent = KayakLatentVariables(model)

        # Create a network model to connect the GLMs
        self._network = KayakNetwork(model, self.latent)

        # Create GLMs for each neuron
        self._glms = []
        for n in range(model['N']):
            self._glms.append(KayakGlm(n, model, self.network, self.latent))

        # Define Kayak objects for the prior, likelihood, and joint.
        self.log_prior = self.latent.log_p
        self.log_prior += self.network.log_p

        for glm in self._glms:
            self.log_prior += glm.log_prior

        self.ll = 0.0
        # Add the likelihood from each GLM
        for glm in self._glms:
            self.ll += glm.ll

        self.log_p = self.log_prior + self.ll

    @property
    def latent(self):
        return self._latent

    @property
    def network(self):
        return self._network

    @property
    def glms(self):
        return self._glms

    def compute_log_prior(self, vars):
        """ Compute the log joint probability under a given set of variables
        """
        # lp = 0.0
        # lp += self.latent.log_p.value
        # lp += self.network.log_p.value
        #
        # for glm in self._glms:
        #     lp += glm.log_prior.value
        #
        # return lp
        print "WARNING: IGNORING GIVEN VARS"
        return self.log_prior.value

    def compute_ll(self, vars):
        """ Compute the log likelihood under a given set of variables
        """
        # ll = 0.0
        #
        # # Add the likelihood from each GLM
        # for glm in self._glms:
        #     ll += glm.ll.value
        # return ll
        print "WARNING: IGNORING GIVEN VARS"
        return self.ll.value

    def compute_log_p(self, vars):
        """ Compute the log joint probability under a given set of variables
        """
        print "WARNING: IGNORING GIVEN VARS"
        return self.log_p.value

    def set_hyperparameters(self, model):
        """ Set the hyperparameters of the model
        """
        self.latent.set_hyperparameters(model)
        self.network.set_hyperparameters(model)
        for glm in self.glms:
            glm.set_hyperparameters(model)

    def get_parameters(self):
        """ Get a list of all variables
        """
        v = {}
        v['latent'] = self.latent.get_variables()
        v['net'] = self.network.get_variables()
        for n in xrange(self.N):
            v['glm_%d' % n] = self.glms[n].get_variables()
        return v

    def set_parameters(self, values):
        params = self.get_parameters()
        self._set_parameter_helper(params, values)

    def _set_parameter_helper(self, curr_params, curr_values):
        """ Helper function to recursively set the value of parameters in hierarchical dict
        """
        for (k,v) in curr_values.items():
            if isinstance(v,dict):
                self._set_parameter_helper(curr_params[k], v)
            else:
                curr_params[k].value = v

    def get_state(self):
        """ Get a list of all variables
        """
        v = {}
        v['latent'] = self.latent.get_state()
        v['net'] = self.network.get_state()
        for n in xrange(self.N):
            v['glm_%d' % n] = self.glms[n].get_state()
        return v

    def eval_state(self, vars):
        self.set_parameters(vars)
        state_vars = self.get_state()

        def _eval_state_helper(curr_state_vars):
            """ Helper function to recursively evaluate state variables
            """
            state = {}
            for (k,v) in curr_state_vars.items():
                if isinstance(v,dict):
                    state[k] = _eval_state_helper(v)
                else:
                    state[k] = v.value
            return state

        # return _eval_state_helper(state_vars)

        # DEBUG
        state = {}
        state['latent'] = _eval_state_helper(state_vars['latent'])
        state['net'] = _eval_state_helper(state_vars['net'])
        state['glms'] = {}
        for n,glm in enumerate(self.glms):
            state['glms'][n] = _eval_state_helper(state_vars['glm_%d' % n])
        # END DEBUG

        # Finally, evaluate the log probability and the log likelihood
        state['logprior'] = self.compute_log_prior(vars)
        state['ll'] = self.compute_ll(vars)
        state['logp'] = state['ll'] + state['logprior']

        return state

    def sample(self):
        """
        Sample parameters of the GLM from the prior
        """
        v = {}
        v['latent'] = self.latent.sample(v)
        v['net'] = self.network.sample(v)
        for n,glm in enumerate(self.glms):
            v['glm_%d' % n] = glm.sample(v)

        return v

    def preprocess_data(self, data):
        """
        Preprocess the data to compute filtered stimuli, spike trains, etc.
        """
        assert isinstance(data, dict), 'Data must be a dictionary'
        self.latent.preprocess_data(data)
        self.network.preprocess_data(data)
        for glm in self.glms:
            glm.preprocess_data(data)
        data['preprocessed'] = True
        return data

    def set_data(self, data):
        """
        Condition on the data
        """
        assert 'preprocessed' in data and data['preprocessed'] == True, \
            'Data must be preprocessed before it can be set'
        self.latent.set_data(data)
        self.network.set_data(data)
        for glm in self.glms:
            glm.set_data(data)

    def simulate(self, vars, (T_start,T_stop), dt, stim, dt_stim):
        """ Simulate spikes from a network of coupled GLMs
        :param vars - the variables corresponding to each GLM
        :type vars    list of N variable vectors
        :param dt    - time steps to simulate

        :rtype TxN matrix of spike counts in each bin
        """
        # Set the parameter values
        self.set_parameters(vars)

        # Initialize the background rates
        N = self.model['N']
        t = np.arange(T_start, T_stop, dt)
        t_ind = np.arange(int(T_start/dt), int(T_stop/dt))
        assert len(t) == len(t_ind)
        nT = len(t)

        # Initialize the background rate
        X = np.zeros((nT,N))
        for n in np.arange(N):
            X[:,n] = self.glms[n].bias_model.I_bias.value

        # Add stimulus induced currents if given
        temp_data = {'S' : np.zeros((nT, N)),
                     'stim' : stim,
                     'dt_stim': dt_stim}
        self.add_data(temp_data)
        for n in np.arange(N):
            X[:,n] += self.glms[n].bkgd_model.I_stim.value

        print "Max background rate: %s" % str(self.glms[0].nlin_model.f_nlin(np.amax(X)))

        # Remove the temp data from the population data sequences
        self.data_sequences.pop()

        # Get the impulse response functions
        imps = []
        for n_post in np.arange(N):
            imps.append(self.glms[n_post].imp_model.impulse.value)
        imps = np.transpose(np.array(imps), axes=[1,0,2])
        T_imp = imps.shape[2]

        # Debug: compute effective weights
        # tt_imp = dt*np.arange(T_imp)
        # Weff = np.trapz(imps, tt_imp, axis=2)
        # print "Effective impulse weights: "
        # print Weff

        # Iterate over each time step and generate spikes
        S = np.zeros((nT,N))
        acc = np.zeros(N)
        thr = -np.log(np.random.rand(N))

        # TODO: Handle time-varying weights appropriately
        time_varying_weights = False
        if not time_varying_weights:
            At = np.tile(np.reshape(self.network.graph.A.value,
                                    [N,N,1]),
                         [1,1,T_imp])

            Wt = np.tile(np.reshape(self.network.weights.W.value,
                                    [N,N,1]),
                         [1,1,T_imp])

        # Count the number of exceptions arising from more spikes per bin than allowable
        n_exceptions = 0
        for t in np.arange(nT):
            # Update accumulator
            if np.mod(t,10000)==0:
                print "Iteration %d" % t
            # TODO Handle nonlinearities with variables
            lam = self.glms[0].nlin_model.f_nlin(X[t,:])
            acc = acc + lam*dt

            # Spike if accumulator exceeds threshold
            i_spk = acc > thr
            S[t,i_spk] += 1
            n_spk = np.sum(i_spk)

            # Compute the length of the impulse response
            t_imp = np.minimum(nT-t-1,T_imp)

            # Get the instantaneous connectivity
            if time_varying_weights:
                # TODO: Really get the time-varying weights
                At = np.tile(np.reshape(self.network.graph.A.value,
                                        [N,N,1]),
                             [1,1,t_imp])

                Wt = np.tile(np.reshape(self.network.weights.W.value,
                                        [N,N,1]),
                             [1,1,t_imp])

            # Iterate until no more spikes
            # Cap the number of spikes in a time bin
            max_spks_per_bin = 10
            while n_spk > 0:
                if np.any(S[t,:] >= max_spks_per_bin):
                    n_exceptions += 1
                    break
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

                #if np.any(S[t,:]>10):
                #    import pdb
                #    pdb.set_trace()
                #    raise Exception("More than 10 spikes in a bin! Decrease variance on impulse weights or decrease simulation bin width.")

        # DEBUG:
        tt = dt * np.arange(nT)
        lam = np.zeros_like(X)
        for n in np.arange(N):
            lam[:,n] = self.glms[0].nlin_model.f_nlin(X[:,n])

        print "Max firing rate (post sim): %f" % np.max(lam)
        E_nS = np.trapz(lam,tt,axis=0)
        nS = np.sum(S,0)

        print "Sampled %s spikes." % str(nS)
        print "Expected %s spikes." % str(E_nS)

        if np.any(np.abs(nS-E_nS) > 3*np.sqrt(E_nS)):
            print "ERROR: Actual num spikes (%s) differs from expected (%s) by >3 std." % (str(nS),str(E_nS))

        print "Number of exceptions arising from multiple spikes per bin: %d" % n_exceptions

        return S,X

