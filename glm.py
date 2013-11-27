import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
    
import numpy as np
import utils.poisson_process as pp
import scipy.optimize as opt
import scipy.linalg as linalg

from components.network import *

from components.bkgd import *
from components.bias import *
from components.impulse import *
from components.nlin import *

class NetworkGlm:
    """
    Network of connected GLMs. 
    """
    def __init__(self, model):
        """
        Initialize the network GLM with given model. What needs to be set?
        """
        self.model = model
        N = model['N']

        # Create a network model to connect the GLMs
        self.network = Network(model)

        # Create GLMs for each neuron
        self.glms = []
        for n in np.arange(N):
            self.glms.append(Glm(n, model, self.network))

        # Evalualte the total log probability
        log_p_glms = map(lambda glm: T.shape_padright(glm.log_p, n_ones=1),
                         self.glms)

        # Concatenate the log posteriors into a vector
        self.log_p = T.sum(T.concatenate(log_p_glms, axis=0)) + \
                     self.network.log_p
        
        # Concatenate all variables into a list
        self.vars = reduce(lambda vacc,glm: vacc+[glm.vars], self.glms,
                           self.network.vars)

        # Compute gradients of the joint log probability
        self.compute_gradients()

    def compute_gradients(self):
        """ Compute gradients of the joint log posterior distribution over GLM
            parameters and networks. Try to take advantage of conditional
            independencies between the GLMs. For example, The background and impulse
            responses may be conditionally independent given the network.
        """

        # Most general form: compute the gradient of the joint log posterior
        self.g_vars = T.grad(self.log_p, self.vars)

        # TODO: Compute this for all pairs of input variables
        # Finally, compute the Hessian
        self.H_vars,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
                               sequences=T.arange(self.g_vars.shape[0]),
                               non_sequences=[self.g_vars, self.vars])

        # Create callable functions to compute firing rate, log likelihood, and gradients
        self.f_lp = theano.function(self.vars, self.log_p)
        self.g_lp = theano.function(self.vars, self.g_vars)
        self.H_lp = theano.function(self.vars, self.H_vars)

    def set_data(self, data):
        """
        Condition on the data
        """
        self.network.set_data(data)

        for glm in self.glms:
            glm.set_data(data)
            
    def sample(self):
        """
        Sample parameters of the GLM from the prior
        """
        vars = []

        vars.append(self.network.sample())

        for glm in self.glms:
            vars.append(glm.sample())
            
        return vars
        
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

        # Initialize the background rate
        X = np.zeros((nT,N))
        for n in np.arange(N):
            X[:,n] = self.glms[n].bias_model.f_I_bias(vars[n])

        # Add stimulus induced currents if given
        for n in np.arange(N):
            X[:,n] += self.glms[n].bkgd_model.f_I_stim(vars[n])[t_ind]

        # Get the impulse response functions
        imps = []
        for n_pre in np.arange(N):
            imps.append(map(lambda n_post: self.glms[n_post].imp_model.imp_models[n_pre].f_impulse(vars[n_post]),
                            np.arange(N)))
        imps = np.array(imps)
        T_imp = imps.shape[2]

        import pdb
        pdb.set_trace()

        # Iterate over each time step and generate spikes
        S = np.zeros((nT,N))
        acc = np.zeros(N)
        thr = -np.log(np.random.rand(N))
            
        for t in np.arange(nT):
            # Update accumulator
            if np.mod(t,1000)==0: 
                print "Iteration %d" % t
            lam = np.array(map(lambda n: self.glms[n].nlin_model.f_nlin(X[t,n]),
                           np.arange(N)))
            acc = acc + lam*dt
            
            # Spike if accumulator exceeds threshold
            i_spk = acc > thr
            S[t,i_spk] += 1
            n_spk = np.sum(i_spk)
    
            # Compute the length of the impulse response
            t_imp = np.minimum(nT-t-1,T_imp)
    
            # Iterate until no more spikes
            while n_spk > 0:
                # Add impulse response to activation of other neurons)
                X[t+1:t+t_imp+1,:] += np.sum(imps[i_spk,:,:t_imp],0).T
                
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
            lam[:,n] = self.glms[n].nlin_model.f_nlin(X[:,n])

        E_nS = np.trapz(lam,tt,axis=0)
        nS = np.sum(S,0)

        print "Sampled %s spikes." % str(nS)
    
        if np.any(np.abs(nS-E_nS) > 3*np.sqrt(E_nS)):
            print "ERROR: Actual num spikes (%d) differs from expected (%d) by >3 std." % (E_nS,nS)
            import pdb
            pdb.set_trace()
    

        return S,X
    
    def fit(self, x0=None):
        """ Fit the GLM using BFGS or other scipy optimization package
        """
        N = self.model['N']

        # Draw initial state from prior if not given
        if x0 is None:
            x0 = self.sample()

        # Determine the shapes of the parameters
        shapes = map(lambda x: x.shape, x0)

        # Concatenate the states int

        x_opts = []
        for n in np.arange(N):
            print "Fitting GLM %d" % n

            nll = lambda x: -1.0 * self.glms[n].f_lp(x)
            grad_nll = lambda x: -1.0*self.glms[n].g_lp(x)
            hess_nll = lambda x: -1.0*self.glms[n].H_lp(x)

            x_opt = opt.fmin_ncg(nll,x0[n],
                                 fprime=grad_nll,
                                 fhess=hess_nll,
                                 disp=True)
            x_opts.append(x_opt)

        return x_opts

class Glm:
    def __init__(self, n, model, network=None):
        """
        Create a GLM for the spikes on the n-th neuron out of N
        This corresponds to the spikes in the n-th column of data["S"]
        """
        # Define the Poisson regression model
        self.n = n
        self.dt = theano.shared(name='dt',
                                value=1.0)
        self.S = theano.shared(name='S',
                               value=np.zeros((1, model['N'])))

        # Concatenate the variables into one long vector
        self.vars = T.dvector(name='vars')
        v_offset = 0

        # Define a bias to the membrane potential
        #self.bias_model = ConstantBias(vars, v_offset)
        self.bias_model = create_bias_component(model, self.vars, v_offset)
        v_offset += self.bias_model.n_vars
       
        # Define stimulus and stimulus filter
        self.bkgd_model = create_bkgd_component(model, self.vars, v_offset)
        v_offset += self.bkgd_model.n_vars

        # Create a list of impulse responses for each incoming connections
        self.imp_model = create_impulse_component(model, self.vars, v_offset, n)
        v_offset += self.imp_model.n_vars

        # If a network is given, weight the impulse response currents and sum them up
        if network is not None:
            # Compute the effective incoming weights
            W_eff = network.graph.A[:,n] * network.weights.W[:,n]
        else:
            W_eff = np.ones((model['N'],))

        I_net = T.dot(self.imp_model.I_imp, W_eff)

        # Rectify the currents to get a firing rate
        self.nlin_model = create_nlin_component(model, self.vars, v_offset)
        lam = self.nlin_model.nlin(self.bias_model.I_bias +
                                   self.bkgd_model.I_stim +
                                   I_net)

        # Compute the log likelihood under the Poisson process
        ll = T.sum(-self.dt*lam + T.log(lam)*self.S[:,n])

        # Compute the log prior
        lp_bias = self.bias_model.log_p
        lp_bkgd = self.bkgd_model.log_p
        lp_imp = self.imp_model.log_p
        lp_nlin = self.nlin_model.log_p
        self.log_p = ll + lp_bias + lp_bkgd + lp_imp + lp_nlin

        # Compute the gradient of the log likelihood wrt vars
        all_vars = [self.vars] + network.vars
        g_vars = T.grad(self.log_p, all_vars)
        #
        ## Finally, compute the Hessian
        #H_vars,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
        #                       sequences=T.arange(g_vars.shape[0]),
        #                       non_sequences=[g_vars, self.vars])
        #
        ## Create callable functions to compute firing rate, log likelihood, and gradients
        #theano.config.on_unused_input = 'ignore'
        self.f_lam = theano.function(all_vars, lam)
        self.f_log_p = theano.function(all_vars, self.log_p)
        #self.g_lp = theano.function([self.vars, network.weights.W], g_vars)
        #self.H_lp = theano.function([self.vars, network.weights.W], H_vars)

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
        vars = []
        
        # Sample bias
        bias = self.bias_model.sample()
        vars = np.concatenate((vars,[bias]))
        
        # Sample background weights
        w_stim = self.bkgd_model.sample()
        vars = np.concatenate((vars,w_stim))
        
        # Sample impulse responses ...
        net_vars = self.imp_model.sample()
        vars = np.concatenate((vars,net_vars))

        return vars
    
    def params(self):
        params = {}
        params['bkgd'] = self.bkgdmodel.params()
        params['imp'] = self.imp_model.params()
        
        return params
