import theano
import theano.tensor as T

from impulse import *

class Network:
    """ Filter the stimulus and expose the filtered stimulus
    """

    def __init__(self, n, N, vars, v_offset, **kwargs):
        """ Initialize the filtered stim model
        """
        prms = {'T_max' : 10,
                'basis_type' : 'cosine',
                'orth' : False,
                'norm' : True,
                'n_cos' : 3,
                'mu_w' : 0,
                'sig_w' : 1}
        prms.update(**kwargs)

        self.mu_w = prms['mu_w']
        self.sig_w = prms['sig_w']

        # Create a basis for the stimulus response
        self.basis = create_impulse_basis(**prms)
        # Compute the number of parameters
        (_,B) = self.basis.shape
        
        # Keep track of the number of variables
        self.n_vars = 0
        
        # Initialize a potentially dynamic weight matrix
        self.W = theano.shared(np.ones((1,N)))
        
        # Define impulse models for each presynaptic neuron
        self.imp_models = []
        for n_pre in np.arange(N):
            imp_model = BasisImpulse(n_pre, n,
                                     vars, 1,
                                     v_offset,
                                     basis=self.basis,
                                     **prms)
            self.imp_models.append(imp_model)
            
            self.n_vars += imp_model.n_vars
            v_offset += imp_model.n_vars
        
        # Get the total network current by adding the incoming currents
        I_irs = [T.constant(0.)]
        for n_pre in np.arange(N):
            #  Multiply by the weight matrix
            I_pre = self.W[:,n_pre] * self.imp_models[n_pre].I_ir
            
            I_irs.append(I_irs[-1] + I_pre)
            
        self.I_net = I_irs[-1]
        
        # Compute the log prior
        lp_imps = [T.constant(0.)]
        for n_pre in np.arange(N):
            lp_imps.append(lp_imps[-1] + self.imp_models[n_pre].log_p)
            
        self.log_p = lp_imps[-1]
        
    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        # Set data for impulse responses
        for imp_model in self.imp_models:
            imp_model.set_data(data)
            
            
        # TODO - SET THE WEIGHTS FROM THE OUTPUT OF A PARTICLE FILTER
        self.W.set_value(np.ones_like(data["S"]))

    def sample(self):
        """
        return a sample of the variables
        """
        vars = []
        
        # Sample impulse responses
        for imp_model in self.imp_models:
            w_imp = imp_model.sample()
            vars = np.concatenate((vars,w_imp))
        return vars

    def params(self):
        return {'basis' : self.basis}