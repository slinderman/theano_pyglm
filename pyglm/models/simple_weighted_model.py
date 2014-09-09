"""
Simple GLM with two neurons connected by a complete network with Gaussian weights.
"""
SimpleWeightedModel = \
{
    # Number of neurons (parametric model!)
    'N' : 2,
    
    # Parameters of the nonlinearity
    'nonlinearity' :
        {
            'type' : 'exp'
        },
    
    # Parameters of the bias    
    'bias' :
        {
            'type' : 'constant',
            'mu' : -2,
            'sigma' : 0.1
        },

    # Parameters of the background model
    'bkgd' :
        {
            'type' : 'basis',
            'D_stim' : 1,       # Dimensionality of the stimulus
            'dt_max' : 500,
            'mu' : 0,
            'sigma' : 0.5,
            'basis' :
                {
                    'type' : 'cosine',
                    'n_eye' : 0,
                    'n_cos' : 3,
                    'a': 1.0/120,
                    'b': 0.5,
                    'orth' : False,
                    'norm' : True
                }
        },

    # Parameters of the impulse responses
    'impulse' :
        {
            'type' : 'dirichlet',
            'dt_max' : 50,
            'alpha' : 1,
            'basis' :
                {
                    'type' : 'cosine',
                    'n_eye' : 0,
                    'n_cos' : 3,
                    'a': 1.0/120,
                    'b': 0.5,
                    'orth' : False,
                    'norm' : True
                }
        },

    # Parameters of the network
    'network' :
        {
            'weight' :
                {
                    'type' : 'gaussian',
                    'mu' : 0.0,
                    'sigma' : 0.5
                },

            'graph' :
                {
                    'type' : 'complete'
                }
        },
}


