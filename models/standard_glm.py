"""
Simple GLM with two neurons connected by a complete network with Gaussian weights.
"""
StandardGlm = \
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
            'mu' : 0,
            'sigma' : 0.1
        },

    # Parameters of the background model
    'bkgd' :
        {
            'type' : 'basis',
            'D_stim' : 1,       # Dimensionality of the stimulus
            'dt_max' : 0.3,
            'mu' : 0.0,
            'sigma' : 0.1,
            'basis' :
                {
                    'type' : 'cosine',
                    'n_eye' : 0,
                    'n_cos' : 3,
                    'a': 1.0/120,
                    'b': 0.5,
                    'orth' : True,
                    'norm' : False
                }
        },

     # Parameters of the impulse responses
     'impulse' :
         {
             'type' : 'basis',
             'dt_max' : 0.3,
             'mu' : 0.0,
             'sigma' : 0.3,
             'basis' :
                 {
                     'type' : 'cosine',
                     'n_eye' : 0,
                     'n_cos' : 3,
                     'a': 1.0/120,
                     'b': 0.5,
                     'orth' : True,
                     'norm' : False
                 }
         },
    # Parameters of the network
    'network' :
        {
            'weight' :
                {
                    'type' : 'constant',
                    'value' : 1.0
                },

            'graph' :
                {
                    'type' : 'complete'
                }
        },
}


