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
            'type' : 'explinear'
        },
    
    # Parameters of the bias    
    'bias' :
        {
            'type' : 'constant',
            'mu' : 20,
            'sigma' : 0.1
        },

    # Parameters of the background model
    'bkgd' :
        {
            #'type' : 'basis',
            'type' : 'none',
            'D_stim' : 1,       # Dimensionality of the stimulus
            'dt_max' : 0.3,
            'prior' : 
                {
                    'type' : 'spherical_gaussian',
                    'mu' : 0.0,
                    'sigma' : 0.01,
                },
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
             'dt_max' : 0.2,
             'mu' : 0.0,
             'sigma' : 200.0,
             'basis' :
                 {
                     'type' : 'cosine',
                     'n_eye' : 0,
                     'n_cos' : 5,
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


