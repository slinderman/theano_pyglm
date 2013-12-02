"""
Simple GLM with two neurons connected by a complete network with Gaussian weights.
"""
SpatiotemporalGlm = \
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
            'type' : 'spatiotemporal',
            'D_stim' : 3,       # Dimensionality of the stimulus
            'dt_max' : 500,
            'mu' : 0,
            'sigma' : 0.5,
            'temporal_basis' :
                {
                    'type' : 'cosine',
                    'n_eye' : 0,
                    'n_cos' : 3,
                    'a': 1.0/120,
                    'b': 0.5,
                    'orth' : False,
                    'norm' : True
                },
            'spatial_basis' :
                {
                    'type' : 'gaussian',
                    'n_eye' : 0,
                    'n_gauss' : 3,
                    'a': 1.0/120,
                    'b': 0.5,
                    'orth' : False,
                    'norm' : True
                }
        },

     # Parameters of the impulse responses
     'impulse' :
         {
             'type' : 'basis',
             'dt_max' : 10,
             'mu' : 0.0,
             'sigma' : 0.33,
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
                    'type' : 'constant',
                    'value' : 1.0
                },

            'graph' :
                {
                    'type' : 'complete'
                }
        },
}


