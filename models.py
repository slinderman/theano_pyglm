"""
Simple GLM
"""
SimpleModel = \
{
    # Number of neurons (parametric model!)
    'N' : 2,
    # Dimensionality of the stimulus
    'D_stim' : 1,
    
    # Parameters of the nonlinearity
    'nonlinearity' :
        {
            'type' : 'exp'
        },
    
    # Parameters of the bias    
    'bias' :
        {
            'type' : 'constant',
            'mu' : -3,
            'sigma' : 0.1
        },

    # Parameters of the background model
    'bkgd' :
        {
            'type' : 'basis',
            'T_max' : 100,
            'basis' : 'cosine',
            'mu' : 0,
            'sigma' : 0.1
        },

    # Parameters of the impulse responses
    'impulse' :
        {
            'type' : 'basis',
            'T_max' : 10,
            'basis' : 'cosine',
            'mu' : 0,
            'sigma' : 1
        },

    # Parameters of the network
    'network' :
        {
            'type' : 'complete'
        },

    # Parameters of the basis functions
    'basis' :
        {
            'n_eye' : 0,
            'n_cos' : 3,
            'a': 1.0/120,
            'b': 0.5,
            'orth' : False,
            'norm' : True
        }
}


