__author__ = 'Scott'

"""
Simple GLM
"""
SimpleModel = \
{
    'bias' :
        {
            'type' : 'constant',
            'mu' : -3,
            'sigma' : 0.1
        },

    'bkgd' :
        {
            'type' : 'basis',
            'T_max' : 100,
            'basis' : 'cosine',
            'mu' : 0,
            'sigma' : 0.1
        },

    'impulse' :
        {
            'type' : 'basis',
            'T_max' : 10,
            'basis' : 'cosine',
            'mu' : 0,
            'sigma' : 1
        },

    'network' :
        {
            'type' : 'complete'
        },

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


