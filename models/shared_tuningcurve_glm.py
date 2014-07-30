"""
Simple GLM with two neurons connected by a complete network with Gaussian weights.
"""
SharedTuningCurveGlm = \
{
    # Number of neurons (parametric model!)
    'N' : 2,

    'latent' :
        {
            'sharedtuningcurves' :
                {
                    'type' : 'latent_type_with_tuning_curves',
                    'name' : 'sharedtuningcurve_provider',

                    # Latent type params
                    'N' : 2,
                    'R' : 2,
                    'alpha0' : 1.0,

                    # Tuning curve params
                    'dt_max' : 0.3,
                    'mu' : 0.0,
                    'sigma' : 5.0,
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
                    #'spatial_basis' :
                    #    {
                    #        'type' : 'gaussian',
                    #        'n_eye' : 0,
                    #        'n_gauss' : (3,),
                    #        'orth' : False,
                    #        'norm' : True
                    #    }
                    'spatial_shape' : (2,2),
                    'spatial_basis' :
                        {
                            'type' : 'identity',
                            'n_eye' : 4
                        }
                }
        },
    
    # Parameters of the nonlinearity
    'nonlinearity' :
        {
            'type' : 'explinear'
        },
    
    # Parameters of the bias    
    'bias' :
        {
            'type' : 'constant',
            'mu' : 10.0,
            'sigma' : 1.0
        },

    # Parameters of the background model
    'bkgd' :
        {
            'type' : 'sharedtuningcurve',
            'tuningcurves' : 'sharedtuningcurve_provider',
        },

     # Parameters of the impulse responses
     'impulse' :
         {
             'type' : 'basis',
            'dt_max' : 0.2,
            'prior' :
                {
                    'type' : 'group_lasso',
                    # 'type' : 'gaussian',
                    'mu' : 0.0,
                    'sigma' : 0.5,
                    'lam' : 1.0
                },
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


