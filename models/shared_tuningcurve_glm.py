"""
Simple GLM with two neurons connected by a complete network with Gaussian weights.
"""
import numpy as np
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
                    'N' : 3,
                    'R' : 2,
                    'alpha0' : 10.0,

                    # Tuning curve params
                    'dt_max' : 0.3,
                    'mu' : 0.0,
                    'sigma' : 2.0,
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
                },

            'latent_location':
                {
                    'type' : 'latent_location',
                    'name' : 'location_provider',
                    'N' : 3,
                    'N_dims' : 2,
                    'location_prior' :
                        {
                            'type' : 'joint_categorical',
                            'min1' : 0,
                            'max1' : 4,
                            'p1' : np.ones(5)/5.0,
                            'min2' : 0,
                            'max2' : 4,
                            'p2' : np.ones(5)/5.0
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
            'locations' : 'location_provider'
        },

     # Parameters of the impulse responses
     # 'impulse' :
     #     {
     #         'type' : 'basis',
     #        'dt_max' : 0.2,
     #        'prior' :
     #            {
     #                'type' : 'group_lasso',
     #                # 'type' : 'gaussian',
     #                'mu' : 0.0,
     #                'sigma' : 0.5,
     #                'lam' : 1.0
     #            },
     #        'basis' :
     #            {
     #                'type' : 'cosine',
     #                'n_eye' : 0,
     #                'n_cos' : 5,
     #                'a': 1.0/120,
     #                'b': 0.5,
     #                'orth' : True,
     #                'norm' : False
     #            }
     #     },
     # Parameters of the impulse responses
    'impulse' :
        {
            # 'type' : 'normalized',
            'type' : 'dirichlet',
            'dt_max' : 0.2,
            'alpha' : 1,
            'basis' :
                {
                    'type' : 'cosine',
                    'n_eye' : 0,
                    'n_cos' : 5,
                    #'type' : 'exp',
                    #'n_eye' : 0,
                    #'n_exp' : 5,
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
                    'prior' :
                     {
                         'type' : 'gaussian',
                         'mu' : 0.0,
                         'sigma' : 2.0
                     },
                    'refractory_prior' :
                    {
                        'type' : 'gaussian',
                        'mu' : -1.0,
                        'sigma' : 0.25
                    }
                },

            'graph' :
                {
                    'type' : 'erdos_renyi',
                    'rho' : 0.5,
                    'rho_refractory' : 1.0
                }
        },
}


