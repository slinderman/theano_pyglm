"""
Make models from a template
"""
import numpy as np

from standard_glm import StandardGlm
from spatiotemporal_glm import SpatiotemporalGlm
from simple_weighted_model import SimpleWeightedModel
from simple_sparse_model import SimpleSparseModel
from sparse_weighted_model import SparseWeightedModel
from sbm_weighted_model import SbmWeightedModel

import copy

def make_model(template, N=None):
    """ Construct a model from a template and update the specified parameters
    """
    if isinstance(template, str):
        # Create the specified model
        if template.lower() == 'standard_glm' or \
           template.lower() == 'standardglm':
            model = copy.deepcopy(StandardGlm)
        elif template.lower() == 'spatiotemporal_glm':
            model = copy.deepcopy(SpatiotemporalGlm)
        elif template.lower() == 'simple_weighted_model' or \
             template.lower() == 'simpleweightedmodel':
            model = copy.deepcopy(SimpleWeightedModel)
        elif template.lower() == 'simple_sparse_model' or \
             template.lower() == 'simplesparsemodel':
            model = copy.deepcopy(SimpleSparseModel)
        elif template.lower() == 'sparse_weighted_model' or \
             template.lower() == 'sparseweightedmodel':
            model = copy.deepcopy(SparseWeightedModel)
        elif template.lower() == 'sbm_weighted_model' or \
             template.lower() == 'sbmweightedmodel':
            model = copy.deepcopy(SbmWeightedModel)

    elif isinstance(template, dict):
        model = copy.deepcopy(template)
    else:
        raise Exception("Unrecognized template model!")

    # Override template model parameters
    if N is not None:
        model['N'] = N

    # TODO Update other parameters as necessary

    return model

def stabilize_sparsity(model):
    """ Adjust the sparsity level for simple weighted models
        with Gaussian weight models and Bernoulli adjacency matrices.
        The variance of a Gaussian N(0,sigma) times a Bernoulli(rho) is 
            E[B*N] = 0
            Var[B*N] = E[(B*N)**2] = rho*sigma^2 + (1-rho)*0 = rho*sigma^2

        Hence the eigenvalues will be distributed in a complex disk of radius
            \sqrt(N) * \sqrt(rho) * sigma 
        For this to be less than (1-delta), we have
            \sqrt(rho) < (1-delta)/sqrt(N)/sigma
            rho < (1-delta)**2 / N / sigma**2 
    """
    N = model['N']
    imp_model = model['impulse']
    weight_model = model['network']['weight']
    graph_model = model['network']['graph']
    if weight_model['type'].lower() == 'gaussian':
        maxeig = 0.7
        # If we have a refractory bias on the diagonal weights then
        # we can afford slightly stronger weights
        if 'mu_refractory' in weight_model:
            maxeig -= weight_model['mu_refractory']

        sigma = weight_model['sigma']

        if graph_model['type'].lower() == 'erdos_renyi':
            stable_rho = maxeig**2/N/sigma**2
            stable_rho = np.minimum(stable_rho, 1.0)
            print "Setting sparsity to %.2f for stability." % stable_rho
            graph_model['rho'] = stable_rho
        
        elif graph_model['type'].lower() == 'sbm':
            # Things are trickier in the SBM case because the entries in A
            # are not iid. But, we can still make an independence approximation
            # and set the SBM sparsity distribution such that the average
            # sparsity is the desired level for an ER graph.
            stable_rho = maxeig**2/N/sigma**2
            stable_rho = np.minimum(stable_rho, 1.0)

            # The mean of a beta is b0/(b0+b1) \approx \rho
            # b0 \approx b0*\rho + b1*\rho
            # b0 \approx b1\rho/(1-\rho)
            # Letting b1 = 1, this gives us a value for rho
            graph_model['b0'] = stable_rho/(1.0-stable_rho)

            print "Setting b0 to %.2f to achive sparsity of %.2f." % (graph_model['b0'],stable_rho)

        #elif weight_model['type'].lower() == 'constant' and \
        #     imp_model['type'].lower() == 'basis':
        #    sigma = imp_model['sigma']
        #    maxeig = 0.7
        #    
        #    # TODO Figure out how sigma actually relates to the eigenvalues
        #    stable_rho = maxeig/N/sigma**2
        #    stable_rho = np.minimum(stable_rho, 1.0)
        #    print "Setting sparsity to %.2f for stability." % stable_rho
        #    graph_model['rho'] = stable_rho

        
