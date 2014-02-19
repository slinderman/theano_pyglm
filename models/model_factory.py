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
            # are not iid. But, we can still make some approximations by
            # thinking about the mean and variance within a single block, wherein
            # the entries really are i.i.d. Then we scale the eigs of
            # each block by N/R, as if the blocks were equal size and the
            # interdependencies between blocks did not matter. Obviously,
            # this is a hack.
            R = graph_model['R']
            if weight_model['type'].lower() == 'sbm':
                sig_mu = weight_model['sigma_mu']
                sig_w = weight_model['sigma_w']
            elif weight_model['type'].lower() == 'gaussian':
                sig_mu = 0.0
                sig_w = weight_model['sigma']
            else:
                raise Exception("Unrecognized weight model for SBM graph: %s" % weight_model['type'])

            var_AW =  1./4. * (3.*sig_mu)**2 + sig_w**2
            mean_lam_max = sig_mu * np.sqrt(R) * N/float(R) + 3*sig_w
            sig_lam_max = np.sqrt(var_AW)
            ub_lam_max = mean_lam_max + 3*sig_lam_max

            var_B = (((1.0-mean_lam_max)/3.0)**2 - sig_w**2) / (3*sig_mu)**2

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

        
