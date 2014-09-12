"""
Make models from a template
"""
import numpy as np
from scipy.optimize import nnls

from standard_glm import StandardGlm
from spatiotemporal_glm import SpatiotemporalGlm
from shared_tuningcurve_glm import SharedTuningCurveGlm
from simple_weighted_model import SimpleWeightedModel
from simple_sparse_model import SimpleSparseModel
from sparse_weighted_model import SparseWeightedModel
from sbm_weighted_model import SbmWeightedModel
from distance_weighted_model import DistanceWeightedModel

import copy

def make_model(template, N=None, dt=None):
    """ Construct a model from a template and update the specified parameters
    """
    if isinstance(template, str):
        # Create the specified model
        if template.lower() == 'standard_glm' or \
           template.lower() == 'standardglm':
            model = copy.deepcopy(StandardGlm)
        elif template.lower() == 'spatiotemporal_glm':
            model = copy.deepcopy(SpatiotemporalGlm)
        elif template.lower() == 'shared_tuning_curve':
            model = copy.deepcopy(SharedTuningCurveGlm)
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
        elif template.lower() == 'distance_weighted_model' or \
             template.lower() == 'distanceweightedmodel':
            model = copy.deepcopy(DistanceWeightedModel)
        else:
            raise Exception("Unrecognized template model: %s!" % template)
    elif isinstance(template, dict):
        model = copy.deepcopy(template)
    else:
        raise Exception("Unrecognized template model!")

    # Override template model parameters
    if N is not None:
        model['N'] = N

    if dt is not None:
        model['dt'] = dt

    # # Update other parameters as necessary
    # if template.lower() == 'distance_weighted_model' or \
    #    template.lower() == 'distanceweightedmodel':
    #     #model['network']['graph']['location_prior']['sigma'] = N/2.0/3.0
    #     model['network']['graph']['location_prior']['mu'] = \
    #         np.tile(np.arange(N).reshape((N,1)),
    #                 [1,model['network']['graph']['N_dims']]).ravel()

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

    delta = 0.3
    maxeig = 1.0-delta

    if graph_model['type'].lower() == 'erdos_renyi':
        if weight_model['prior']['type'].lower() == 'gaussian':

            # If we have a refractory bias on the diagonal weights then
            # we can afford slightly stronger weights
            if 'refractory_prior' in weight_model:
                maxeig -= weight_model['refractory_prior']['mu']

            sigma = weight_model['prior']['sigma']
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

        # var_AW =  1./4. * (3.*sig_mu)**2 + sig_w**2
        # mean_lam_max = sig_mu * np.sqrt(R) * N/float(R) + 3*sig_w
        # sig_lam_max = np.sqrt(var_AW)
        # ub_lam_max = mean_lam_max + 3*sig_lam_max
        #
        # var_B = (((1.0-mean_lam_max)/3.0)**2 - sig_w**2) / (3*sig_mu)**2
        #
        # print "Setting b0 to %.2f to achive sparsity of %.2f." % (graph_model['b0'],stable_rho)

        #elif weight_model['type'].lower() == 'constant' and \
        #     imp_model['type'].lower() == 'basis':
        #    sigma = imp_model['sigma']
        #    maxeig = 0.7
        #    
        #    # TODO Figure out how sigma actually relates to the eigenvalues
        #    stable_rho = maxeig/N/sigma**2
        #    stable_rho = np.minimum(stable_rho, 1.0)
        #    print "Setting sparsity to %.2f for stability." % stable_rho
        #    graph_model['rho'] = stable_rho\

    elif graph_model['type'].lower() == 'distance':
        # It's hard to characterize this exactly, but suppose that we had an exact
        # banded adjacency matrix where the -b-th lower diagonal to the +b-th upper diagonal
        # are normally distributed, and the rest of the matrix is zero.
        #
        # The max eigenvalue of this matrix tends to grow proportionally to log N
        # and b, and is scaled by the standard deviation of the weights, sigma.
        # The mean refractory weight adds a bias to the max eigenvalue.
        #
        # To estimate the bandwidth (i.e. 2b in this example), suppose that the
        # probability of connection p(A_n,n') decays exponentially with |n-n'| at
        # length scale d. That is, when |n-n'| > d, the probability has decreased to
        # about 1/3. Hence, at distance 2d, the probability of interaction is about 10%.
        # Thus, let's say the bandwidth of this matrix is roughly 2*(2d) = 4d,
        #
        # So, given the bandwidth of 4d and the matrix size N, we will scale the weight
        # distribution to hopefully achieve stability
        b = 8.0 * graph_model['delta']

        # Constant scaling factors
        # max eig ~= mu_ref + .25*sigma_W * (b+2*log(N))
        # So set sigma_W ~= 4*(1-mu_ref)/(b+2*log N)
        if weight_model['type'].lower() == 'gaussian':
            mu_ref = weight_model['refractory_prior']['mu']
            sig_w = 4.0*(1.0-mu_ref)/(b+2.0*np.log(model['N']))

            print "Setting sig_w to %.3f to ensure stability" % sig_w
            weight_model['prior']['sigma'] = sig_w
        else:
            raise Exception("Unrecognized weight model for distance graph: %s" % weight_model['type'])

def check_stability(model, x, N):
    """
    Check the stability of model parameters
    """

    if model['network']['weight']['type'].lower() == 'gaussian':
        Weff = x['net']['graph']['A'] * np.reshape(x['net']['weights']['W'], (N,N))
        maxeig = np.amax(np.real(np.linalg.eig(Weff)[0]))
        print "Max eigenvalue of Weff: %.2f" % maxeig
        return maxeig < 1
    else:
        print "Check stability: unrecognized model type. Defaulting to true."
        return True

def convert_model(from_popn, from_model, from_vars, to_popn, to_model, to_vars):
    """ Convert from one model to another model of a different type
        Generally this will involve projecting impulse responses, etc.
        It's hairy business.
    """

    # Idea: Get the state of the GLMs, e.g. the impulse responses, etc.
    #       Project those states onto the parameters of the to-model
    N = from_popn.N
    # import pdb; pdb.set_trace()
    from_state = from_popn.eval_state(from_vars)
    to_state = to_popn.eval_state(to_vars)

    conv_vars = None
    if from_model['impulse']['type'].lower() == 'basis':
        if to_model['impulse']['type'].lower() == 'normalized' or \
           to_model['impulse']['type'].lower() == 'dirichlet':
            import copy
            conv_vars = copy.deepcopy(to_vars)

            # To convert from basis -> normalized, project the impulse
            # responses onto the normalized basis, divide by the area
            # under the curve to get the weight.
            W = np.zeros((N,N))
            for n2 in np.arange(N):
                B = to_state['glms'][n2]['imp']['basis'].shape[1]
                w_ir_n2 = np.zeros((N,B))
                for n1 in np.arange(N):
                    # Solve a nonnegative least squares problem
                    (w_ir_n1n2p, residp) = nnls(to_state['glms'][n2]['imp']['basis'],
                                                from_state['glms'][n2]['imp']['impulse'][n1,:])
                    (w_ir_n1n2n, residn) = nnls(to_state['glms'][n2]['imp']['basis'],
                                                -1.0*from_state['glms'][n2]['imp']['impulse'][n1,:])

                    # Take the better of the two solutions
                    if residp < residn:
                        Wsgn = 1.0
                        w_ir_n1n2 = w_ir_n1n2p
                    else:
                        Wsgn = -1.0
                        w_ir_n1n2 = w_ir_n1n2n

                    # Normalized weights must be > 0, sum to 1
                    w_ir_n1n2 = w_ir_n1n2
                    w_ir_n1n2 = np.clip(w_ir_n1n2,0.001,np.Inf)
                    # Normalize the impulse response to get a weight
                    W[n1,n2] = Wsgn*np.sum(w_ir_n1n2)

                    # Set impulse response to normalized impulse response
                    w_ir_n2[n1,:] = w_ir_n1n2 / np.sum(w_ir_n1n2)

                # Update to_vars
                if to_model['impulse']['type'].lower() == 'normalized':
                    conv_vars['glms'][n2]['imp']['w_lng'] = np.log(w_ir_n2.flatten())
                if to_model['impulse']['type'].lower() == 'dirichlet':
                    for n1 in range(N):
                        # Scale up the weights such that the average is preserved
                        alpha = to_popn.glm.imp_model.alpha
                        B = to_popn.glm.imp_model.B
                        conv_vars['glms'][n2]['imp']['g_%d' % n1] = alpha * B * w_ir_n2[n1,:]

            # Update to_vars
            conv_vars['net']['weights']['W'] = W.flatten()

            # Threshold the adjacency matrix to start with the right level of sparsity
            if 'rho' in to_model['network']['graph'].keys():
                W_sorted = np.sort(np.abs(W.ravel()))
                thresh = W_sorted[np.floor((1.0-2.0*to_model['network']['graph']['rho'])*(N**2-N)-N)]
                conv_vars['net']['graph']['A'] = (np.abs(W) >= thresh).astype(np.int8)
            else:
                conv_vars['net']['graph']['A'] = np.ones((N,N), dtype=np.int8)

            # Update simple other parameters
            for n in np.arange(N):
                conv_vars['glms'][n]['bias']['bias'] = from_vars['glms'][n]['bias']['bias']

    # Update background params
    if 'sharedtuningcurves' in to_model['latent'] and \
        from_model['bkgd']['type'] == 'spatiotemporal':
        convert_stimulus_filters_to_sharedtc(from_popn, from_model, from_vars,
                                             to_popn, to_model, conv_vars)

    return conv_vars

def convert_stimulus_filters_to_sharedtc(from_popn, from_model, from_vars, to_popn, to_model, to_vars):
    """
    Convert a set of stimulus filters to a shared set of tuning curves
    """
    # Get the spatial component of the stimulus filter for each neuron
    N = from_popn.N
    R = to_model['latent']['sharedtuningcurves']['R']
    from_state = from_popn.eval_state(from_vars)

    locs = np.zeros((N,2))
    local_stim_xs = []
    local_stim_ts = []
    for n in range(N):
        s_glm = from_state['glms'][n]


        # to_state = to_popn.eval_state(to_vars)
        assert 'stim_response_x' in s_glm['bkgd']

        # Get the stimulus responses
        stim_x = s_glm['bkgd']['stim_response_x']
        stim_t = s_glm['bkgd']['stim_response_t']
        loc_max = np.argmax(np.abs(stim_x))

        if stim_x.ndim == 2:
            locsi, locsj = np.unravel_index(loc_max, stim_x.shape)
            locs[n,0], locs[n,1] = locsi.ravel(), locsj.ravel()

        # Get the stimulus response in the vicinity of the mode
        # Create a meshgrid of the correct shape, centered around the max
        max_rb = to_model['latent']['latent_location']['location_prior']['max0']
        max_ub = to_model['latent']['latent_location']['location_prior']['max1']
        gsz = to_model['latent']['sharedtuningcurves']['spatial_shape']
        gwidth = (np.array(gsz) - 1)//2
        lb = max(0, locs[n,0]-gwidth[0])
        rb = min(locs[n,0]-gwidth[0]+gsz[0], max_rb)
        db = max(0, locs[n,1]-gwidth[1])
        ub = min(locs[n,1]-gwidth[1]+gsz[1], max_ub)
        grid = np.ix_(np.arange(lb, rb).astype(np.int),
                      np.arange(db, ub).astype(np.int))

        # grid = grid.astype(np.int)
        # Add this local filter to the list
        local_stim_xs.append(stim_x[grid])
        local_stim_ts.append(stim_t)

    # Cluster the local stimulus filters
    from sklearn.cluster import KMeans
    flattened_filters_x = np.array(map(lambda f: f.ravel(), local_stim_xs))
    flattened_filters_t = np.array(map(lambda f: f.ravel(), local_stim_ts))
    km = KMeans(n_clusters=R)
    km.fit(flattened_filters_x)
    Y = km.labels_
    print 'Filter cluster labels from kmeans: ',  Y

    # Initialize type based on stimulus filter
    to_vars['latent']['sharedtuningcurve_provider']['Y'] = Y

    # Initialize shared tuning curves (project onto the bases)
    from pyglm.utils.basis import project_onto_basis
    for r in range(R):
        mean_filter_xr = flattened_filters_x[Y==r].mean(axis=0)
        mean_filter_tr = flattened_filters_t[Y==r].mean(axis=0)

        # Project the mean filters onto the basis
        to_vars['latent']['sharedtuningcurve_provider']['w_x'][:,r] = \
            project_onto_basis(mean_filter_xr,
                               to_popn.glm.bkgd_model.spatial_basis).ravel()

        # Temporal part of the filter
        temporal_basis = to_popn.glm.bkgd_model.temporal_basis
        t_temporal_basis = np.arange(temporal_basis.shape[0])
        t_mean_filter_tr = np.linspace(0, temporal_basis.shape[0]-1, mean_filter_tr.shape[0])
        interp_mean_filter_tr = np.interp(t_temporal_basis, t_mean_filter_tr, mean_filter_tr)
        to_vars['latent']['sharedtuningcurve_provider']['w_t'][:,r] = \
            project_onto_basis(interp_mean_filter_tr, temporal_basis).ravel()


    # Initialize locations based on stimuls filters
    to_vars['latent']['location_provider']['L'] = locs.ravel().astype(np.int)

