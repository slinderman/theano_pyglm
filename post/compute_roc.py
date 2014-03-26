""" Compute an roc curve for synthetic data with ground truth
"""
import os
import cPickle
import numpy as np
from sklearn.metrics import roc_curve

from plotting.roc import plot_roc_curve
from population import Population
from models.model_factory import make_model

import matplotlib
matplotlib.use('Agg')       # To enable saving remotely
import matplotlib.pyplot as plt

def compute_roc_from_std_glm(data, x_map):
    """ Compute the ROC curve for a standard glm with linear basis functions
        for the impulse response. The probability of a connection is proportional
        to the area under the impulse response (L1 norm)
    """
    # Get the weights of the impulse responses
    W_inf = x_map['net']['weights']['W'] * x_map['net']['graph']['A']
    N = W_inf.shape[0]
    
    s_imps = []
    for n_post in np.arange(N):
        s_imp_row = []
        
        s_imp_n = x_map['glms'][n_post]['imp']['impulse']
        for n_pre in np.arange(N):
            w = W_inf[n_pre,n_post]
            s_imp_row.append(w*s_imp_n[n_pre,:])

        s_imps.append(s_imp_row)

    # Convert to numpy array with pre x post x T_imp 
    s_imps = np.array(s_imps)    
    # Transpose so that pre is row index 
    s_imps = np.transpose(s_imps, [1,0,2])

    W_imp = np.sum(s_imps,2)
    W_imp_max = np.amax(np.abs(W_imp))

    # Define the probability of an edge to be proportional to the L1 norm of the impulse
    p_A = np.abs(W_imp)/W_imp_max
    
    # Compute the true and false positive rates
    fpr, tpr, thresholds = roc_curve(data['net']['graph']['A'].ravel(), p_A.ravel())

    return (tpr, fpr)

def compute_roc_from_sparse_glm_smpls(data, x_smpls):
    """ Compute the ROC curve given samples from a sparse weighted model
        We can estimate p(A_{k,k'} | data) from our samples
    """
    # Compute the inferred connectivity matrix
    p_A = np.zeros_like(x_smpls[0]['net']['graph']['A'])
    for smpl in x_smpls:
        p_A += smpl['net']['graph']['A']
    p_A /= float(len(x_smpls))
    
    # Compute the true and false positive rates
    fpr, tpr, thresholds = roc_curve(data['net']['graph']['A'].ravel(), p_A.ravel())
    
    return (tpr, fpr)
    
def postprocess(popn, x_inf, popn_true, x_true, options):
    """ Compute an ROC curve from a set of inferred samples and the true state
    """
    true_state = popn_true.eval_state(x_true)

    # Make sure we have a list of x's
    if not isinstance(x_inf, list):
        x_inf = [x_inf]

    inf_state = [popn.eval_state(x) for x in x_inf]


    # Check if the inference model is a standard GLM or a network GLM
    if
    # Now compute the true and false positive rates for MAP
    (map_tpr, map_fpr) = compute_roc_from_std_glm(true_state, map_state)
    map_tprs.append(map_tpr)
    map_fprs.append(map_fpr)

    print "Loading mcmc estimate"
    x_mcmc = None
    with open(os.path.join(data_dir, d, 'mcmc', 'results.pkl'), 'r') as f:
        x_mcmc = cPickle.load(f)


    model_mcmc = make_model('sparse_weighted_model', N=data['N'])
    popn_mcmc = Population(model_mcmc)
    popn_mcmc.set_data(data)

    # Evaluate the state
    mcmc_state = []
    for x in x_mcmc:
        mcmc_state.append(popn_mcmc.eval_state(x))


    # Now compute the true and false positive rates for MCMC
    # For MCMC results, only consider the tail of the samples
    N_samples = len(mcmc_state)
    sample_frac = 0.2
    start_smpl = int(np.floor(N_samples - sample_frac*N_samples))
    (mcmc_tpr, mcmc_fpr) = compute_roc_from_sparse_glm_smpls(true_state, mcmc_state[start_smpl:])
    mcmc_tprs.append(mcmc_tpr)
    mcmc_fprs.append(mcmc_fpr)


    # Pickle the roc results
    with open(PKL_FNAME, 'w') as f:
        # TODO Dump the MCMC results too
        cPickle.dump({'map_tprs' : map_tprs,
                      'map_fprs' : map_fprs},
                     f,
                     protocol=-1)

    # Plot the actual ROC curve
    # Subsample to get about 10 errorbars
    subsample = N*N//10
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_roc_curve(map_tprs, map_fprs, ax=ax, color='b', subsample=subsample)
    # plot_roc_curve(mcmc_tprs, mcmc_fprs, ax=ax, color='r', subsample=subsample)
    fname = os.path.join(PLOTDIR,'roc_N%dT%d.pdf' % (N,T))
    print "Saving ROC to %s" % fname
    f.savefig(fname)
    plt.close(f)

if __name__ == '__main__':
    (options, _) = parse_cmd_line_args()
    parse_results(DATADIR, options)
