"""
Plotting for ROC curves and link prediction tests
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(tprs, fprs, color='k', ax=None):
    """ Plot an ROC curve for the given true and false positive rates.
        If multiple rates are given, e.g. corresponding to multiple
        networks inferred using the same procedure, compute error bars
        (both horizontal and vertical) for the ROC curve.

        Plot in specified color, default black.

        Plot on the specified axes, or create a new axis necessary.
    """
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    if not isinstance(tprs, list):
        tprs = [tprs]
    if not isinstance(fprs, list):
        fprs = [fprs]

    # Make sure all tprs and fprs are the same length
    N = tprs[0].size
    for tpr in tprs:
        if not tpr.shape == (N,):
            raise Exception("All TPRs must be vectors of length %d." % N)
    for fpr in fprs:
        if not fpr.shape == (N,):
            raise Exception("All FPRs must be vectors of length %d." % N)

    # Stack tprs and fprs to make matrices
    tprs = np.concatenate(tprs, axis=1)
    fprs = np.concatenate(fprs, axis=1)

    # Compute error bars (for both tpr and fpr)
    mean_tprs = np.mean(tprs, axis=1)
    std_tprs = np.std(tprs, axis=1)

    mean_fprs = np.mean(fprs, axis=1)
    std_fprs = np.std(fprs, axis=1)

    # Plot the error bars
    plt.errorbar(mean_fprs, mean_tprs,
                 xerr=std_fprs, yerr=std_tprs,
                 ecolor=color, axes=ax)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim((-0.05,1))
    plt.ylim((-0.05,1))

    return ax