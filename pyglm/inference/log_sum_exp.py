import numpy as np


def log_sum_exp_sample(lnp):
    """
    Sample uniformly from a vector of unnormalized log probs using
    the log-sum-exp trick
    """
    assert np.ndim(lnp) == 1, "ERROR: logSumExpSample requires a 1-d vector"
    lnp = np.ravel(lnp)
    N = np.size(lnp)

    # Use logsumexp trick to calculate ln(p1 + p2 + ... + pR) from ln(pi)'s
    max_lnp = np.max(lnp)
    denom = np.log(np.sum(np.exp(lnp-max_lnp))) + max_lnp
    p_safe = np.exp(lnp - denom)

    # Normalize the discrete distribution over blocks
    sum_p_safe = np.sum(p_safe)
    if sum_p_safe == 0 or not np.isfinite(sum_p_safe):
        print "Total probability for logSumExp is not valid! %f" % sum_p_safe
        raise Exception("Invalid input. Probability infinite everywhere.")

    # Randomly sample a block
    choice = -1
    u = np.random.rand()
    acc = 0.0
    for n in np.arange(N):
        acc += p_safe[n]
        if u <= acc:
            choice = n
            break

    if choice == -1:
        raise Exception("Invalid choice in logSumExp!")

    return choice
