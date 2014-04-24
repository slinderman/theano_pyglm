"""
Adaptive rejection sampling
"""
import numpy as np

def adaptive_rejection_sample(f, xs, v_xs, _rejects=0):
    """
    Recurisve implementation of derivative-free adaptive rejection sampling
    for 1-dimensional log-concave distributions.
    Parameters:
    f:  function that computes log probability
    xs: points where the log prob has previously been computed. Must contain
        at least two points, the lower bound for x and the upper bound for x.
    vs: log prob at xs

    Private params:
    _rejects: Depth of recursion
    """
    N = len(xs)
    assert N >= 2
    assert xs.ndim == 1

    # Make sure xs is sorted
    if not np.all(xs[1:] > xs[:-1]):
        perm = np.argsort(xs)
        xs = xs[perm]
        v_xs = v_xs[perm]

    # If N < 3, evaluate at the midpoint
    if N < 3:
        x_mid = (xs[1]+xs[0])/2.0
        v_mid = f(x_mid)
        return adaptive_rejection_sample(f,
                                         np.concatenate((xs, x_mid)),
                                         np.concatenate((v_xs, v_mid)))

    # # Define the envelope. For each interval (W_nns[1:2]), ..., (W_nns[N-3:N-2])
    # # compute the chords joining log_L[1:2], ..., log_L[N-3:N-2]
    # i = np.arange(1, N - 2)
    # # dxs[j] = W_nns[j+1]-W_nns[j] for j = 0...N-2
    # dxs = xs[1:] - xs[:-1]
    # # dfs[j] = log_f[j+1]-log_f[j] for j = 0...N-2
    # dfs = v_xs[1:] - v_xs[:-1]
    # slopes = dfs / dxs
    #
    # # We extrapolate tangent curves from W_nns[0:1],...,W_nns[N-4:N-3] on the left
    # # and W_nns[3:2],...,W_nns[N-1:N-2] on the right to upper bound the interval.
    # # These tangent curves intersect at points z_1...z_{N-2}
    # # Doing some algebra yields
    # zs = (v_xs[i + 1] - xs[i + 1] / dxs[i + 1] * dfs[i + 1] - v_xs[i - 1] + xs[i - 1] / dxs[i - 1] * dfs[i - 1])
    # zs /= (slopes[i - 1] - slopes[i + 1])
    #
    # # The envelope is now piecewise linear with knots at W_nns and zs
    # # Evaluate the envelope at the zs
    # v_zs = v_xs[i] + slopes * (zs - xs[i])
    #
    # knots = np.concatenate((xs, zs))
    # v_knots = np.concatenate((v_xs, v_zs))
    # perm = np.argsort(knots)
    # knots = knots[perm]
    # v_knots = v_knots[perm]
    #
    # # Offsets are the left knots
    # bs = knots[:-1]
    #
    # # Compute the slopes at all knots but the last
    # dknots = knots[1:] - knots[:-1]
    # dv_knots = v_knots[1:] - v_knots[:-1]
    # ms = dv_knots / dv_knots
    knots, bs, ms = _compute_envelope(xs, v_xs)
    dknots = knots[1:] - knots[:-1]

    # Now compute the area under each exponentiated line segment
    prs = np.exp(bs)/ms *(np.exp(ms*dknots) - 1.0)
    Z = np.sum(prs)
    prs /= Z

    # Compute the CDF
    # TODO: Handle boundaries
    cdf = np.cumsum(prs)

    # Sample a line segment
    u1 = np.random.rand()
    seg = len(knots)-1
    for j in range(len(knots)):
        if u1 < cdf[j]:
            seg = j
            break


    # Inverse sample the cdf
    u2 = np.random.rand()
    x_prop = np.log(u2*(np.exp(ms[seg]*knots[seg+1]) - np.exp(ms[seg]*knots[seg])) +
                    np.exp(ms[seg]*knots[seg]) ) \
             / ms[seg]

    # Sample under the envelope and accept or reject
    px_prop = np.exp(bs[seg] + ms[seg]*(x_prop-knots[seg])) / Z
    fx_prop = f(x_prop)
    u3 = np.random.rand() * px_prop
    if u3 > fx_prop:
        # Reject: add x_prop and fx_prop to our list of points
        return adaptive_rejection_sample(f,
                                         np.concatenate((xs, x_prop)),
                                         np.concatenate((v_xs, fx_prop)),
                                         _rejects=_rejects+1)
    else:
        # DEBUG
        print "Accepted after %d rejects." % _rejects
        return x_prop

def _compute_envelope(xs, v_xs):
    """
    Compute an envelope that upper bounds the log-concave function
    """
    N = len(xs)
    # Define the envelope. For each interval (W_nns[1:2]), ..., (W_nns[N-3:N-2])
    # compute the chords joining log_L[1:2], ..., log_L[N-3:N-2]
    i = np.arange(1, N - 2)
    # dxs[j] = W_nns[j+1]-W_nns[j] for j = 0...N-2
    dxs = xs[1:] - xs[:-1]
    # dfs[j] = log_f[j+1]-log_f[j] for j = 0...N-2
    dfs = v_xs[1:] - v_xs[:-1]
    slopes = dfs / dxs

    # We extrapolate tangent curves from W_nns[0:1],...,W_nns[N-4:N-3] on the left
    # and W_nns[3:2],...,W_nns[N-1:N-2] on the right to upper bound the interval.
    # These tangent curves intersect at points z_1...z_{N-2}
    # Doing some algebra yields
    zs = (v_xs[i + 1] - xs[i + 1] / dxs[i + 1] * dfs[i + 1] - v_xs[i - 1] + xs[i - 1] / dxs[i - 1] * dfs[i - 1])
    zs /= (slopes[i - 1] - slopes[i + 1])

    # The envelope is now piecewise linear with knots at W_nns and zs
    # Evaluate the envelope at the zs
    v_zs = v_xs[i] + slopes * (zs - xs[i])

    knots = np.concatenate((xs, zs))
    v_knots = np.concatenate((v_xs, v_zs))
    perm = np.argsort(knots)
    knots = knots[perm]
    v_knots = v_knots[perm]

    # Offsets are the left knots
    bs = knots[:-1]

    # Compute the slopes at all knots but the last
    dknots = knots[1:] - knots[:-1]
    dv_knots = v_knots[1:] - v_knots[:-1]
    ms = dv_knots / dv_knots

    return knots, bs, ms

def test_ars():
    """
    Test ARS on a Gaussian distribution
    """
    f = lambda x: -0.5 * x**2
    x_init = np.array([-1, 0, 1])
    v_init = f(x_init)

    knots, bs, ms = _compute_envelope(x_init, v_init)

    import matplotlib.pyplot as plt
    plt.figure()
    xx = np.linspace(-3,3)
    plt.plot(xx, f(xx), '-b')
    for i in