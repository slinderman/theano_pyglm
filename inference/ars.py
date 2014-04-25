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
    if N < 5:
        x_props = xs[0] + (xs[-1]-xs[0])*np.random.rand(5-N)
        v_props = np.zeros_like(x_props)
        for j in range(N-5):
            v_props[j] = f(x_props[j])

        # x_mid = (xs[1]+xs[0])/2.0
        # v_mid = f(x_mid)
        return adaptive_rejection_sample(f,
                                         np.concatenate((xs, x_props)),
                                         np.concatenate((v_xs, v_props)))

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
    knots, v_knots, bs, ms = _compute_envelope(xs, v_xs)
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
        print "Accepted after #d rejects." # _rejects
        return x_prop

def _compute_envelope(xs, v_xs):
    """
    Compute an envelope that upper bounds the log-concave function
    """
    N = len(xs)
    # Define the envelope. For each interval (W_nns[1:2]), ..., (W_nns[N-3:N-2])
    # compute the chords joining log_L[1:2], ..., log_L[N-3:N-2]
    i = np.arange(1, N -1)
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
    v_zs = v_xs[i] + slopes[i] * (zs - xs[i])

    knots = np.concatenate((xs, zs))
    v_knots = np.concatenate((v_xs, v_zs))
    perm = np.argsort(knots)
    knots = knots[perm]
    v_knots = v_knots[perm]

    # Offsets are the values left knots
    bs = v_knots[:-1]

    # Compute the slopes at all knots but the last
    dknots = knots[1:] - knots[:-1]
    dv_knots = v_knots[1:] - v_knots[:-1]
    ms = dv_knots / dv_knots

    return knots, v_knots, bs, ms

def ars_pmtk(func, xs, v_xs, domain, debug=True):
    #   Copied from Probabilistic Machine Learning Toolkit, see below.
    #    ARS - Adaptive Rejection Sampling
    #          sample perfectly & efficiently from a univariate log-concave
    #          function
    # 
    #    func        a function handle to the log of a log-concave function.
    #                   evaluated as: func(x, varargin{:}), where x could  be a
    #                   vector
    # 
    #    domain      the domain of func. may be unbounded.
    #                   ex: [1,inf], [-inf,inf], [-5, 20]
    # 
    #    a,b         two points on the domain of func, a<b. if domain is bounded
    #                   then use a=domain[0], b=domain[1]. if domain is
    #                   unbounded on the left, the derivative of func for x=<a
    #                   must be positive. if domain is unbounded on the right,
    #                   the derivative for x>=b must be negative.
    # 
    #                   ex: domain = [1,inf], a=1, b=20 (ensuring that
    #                   func'(x>=b)<0
    # 
    #    nSamples    number of samples to draw
    # 
    #    varargin    extra arguments passed directly to func
    # 
    # PMTKauthor   Daniel Eaton
    #  danieljameseaton@gmail.com
    # PMTKdate 2006

    if domain[0] >= domain[1]:
        raise Exception('invalid domain')

    a = xs[0]
    b = xs[-1]
    if a>=b or np.isinf(a) or np.isinf(b) or a<domain[0] or b>domain[1]:
        raise Exception('invalid a & b')
    
    # Check gradients at left and right boundary of domain
    # numDerivStep = 1e-3
    # S = np.array([a, a+numDerivStep, b-numDerivStep, b])
    #
    # if domain[0] == -np.Inf:
    #     # ensure the derivative there is positive
    #     f = func(S[0:2])
    #     if (f[1]-f[0])<=0:
    #         raise Exception('derivative at a must be positive, since the domain is unbounded to the left')
    #
    #
    # if domain[1]== np.Inf:
    #     # ensure the derivative there is negative
    #     f = func(S[2:])
    #     if (f[1]-f[0]) >= 0:
    #         raise Exception('derivative at b must be negative, since the domain is unbounded to the right')
    
    # initialize a mesh on which to create upper & lower hulls
    if len(xs) < 5:
        x_prop = a + (b-a)*np.random.rand(5-len(xs))
        v_prop = func(x_prop)
        xs = np.concatenate((xs, x_prop))
        v_xs = np.concatenate((v_xs, v_prop))
        perm = np.argsort(xs)
        xs = xs[perm]
        v_xs = v_xs[perm]

    lowerHull, upperHull = arsComputeHulls(xs, v_xs, domain)

    rejects = 0
    while True:
        if debug:
            arsPlot(upperHull, lowerHull, domain, xs, v_xs, func)

        import pdb; pdb.set_trace()
        # sample x from Hull
        x = arsSampleUpperHull( upperHull )

        lhVal, uhVal = arsEvalHulls( x, lowerHull, upperHull )
    
        U = np.random.rand()
    
        meshChanged = False # flag to indicate if a new point has been added to the mesh
    
        # three cases for acception/rejection
        if U<=lhVal/uhVal:
            # accept, u is below lower bound
            if debug:
                print "Sample found after %d rejects" % rejects
            return x

        # Otherwise we must compute the actual function
        vx = func(x)
        if U<=vx/uhVal:
            # accept, u is between lower bound and f
            if debug:
                print "Sample found after %d rejects" % rejects
            return x
        else:
            # reject, u is between f and upper bound
            meshChanged = True
    
        if meshChanged:
            xs = np.concatenate((xs, [x]))
            v_xs = np.concatenate((v_xs, [vx]))
            perm = np.argsort(xs)
            xs = xs[perm]
            v_xs = v_xs[perm]

            # Recompute the hulls
            lowerHull, upperHull = arsComputeHulls(xs, v_xs, domain)
    
        if debug:
            print 'reject %d' % rejects
            
        rejects += 1


class Hull:
    def __init__(self):
        self.m = None
        self.b = None
        self.left = None
        self.right = None
        self.pr = None

def arsComputeHulls(S, fS, domain):
    # compute lower piecewise-linear hull
    # if the domain of func is unbounded to the left or right, then the lower
    # hull takes on -inf to the left or right of the end points of S

    lowerHull = []
    for li in np.arange(len(S)-1):
        h = Hull()
        h.m = (fS[li+1]-fS[li])/(S[li+1]-S[li])
        h.b = fS[li] - h.m*S[li]
        h.left = S[li]
        h.right = S[li+1]
        lowerHull.append(h)
    
    # compute upper piecewise-linear hull
    upperHull = []
    
    if np.isinf(domain[0]):
        # first line (from -infinity)
        m = (fS[1]-fS[0])/(S[1]-S[0])
        b = fS[0] - m*S[0]
        pr = np.exp(b)/m * ( np.exp(m*S[0]) - 0 ) # integrating in from -infinity
        h = Hull()
        h.m = m
        h.b = b
        h.pr = pr
        h.left = -np.Inf
        h.right = S[0]
        upperHull.append(h)

    # second line
    m = (fS[2]-fS[1])/(S[2]-S[1])
    b = fS[1] - m*S[1]
    pr = np.exp(b)/m * ( np.exp(m*S[1]) - np.exp(m*S[0]) )
    # Append upper hull for second line
    h = Hull()
    h.m = m
    h.b = b
    h.pr = pr
    h.left = S[0]
    h.right = S[1]
    upperHull.append(h)
    
    # interior lines
    # there are two lines between each abscissa
    for li in np.arange(1,len(S)-2):
    
        m1 = (fS[li]-fS[li-1])/(S[li]-S[li-1])
        b1 = fS[li] - m1*S[li]
    
        m2 = (fS[li+2]-fS[li+1])/(S[li+2]-S[li+1])
        b2 = fS[li+1] - m2*S[li+1]
    
        ix = (b1-b2)/(m2-m1) # compute the two lines' intersection
    
        pr1 = np.exp(b1)/m1 * ( np.exp(m1*ix) - np.exp(m1*S[li]) )
        h = Hull()
        h.m = m1
        h.b = b1
        h.pr = pr1
        h.left = S[li]
        h.right = ix
        upperHull.append(h)
    
        pr2 = np.exp(b2)/m2 * ( np.exp(m2*S[li+1]) - np.exp(m2*ix) )
        h = Hull()
        h.m = m2
        h.b = b2
        h.pr = pr2
        h.left = ix
        h.right = S[li+1]
        upperHull.append(h)

    # second last line
    m = (fS[-2]-fS[-3])/(S[-2]-S[-3])
    b = fS[-2] - m*S[-2]
    pr = np.exp(b)/m * ( np.exp(m*S[-1]) - np.exp(m*S[-2]) )
    h = Hull()
    h.m = m
    h.b = b
    h.pr = pr
    h.left = S[-2]
    h.right = S[-1]
    upperHull.append(h)

    if np.isinf(domain[1]):
        # last line (to infinity)
        m = (fS[-1]-fS[-2])/(S[-1]-S[-2])
        b = fS[-1] - m*S[-1]
        pr = np.exp(b)/m * ( 0 - np.exp(m*S[-1]) )
        h = Hull()
        h.m = m
        h.b = b
        h.pr = pr
        h.left = S[-1]
        h.right = np.Inf
        upperHull.append(h)


    # TODO: Make these reductions and maps
    Z = 0
    for h in upperHull:
        Z += h.pr

    for h in upperHull:
        h.pr /= Z

    return lowerHull, upperHull

def arsSampleUpperHull(upperHull):
    cdf = np.cumsum(np.array([h.pr for h in upperHull]))

    # randomly choose a line segment
    U = np.random.rand()
    for li in np.arange(len(upperHull)):
        if U < cdf[li]:
            break

    # sample along that line segment
    U = np.random.rand()

    m = upperHull[li].m
    b = upperHull[li].b
    left = upperHull[li].left
    right = upperHull[li].right

    x = np.log( U*(np.exp(m*right) - np.exp(m*left)) + np.exp(m*left) ) / m

    if np.isinf(x) or np.isnan(x):
        raise Exception('sampled an infinite or NaN x')

    return x

def arsEvalHulls( x, lowerHull, upperHull ):

    # lower bound
    lhVal = None
    if x< np.amin(np.array([h.left for h in lowerHull])):
        lhVal = -np.Inf
    elif x>np.amax(np.array([h.right for h in lowerHull])):
        lhVal = -np.Inf
    else:
        for h in lowerHull:
            left = h.left
            right = h.right

            if x>=left and x<=right:
                lhVal = h.m*x + h.b
                break

    # upper bound
    uhVal = None
    for h in upperHull:
        left = h.left
        right = h.right

        if x>=left and x<=right:
            uhVal = h.m*x + h.b
            break

    return lhVal, uhVal

def arsPlot(upperHull, lowerHull, domain, S, fS, func):
    import matplotlib.pyplot as plt

    Swidth = S[-1]-S[0]
    plotStep = Swidth/1000.0
    ext = 0.15*Swidth; # plot this much before a and past b, if the domain is infinite

    left = S[0]; right = S[-1]
    if np.isinf(domain[0]):
        left -= ext
    if np.isinf(domain[1]):
        right += ext

    x = np.arange(left, right, plotStep)
    fx = func(x)


    plt.plot(x,fx, 'k-')
    plt.plot(S, fS, 'ko')
    plt.title('ARS')

    # plot lower hull
    for h in lowerHull[:-1]:
        m = h.m
        b = h.b

        x = np.arange(h.left, h.right, plotStep)
        plt.plot( x, m*x+b, 'b-' )

    # plot upper bound

    # first line (from -infinity)
    if np.isinf(domain[0]):
        x = np.arange(upperHull[0].right-ext, upperHull[0].right, plotStep)
        m = upperHull[0].m
        b = upperHull[0].b
        plt.plot( x, x*m+b, 'r-')

    # middle lines
    for li in np.arange(1, len(upperHull)-1):

        x = np.arange(upperHull[li].left, upperHull[li].right, plotStep)
        m = upperHull[li].m
        b = upperHull[li].b
        plt.plot( x, x*m+b, 'r-')

    # last line (to infinity)
    if np.isinf(domain[1]):
        x = np.arange(upperHull[-1].left, (upperHull[-1].left+ext), plotStep)
        m = upperHull[-1].m
        b = upperHull[-1].b
        plt.plot( x, x*m+b, 'r-')

    plt.show()

def test_ars():
    """
    Test ARS on a Gaussian distribution
    """
    from scipy.stats import norm
    mu = 0
    sig = 1
    p = norm(mu, sig).pdf

    f = lambda x: -x**2
    x_init = np.array([-2, -1.999, -0.995, 0, 0.995, 1.999, 2.0])
    v_init = f(x_init)

    N_samples = 1000
    smpls = np.zeros(N_samples)
    for s in np.arange(N_samples):
        smpls[s] =  ars_pmtk(f, x_init, v_init, (-np.Inf, np.Inf), debug=False)

    import matplotlib.pyplot as plt
    plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()
    # knots, v_knots, bs, ms = _compute_envelope(x_init, v_init)
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # xx = np.linspace(-3,3)
    # plt.plot(xx, f(xx), '-b')
    # for i in range(len(bs)):
    #     plt.plot(knots[i:i+2], v_knots[i:i+2], '-r')
    # plt.show()


if __name__ == '__main__':
    test_ars()