"""
Adaptive rejection sampling
"""
import numpy as np
from scipy.misc import logsumexp

def adaptive_rejection_sample(func, xs, v_xs, domain, debug=False, check_inputs=True):
    #   Modified from Probabilistic Machine Learning Toolkit by Scott Linderman.
    #   See below for original author info.
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

    if check_inputs:
        _check_concavity(xs, v_xs)

        # Check gradients at left and right boundary of domain
        numDerivStep = 1e-3
        S = np.array([a, a+numDerivStep, b-numDerivStep, b])

        if domain[0] == -np.Inf:
            # ensure the derivative there is positive
            f = func(S[0:2])
            if (f[1]-f[0])<=0:
                raise Exception('derivative at a must be positive, since the domain is unbounded to the left')


        if domain[1]== np.Inf:
            # ensure the derivative there is negative
            f = func(S[2:])
            if (f[1]-f[0]) >= 0:
                raise Exception('derivative at b must be negative, since the domain is unbounded to the right')

    # initialize a mesh on which to create upper & lower hulls
    if len(xs) < 5:
        x_prop = a + (b-a)*np.random.rand(5-len(xs))
        v_prop = func(x_prop)
        xs = np.concatenate((xs, x_prop))
        v_xs = np.concatenate((v_xs, v_prop))
        perm = np.argsort(xs)
        xs = xs[perm]
        v_xs = v_xs[perm]

    # Compute the hull at xs
    lowerHull, upperHull = _ars_compute_hulls(xs, v_xs, domain)

    rejects = 0
    while True:
        if debug:
            _ars_plot(upperHull, lowerHull, domain, xs, v_xs, func)

        # sample x from Hull
        x = _ars_sample_upper_hull(upperHull)
        # Evaluate upper and lower hull at x
        lhVal, uhVal = _ars_eval_hulls(x, lowerHull, upperHull)

        # Sample under the upper hull at x and see if we accept or reject
        u = np.log(np.random.rand())
        # Three cases for acception/rejection
        if u <= lhVal - uhVal:
            # accept, u is below lower bound
            if debug:
                print "Sample found after %d rejects" % rejects
            return x

        # Otherwise we must compute the actual function
        vx = func(x)
        if u <= vx - uhVal:
            # accept, u is between lower bound and f
            if debug:
                print "Sample found after %d rejects" % rejects
            return x

        # If we made it this far, we rejected.
        # Now we have another evaluation that we can add to our hull though.
        xs = np.concatenate((xs, [x]))
        v_xs = np.concatenate((v_xs, [vx]))
        perm = np.argsort(xs)
        xs = xs[perm]
        v_xs = v_xs[perm]

        # Recompute the hulls
        lowerHull, upperHull = _ars_compute_hulls(xs, v_xs, domain)
    
        if debug:
            print 'reject %d' % rejects
            
        rejects += 1

def _check_concavity(xs, v_xs):
    """ 
    Check whether a set of points is concave according to second derivative test
    """
    g = np.diff(v_xs)/np.diff(xs)
    g2 = np.diff(g)/np.diff(xs[:-1])
    if not np.all(g2<=0):
        raise Exception("It looks like the function to be sampled is not log concave!")

class Hull:
    def __init__(self):
        self.m = None
        self.b = None
        self.left = None
        self.right = None
        self.pr = None
        self.lpr = None

def _signed_lse(m, b, a1, a0):
    """
    Compute log[ e^{b}/m * (e^{m*a1} - e^{m*a2}) ]
    """
    # Make sure that the term inside the log is postiive.
    # If m>0: m*a1 > m*a2, aka a1 > a2
    # if m<0: m*a1 < m*a2, aka a1 > a2
    sgn = np.sign(m)
    assert a1 > a0, "a1 must be greater than a2!"

    if np.allclose(m,0.0):
        return b + np.log(a1-a0)

    # Now we can work with absolute value of m and e^{a1} - e^{a2}
    am = np.maximum(m*a1, m*a0)
    se = np.exp(m*a1-am)-np.exp(m*a0-am)
    lse = b - np.log(m*sgn) + am + np.log(se*sgn)

    if not np.isfinite(lse):
        print "LSE is not finite"
        print "lse: %f" % lse
        print "m: %f" % m
        print "b: %f" % b
        print "a1: %f" % a1
        print "a2: %f" % a0
        print "am: %f" % am
        print "se: %f" % se
    
    return lse

def _ars_compute_hulls(S, fS, domain):
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
        # pro = np.exp(b)/m * ( np.exp(m*S[0]) - 0 ) # integrating in from -infinity
        lnpr = b - np.log(m) + m*S[0]
        h = Hull()
        h.m = m
        h.b = b
        h.lnpr = lnpr
        h.left = -np.Inf
        h.right = S[0]
        upperHull.append(h)

    # second line
    m = (fS[2]-fS[1])/(S[2]-S[1])
    b = fS[1] - m*S[1]
    # pro = np.exp(b)/m * ( np.exp(m*S[1]) - np.exp(m*S[0]) )
    lnpr = _signed_lse(m, b, S[1], S[0])
    # Append upper hull for second line
    h = Hull()
    h.m = m
    h.b = b
    h.lnpr = lnpr
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

        # compute the two lines' intersection
        # Make sure it's in the valid range
        ix = (b1-b2)/(m2-m1)
        assert ix >= S[li] and ix <= S[li+1]
    
        # pro = np.exp(b1)/m1 * ( np.exp(m1*ix) - np.exp(m1*S[li]) )
        lnpr1 = _signed_lse(m1, b1, ix, S[li])
        h = Hull()
        h.m = m1
        h.b = b1
        h.lnpr = lnpr1
        h.left = S[li]
        h.right = ix
        upperHull.append(h)
    
        # pro = np.exp(b2)/m2 * ( np.exp(m2*S[li+1]) - np.exp(m2*ix) )
        lnpr2 = _signed_lse(m2, b2, S[li+1], ix)
        h = Hull()
        h.m = m2
        h.b = b2
        h.lnpr = lnpr2
        h.left = ix
        h.right = S[li+1]
        upperHull.append(h)

    # second to last line (m<0)
    m = (fS[-2]-fS[-3])/(S[-2]-S[-3])
    b = fS[-2] - m*S[-2]
    # pro = np.exp(b)/m * ( np.exp(m*S[-1]) - np.exp(m*S[-2]) )
    lnpr = _signed_lse(m, b, S[-1], S[-2])
    h = Hull()
    h.m = m
    h.b = b
    h.lnpr = lnpr
    h.left = S[-2]
    h.right = S[-1]
    upperHull.append(h)

    if np.isinf(domain[1]):
        # last line (to infinity)
        m = (fS[-1]-fS[-2])/(S[-1]-S[-2])
        b = fS[-1] - m*S[-1]
        # pro = np.exp(b)/m * ( 0 - np.exp(m*S[-1]) )
        lnpr = b - np.log(np.abs(m)) + m*S[-1]
        h = Hull()
        h.m = m
        h.b = b
        h.lnpr = lnpr
        h.left = S[-1]
        h.right = np.Inf
        upperHull.append(h)


    lnprs = np.array([h.lnpr for h in upperHull])
    lnZ = logsumexp(lnprs)
    prs = np.exp(lnprs - lnZ)
    for (i,h) in enumerate(upperHull):
        h.pr = prs[i]

    if not np.all(np.isfinite(prs)):
        print "ARS prs contains Inf or NaN"
        print lnprs
        print lnZ
        print prs
        raise Exception("ARS prs contains Inf or NaN")

    return lowerHull, upperHull

def _ars_sample_upper_hull(upperHull):
    prs = np.array([h.pr for h in upperHull])
    if not np.all(np.isfinite(prs)):
        print prs
        raise Exception("ARS prs contains Inf or NaN")

    cdf = np.cumsum(prs)
    if not np.all(np.isfinite(cdf)):
        print cdf
        raise Exception("ARS cumsum Inf or NaN")
    
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

    # x = np.log( U*(np.exp(m*right) - np.exp(m*left)) + np.exp(m*left) ) / m
    # If we sampled an interior edge then we can do the sampling in a more 
    # stable manner.
    if np.isfinite(left) and np.isfinite(right):
        # x = np.log( U*(np.exp(m*right) - np.exp(m*left)) + np.exp(m*left) ) / m
        # x = left + np.log(U*(np.exp(m*(right-left))-1) + 1) / m
        # x = left + np.log(np.exp(np.log(U) + m*(right-left))-np.exp(np.log(U)) + np.exp(0))/ m
        # lnu = np.log(U)
        # v = lnu + m*(right-left)
        # x = left + np.log(np.exp(v) - np.exp(lnu) + np.exp(0))/ m
        # vmax = np.amax([v, lnu, 0])
        # x = left + np.log(np.exp(vmax)*(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax)))/m
        # x = left + vmax + np.log(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax))/m

        # First check if the slope is zero, because then our integrals are undefined
        if np.allclose(m, 0.0):
            # If so, the pdf is flat in this interval so sample uniformly
            x = left + np.random.rand() * (right-left)
        else:
            # Otherwise, inverse sample an exponential distribution
            lnu = np.log(U)
            v = lnu + m*(right-left)
            vmax = np.amax([v, lnu, 0])
            x = left + vmax/m + np.log(np.exp(v-vmax) - np.exp(lnu-vmax) + np.exp(-vmax))/m

    # If the left edge is -Inf, we need to be smarter
    elif np.isinf(left):
        assert m > 0
        x = right + np.log(U) / m
    # Same for the right edge being +Inf
    else:
        # x = np.log( U*(- np.exp(m*left)) + np.exp(m*left) ) / m
        # x = np.log( np.exp(m*left)*(1-U)) / m
        assert m < 0
        x = left + np.log(1-U) / m

    if np.isinf(x) or np.isnan(x):
        # import pdb; pdb.set_trace()
        
        raise Exception('sampled an infinite or NaN x. Left=%.3f. Right=%.3f. m=%.3f. b=%.3f. cdf=%s. U=%.3f' % (left, right, m, b, str(cdf), U))

    return x

def _ars_eval_hulls( x, lowerHull, upperHull ):

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

def _ars_plot(upperHull, lowerHull, domain, S, fS, func):
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

    f = lambda x: -0.5*x**2
    x_init = np.array([-2, -1.999, -0.995, 0, 0.995, 1.999, 2.0])
    v_init = f(x_init)

    N_samples = 20000
    smpls = np.zeros(N_samples)
    for s in np.arange(N_samples):
        smpls[s] =  adaptive_rejection_sample(f, x_init, v_init, (-np.Inf, np.Inf), debug=False)

    import matplotlib.pyplot as plt
    f = plt.figure()
    _, bins, _ = plt.hist(smpls, 20, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p(bincenters), 'r--', linewidth=1)
    plt.show()
    f.savefig('ars_test.pdf')
    plt.close(f)

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
