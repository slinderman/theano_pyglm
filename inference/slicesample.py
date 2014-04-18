"""
Slice sampling implementation from Ryan Adams
http://machinelearningmisc.blogspot.com/
10.1.2013
"""
import numpy
import numpy.random

import logging
log = logging.getLogger("global_log")

def slicesample(xx, llh_func, last_llh=None, step=1, step_out=True, x_l=None, x_r=None):
    xx = numpy.atleast_1d(xx)
    dims = xx.shape[0]
    perm = range(dims)
    numpy.random.shuffle(perm)

    if isinstance(step, int) or isinstance(step, float) or \
        isinstance(step, numpy.int) or isinstance(step, numpy.float):
        step = numpy.tile(step, dims)
    elif isinstance(step, tuple) or isinstance(step, list):
        step = numpy.array(step)
 
    if last_llh is None:
        last_llh = llh_func(xx)
 
    for d in perm:
        llh0   = last_llh + numpy.log(numpy.random.rand())
        rr     = numpy.random.rand(1)
        if x_l is None:
            x_l    = xx.copy()
            x_l[d] = x_l[d] - rr*step[d]
        else:
            x_l = numpy.atleast_1d(x_l)
            assert x_l.shape == xx.shape
            assert numpy.all(x_l <= xx)
        if x_r is None:
            x_r    = xx.copy()
            x_r[d] = x_r[d] + (1-rr)*step[d]
        else:
            x_r = numpy.atleast_1d(x_r)
            assert x_r.shape == xx.shape
            assert numpy.all(x_r >= xx)
         
        if step_out:
            llh_l = llh_func(x_l)
            while llh_l > llh0:
                x_l[d] = x_l[d] - step[d]
                llh_l  = llh_func(x_l)
            llh_r = llh_func(x_r)
            while llh_r > llh0:
                x_r[d] = x_r[d] + step[d]
                llh_r  = llh_func(x_r)

        try:
            assert numpy.isfinite(llh0)
            assert numpy.isfinite(llh_l)
            assert numpy.isfinite(llh_r)
        except:
            import pdb; pdb.set_trace()

        x_cur = xx.copy()
        n_steps = 0
        while True:
            xd       = numpy.random.rand()*(x_r[d] - x_l[d]) + x_l[d]
            x_cur[d] = xd
            # print "x_l: %f \tx_curr: %f\tx_r: %f" % (x_l[d], x_cur[d], x_r[d])
            last_llh = llh_func(x_cur)
            if last_llh > llh0:
                xx[d] = xd
                break
            elif xd > xx[d]:
                x_r[d] = xd
            elif xd < xx[d]:
                x_l[d] = xd
            else:
                raise Exception("Slice sampler shrank too far.")
            n_steps += 1

            if n_steps > 20:
                import pdb; pdb.set_trace()
    return xx, last_llh