"""
Slice sampling implementation from Ryan Adams
http://machinelearningmisc.blogspot.com/
10.1.2013
"""
import numpy
import numpy.random

import logging
log = logging.getLogger("global_log")

def slicesample(xx, llh_func, last_llh=None, sigma=1, step_out=True):
    dims = xx.shape[0]
    perm = range(dims)
    numpy.random.shuffle(perm)
         
    if (type(sigma).__name__ == 'int') or (type(sigma).__name__ == 'float'):
        sigma = numpy.tile(sigma, dims)
    elif (type(sigma).__name__ == 'tuple') or (type(sigma).__name__ == 'list'):
        sigma = numpy.array(sigma)
 
    if last_llh is None:
        last_llh = llh_func(xx)
 
    for d in perm:
        llh0   = last_llh + numpy.log(numpy.random.rand())
        rr     = numpy.random.rand(1)
        x_l    = xx.copy()
        x_l[d] = x_l[d] - rr*sigma[d]
        x_r    = xx.copy()
        x_r[d] = x_r[d] + (1-rr)*sigma[d]
         
        if step_out:
            llh_l = llh_func(x_l)
            while llh_l > llh0:
                x_l[d] = x_l[d] - sigma[d]
                llh_l  = llh_func(x_l)
            llh_r = llh_func(x_r)
            while llh_r > llh0:
                x_r[d] = x_r[d] + sigma[d]
                llh_r  = llh_func(x_r)
 
        x_cur = xx.copy()
        while True:
            xd       = numpy.random.rand(1)*(x_r[d] - x_l[d]) + x_l[d]
            x_cur[d] = xd
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
    return xx, last_llh