# cython: boundscheck=False
# cython: wraparound=False

"""
Cythonized nonlinearities for speed.
"""
from libc.math cimport exp
from cython.parallel import parallel, prange

# Inline the nonlinearity calculations
cdef inline double nlin(double xx) nogil: return exp(xx) if xx < 0 else 1+xx
cdef inline double grad_nlin(double xx) nogil: return exp(xx) if xx < 0 else 1.0


cpdef explinear(double[:,::1] x, double[:,::1] lam):
    """
    Compute the nonlinearity
    """
    cdef int N = x.shape[0]
    cdef int n

    with nogil:
        for n in prange(N):
            lam[n,0] = nlin(x[n,0])

cpdef grad_explinear(double[:,::1] x, double[:,::1] dx):
    """
    Compute the nonlinearity
    """
    cdef int N = x.shape[0]
    cdef int n

    with nogil:
        for n in prange(N):
            dx[n,0] = grad_nlin(x[n,0])