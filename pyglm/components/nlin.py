import numpy as np

import theano.tensor as T

import kayak as kyk

from pyglm.components.component import Component


def create_nlin_component(model):
    type = model['nonlinearity']['type'].lower()
    if type == 'exp':
        nlin = ExpNonlinearity(model)
    elif type == 'explinear':
        nlin = TheanoExpLinearNonlinearity(model)
    else:
        raise Exception("Unrecognized nonlinearity model: %s" % type)
    return nlin

class ExpNonlinearity(Component):
    """ Standard exponential nonlinearity.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self.nlin = T.exp
        self._log_p = T.constant(0.)
        self.f_nlin = np.exp

    @property
    def log_p(self):
        return self._log_p


class TheanoExpLinearNonlinearity(Component):
    """ Exponential nonlinearity (\lambda=e^x) for x<0,
        Linear (\lambda=1+x) for x>0.
        This is nice because it satisfies a Lipschitz bound of 1.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self._log_p = T.constant(0.)

    @property
    def log_p(self):
        return self._log_p

    def nlin(self, x):
        return T.log(1.0+T.exp(x))

    def f_nlin(self, x):
        return np.log(1.0 + np.exp(x))

class KayakSoftRectLinearNonlinearity(Component):
    """ Soft rectified linear = log(1+exp(x))
        This is nice because it satisfies a Lipschitz bound of 1.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self._log_p = kyk.Constant(0.)

    @property
    def log_p(self):
        return self._log_p

    def nlin(self, x):
        return kyk.ElemLog(1.0 + kyk.ElemExp(x))

    def f_nlin(self, x):
        return np.log(1.0 + np.exp(x))


class KayakExpNonlinearity(Component):
    """ Exponential nonlinearity (\lambda=e^x) for x<0,
        Linear (\lambda=1+x) for x>0.
        This is nice because it satisfies a Lipschitz bound of 1.
    """

    def __init__(self, model):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        self._log_p = kyk.Constant(0.)

    @property
    def log_p(self):
        return self._log_p

    def nlin(self, x):
        return kyk.ElemExp(x)

    def f_nlin(self, x):
        return np.exp(x)

from pyglm.components.nonlinearities import explinear, grad_explinear
class _ExpLinearDifferentiable(kyk.Differentiable):
    """
    Create a kayak node to evaluate this nonlinearity.
    We use Cython to speed up the computation
    """

    __slots__ = ['x']
    def __init__(self, x):
        super(_ExpLinearDifferentiable, self).__init__([x])
        self.x = x

    def _compute_value(self):
        lam = np.empty(self.x.shape)
        explinear(self.x.value, lam)
        return lam

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            dx = np.empty(self.x.shape)
            grad_explinear(self.x.value, dx)
            return d_out_d_self * dx
        else:
            raise Exception("Not a parent of me")

class KayakExpLinearNonlinearity(Component):
    """ Exponential nonlinearity (\lambda=e^x) for x<0,
        Linear (\lambda=1+x) for x>0.
        This is nice because it satisfies a Lipschitz bound of 1.
    """
    def __init__(self, model):
        self._log_p = kyk.Constant(0.)

    @property
    def log_p(self):
        return self._log_p

    def nlin(self, x):
        return _ExpLinearDifferentiable(x)

    def f_nlin(self, x):
        from nonlinearities import explinear
        xa = np.reshape(x, (np.size(x), 1)).copy(order='C')
        lam = np.empty(xa.shape, order='C')
        explinear(xa, lam)
        lam = np.reshape(lam, np.shape(x))
        if np.isscalar(x):
            return np.asscalar(lam)
        else:
            return lam