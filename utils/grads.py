""" Gradient helper. We want to compute the gradient of the log probability
    wrt a set of parameters simultaneously. Similarly, we want to compute 
    the Hessian of the log prob wrt each pair of parameters. The result should 
    be a theano vector or matrix, respectively.
"""
import theano
import theano.tensor as T

def grad_wrt_list(cost, wrt_list):
    """
    Compute gradient of cost wrt the variables in wrt_list.
    Return a concatenated vector of the results
    """
    if wrt_list == []:
        return T.constant(0.),[]

    g_list = T.grad(cost, wrt_list)

    for (n,g) in enumerate(g_list):
        if g.ndim < 1:
            g_list[n] = T.shape_padright(g, n_ones=1)
        elif g.ndim > 1:
#            raise Exception("Gradients can only be taken wrt vectors.")
            g_list[n] = T.flatten(g)
    
    g_vec = T.concatenate(g_list)
    
    return g_vec, g_list
    
def hessian_wrt_list(cost, wrt_list, g_list=None):
    """
    Compute gradient of cost wrt the variables in wrt_list.
    Return a concatenated vector of the results
    """
    if wrt_list == []:
        return T.constant(0.)
    
    if g_list is None:
        g_list = T.grad(cost, wrt_list)
    # Compute the hessian
    H_rows = []
    for gv1 in g_list:
        H_v1 = []
        for v2 in wrt_list:
            # Ensure v2 is a vector
            if v2.ndim < 1:
                v2 = T.shape_padright(v2, n_ones=1)
            elif v2.ndim > 1:
                v2 = T.flatten(v2)
            
            # Compute dgv1/dv2
            H_v1v2,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
                                   sequences=T.arange(gv1.shape[0]),
                                   non_sequences=[gv1, v2])
            if H_v1v2.ndim < 2 :
                H_v1v2 = T.shape_padright(H_v1v2, n_ones=(2-H_v1v2.ndim))
            elif H_v1v2.ndim > 2:
                raise Exception("Hessians must be at most 2D")
            H_v1.append(H_v1v2)
            
        H_rows.append(T.concatenate(H_v1, axis=1))
    
    # Concatenate the Hessian blocks into a matrix
    H = T.concatenate(H_rows, axis=0)
    
    return H

def hessian_rop_wrt_list(cost, wrt_list, v, g_vec=None, g_list=None):
    """
    Compute an expression for the Hessian of cost with respect to wrt_list,
    right-multiplied by a column vector v.
    """
    if wrt_list == []:
        raise Exception("wrt_list must not be empty!")

    if g_vec is None:
        if g_list is None:
            g_list = T.grad(cost, wrt_list)
        g_vec = T.concatenate(g_list, axis=0)

    # Compute the Hessian \dot vector Rop
    wrt_flat = []
    for wrt in wrt_list:
        if wrt.ndim < 1:
            wrt = T.shape_padright(wrt, n_ones=1)
        elif wrt.ndim > 1:
            wrt = T.flatten(wrt)
        wrt_flat.append(wrt)

    # Concatenate wrt into a single vector
    wrt = T.concatenate(wrt_flat, axis=0)

    # Compute the Rop
    Hv = T.Rop(g_vec, wrt, v)
    return Hv

def differentiable(syms):
    """
    Return the subset of the syms dictionary containing only differentiable variables
    """
    diff = {}
    for (k,v) in syms.items():
        vnew = None
        if isinstance(v,dict):
            # Recurse on the sub dictionary
            vnew  = differentiable(v)
        elif hasattr(v,'dtype'):
            if 'float' in v.dtype:
                vnew = v
        else:
            import pdb
            pdb.set_trace()
            raise Exception('Unrecognized value: %s' % str(v))

        if vnew is not None:
            diff[k] = v

    # Check if the dict is empty
    if not any(diff):
        return None
    else:
        return diff
