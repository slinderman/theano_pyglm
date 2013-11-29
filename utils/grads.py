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
    g_list = T.grad(cost, wrt_list)
    
    for (n,g) in enumerate(g_list):
        if g.ndim < 1:
            g_list[n] = T.shape_padright(g,nones=1)
        elif g.ndim > 1:
            raise Exception("Gradients can only be taken wrt vectors.")
        
    g_vec = T.concatenate(g_list)
    
    return g_vec, g_list
    
def hessian_wrt_list(cost, wrt_list, g_list=None):
    """
    Compute gradient of cost wrt the variables in wrt_list.
    Return a concatenated vector of the results
    """
    if g_list is None:
        g_list = T.grad(cost, wrt_list)
    # Compute the hessian
    H_rows = []
    for gv1 in g_list:
        H_v1 = []
        for v2 in wrt_list:
            # Compute dgv1/dv2
            H_v1v2,_ = theano.scan(lambda i, gy, x: T.grad(gy[i], x),
                                   sequences=T.arange(gv1.shape[0]),
                                   non_sequences=[gv1, v2])
            if H_v1v2.ndim < 2 :
                H_v1v2 = T.shape_padright(H_v1v2,nones=(2-H_v1v2.ndim))
            elif H_v1v2.ndim > 2:
                raise Exception("Hessians must be at most 2D")
            H_v1.append(H_v1v2)
            
        H_rows.append(T.concatenate(H_v1, axis=1))
    
    # Concatenate the Hessian blocks into a matrix
    H = T.concatenate(H_rows, axis=0)
    
    return H