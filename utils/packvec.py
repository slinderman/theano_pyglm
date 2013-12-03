import numpy as np

def pack(var_list):
    """ Pack a list of variables (as numpy arrays) into a single vector
    """
    vec = np.zeros((0,))
    shapes = []
    for var in var_list:
        assert isinstance(vec, np.ndarray), "Can only pack numpy arrays!"
        sz = var.size
        shp = var.shape
        assert sz == np.prod(shp), "Just making sure the size matches the shape"
        shapes.append(shp)
        vec = np.concatenate((vec, np.reshape(var, (sz,))))
    return vec, shapes


def unpack(vec, shapes):
    """ Unpack a vector of variables into an array
    """
    off = 0
    var_list = []
    for shp in shapes:
        sz = np.prod(shp)
        var_list.append(np.reshape(vec[off:off + sz], shp))
        off += sz
    assert off == len(vec), "Unpack was called with incorrect shapes!"
    return var_list
