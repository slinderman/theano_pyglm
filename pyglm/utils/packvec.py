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

def packdict(var_dict, on_unpackable_type='raise'):
    """ Pack a dictionary of variables (as numpy arrays) into a single vector
    """
    vec = np.zeros((0,))
    shapes = {}
    # This sorting is important!
    for (var, val) in sorted(var_dict.items(), key=lambda t: t[0]):
        if isinstance(val, dict):
            # Recurse on sub dictionary
            svec, sshapes = packdict(val)
            vec = np.concatenate((vec,svec))
            shapes[var] = sshapes
        elif val == []:
            continue
        else:
            if not isinstance(val, np.ndarray):
                if on_unpackable_type.lower() == 'raise':
                    raise Exception("Can only pack numpy arrays!")
                else:
                    print "Can only pack numpy arrays! Attempting to " \
                          "pack type %s." % str(type(val))
                    val = np.asarray(val)
            sz = val.size
            shp = val.shape
            shapes[var] = shp
            vec = np.concatenate((vec, np.reshape(val, (sz,))))
        
    return vec, shapes


def dictshapes(var_dict, on_unpackable_type='raise'):
    """ Pack a dictionary of variables (as numpy arrays) into a single vector
    """
    shapes = {}
    # This sorting is important!
    for (var, val) in sorted(var_dict.items(), key=lambda t: t[0]):
        if isinstance(val, dict):
            # Recurse on sub dictionary
            sshapes = dictshapes(val)
            shapes[var] = sshapes
        elif val == []:
            continue
        else:
            if not isinstance(val, np.ndarray):
                if on_unpackable_type.lower() == 'raise':
                    raise Exception("Can only pack numpy arrays!")
                else:
                    print "Can only pack numpy arrays! Attempting to " \
                          "pack type %s." % str(type(val))
            shp = val.shape
            shapes[var] = shp

    return shapes


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

def unpackdict(vec, shapes):
    """ Unpack a dictionary of variables (as numpy arrays) into a single vector
    """
    vars,_ = unpackdict_helper(vec,shapes,0)
    return vars

def unpackdict_helper(vec, shapes, offset):
    """ Unpack dictionary recursion helper
    """
    vars = {}
    sz = 0
    # This sorting is important!
    for (var, shp) in sorted(shapes.items(), key=lambda t: t[0]):
        if isinstance(shp, dict):
            # Recurse on sub dictionary
            vars[var],ssz = unpackdict_helper(vec, shp, offset)
            sz += ssz
            offset += ssz
        elif isinstance(shp, tuple):
            ssz = np.prod(shp)
            vars[var] = np.reshape(vec[offset:offset+ssz], shp)
            sz += ssz
            offset += ssz
        else:
            raise Exception("Can only unpack shape tuples!")
    return vars, sz


def get_vars(syms, vars):
    """ Extract variables for corresponding syms
    """
    ve = {}
    for (k,v) in syms.items():
        assert k in vars.keys(), "ERROR: syms key %s not found in vars!" % k
        if isinstance(v,dict):
            ve[k] = get_vars(v, vars[k])
        else:
            ve[k] = vars[k]
    return ve

def set_vars(syms, vars, vals):
    if isinstance(syms, dict):
        for (k,v) in syms.items():
            assert k in vars.keys(), "ERROR: syms key %s not found in vars!" % k
            assert k in vals.keys(), "ERROR: syms key %s not found in vals!" % k
            if isinstance(v,dict):
                vars[k] = set_vars(v, vars[k], vals[k])
            else:
                vars[k] = vals[k]
    elif syms in vars:
        vars[syms] = vals
    else:
        raise Exception("Can only set variables for a dictionary of symbolic vars" \
                        "or a specific key in vars")
        
    return vars

def get_shapes(x, syms):
    xv = get_vars(syms, x)
    _,shapes = packdict(xv)
    return shapes