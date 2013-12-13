import numpy as np

def average_list_of_dicts(smpls):
    """ Average a list of dictionaries, e.g. a list of samples from
        the MCMC loop. The dictionary elements may themselves be
        dictionaries, so work recursively and only average the leaves.
    """
    N_smpls = len(smpls)
    import copy
    avg = copy.deepcopy(smpls[0])

    def inc_avg(avg, smpl):
        """ Helper to recrusively add to the average
        """
        if isinstance(smpl, dict):
            for (key,val) in smpl.items():
                if isinstance(val, dict) or \
                  isinstance(val, list):
                  # Recurse if the entry is another dict
                    avg[key] = inc_avg(avg[key], val)
                elif isinstance(val, np.ndarray):
                    # Otherwise increment the value
                    avg[key] = avg[key].astype(np.float) + \
                               val.astype(np.float)
                else:
                    # Who knows if this will work without casting!
                    avg[key] += val
        
        elif isinstance(smpl, list):
            for (i,val) in enumerate(smpl):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    avg[i] = inc_avg(avg[i], val)
                elif isinstance(val, np.ndarray):
                    avg[i] = avg[i].astype(np.float) + \
                             val.astype(np.float)
                else:
                    # Who knows if this will work without casting!
                    avg[i] += float(val)
        return avg

    for smpl in smpls[1:]:
        avg = inc_avg(avg, smpl)

    def norm_avg(avg, N_smpls):
        """ Normalize the average by dividing by N_smpls
        """
        if isinstance(avg, dict):
            for (key,val) in avg.items():
                if isinstance(val, dict) or \
                   isinstance(val, list):
                    avg[key] = norm_avg(val, N_smpls)
                else:
                    avg[key] /= float(N_smpls)
        elif isinstance(avg, list):
            for (i,val) in enumerate(avg):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    avg[i] = norm_avg(val, N_smpls)
                else:
                    avg[i] /= float(N_smpls)
                        
        return avg

    avg = norm_avg(avg, N_smpls)
    return avg

def variance_list_of_dicts(smpls, avg=None):
    """ Compute the variance of a list of dictionaries, e.g. a list of samples from
        the MCMC loop. The dictionary elements may themselves be
        dictionaries, so work recursively and only average the leaves.
    """
    N_smpls = len(smpls)

    if avg is None:
        avg = average_list_of_dicts(smpls)

    import copy
    centered_smpls = copy.deepcopy(smpls)
    
    def centersq(smpl, avg):
        """ Helper to recrusively subtract the mean from a sample
            and square
        """
        if isinstance(smpl, dict):
            for (key,val) in smpl.items():
                if isinstance(val, dict) or \
                  isinstance(val, list):
                  # Recurse if the entry is another dict
                    smpl[key] = centersq(val, avg[key])
                elif isinstance(val, np.ndarray):
                    # Otherwise subtract the average and square
                    smpl[key] = (val.astype(np.float) - \
                                 avg[key].astype(np.float))**2
                else:
                    # Who knows if this will work without casting!
                    smpl[key] = (val-avg[key])**2
        
        elif isinstance(smpl, list):
            for (i,val) in enumerate(smpl):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    smpl[i] = centersq(val, avg[i])
                elif isinstance(val, np.ndarray):
                    smpl[i] = (val.astype(np.float) - \
                               avg[key].astype(np.float))**2
                else:
                    # Who knows if this will work without casting!
                    smpl[i] += (val-avg[key])**2
        return smpl

    for smpl in centered_smpls:
        smpl = centersq(smpl, avg)

    # Compute the variance 
    var = average_list_of_dicts(centered_smpls)
    return var

def std_list_of_dicts(smpls, avg=None):
    """ Compute the std of a list of dictionaries, e.g. a list of samples from
        the MCMC loop. The dictionary elements may themselves be
        dictionaries, so work recursively and only average the leaves.
    """
    N_smpls = len(smpls)
    var = variance_list_of_dicts(smpls, avg=avg)

    def sqrt_helper(smpl):
        """ Helper to recursively take the square root 
        """
        if isinstance(smpl, dict):
            for (key,val) in smpl.items():
                if isinstance(val, dict) or \
                  isinstance(val, list):
                  # Recurse if the entry is another dict
                    smpl[key] = sqrt_helper(val)
                else:
                    # Take the square root
                    smpl[key] = np.sqrt(val)
        
        elif isinstance(smpl, list):
            for (i,val) in enumerate(smpl):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    smpl[i] = sqrt_helper(val)
                else:
                    smpl[i] = np.sqrt(val)
        return smpl

    std = sqrt_helper(var)
    return std

