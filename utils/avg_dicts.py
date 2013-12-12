
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
                else:
                    # Otherwise increment the value
                    avg[key] += val
        
        elif isinstance(smpl, list):
            for (i,val) in enumerate(smpl):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    avg[i] = inc_avg(avg[i], val)
                else:
                    avg[i] += val
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
                    avg[key] /= N_smpls
        elif isinstance(avg, list):
            for (i,val) in enumerate(avg):
                if isinstance(val, list) or \
                   isinstance(val, dict):
                    # Recurse if the entry is another iterable
                    avg[i] = norm_avg(val, N_smpls)
                else:
                    avg[i] /= N_smpls
                        
        return avg

    avg = norm_avg(avg, N_smpls)
    return avg
