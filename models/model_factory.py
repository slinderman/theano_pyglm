"""
Make models from a template
"""
from simple_weighted_model import SimpleWeightedModel
from standard_glm import StandardGlm
import copy

def make_model(template, N=None):
    """ Construct a model from a template and update the specified parameters
    """
    if isinstance(template, str):
        # Create the specified model
        if template.lower() == 'standard_glm' or \
           template.lower() == 'standardglm':
            model = copy.deepcopy(StandardGlm)
        elif template.lower() == 'simple_weighted_model' or \
             template.lower() == 'simpleweightedmodel':
            model = copy.deepcopy(SimpleWeightedModel)

    elif isinstance(template, dict):
        model = copy.deepcopy(template)
    else:
        raise Exception("Unrecognized template model!")

    # Override template model parameters
    if N is not None:
        model['N'] = N

    # TODO Update other paramters as necessary

    return model