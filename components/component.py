class Component:
    """
    """
    
    def __init__(self, model):
        """ Initialize the component with the parameters from the given model.
        """
        pass
    
    def get_variables(self):
        """ Get a dictionary of (name : Theano variable) items for all the
            symbolic variables associated with this component.
        """
        return {}
    
    def get_state(self):
        return {}

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        pass
    
    def set_hyperparameters(self, model):
        """ Set hyperparameters of the model
        """
        pass

    def sample(self):
        """
        return a sample of the variables
        """
        return []
