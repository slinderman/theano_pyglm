class Component:
    """
    """
    
    def __init__(self, model, vars, offset):
        """ Initialize the component with the parameters from the given model,
        the vector of symbolic variables, vars, and the offset into that vector, offset.
        """
        pass

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        pass

    def sample(self):
        """
        return a sample of the variables
        """
        pass

    def params(self):
        """
        Get the parameters of this component?
        """
        pass