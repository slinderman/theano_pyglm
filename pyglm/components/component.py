class Component(object):
    """
    """

    @property
    def log_p(self):
        """
        Compute the log probability of this component
        :return:
        """
        raise NotImplementedError()

    def get_variables(self):
        """ Get a dictionary of (name : Theano variable) items for all the
            symbolic variables associated with this component.
        """
        return {}
    
    def get_state(self):
        return {}

    def preprocess_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        pass

    def set_data(self, data):
        """ Set the shared memory variables that depend on the data
        """
        pass
    
    def set_hyperparameters(self, model):
        """ Set hyperparameters of the model
        """
        pass

    def sample(self, acc):
        """
        return a sample of the variables
                """
        return {}
