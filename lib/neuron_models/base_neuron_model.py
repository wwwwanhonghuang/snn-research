class AbstractNeuron(object):
    def __init__(self, hyperparameters = None, states = None, output_index = 0):
        self._hyperparameters = hyperparameters
        self._states = states
        self.output_index = 0
    
    @property
    def states(self):
        return self._states

    @property
    def hyperparameters(self):
        return self._hyperparameters
    
    def set_states(self, states):
        self._states = states
        
    def set_hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def reset_states(self):
        raise NotImplementedError
    
    def update_state(self, u):
        raise NotImplementedError
    
    def __call__(self, u):
        self.update_state(u)
        if self.output_index == -1:
            return self._states
        return self._states, self._states[self.output_index]
        