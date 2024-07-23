class AbstractNeuron(object):
    def __init__(self, hyperparameters = None, states = None, output_index = 0):
        self._hyperparameters = hyperparameters
        self._states = states
        self.output_index = 0
        self._cached_states = None
    
    @property
    def states(self):
        return self._states

    @property
    def hyperparameters(self):
        return self._hyperparameters
    
    @property
    def cached_states(self):
        return self._cached_states
    
    def set_cached_states(self, states):
        self._cached_states = states
        
    def set_hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def reset_states(self):
        raise NotImplementedError
    
    def update_state(self, u):
        raise NotImplementedError
    
    def reset_time(self, t = 0):
        self.t = t
    
    def write_back_states(self):
        if self.cached_states == None:
            raise ValueError("No updated states are stored in cache.")
        self._states =  self.cached_states
        self._cached_states = None
    
    def initialize_states(self):
        raise NotImplementedError
    
    def __call__(self, u):
        if(self.cached_states != None):
            raise RuntimeError("cached output should write back to `states` property by `step_to_next_states()` before further update.")
        self.update_state(u)
        if self.output_index == -1:
            return self.cached_states
        return self.cached_states, self.cached_states[self.output_index]
        