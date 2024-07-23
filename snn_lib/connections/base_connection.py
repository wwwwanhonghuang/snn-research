class AbstractConnection(object):
    def __init__(self):
        self._cached_states = None
        self._states = None
    
    def do_update_states(self):
        if (self._cached_states == None):
            raise ValueError("Cache is empty.")
        self._states = self._cached_states
        self._cached_states = None
        
    @property
    def states(self):
        return self._states
    
    @property
    def cached_states(self):
        return self._cached_states
    
    def initialize(self):
        raise NotImplemented
    
    def pseudo_update_states(self, u = None):
        raise NotImplemented
    
    def get_output(self, u = None):
        raise NotImplemented
    
    def __call__(self, u):
        if self.states == None:
            raise ValueError("States have not been initialized yet.")
        if(self.cached_states != None):
            raise RuntimeError("cached output should write back to `states` property by `do_update_states()` before further update.")
        connection_states = self.pseudo_update_states(u)
        connection_output = self.get_output(u)
        return connection_states, connection_output
        