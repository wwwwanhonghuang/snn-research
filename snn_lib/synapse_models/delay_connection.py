import numpy as np
from snn_lib.synapse_models.base_synapse import AbstractSynapse

class DelayConnection(AbstractSynapse):
    def __init__(self, N, delay, dt=1e-4):
        super().__init__()
        nt_delay = round(delay / dt)
        self.N = N
        self.delay = delay
        self.dt = dt
        self.nt_delay = nt_delay
        self.initialize()
        
    
    def initialize(self):
        self._states = np.zeros((self.N, self.nt_delay))
        self._cached_states = self._states
    
    def get_output(self, u=None):
        out = self._cached_states[:, -1]
        return out
    
    def pseudo_update_states(self, u=None):
        states = self.states
        states[:, 1:] = states[:, :-1]
        states[:, 0] = u
        self._cached_states = states
        return self._cached_states
    
  
    
    