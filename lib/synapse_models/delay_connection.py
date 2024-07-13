import numpy as np
from lib.synapse_models.base_synapse import AbstractSynapse

class DelayConnection(AbstractSynapse):
    def __init__(self, N, delay, dt=1e-4):
        nt_delay = round(delay / dt)
        self.state = np.zeros((N, nt_delay))
        self.N = N
        self.delay = delay
        self.dt = dt
    
    def initialize_states(self):
        nt_delay = round(self.delay / self.dt)
        self.state = np.zeros((self.N, nt_delay))
        
    def __call__(self, x):
        out = self.state[:, -1]
        self.state[:, 1:] = self.state[:, :-1]
        self.state[:, 0] = x
        return out