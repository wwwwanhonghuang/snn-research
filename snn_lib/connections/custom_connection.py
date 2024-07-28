from snn_lib.connections.base_connection import AbstractConnection
from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
import numpy as np

class CustomConnection(AbstractConnection):
    def __init__(self, pre_connection_neuron : AbstractNeuron, post_connection_neuron: AbstractNeuron, connection = None, W = None):
        super().__init__()
        self.pre_connection_neuron = pre_connection_neuron
        self.post_connection_neuron = post_connection_neuron
        self.W = W      
        mask = np.zeros((self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron))
        mask[connection[0],:] += 1
        mask[:, connection[1]] += 1
        mask[mask != 2] = 0  
        mask[mask == 2] = 1 
        self.mask = mask

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def pseudo_update_states(self, u = None):
        self.cache_states(self.states)
        return self.cached_states

    def get_output(self, u):
        out = np.multiply(self.W, self.mask) * u
        return out #self.W @ u
    
    def initialize(self, W = None):
        
        size = (self.post_connection_neuron.n_neuron, self.pre_connection_neuron.n_neuron)
        if not (W is None):
            self.W = W
        else:
            self.W = np.random.rand(size[0], size[1])
        if self.W.shape[0] != size[0] or self.W.shape[1] != size[1]:
            raise ValueError("weight shape = [%d, %d] should equal to [%d, %d]" % (self.W.shape[0], self.W.shape[1], \
                size[0], size[1]))
        
        self._states = [0]
        self._cached_states = None