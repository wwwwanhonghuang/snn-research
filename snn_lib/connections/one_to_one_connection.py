from snn_lib.connections.base_connection import AbstractConnection
import numpy as np
class OneToOneConnection(AbstractConnection):
    def __init__(self, initW = None):
        super().__init__()
        if initW is not None:
            self.initialize(initW)
        else:
            self.initialize()

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def pseudo_update_states(self, u=None):
        return None

    def get_output(self, u):
        return self.W * u #self.W @ u
    
    def initialize(self, W = None):
        if W == None:
            self.W = 0.1 * np.random.rand()
        else:
            self.W = W