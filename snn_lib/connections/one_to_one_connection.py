import numpy as np
class OneToOneConnection:
    def __init__(self, initW = None):
        if initW is not None:
            self.initialize(initW)
        else:
            self.initialize()

    def backward(self, x):
        return self.W * x #self.W.T @ x

    def update_states(self, u=None):
        return None

    def get_output(self, u):
        return self.W * u #self.W @ u
    
    def initialize(self, W = None):
        if W == None:
            self.W = 0.1 * np.random.rand()
        else:
            self.W = W