class AbstractConnection(object):
    def __init__(self):
        pass
    
    
    def initialize(self):
        raise NotImplemented
    
    def update_states(self, u = None):
        raise NotImplemented
    
    def get_output(self, u = None):
        raise NotImplemented
    
    def __call__(self, u):
        connection_states = self.update_states(u)
        connection_output = self.get_output(u)
        return connection_states, connection_output
        