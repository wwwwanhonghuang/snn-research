from snn_lib.network.network import Network
class NetworkBuilder():
    def __init__(self):
        self.neurons = {}
        self.connections = []
        
        
    def add_neuron(self, neuron_id, neuron):
        self.neurons[neuron_id] = neuron
        
    def add_connection(self, connection):
        self.connections.append(connection)
            
    def build_network(self):
        return Network(self.neurons, self.connections)

    