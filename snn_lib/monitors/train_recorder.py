class TrainRecorder():
    NEURON_RECORD = 0
    CONNECTION_RECORD = 1
    def __init__(self, neuron_map, connections) -> None:
        self.neuron_map = neuron_map
        self.connections = connections
        self.neuron_recorders = {
        }
        self._neuron_recorders_item_initializer = {
        }
        self._neuron_recorders_update_function = {
        }
        self.connection_recorders = {
        }
        self._connection_recorders_item_initializer = {
        }
        self._connection_recorders_update_function = {
        }
        self.requisites = {
            
        }
    
    def add_neuron_recorder(self, record_id, item_initializer, update_function):
        self.neuron_recorders[record_id] = {}
        self._neuron_recorders_item_initializer[record_id] = item_initializer
        self._neuron_recorders_update_function[record_id] = update_function
        
    def add_connection_recorder(self, record_id, item_initializer, update_function):
        self.connection_recorders[record_id] = {}
        self._connection_recorders_item_initializer[record_id] = item_initializer
        self._connection_recorders_update_function[record_id] = update_function
        
    def add_pre_requisite(self, record, requisite):
        if record not in self.requisites:
            self.requisites[record] = []
        self.requisites[record].append(requisite)
        
    def update_all_recorders(self, t):
        for recorder_id in self.neuron_recorders:
            if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == TrainRecorder.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1])
                    elif prerequisite[0] == TrainRecorder.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1])            
            self.update_neuron_recorder(t, recorder_id)
            
        for recorder_id in self.connection_recorders:
            if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == TrainRecorder.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1])
                    elif prerequisite[0] == TrainRecorder.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1])            
            self.update_connection_recorder(t, recorder_id)
            
    def update_neuron_recorder(self, t, recorder_id):
        pass
    
    def update_connection_recorder(self, t, recorder_id):
        pass