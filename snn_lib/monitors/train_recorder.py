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
    
    def get_get_enuron_recorder(self, recorder_id):
        return self.neuron_recorders[recorder_id]['items']
    
    def add_neuron_recorder(self, record_id, item_initializer, update_function):
        self.neuron_recorders[record_id] = {'updated': False, 'items': {}}
        self._neuron_recorders_item_initializer[record_id] = item_initializer
        self._neuron_recorders_update_function[record_id] = update_function
        
    def add_connection_recorder(self, record_id, item_initializer, update_function):
        self.connection_recorders[record_id] = {'updated': False, 'items': {}}
        self._connection_recorders_item_initializer[record_id] = item_initializer
        self._connection_recorders_update_function[record_id] = update_function
        
    def add_pre_requisite(self, record, requisite):
        if record not in self.requisites:
            self.requisites[record] = []
        self.requisites[record].append(requisite)
        
    def update_all_recorders(self, t, arg = None):
        for recorder_id in self.neuron_recorders:     
            self.update_neuron_recorder(t, recorder_id)
            
        for recorder_id in self.connection_recorders: 
            self.update_connection_recorder(t, recorder_id)
        
    def update_neuron_recorder(self, t, recorder_id, arg = None):
        if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == TrainRecorder.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1], arg)
                    elif prerequisite[0] == TrainRecorder.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1], arg)
        recorder = self.neuron_recorders[recorder_id]
        update_function = self._neuron_recorders_update_function[recorder_id]

        for neuron_id in self.neuron_map:
            new_value = update_function(t, recorder['items'], self.neuron_map[neuron_id], recorder['items'][neuron_id], arg)
            recorder['items'][neuron_id] = new_value
        self.neuron_recorders[recorder_id]['updated'] = True
    
    def update_connection_recorder(self, t, recorder_id, arg = None):
        if recorder_id in self.requisites:
                prerequisites = self.requisites[recorder_id]
                for prerequisite in prerequisites:
                    if prerequisite[0] == TrainRecorder.NEURON_RECORD:
                        self.update_neuron_recorder(t, prerequisite[1], arg)
                    elif prerequisite[0] == TrainRecorder.CONNECTION_RECORD:
                        self.update_connection_recorder(t, prerequisite[1], arg)
        
        recorder = self.connection_recorders[recorder_id]
        update_function = self._neuron_recorders_update_function[recorder_id]

        for connection in self.connections:
            new_value = update_function(t, recorder['items'], \
                connection, recorder['items'][connection[0] + "_" + connection[1]], arg)
            recorder['items'][connection[0] + "_" + connection[1]] = new_value
        self.neuron_recorders[recorder_id]['updated'] = True
    
    def initialize_recorders(self):
        for recorder_id in self.neuron_recorders:
            recorder = self.neuron_recorders[recorder_id]
            init_func = self._neuron_recorders_item_initializer[recorder_id]
            for neuron_id in self.neuron_map:
                recorder[neuron_id] = init_func()
            
        for recorder_id in self.connection_recorders:
            recorder = self.neuron_recorders[recorder_id]
            init_func = self._connection_recorders_item_initializer[recorder_id]
            for connection in self.connections:
                recorder[connection[0] + "_" + connection[1]] = init_func()

    def _finish_update(self):
        for recorder_id in self.neuron_recorders:
            self.neuron_recorders[recorder_id]['updated'] = False
            