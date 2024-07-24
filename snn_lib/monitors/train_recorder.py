class TrainRecorder():
    def __init__(self) -> None:
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
        
        