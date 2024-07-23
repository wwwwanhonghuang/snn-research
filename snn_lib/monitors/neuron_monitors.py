class NeuronOutputMonitor(object):
    def __init__(self):
        self.neuron_output_records = {}

    def record_neuron_output(self, t, neuron, neuron_id):
        if neuron_id not in self.neuron_output_records:
            self.neuron_output_records[neuron_id] = {}
        out = neuron.states[neuron._output_index]
        self.neuron_output_records[neuron_id][t] = out
        
    def clear_record(self):
        self.neuron_output_records = {}
        
    def get_dataframe_record(self):
        import pandas as pd
        return pd.DataFrame(self.neuron_output_records)

class NeuronMembranePotentialMonitor(object):
    def __init__(self):
        self.neuron_output_records = {}

    def record_neuron_membrane_potential(self, t, neuron, neuron_id):
        if neuron_id not in self.neuron_output_records:
            self.neuron_output_records[neuron_id] = {}
        v = neuron.states[neuron.INDEX_V]
        self.neuron_output_records[neuron_id][t] = v
        
    def clear_record(self):
        self.neuron_output_records = {}
        
    def get_dataframe_record(self):
        import pandas as pd
        return pd.DataFrame(self.neuron_output_records)
