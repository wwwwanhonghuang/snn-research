class NeuronOutputMonitor(object):
    def __init__(self):
        self.neuron_output_records = []

    def record_neuron_output(self, t, neuron_map):
        record_map = {}
        for neuron_id in neuron_map:
            neuron = neuron_map[neuron_id]
            out = neuron.states[neuron._output_index]
            record_map[neuron_id] = (t, out)
        self.neuron_output_records.append(record_map)
    def clear_record(self):
        self.neuron_output_records = []
        
    def get_dataframe_record(self, neuron_map):
        import pandas as pd
        total_record = {neuron_id:[record[neuron_id] for record in self.neuron_output_records] for neuron_id in neuron_map}
        return pd.DataFrame(total_record)

class NeuronMembranePotentialMonitor(object):
    def __init__(self):
        self.neuron_output_records = []

    def record_neuron_output(self, t, neuron_map):
        record_map = {}
        for neuron_id in neuron_map:
            neuron = neuron_map[neuron_id]
            out = neuron.states[neuron.INDEX_V]
            record_map[neuron_id] = (t, out)
        self.neuron_output_records.append(record_map)
    def clear_record(self):
        self.neuron_output_records = []
        
    def get_dataframe_record(self, neuron_map):
        import pandas as pd
        total_record = {neuron_id:[record[neuron_id] for record in self.neuron_output_records] for neuron_id in neuron_map}
        return pd.DataFrame(total_record)
