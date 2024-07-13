class NeuronPlot():
    import matplotlib.pyplot as plt

    def plot_possion_neuron_raster(self, possion_neuron):
        t = np.arange(possion_neuron.time_steps) * possion_neuron.dt
        plt.figure(figsize=(5, 4))
        for i in range(possion_neuron.N):
            plt.plot(t, possion_neuron.spikes[:, i] * (i + 1), 'ko', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron index')
        plt.xlim(0, possion_neuron.simulation_time_duration); plt.ylim(0.5, 10 + 0.5)
        plt.show() 

    