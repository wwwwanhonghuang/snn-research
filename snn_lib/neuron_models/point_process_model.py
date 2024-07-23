from snn_lib.neuron_models.base_neuron_model import AbstractNeuron
from typing import Callable, List
from numpy.typing import NDArray
import numpy as np

class PointProcessNeuron(AbstractNeuron):
 
    def __init__(self, N, fr_generator, hyperparameters):
        super().__init__()

        ## Hyperparameters
        self.set_hyperparameters({
                'dt': hyperparameters.get('dt', 1e-4), 
                'simulation_time_duration': hyperparameters.get('simulation_time_duration', 1),
                'tref': hyperparameters.get('tref', 0)
            })

        ## parameters
        self.N = N
        self.time_steps = (int)(np.ceil(self.simulation_time_duration / self.dt))
        if isinstance(fr_generator, int):
            self.fr = np.array([[fr_generator]] * self.time_steps)
        elif isinstance(fr_generator, float):
            self.fr = np.array([[fr_generator]] * self.time_steps)
        elif isinstance(fr_generator, Callable):
            self.fr = np.expand_dims(fr_generator(np.arange(self.time_steps)  * self.dt), 1)
        elif isinstance(fr_generator, List):
            self.fr = np.array(fr_generator)
        elif isinstance(fr_generator, NDArray):
            self.fr = fr_generator
        else:
            raise ValueError

        self.INDEX_NEURONS_V = 0
        self.INDEX_T = 1
        self.INDEX_TLAST = 2

        self.initialize()

        self.reset_spikes()

    def _generate_spikes(self):
        spikes = np.where(np.random.rand(self.time_steps, self.N) < self.fr * self.dt, 1, 0)
        self.spikes = spikes
        print("Num. of spikes:", np.sum(spikes))
        print("Firing rate:", np.sum(spikes)/(self.N * self.simulation_time_duration))
        
    @property
    def dt(self):
        return self._hyperparameters['dt']
    
    @property
    def simulation_time_duration(self):
        return self._hyperparameters['simulation_time_duration']

    @property
    def tref(self):
        return self._hyperparameters['tref']


    def initialize(self):
        self.cache_states([[], 0, np.array([[-self.tref - 1]] * self.N)])
        self._states = self._cached_states
        
    def reset_spikes(self):        
        self._generate_spikes()
        
    def pseudo_update_states(self, u = None):
        t = self.states[self.INDEX_T]
        tlast = self.states[self.INDEX_TLAST]
        t += 1
        neurons_v = self.spikes[t]
        neurons_v = neurons_v * (self.dt * t > (tlast + self.tref))
        tlast = tlast * (1 - neurons_v) + self.dt * t * neurons_v
        
        self.cache_states([neurons_v, t, tlast])
        return self.cached_states
  

class PossionProcessNeuron(PointProcessNeuron):
    def __init__(self, N, frequency, hyperparameters):
        super().__init__(N, frequency, hyperparameters)

class GammaProcessNeuron(PointProcessNeuron):
    def __init__(self, N, frequency, hyperparameters):
        self.k = hyperparameters.get("k", 12)
        self.fr = frequency

        super().__init__(N, frequency, hyperparameters)
        
    def _generate_spikes(self):
        theta = 1 / (self.k * (self.fr * self.dt)) # fr = 1 / ( k * theta)
        invervals = np.random.gamma(shape = self.k, scale = theta, size = (self.time_steps, self.N))
        spike_time = np.cumsum(invervals, axis = 0) # ISI を累積
        spike_time = spike_time.astype(np.int32)
        spikes = np.zeros((self.time_steps, self.N))
        spike_time[spike_time > self.time_steps - 1] = 0
        for i in range(self.N):
            spikes[spike_time[:, i], i] = 1
        spikes[0] = 0 
        print("Num. of spikes:", np.sum(spikes))
        print("Firing rate:", np.sum(spikes)/(self.N * self.simulation_time_duration))
