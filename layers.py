from Neuron import Neuron


class Layer:
    def __init__(self, num_neurons, function):
        self.num_neurons = num_neurons
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron())
        self.function = function # Name of function. Ex: sigmoid, relu, tanh etc.

        print( self.__repr__() + "  --initialized")

    def put_values(self, values):
        for i, v in enumerate(values):
            self.neurons[i].value = v

    def __repr__(self):
        return "Layer:  size = " + str(self.num_neurons) + "  function = " + str(self.function)
