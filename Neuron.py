class Neuron:
    def __init__(self):
        self.value = None

        print(self.__repr__() + "  --initialized!")

    def __repr__(self):
        return "Neuron:  value = " + str(self.value)
