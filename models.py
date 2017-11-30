from Neuron import Neuron
import numpy as np
from random import uniform


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1 - np.tanh(x)**2
def softmax(x):
    # print(str(x.shape))
    sum = 0
    for n in x:
        sum += np.e**n
    return [np.e**m/sum for m in x]

def compute_loss(output, expected_output):
    # return difference squared
    if len(output) != len(expected_output):
        raise Exception("Expected output does not match the predicted output in size." \
                        " \n Size of output: " + str(len(output)) + " Size of expected output: " + str(len(expected_output)))
    sum = 0
    for i in range(len(output)):
        sum += (output[i] - expected_output[i])**2
    return sum





class Sequential:
    def __init__(self):
        print("Initialized model!")
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        self.weights = []
        for i in range(1, len(self.layers)):
            w = np.random.random((self.layers[i - 1].num_neurons, self.layers[i].num_neurons)) #(h, w)
            # Make some weights negative
            rows, cols = w.shape
            for x in range(0, rows):
                for y in range(0, cols):
                    if uniform(0, 1) > 0.5:
                        w[x, y] *= -1
            self.weights.append(w)
            print(w)

    # Helper for run
    def multiply(self, l, value):
        # print("Row: " + str(l))
        for i in range(len(l)):
            l[i] = l[i] * value
        # print("times : " + str(value) + " is \n" + str(l))
        return l

    def run(self, inputs):
        print("Running...")
        if len(inputs) != self.layers[0].num_neurons:
            raise Exception("Length of inputs not equal to length of input layer.")
        self.layers[0].put_values(inputs)
        for i, w in enumerate(self.weights):
            print("\n\nForward propagating weight number " + str(i))
            next_inputs = []
            rows = []
            for row_num in range(0, w.shape[0]):
                row = w[row_num]
                # print("Row: " + str(row))
                row_copy = row.copy()
                row_copy = self.multiply(row_copy, self.layers[i].neurons[row_num].value)
                rows.append(row_copy)
            for col_num in range(rows[0].shape[0]):
                sum = 0
                for r in rows:
                    sum += r[col_num]
                if i == len(self.weights) - 1:
                    next_inputs.append(sum)
                else:
                    next_inputs.append(sigmoid(sum))
            self.layers[i+1].put_values(next_inputs)

        s = ''
        p = []
        for n in self.layers[len(self.layers) - 1].neurons:
            p.append(n.value)
            s += str(n.value) + " , "

        print("\n\nOutput: " + s[:-3])
        print("Probabilities: " + str(softmax(p)))

        print("Running complete!")
        return softmax(p)