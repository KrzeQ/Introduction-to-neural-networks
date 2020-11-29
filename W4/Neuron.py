import numpy as np
import matplotlib.pyplot as plt


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Example(object):

    def __init__(self, armLength=10):
        self.center = Point(0, 0)
        self.armLength = armLength
        self.input = []
        self.output = []

    def generate(self, noOfExamples):
        for i in range(noOfExamples):
            point = self.generatePoint(self.center)
            self.input.append([point.x, point.y])
        return self.input, self.output

    def generatePoint(self, center):
        alpha = np.random.random() * np.pi
        beta = np.random.random() * np.pi
        self.output.append([alpha, beta])

        tempPoint = self.translate(self.center, alpha)
        finalPoint = self.translate(tempPoint, np.pi - beta + alpha)
        return finalPoint

    def translate(self, center, angle):
        return Point(center.x + self.armLength * np.sin(angle), center.y - self.armLength * np.cos(angle))


class Neuron(object):
    def __init__(self, no_of_inputs, eta=0.1):
        self.no_of_inputs = no_of_inputs
        self.eta = eta
        self.weights = np.random.random(self.no_of_inputs) - 0.5
        self.activation = 0
        self.sigma = 0
        self.delta = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.sigma * (1 - self.sigma)

    def output(self, x):
        activation = 0
        for i in range(self.no_of_inputs):
            activation += self.weights[i] * x[i]
        self.activation = activation
        self.sigma = self.sigmoid(activation)
        return self.sigma.copy()  # do sprawdzenia

    def update_weigts(self, x):
        for i in range(self.no_of_inputs):
            self.weights[i] -= self.eta * self.delta * x[i]


class Layer(object):
    def __init__(self, layer_size, prev_layer_size, eta=0.1):
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.eta = eta
        self.neurons = []
        self.input = []
        for i in range(layer_size):
            self.neurons.append(Neuron(prev_layer_size, eta=self.eta))

    def output(self, x):
        result = []
        for i in range(self.layer_size):
            result.append(self.neurons[i].output(x))
        return np.array(result)

    def update_weights(self, x):
        for i in range(self.layer_size):
            self.neurons[i].update_weigts(x)


class NeuralNetwork(object):
    def __init__(self, structure, iterations=100, eta=0.1):
        self.structure = structure
        self.iterations = iterations
        self.eta = eta
        self.layers = []
        self.errors = []
        self.output = []
        for i in range(len(self.structure) - 1):
            self.layers.append(Layer(self.structure[i + 1], self.structure[i]))
        # structure = [2, 10, 10, 3]

    def train(self, training_data_x, training_data_y):
        for i in range(self.iterations):
            index = int(np.random.random() * len(training_data_x))
            input = training_data_x[index]
            output = training_data_y[index]

            self.forward(input)
            self.backward(output)

    def forward(self, input):
        self.layers[0].input = input.copy()
        for i in range(len(self.structure) - 2):
            input = self.layers[i].output(input)
            self.layers[i + 1].input = input.copy()
        self.output = self.layers[i + 1].output(self.layers[i + 1].input).copy()

    def backward(self, output):
        last_layer = len(self.layers) - 1
        for j in range(self.layers[last_layer].layer_size):
            epsilon = output[j] - self.output[j]
            self.layers[last_layer].neurons[j].delta = epsilon * self.layers[last_layer].neurons[j].sigmoid_derivative()
        self.layers[last_layer].update_weights(self.layers[last_layer].input)

        for l in reversed(range(len(self.layers) - 1)):
            for j in range(self.layers[l].layer_size):
                epsilon = 0
                for k in range(self.layers[l+1].layer_size):
                    epsilon += self.layers[l+1].neurons[k].weights[j] * self.layers[l+1].neurons[k].delta
                self.layers[l].neurons[j].delta = epsilon * self.layers[l].neurons[j].sigmoid_derivative()
            self.layers[l].update_weights(self.layers[l].input)


def main():
    structure = [2, 10, 10, 3]
    nn = NeuralNetwork(structure)
    nn.forward(np.random.random(2))
    print(nn.layers[0].neurons[0].delta)
    print(nn.output)


if __name__ == "__main__":
    main()
