import numpy as np
import random
import matplotlib.pyplot as plt


def fourier_transform(x): # transforomata fouriera
    a = np.abs(np.fft.fft(x))
    return a/np.max(a)


def standarise_features(x): # standaryzacja - problemy
    return (x - np.mean(x))/np.std(x)


class Perceptron(object):

    def __init__(self, no_of_input, number, learning_rate=0.001, iterations=10000, biased=False):
        self.no_of_input = no_of_input
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.random.random(2 * self.no_of_input + 1) / 1000 # losowe wagi początkowe
        self.errors = []
        self.biased = biased # opcjonalne dodanie biasu
        self.number = number

    def add_bias(self, x):
        if self.biased:
            return 0
        else:
            return x

    def train(self, training_data_x, training_data_y):
        # preprocessed_training_data_x = standarise_features(training_data_x) # problemy ze standaryzacją
        for _ in range(self.iterations):
            e = 0

            randomize_list = list(zip(training_data_x, training_data_y))
            random.shuffle(randomize_list)
            training_data_x, training_data_y = zip(*randomize_list)

            for x, y in zip(training_data_x, training_data_y):
                x_ext = np.concatenate([x, fourier_transform(x)])
                out = self.output(x)
                self.weights[1:] += self.learning_rate * (y - out) * x_ext * self.derivative(out)
                self.weights[0] += self.learning_rate * (y - out) * self.derivative(out)
                e += 0.5 * (y - out) ** 2
            self.errors.append(e)
        plt.plot(range(len(self.errors)), self.errors, label=str(self.number))
        plt.ylim(-0.5, 2.0)
        plt.legend()
        plt.savefig('errors.pdf')

    def activation(self, x):  # Zadanie: dodanie funkcji aktywacji -> zmiana pochodnej
        return 1 / (1 + np.exp(-x))
        # return x

    def output(self, input):
        inp = np.concatenate([input, fourier_transform(input)])
        summation = self.activation(np.dot(self.weights[1:], inp) + self.add_bias(self.weights[0]))
        return summation

    def derivative(self, input): # pochodna
        return self.activation(input) * (1 - self.activation(input))

