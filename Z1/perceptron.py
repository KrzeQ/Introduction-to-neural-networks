import numpy as np
import random


class Perceptron(object):

    def __init__(self, no_of_inputs, learning_rate=0.01, iterations=3000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.rand(self.no_of_inputs + 1)/100  # ZD1 - losowosc

    def train_spla(self, training_data, labels):  # ZD3
        for _ in range(self.iterations):
            data_as_list = list(zip(training_data, labels))
            random.shuffle(data_as_list)
            training_data, labels = zip(*data_as_list)
            for input, label in zip(training_data, labels):
                input_noisy = self.noisy(input)
                prediction = self.output(input_noisy)
                err = label - prediction

                if err != 0.0:
                    self.weights[1:] += err * self.learning_rate * input_noisy
                    self.weights[0] += err * self.learning_rate
        print(self.weights)

    def train_pla(self, training_data, labels):  # ZD4 - PLA
        best_set_of_weights = np.copy(self.weights)
        longest_lifetime = 0
        life_of_weight = 0

        for _ in range(self.iterations):
            data_as_list = list(zip(training_data, labels))
            random.shuffle(data_as_list)
            training_data, labels = zip(*data_as_list)
            for input, label in zip(training_data, labels):
                input_noisy = self.noisy(input)
                prediction = self.output(input_noisy)
                err = label - prediction

                if err == 0:
                    life_of_weight += 1
                else:
                    if life_of_weight > longest_lifetime:
                        best_set_of_weights = np.copy(self.weights)
                        longest_lifetime = life_of_weight
                    life_of_weight = 0
                    self.weights[1:] += err * self.learning_rate * input_noisy
                    self.weights[0] += err * self.learning_rate

        if life_of_weight > longest_lifetime:
            best_set_of_weights = np.copy(self.weights)

        self.weights = np.copy(best_set_of_weights)
        print(self.weights)

    def train_rpla(self, training_data, labels):  # ZD4 - RPLA
        best_set_of_weights = np.copy(self.weights)
        longest_lifetime = 0
        life_of_weight = 0
        best_number_of_correct_predictions = 0

        for _ in range(self.iterations):
            data_as_list = list(zip(training_data, labels))
            random.shuffle(data_as_list)
            training_data, labels = zip(*data_as_list)
            for input, label in zip(training_data, labels):
                input_noisy = self.noisy(input)
                prediction = self.output(input_noisy)
                err = label - prediction

                if err == 0:
                    life_of_weight += 1

                else:
                    correct_predictions = self.check_number_of_correct_predictions(data_as_list)
                    if life_of_weight > longest_lifetime and correct_predictions > best_number_of_correct_predictions:
                        best_set_of_weights = np.copy(self.weights)
                        longest_lifetime = life_of_weight
                        best_number_of_correct_predictions = correct_predictions
                    life_of_weight = 0
                    self.weights[1:] += err * self.learning_rate * input_noisy
                    self.weights[0] += err * self.learning_rate

        if life_of_weight > longest_lifetime:
            best_set_of_weights = np.copy(self.weights)

        self.weights = np.copy(best_set_of_weights)
        print(self.weights)

    def output(self, input):
        summation = np.dot(self.weights[1:], input) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def check_number_of_correct_predictions(self, data_as_list):
        # sprawdzanie ilosci poprawnych przykladów
        result = 0
        for input, label in data_as_list:
            prediction = self.output(input)
            err = label - prediction
            if err != 0:
                result += 1
        return result

    def check_predictions(self, data_as_list):  # ZD3 - warunek stopu
        # sprawdzanie czy wszystkie przyklady sa poprawne
        for input, label in data_as_list:
            prediction = self.output(input)
            err = label - prediction
            if err != 0:
                return False
        return True

    def noisy(self, input):
        input_copy = np.copy(input)
        for i in range(len(input_copy)):
            if random.random() < 0.01:
                input_copy[i] = input_copy[i] * (-1) + 1
        return input_copy

