from PIL import Image
import numpy as np
import csv
import random
import os
class neuron:
    def __init__(self, activation_level):
        self.__activation_level = activation_level
        self.__weights = []
        self.__bias = []

    def activation_level_getter(self):
        return self.__activation_level

    def activation_level_setter(self, new_level):
        self.__activation_level = new_level

    def weights_getter(self):
        return self.__weights

    def weights_setter(self, new_weight):
        self.__weights = new_weight

    def bias_getter(self):
        return self.__bias

    def bias_setter(self, new_bias):
        self.__bias = new_bias

    def calc_new_activation(self, inputs):
        new_level = (np.matmul(inputs, self.weights_getter()).clip(-500, 500))
        new_level = (1/(1+np.exp(-new_level)))
        #print("calc_new_activation", new_level)
        return new_level
class layer:
    def __init__(self, name):
        self.__layer_name = name
        self.__neurons = []

    def layer_name_getter(self):
        return self.__layer_name

    def neuron_getter(self):
        return self.__neurons

class network:

    learning_rate = 0.3

    def __init__(self, network_name):
        self.__network_name = network_name
        self.__layers = []
        self.__data_set = [[], []]

    def data_load(self, new_dataset):
        self.__data_set = new_dataset

    def data_getter(self):
        return self.__data_set

    def input_getter(self):
        return self.__data_set[0]

    def input_setter(self, new_input):
        self.__data_set[0] = new_input

    def output_getter(self):
        return self.__data_set[1]

    def output_setter(self, new_output):
        self.__data_set[1] = new_output

    def network_name_getter(self):
        return self.__network_name

    def layers_getter(self):
        return self.__layers

    def layers_adder(self, new_layer):
        self.__layers.append(new_layer)

    def network_startup(self, input_neurons:type = int, output_neurons:type = int, between_layers_number:type = int, neurons_per_layer:type = list[int]):
        all_neurons_number = [0, input_neurons]
        for number in neurons_per_layer:
            all_neurons_number.append(number)
        all_neurons_number.append(output_neurons)
        print(all_neurons_number)
        neurons_per_layer = all_neurons_number
        for x1 in range(1, len(neurons_per_layer)):
            new_layer = layer(self.network_name_getter()+"_layer"+str(x1))
            for x11 in range(0, neurons_per_layer[x1]):
                new_neuron1 = neuron(0)
                new_layer.neuron_getter().append(new_neuron1)
                if x1 == 1:
                    weights = [1]
                    bias = [0]
                else:
                    weights = [2]*neurons_per_layer[x1-1]
                    bias = [0]*neurons_per_layer[x1-1]
                new_neuron1.weights_setter(weights)
                new_neuron1.bias_setter(bias)
            self.layers_adder(new_layer)

    def easy_startup(self, neurons_per_layer: type = list[int]):
        self.input_setter(self.inputs_standardizer())
        self.output_setter(self.outputs_to_vector())
        input_neurons = len(self.input_getter())
        output_neurons = len(self.output_getter()[0])
        between_layers = len(neurons_per_layer)
        self.network_startup(input_neurons, output_neurons, between_layers, neurons_per_layer)

    def network_structure(self):
        for layers in self.layers_getter():
            print(layers.layer_name_getter())
            for neurons in layers.neuron_getter():
                print(neurons.activation_level_getter(), neurons.weights_getter(), neurons.bias_getter())

    @staticmethod
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return network.quicksort(left) + middle + network.quicksort(right)


    @staticmethod
    def csv_to_array(csv_file):
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            inputs_array = []
            outputs_array = []
            for line in csv_reader:
                input = line[0:len(line)-1]
                input = [float(inputs) for inputs in input]
                output = line[len(line)-1:]
                output = [int(outputs) for outputs in output]
                inputs_array.append(input)
                outputs_array.append(output)
            return [inputs_array, outputs_array]

    def outputs_to_vector(self):
        outputs_as_a_vector = []
        outputs = self.output_getter()
        op_dict = {}
        for op in outputs:
            op_dict[op[0]] = 'placeholder'
        for op in outputs:
            new_vector = [0]*len(op_dict.keys())
            new_vector[op[0]] = 1
            outputs_as_a_vector.append(new_vector)
        return outputs_as_a_vector

    def inputs_standardizer(self):
        features = len(self.data_getter()[0][0])
        n = len(self.data_getter()[0])
        inputs = self.data_getter()[0]
        means = [0]*features
        std = [0]*features
        for rows in range(0, n):
            for cols in range(0, features):
                means[cols] += inputs[rows][cols]
        means = [mean/n for mean in means]
        for rows in range(0, n):
            for cols in range(0, features):
                std[cols] += np.square(inputs[rows][cols]-means[cols])
        std = [np.sqrt(x/(n-1))for x in std]
        for rows in range(0, n):
            for cols in range(0, features):
                inputs[rows][cols] = (inputs[rows][cols]-means[cols])/std[cols]
        return inputs

    def back_propagation(self):
        layers = self.layers_getter()
        gradients = []
        for layer_number in range(len(layers)):
            current_layer = len(layers) - layer_number
            for neuron_number in range(len(layers[current_layer].neuron_getter())):
                pass





    def forward_pass(self, input):
        layers = self.layers_getter()
        for layer_number in range(len(layers)):
            current_layer = layers[layer_number]
            if layer_number == 0:
                for neuron, input_value in zip(current_layer.neuron_getter(), input):
                    neuron.activation_level_setter(neuron.calc_new_activation([input_value]))
            else:
                previous_layer = layers[layer_number-1]
                activation_levels = [neuron.activation_level_getter() for neuron in previous_layer.neuron_getter()]
                for neuron in current_layer.neuron_getter():
                    neuron.activation_level_setter(neuron.calc_new_activation(activation_levels))

        final_layer = layers[-1]
        output = [neuron.activation_level_getter() for neuron in final_layer.neuron_getter()]
        return output


    def find_cost(self, input):
        cost = self.forward_pass(input)



    def train_network(self):
        input = self.input_getter()
        output = self.output_getter()
        cost_sum = 0
        for record in zip(input, output):
            prediction = self.forward_pass(record[0])
            actual = record[1]
            cost = np.square(np.subtract(prediction, actual))
            cost_sum += cost
            print(cost)
        cost_sum /= len(input)
        print(cost_sum)





network1 = network("network1")

network1.data_load(network.csv_to_array('test_data3.csv'))
network1.outputs_to_vector()
network1.inputs_standardizer()
network1.easy_startup([8, 5])

#print(network1.network_structure())

network1.train_network()
