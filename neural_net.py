import random as rd
import numpy as np
import csv
import pickle
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
        # print("Inputs shape:", np.shape(inputs))
        # print("Weights shape:", np.shape(self.weights_getter()))
        # print("Bias value before addition:", self.bias_getter())
        new_level = (np.matmul(inputs, self.weights_getter()))
        # print(self.bias_getter())
        new_level += self.bias_getter()
        new_level = new_level.clip(-500, 500)
        # print(f"new level{new_level}")
        new_level = (1/(1+np.exp(-new_level)))
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

    def __init__(self, network_name):
        self.__network_name = network_name
        self.__layers = []
        self.__data_set = [[], []]
        self.__learning_rate = 0.1

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

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

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
                else:
                    weights = [rd.uniform(-1, 1)]*neurons_per_layer[x1-1]
                bias = rd.uniform(-1, 1)
                new_neuron1.weights_setter(weights)
                new_neuron1.bias_setter(bias)
            self.layers_adder(new_layer)

    def easy_startup(self, neurons_per_layer, input_size, output_size):
        self.input_setter(self.inputs_standardizer())
        self.output_setter(self.outputs_to_vector())
        between_layers = len(neurons_per_layer)
        self.network_startup(input_size, output_size, between_layers, neurons_per_layer)

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
        unique_classes = list(
            set(op[0] if isinstance(op, (list, np.ndarray)) else op for op in outputs)) # Extract unique class labels

        class_index_map = {label: idx for idx, label in enumerate(unique_classes)}
        print(class_index_map)

        for op in outputs:  # Check if op is a list or a scalar
            print("op:", op, type(op))
            if isinstance(op, (list, np.ndarray)) and len(op) > 0:
                class_label = op[0]
            elif isinstance(op, (np.int64, np.int32, np.float64, np.float32)):
                class_label = op
            else:
                print("Unexpected output format:", op)
                continue  # Skip this output if the format is unexpected

            # Initialize the one-hot vector
            new_vector = [0] * len(class_index_map)
            if class_label in class_index_map:
                new_vector[class_index_map[class_label]] = 1
            else:
                print(f"Class label {class_label} not found in mapping.")

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
        for i in range(len(std)):
            if std[i] == 0:
                std[i] = 1
        for rows in range(0, n):
            for cols in range(0, features):
                inputs[rows][cols] = (inputs[rows][cols]-means[cols])/std[cols]
        return inputs

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


    def back_propagation(self, target_output):
        layers = self.layers_getter()
        final_layer = layers[-1]
        # print("final layer :",final_layer.neuron_getter())
        output_activations = [neurons.activation_level_getter() for neurons in final_layer.neuron_getter()]

        output_error = np.subtract(output_activations, target_output)
        output_derivative = [
            neurons.activation_level_getter() * (1-neurons.activation_level_getter())
            for neurons in final_layer.neuron_getter()
        ]
        #print("output error", output_error)
        #print("output derivatives", output_derivative)
        #print(f"Target output length: {len(target_output)}")
        #print(f"Output activations length: {len(output_activations)}")
        deltas = [output_error[i]*output_derivative[i] for i in range(len(output_error))]

        for layers_idx in reversed(range(1, len(layers))):
            current_layer = layers[layers_idx]
            previous_layer = layers[layers_idx-1]

            current_weights = [neurons.weights_getter() for neurons in current_layer.neuron_getter()]
            previous_activations = [neurons.activation_level_getter() for neurons in current_layer.neuron_getter()]
            previous_derivatives = [neurons.activation_level_getter()*(1-neurons.activation_level_getter()) for neurons in previous_layer.neuron_getter()]

            for i, neurons in enumerate(current_layer.neuron_getter()):
                for j in range(len(neurons.weights_getter())):
                    neurons.weights_setter(
                        np.subtract(
                            neurons.weights_getter(), deltas[i] * previous_activations[i]
                        )
                    )
                new_bias = neurons.bias_getter() - self.__learning_rate * deltas[i]
                neurons.bias_setter(new_bias)
                #print(f"deltas{deltas[i]}")
            if layers_idx > 1:
                new_deltas = []
                for j in range(len(previous_layer.neuron_getter())):
                    cost = 0
                    for i in range(len(deltas)):
                        cost += sum([deltas[i] * current_weights[i][j]])
                    new_deltas.append(cost*previous_derivatives[j])
                deltas = new_deltas



    def train_network(self, epochs):
        inputs = self.input_getter()
        targets = self.output_getter()
        #print("targets", targets)

        for epoch in range(epochs):
            sum_cost = 0
            for record in zip(inputs, targets):
                input_vectors, target_outputs = record
                prediction = self.forward_pass(input_vectors)

                cost = np.square(np.subtract(prediction, target_outputs))
                sum_cost += np.sum(cost)

                self.back_propagation(target_outputs)
            avg_cost = sum_cost/len(inputs)
            print(f"Epoch {epoch+1}/{epochs} : Cost = {avg_cost}")

    def softmax(self, log_its):
        e_log_its = np.exp(log_its - np.max(log_its))
        return e_log_its / e_log_its.sum(axis=0)

    def predict(self, inputs):
        predictions = []
        probabilities = []

        if isinstance(inputs[0], list):
            for input_vector in inputs:
                output = self.forward_pass(input_vector)
                probability = self.softmax(output)
                predicted_class = (np.argmax(probability))
                predictions.append(predicted_class)
                probabilities.append(probability)
        else:  # Single input vector
            output = self.forward_pass(inputs)
            prob = self.softmax(output)  # Calculate probabilities
            predicted_class = np.argmax(prob)  # Get index of the maximum activation
            predictions.append(predicted_class)
            probabilities.append(prob)  # Store probabilities

        return predictions, probabilities

    @staticmethod
    def test_train_split(data):
        batch = 0.2
        inputs, targets = data
        train_amount = int((1 - batch) * len(inputs))

        x_train = inputs[:train_amount]
        y_train = targets[:train_amount]
        x_test = inputs[train_amount:]
        y_test = targets[train_amount:]

        return x_train, y_train, x_test, y_test

    def save_parameters(self, filename):
        parameters = {}

        for layer_idx, layer in enumerate(self.layers_getter()):
            layer_params = []
            for neuron in layer.neuron_getter():
                layer_params.append({
                    'weights': neuron.weights_getter(),
                    'bias': neuron.bias_getter()
                })
            parameters[f'layer_{layer_idx}'] = layer_params

        with open(filename, 'wb') as file:
            pickle.dump(parameters, file)
        print(f"Parameters saved to {filename}.")

    def load_parameters(self, filename):
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)

        for layer_idx, layer in enumerate(self.layers_getter()):
            layer_params = parameters.get(f'layer_{layer_idx}', [])
            for neuron_idx, neuron in enumerate(layer.neuron_getter()):
                if neuron_idx < len(layer_params):
                    neuron.weights_setter(layer_params[neuron_idx]['weights'])
                    neuron.bias_setter(layer_params[neuron_idx]['bias'])

        print(f"Parameters loaded from {filename}.")