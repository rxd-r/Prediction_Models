import csv
import numpy as np

class data:
    def __init__(self, data_name:type = str, data_class:type = int, data_values:type = list):
        self.__data_name = data_name #private string var
        self.__data_class = data_class #private int var
        self.__data_values = data_values #private array var
    #getters and setters
    def get_name(self):
        return self.__data_name

    def set_name(self, new_name):
        self.__data_name = new_name

    def get_data_values(self):
        return self.__data_values

    def get_class(self):
        return self.__data_class

class log_graph:

    learning_rate = 0.3

    def __init__(self, name):
        self.__graph_name = name
        self.__training_data = []
        self.__weights = []

    def get_graph_name(self):
        return self.__graph_name

    def get_weights(self):
        return self.__weights

    def set_weights(self, new_weights):
        self.__weights = new_weights

    def get_training_data(self):
        return self.__training_data

    def init_weights(self):
        new_weights = []
        for i in self.get_training_data()[0].get_data_values():
            new_weights.append(0)
        self.__weights = new_weights
        print("weights, initialized", self.get_weights())

    def standard_scaler(self): #scales every vairable in every dot to mean = 0 and SD = 1
        features = len(self.get_training_data()[0].get_data_values())
        n = len(self.get_training_data())
        means = [0]*features
        std_devs = [0]*features
        for dot in self.get_training_data():
            for cols in range(features):
                means[cols] += dot.get_data_values()[cols]
        means = [(mean / n) for mean in means] #dividing all values by n
        for dot in self.get_training_data():
            for cols in range(features):
                std_devs[cols] +=   (dot.get_data_values()[cols] - means[cols]) ** 2 #Varaince formula
        std_devs = [(std_dev / (len(self.get_training_data()) - 1))**0.5 for std_dev in std_devs]

        for idx, dot in enumerate(self.get_training_data()):
            for idx1 in range(features):
                 self.get_training_data()[idx].get_data_values()[idx1] = ((dot.get_data_values()[idx1] - means[idx1])/ std_devs[idx1]) #changing each data point to new, standardized data

    def calculate_cost(self, dot): #cost function (wasnt used yet)
        p = np.clip(np.matmul(self.get_weights(), dot.get_data_values()), -500,500) #clip so it doesnt get too big
        odds = 1/(1+(np.exp(-p)))
        target = dot.get_class()
        cost = -target*np.log(odds) - (1-target)*np.log(1-odds) #cost function
        return cost

    def cost_for_n(self):
        cost_for_n = 0 #initialize variable string
        for idx, dot in enumerate(self.get_training_data()):
            cost_for_n += log_graph.calculate_cost(self, dot)
        cost_for_n /= len(self.get_training_data())
        return cost_for_n

    def gradiant_desc(self): #finds how wrong model is and change weight accordingly
        new_weights = []
        for idx1, weights in enumerate(self.get_weights()):
            gradiant = 0
            for dot in self.get_training_data():
                p = np.clip(np.matmul(self.get_weights(), dot.get_data_values()), -500, 500)
                odds = 1 / (1 + (np.exp(-p)))
                error = odds - dot.get_class()
                gradiant += error * dot.get_data_values()[idx1]
            gradiant /= len(self.get_training_data()) #averages the wrongness gradient sum
            new_weight = self.get_weights()[idx1] - log_graph.learning_rate * gradiant #gradient descent update rule
            new_weights.append(new_weight)
        self.set_weights(new_weights)

    def auto_descent(self, change_threshold):  # descent into my madness
        slope_change = 1
        while slope_change > change_threshold:
            old_weights = self.get_weights()
            self.gradiant_desc()
            for idx, weights in enumerate(self.get_weights()):
                weight_changes_list = []
                if old_weights[idx] != 0:
                    weight_changes = ((self.get_weights()[idx] - old_weights[idx])/old_weights[idx])*100
                    weight_changes_list.append(weight_changes)
                    slope_change = max(weight_changes_list)
                print(self.get_weights())
    @staticmethod
    def csv_to_graph(csv_file:type = str): #initiates a graph and its data
        new_graph = log_graph(csv_file)
        with open(csv_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader:
                new_dot_name = line[0]
                new_dot_class = float(line[1].strip('')) #strips class which should be 1 or 0
                new_data = []
                for idx in range(2, len(line)):
                    new_data.append(float(line[idx]))
                new_dot = data(new_dot_name, new_dot_class, new_data) #initializes dot
                new_graph.get_training_data().append(new_dot)
        return new_graph

graph2 = log_graph.csv_to_graph('test_data2.csv')
for i in graph2.get_training_data():
    print(data.get_data_values(i))

graph2.standard_scaler()
for i in graph2.get_training_data():
    print(data.get_data_values(i))

graph2.init_weights()
graph2.auto_descent(0.00000000000000000000000000000001)

