import numpy as np
import csv

class dp:
    def __init__(self, name, data, classification):
        self.name = name #private str var
        self.data = data #private list var
        self.clas = classification #private var

    def find_distance(self, graph):
        eu_distance_squared = []
        for idx, dots in enumerate(graph.training_data):
            points = [] #create an array for each point
            eu_distance_squared_sum = 0
            for idx1, data in enumerate(dots.data):
                distance = np.square(self.data[idx1]-graph.training_data[idx].data[idx1]) #distance of each variable squared
                eu_distance_squared_sum += distance
            points.append(graph.training_data[idx].name) #append point name
            points.append(eu_distance_squared_sum) #append distance
            points.append(graph.training_data[idx].clas) #append class
            eu_distance_squared.append(points)

        return eu_distance_squared

    @staticmethod
    def quicksort_2d(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2][1]
        left = [x for x in arr if x[1] < pivot]
        middle = [x for x in arr if x[1] == pivot]
        right = [x for x in arr if x[1] > pivot]
        return dp.quicksort_2d(left) + middle + dp.quicksort_2d(right)

    def choose_nearest_k(self, graph, k: type = int):
        distances_array = dp.find_distance(self, graph) #get distances array
        sorted_distances_array = dp.quicksort_2d(distances_array) #get sorted array
        print("sorted", sorted_distances_array)
        nearest_ns = []
        for i in range(0, k):
            nearest_ns.append(sorted_distances_array[i]) #choose nearest k
        return nearest_ns

class graph:

    def __init__(self, name, training_data):
        self._graph_name = name
        self.training_data = training_data

    @staticmethod
    def csv_to_graph(csv_file:type = str):
        new_graph = graph(csv_file, [])
        with open(csv_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for line in csv_reader:
                new_dot_name = line[0]
                new_dot_class = int(line[1].strip(''))
                new_data = []
                for idx in range(2, len(line)):
                    new_data.append(int(line[idx]))
                new_dot = dp(new_dot_name, new_data, new_dot_class)
                new_graph.training_data.append(new_dot)
        return new_graph

graph2 = graph.csv_to_graph('test_data.csv')
d10 = dp("d10", [11, 12, 13, 14, 15, 16], 1)
print(graph2.training_data)
print(dp.choose_nearest_k(d10, graph2, 3))







