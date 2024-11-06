from neural_net import network
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np

network1 = network("network1")
network1.set_learning_rate(0.01)
digits = load_digits()
x = digits.data
y = digits.target

x_train, y_train, x_test, y_test = network.test_train_split((x, y))
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

data = [x_train_scaled, y_train]
print("x_trained_scaled:", x_train_scaled[0])
print("y_train:", y_train[0])
network1.data_load(data)
network1.easy_startup([64, 32], 728, 10)
print(network1.network_structure())

network1.train_network(3)
print(network1.network_structure())

network1.save_parameters("network_params.pkl")

