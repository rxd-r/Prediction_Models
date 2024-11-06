from neural_net import network
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

network10 = network("network10")

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
network10.data_load(data)

network10.easy_startup([32, 16], 728, 10)
network10.load_parameters("network_params.pkl")
print(network10.network_structure())

network10.set_learning_rate(0.01)
network10.train_network(100)
print(network10.network_structure())

network10.save_parameters("network_params.pkl")