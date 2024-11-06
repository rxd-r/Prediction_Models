import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("MNIST dataset loaded successfully!")

# Print the shape of the data
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

# Example: normalize and one-hot encode labels
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0  # Flatten and normalize
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0      # Flatten and normalize

# One-hot encode the labels
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

print("Data preprocessing completed.")
