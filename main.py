import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Reshape (Flatten) the data from (60000, 28, 28) to (60000, 784)
train_data_reshaped = train_data.reshape(60000, 28 * 28)
test_data_reshaped = test_data.reshape(10000, 28 * 28)

# Apply MinMaxScaler
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data_reshaped)
test_data_norm = scaler.transform(test_data_reshaped)

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the model
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model_4.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])

# Train the model
history_4 = model_4.fit(train_data_norm, train_labels, epochs=20, validation_data=(test_data_norm, test_labels))

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Save the trained model
model_4.save("fashionMNIST.h5")
np.savez("fashion_mnist_data.npz", test_data_norm=test_data_norm, test_labels=test_labels)

print("Model and data saved successfully!")
