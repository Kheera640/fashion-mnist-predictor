import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model("fashionMNIST.h5")

# Load Fashion MNIST dataset (only test data needed)
(_, _), (test_data, test_labels) = fashion_mnist.load_data()

# Preprocess data (reshape & normalize)
test_data_reshaped = test_data.reshape(10000, 28 * 28)
scaler = MinMaxScaler()
test_data_norm = scaler.fit_transform(test_data_reshaped)

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_random_image(model, images, true_labels, classes):
    """
    Plots a random image from the dataset along with its predicted label.
    """
    i = random.randint(0, len(images) - 1)
    target_image = images[i]

    pred_prob = model.predict(target_image.reshape(1, -1))  # Make a prediction
    pred_label = classes[pred_prob.argmax()]
    true_label = classes[true_labels[i]]

    plt.imshow(target_image.reshape(28, 28), cmap=plt.cm.binary)  # Reshape & display

    color = "green" if pred_label == true_label else "red"

    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(
        pred_label, 100 * tf.reduce_max(pred_prob), true_label),
        color=color)

    plt.show()

# Call the function to plot a random image
plot_random_image(model, test_data_norm, test_labels, class_names)
