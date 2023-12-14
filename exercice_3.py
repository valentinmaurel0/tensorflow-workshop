# Exercise 0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# Exercise 1
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Exercise 2
train_images = train_images / 255.0

test_images = test_images / 255.0


# Exercise 3
model = ??? # TODO: Build the model

model.compile(???) # TODO: Compile the model
