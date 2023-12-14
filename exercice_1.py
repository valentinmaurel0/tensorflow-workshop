# Exercise 0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# Exercise 1
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [???] # TODO: Fill in the class names