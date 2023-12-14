# TensorFlow Workshop: Basic Classification with TensorFlow

## Introduction
Welcome to the TensorFlow workshop! This workshop is designed to give you a
basic understanding of machine learning in TensorFlow.

In this workshop, you will build and train a neural network model to classify
images of clothing items. The model will be trained using the Fashion MNIST.

## Requirements
Before starting the workshop, ensure you have the following installed:
- Python 3.7 or later
- NumPy
- Matplotlib
- TensorFlow 2.x

```sh
pip install tensorflow
```

## Exercises

### Exercise 0: Import TensorFlow and other libraries

First, we will need to import TensorFlow.
```python
import tensorflow as tf
```

Next, we will import other libraries that we will use throughout the workshop.
```python
import numpy as np
import matplotlib.pyplot as plt
```

Verify that TensorFlow is installed and working by running the following code:
```python
print(tf.__version__)
```

### Exercise 1: Import the Fashion MNIST dataset

For this workshop, we will be using the Fashion MNIST dataset. This dataset
contains 70,000 grayscale images of clothing items in 10 categories. Each image
is 28x28 pixels in size.

In this case, 60,000 images will be used to train the model, and 10,000 images
will be used to evaluate the model.

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The
labels are an array of integers, ranging from 0 to 9. Each integer corresponds
to a class of clothing.

| Label | Class |
| ----- | ----- |
| 0     | T-shirt/top |
| 1     | Trouser |
| 2     | Pullover |
| 3     | Dress |
| 4     | Coat |
| 5     | Sandal |
| 6     | Shirt |
| 7     | Sneaker |
| 8     | Bag |
| 9     | Ankle boot |


TODO: Create an array of class names.
```python
class_names = [???]
```


### Exercise 2: Preprocess the data

Before training the model, we will need to preprocess the data. First, we will
inspect the first image in the training set.

```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

TODO: As you can see, the pixel values range from 0 to 255. We will need to scale
these values to a range of 0 to 1 before feeding them to the neural network
model. To do this, we will divide the values by 255. It is important that the
training set and the testing set are preprocessed in the same way.
```python
train_images = ???
test_images = ???
```


We can verify that the data is in the correct format by displaying some images
from the training set along with their class names.

```python
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

### Exercise 3: Build the model

Now, we will build the neural network model:

- The first layer transforms the format of the images from a 2d-array (of
    28x28 pixels) to a 1d-array (of 28x28=784 pixels). This layer has no parameters to learn; it only reformats the data.
- The second layer is a densely connected layer of 128 neurons.
- The third layer is a 10-node softmax layer. Each node represents a class of
    clothing. The output of this layer is an array of 10 probability scores that
    sum to 1. Each score represents the probability that the current image
    belongs to one of the 10 classes.

TODO: Create the model.
```python
model = ???
```

After the model is created, we will compile it. When compiling the model, we
will need to specify the following:
- Loss function: This measures how accurate the model is during training. The
    goal is to minimize this function. In this case, we will use
    " Sparse Categorical Crossentropy".
- Optimizer: This is how the model is updated based on the data it sees and its
    loss function. In this case, we will use "Adam".
- Metrics: Used to monitor the training and testing steps. In this case, we will
    use "accuracy".

TODO: Compile the model.
```python
model.compile(???)
```


### Exercise 4: Train the model

Now that the model is built, we will train it using the training set. To do so,
we will need to do the following:

TODO: Feed the training data to the model. 10 epochs is sufficient for this model.

As the model trains, the loss and accuracy metrics will be displayed.


### Exercise 5: Evaluate accuracy

With the trained model, we can now evaluate it using the test set.
In order to do so, we will need to attach a softmax layer to convert the output
to probabilities.

TODO: Attach a softmax layer to the model.
```python
probability_model = tf.keras.Sequential(???)
```

Then, we will use the model to predict the class of each image in the test set.
```python
predictions = probability_model.predict(test_images)
```

The `predictions` array represents the predicted probability of each class for
each image in the test set.

We can see which label has the highest confidence value for the first image in
the test set.
```python
np.argmax(predictions[0])
```

Finally, we can verify that the prediction is correct by comparing it to the
label of the first image in the test set.
```python
test_labels[0] == np.argmax(predictions[0])
```

### Exercise 6: Going further

At this point, you have a basic understanding of machine learning in TensorFlow.

TODO: Try using the model to make predictions about a single image.

TODO: Try experimenting with different models and parameters to see how they
affect the accuracy of the model.


## Conclusion
By the end of this workshop, you will have a basic understanding of machine
learning and TensorFlow, and you will have built and trained a simple neural
network model. For further learning, explore the additional resources provided.
