# handwritten-number-detection-NeuralNetworks-mnist
The goal is to recognize handwritten numbers
MNIST Digit Classification with Neural Networks

This repository contains code for training and evaluating a neural network model on the MNIST dataset. The MNIST dataset consists of a large collection of 28x28 pixel grayscale images of handwritten digits (0-9). The goal is to build a model that can accurately classify these digits.

The code uses TensorFlow and Keras to construct a neural network with two dense layers. The first layer has 150 units and uses the ReLU activation function, while the second layer has 10 units and uses the sigmoid activation function. The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

The dataset is preprocessed by normalizing the pixel values to a range of 0 to 1. The images are then flattened into 1D arrays to serve as input to the neural network.

After training the model for 5 epochs, the code evaluates its performance on the test set and generates a confusion matrix using TensorFlow's built-in function. The confusion matrix provides insights into the model's classification accuracy for each digit class.

The code also includes an alternative implementation without any hidden layers, which can be uncommented for a simplified model.

Feel free to explore and experiment with the code to understand and improve upon the digit classification task using neural networks.

