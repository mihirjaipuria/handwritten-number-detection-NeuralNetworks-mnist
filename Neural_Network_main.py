import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to a range of 0 to 1
X_train = X_train / 255
X_test = X_test / 255

# Reshape the input data to 1D arrays
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(150, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate the model on the test set
model.evaluate(X_test_flattened, y_test)

# Make predictions on a test sample
y_predicted = model.predict(X_test_flattened)
plt.matshow(X_test[1])  # Display the image corresponding to the test sample
np.argmax(y_predicted[1])  # Print the predicted label for the test sample

# Compute the predicted labels for all test samples
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Create a confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
