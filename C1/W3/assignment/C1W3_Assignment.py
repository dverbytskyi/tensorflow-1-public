import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the data

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz")

# Get only training set
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

def reshape_and_normalize(images):
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = images.reshape(-1, images.shape[1], images.shape[2], 1)

    # Normalize pixel values
    images = np.divide(images, 255.0)

    ### END CODE HERE

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

"""
Maximum pixel value after normalization: 1.0
Shape of training set after reshaping: (60000, 28, 28, 1)
Shape of one image after reshaping: (28, 28, 1)
"""


# GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("Reached 99.5% accuracy so cancelling training!")

            self.model.stop_training = True

### END CODE HERE

def convolutional_model():
    ### START CODE HERE

    # Define the model
    model = tf.keras.models.Sequential([
        # Add convolutions and max pooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Add the same layers as before
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')

    ])
    """
    - A Conv2D layer with 32 filters, a kernel_size of 3x3, ReLU activation function and an input shape that matches that of every image in the training set
   - A MaxPooling2D layer with a pool_size of 2x2
   - A Flatten layer with no arguments
   - A Dense layer with 128 units and ReLU activation function
   - A Dense layer with 10 units and softmax activation function
    """
    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Save your untrained model
model = convolutional_model()

# Get number of weights
model_params = model.count_params()

# Unit test to limit the size of the model
assert model_params < 1000000, (
    f'Your model has {model_params:,} params. For successful grading, please keep it ' 
    f'under 1,000,000 by reducing the number of units in your Conv2D and/or Dense layers.'
)

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
"""
Epoch 1/10
1875/1875 [==============================] - 44s 23ms/step - loss: 0.1454 - accuracy: 0.9557
Epoch 2/10
1875/1875 [==============================] - 43s 23ms/step - loss: 0.0452 - accuracy: 0.9858
Epoch 3/10
1875/1875 [==============================] - 44s 23ms/step - loss: 0.0321 - accuracy: 0.9899
Epoch 4/10
1875/1875 [==============================] - 45s 24ms/step - loss: 0.0238 - accuracy: 0.9925
Epoch 5/10
1875/1875 [==============================] - 45s 24ms/step - loss: 0.0181 - accuracy: 0.99411s - l
Epoch 6/10
1873/1875 [============================>.] - ETA: 0s - loss: 0.0145 - accuracy: 0.9953Reached 99.5% accuracy so cancelling training!
1875/1875 [==============================] - 45s 24ms/step - loss: 0.0145 - accuracy: 0.9953
"""


print(f"Your model was trained for {len(history.epoch)} epochs")
# Your model was trained for 6 epochs