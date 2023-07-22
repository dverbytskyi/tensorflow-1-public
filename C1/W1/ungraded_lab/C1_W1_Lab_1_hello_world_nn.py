import tensorflow as tf
import numpy as np  # helps to represent data as arrays easily and to optimize numerical operations
from tensorflow import keras

# Build a simple Sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
"""
1. Creating a Sequential Model: In Keras, a sequential model is a linear stack of layers. The 'Sequential' class 
allows you to create a neural network where each layer has exactly one input tensor and one output tensor. In this 
example, we use 'tf.keras.Sequential' to define our neural network.

2. Adding Layers to the Model: We define the model with a single Dense layer using the 'keras.layers.Dense' function. 
The 'Dense' layer is a standard fully connected layer in a neural network. It has one neuron in this case, 
and the 'units' parameter defines the number of neurons in the layer. The 'input_shape' parameter specifies the shape of 
the input data. Here, we have just one value as input, so the input shape is '[1]'.

3. Summary of the Model: After defining the model, you can use the 'model.summary()' method to get an overview of the 
model's architecture, including the number of trainable parameters in each layer.

Putting it all together, the code creates a simple neural network with a single neuron in a single Dense layer. This 
network can take one input value and produce an output based on the single neuron's computation.
"""
model.summary()
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 1)                 2         
=================================================================
Total params: 2 (8.00 Byte)
Trainable params: 2 (8.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Declare model inputs and outputs for training
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)
"""
Epoch 1/500
1/1 [==============================] - 0s 180ms/step - loss: 2.3879
Epoch 2/500
1/1 [==============================] - 0s 1ms/step - loss: 2.0275
...
Epoch 499/500
1/1 [==============================] - 0s 2ms/step - loss: 2.5063e-05
Epoch 500/500
1/1 [==============================] - 0s 1ms/step - loss: 2.4548e-05
"""

# Make a prediction
print(model.predict([10.0]))  # => [[18.985542]]
