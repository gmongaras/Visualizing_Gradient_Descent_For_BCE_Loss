"""



Note: This file is used to test the forward method.



"""



import tensorflow as tf
from functools import partial
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import special


# A fully connected layer for the neural network to use
class FullyConnected:
    # Initializes a fully connected layer
    # Parameters:
    # - Number of inputs from previous layer
    # - Number of nodes in this layer
    # - The activation function to use
    def __init__(self, numInputs, numNodes, activation):
        # Initialize the number of inputs, number of nodes, and activation function
        self.numInputs = numInputs
        self.numNodes = numNodes
        self.activation = activation

        # Initialize the weights to random values.
        # The matrix will be of size: [self.numNodes, self.numInputs]
        # The values will be between -1 and 1.
        self.weights = np.random.uniform(-1, 1, (self.numNodes, self.numInputs))

        # Initialize the biases to random values
        # The vector will be of size [self.numNodes] as
        # there is 1 bias per node. The value will be between -1 and 1.
        self.biases = np.random.uniform(-1, 1, self.numNodes)
    
    # Given a set of inputs, returns the value of the feed forward method
    # Parameters:
    # - An array of inputs of size [self.numInputs, batchSize]
    # Output:
    # - The output from the layer
    # - The output from the layer before going through the activation function
    def forward(self, inputs):
        # If "self.numInputs" number of inputs was not supplied,
        # return False
        if np.array(inputs).shape[0] != self.numInputs:
            raise NameError("Inputs not correct shape")


        # Get the value of each node by taking the dot of
        # each input and it's weights. The result should be a
        # vector of size [self.numNodes, batchSize]
        #if len(np.array(inputs).shape) == 1: inputs = [inputs]

        #z = [np.matmul(inputs[:,x], self.weights.T) for x in range(0, np.array(inputs).shape[1])]
        z = np.dot(inputs.T, self.weights.T)

        # Add the biases to each node output
        z = np.array(z)+self.biases
        
        # Send each node value through the activation function if
        # the activation function is specified
        # Return the output and z value to be stored in the cache.
        if self.activation == "relu":
            #return np.maximum(z.T, 0), z.T
            return np.where(z.T > 0, z.T, z.T*0.01), z.T
        elif self.activation == "softmax":
            return special.softmax(z, axis=1), z.T
        elif self.activation == "sigmoid":
            return 1/(1 + np.exp(-1*z.T)), z.T
        else:
            return z.T, z.T



class NeuralNetwork():
    # Initializes a neural network
    # Parameters:
    # - number of inputs
    # - number of layers
    # - size of each layer: vector of size [number of layers]
    # - activation for each layer: vector of size [number of layers]
    def __init__(self, numInputs, numLayers, layerSizes, layerActivations):
        # Initialize the number of inputs and number of layers
        self.numInputs = numInputs
        self.numLayers = numLayers

        # Initialize an array to hold each layer object
        self.layers = []

        # Create "numLayers" new layers
        for i in range(0, self.numLayers):
            self.layers.append(FullyConnected(numInputs, layerSizes[i], layerActivations[i]))
            numInputs = layerSizes[i]
    

    # Given a set of inputs, returns the value of the feedforward method
    # By sending the values through each layer's feedforward method
    def forward(self, inputs):
        # This chache holds calculated values from each layer
        # to be used in backpropagation
        cache = dict()

        # Iterate over every layer in the network and send the
        # inputs (or outputs from the previous layer) through
        # the forward pass for each layer. Add each output
        # from each layer to the cache along the way
        for i in range(0, self.numLayers):
            # Add the inputs to the cache
            cache["a" + str(i-1)] = inputs

            # Get the output from the layers
            inputs, z = self.layers[i].forward(inputs)

            # Add the output and z-value to the cache
            cache["a" + str(i)] = inputs
            cache["z" + str(i)] = z
        
        # Return the output value
        return inputs, cache



data = sklearn.datasets.make_moons(noise=0.3, n_samples=100, shuffle=False, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    tf.keras.layers.Dense(16, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    tf.keras.layers.Dense(16, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    tf.keras.layers.Dense(16, activation=partial(tf.nn.leaky_relu, alpha=0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])



print(model(X_train).numpy())





model2 = NeuralNetwork(2, 5, [16, 16, 16, 16, 1], ["relu", "relu", "relu", "relu", "sigmoid"])
model2.layers[0].weights = np.array(model.layers[0].get_weights()[0]).T
model2.layers[0].biases = np.array(model.layers[0].get_weights()[1])
model2.layers[1].weights = np.array(model.layers[1].get_weights()[0]).T
model2.layers[1].biases = np.array(model.layers[1].get_weights()[1])
model2.layers[2].weights = np.array(model.layers[2].get_weights()[0]).T
model2.layers[2].biases = np.array(model.layers[2].get_weights()[1])
model2.layers[3].weights = np.array(model.layers[3].get_weights()[0]).T
model2.layers[3].biases = np.array(model.layers[3].get_weights()[1])
model2.layers[4].weights = np.array(model.layers[4].get_weights()[0]).T
model2.layers[4].biases = np.array(model.layers[4].get_weights()[1])


print(model2.forward(X_train.T)[0])