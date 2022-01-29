import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
from scipy import special
import matplotlib.gridspec as gridspec



# A fully connected layer for the neural network to use
class FullyConnected:
    # Initializes a fully connected layer
    # Parameters:
    #   numInputs - Number of inputs from previous layer
    #   numNodes - Number of nodes in this layer
    #   activation - The activation function to use
    def __init__(self, numInputs, numNodes, activation):
        # Initialize the number of inputs, number of
        # nodes, and activation function
        self.numInputs = numInputs
        self.numNodes = numNodes
        self.activation = activation

        # Initialize the weights to random values.
        # The matrix will be of size:
        #      [self.numNodes, self.numInputs]
        # The values will be between -1 and 1.
        self.weights = np.random.normal(-1, 1,\
                            (self.numNodes, self.numInputs))

        # Initialize the biases to random values
        # The vector will be of size [self.numNodes] as
        # there is 1 bias per node. The value will be between
        # -1 and 1.
        self.biases = np.random.normal(-1, 1, self.numNodes)
    
    # Given a set of inputs, returns the value of the
    #   feed forward method
    # Parameters:
    #   inputs - An array of inputs of size
    #            [self.numInputs, batchSize]
    # Outputs:
    #   The output from the layer
    #   The output from the layer before going through
    #      the activation function
    def forward(self, inputs):
        # If "self.numInputs" number of inputs was not supplied,
        # return False
        if np.array(inputs).shape[0] != self.numInputs:
            raise NameError("Inputs not correct shape")


        # Get the value of each node by taking the dot of
        # each input and it's weights. The result should be a
        # vector of size [self.numNodes, batchSize]
        z = np.dot(inputs.T, self.weights.T)

        # Add the biases to each node output
        z = (np.array(z)+self.biases).T
        
        # Send each node value through the activation function if
        # the activation function is specified
        # Return the output and z value to be stored in the cache.
        if self.activation == "relu":
            return np.where(z > 0, z, z*0.05), z
        elif self.activation == "softmax":
            return special.softmax(z, axis=-1), z
        elif self.activation == "sigmoid":
            return 1/(1 + np.exp(-1*z)), z
        else:
            return z, z



class NeuralNetwork():
    # Initializes a neural network
    # Parameters:
    #   numInputs - number of inputs
    #   numLayers - number of layers
    #   layerSizes - size of each layer: vector of size
    #                [number of layers]
    #   layerActivations - activation for each layer:
    #                      vector of size [number of layers]
    def __init__(self, numInputs, numLayers, layerSizes, \
                 layerActivations):
        # Initialize the number of inputs and number of layers
        self.numInputs = numInputs
        self.numLayers = numLayers

        # Initialize an array to hold each layer object
        self.layers = []

        # Create "numLayers" new layers
        for i in range(0, self.numLayers):
            self.layers.append(FullyConnected(numInputs, \
                               layerSizes[i], layerActivations[i]))
            numInputs = layerSizes[i]
    

    # Given a set of inputs, returns the value of the feedforward
    # method by sending the values through each layer's feedforward
    # method.
    # Parameters:
    #   inputs - An array of inputs of size
    #            [self.numInputs, batchSize]
    # Outputs:
    #   inputs - An array containing an output from each output
    #            node.
    #   cache - A dictionary containing cached values from each
    #           layer in the feedforward method.
    def forward(self, inputs):
        # This cache holds calculated values from each layer
        # to be used in backpropagation
        cache = dict()

        # Iterate over every layer in the network and send the
        # inputs (or outputs from the previous layer) through
        # the forward pass for each layer. Add each outputs
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
    

    # Given the cache and other parameters, this method calculates
    # the derivatives needed to perform one step of backproagation.
    # As the derivatives are calculated, the weights and biases
    # are updated
    # Parameters:
    #   cache - Values from the forward pass which will be needed to
    #           calculate the gradients and update the weights
    #           and biases
    #   X_train - Inputs values used to train the model
    #   y_train - Output values used to train the model
    #   alpha - The learning rate which is the rate at which the
    #           derivatives effect the weights and biases.
    # Outputs:
    #   cache - Values from the forward pass as well as values
    #           from the backward pass.
    def backward(self, cache, X_train, y_train, alpha):
        # Starting at the loss function, calculate the partial
        # derivatveof the loss function
        cache["da" + str(self.numLayers)] = (-y_train/cache["a" + \
                     str(self.numLayers-1)])+((1-y_train)/(1-\
                     cache["a" + str(self.numLayers-1)]))

        # Iterate through each layer starting with the last one
        # and calculate the partial derivatives needed to find 
        # the gradeints for the weights and biases
        for i in reversed(range(0, self.numLayers)):
            # Calculate the derivative of the activation function
            # in terms of z

            # If the activation function is relu, then the
            # derivative is 1 if the output of the layer before the
            # activation function (zi) is greater than 0 or the
            # derivative is 0 if the output of the layer before the
            # activation function (zi) is less than or equal to 0. 
            if self.layers[i].activation == "relu":
                cache["dz" + str(i)] = np.where(cache["z" +\
                                            str(i)] <= 0, -0.05, 1.)
            # If the activation function is sigmoid, then the
            # derivative is given by the following formula:
            # (phat(1-phat))
            # where phat are the outputs (or predictions) for that
            # layer
            elif self.layers[i].activation == "sigmoid":
                cache["dz" + str(i)] = cache["a" + str(i)]*(1 - \
                                                cache["a" + str(i)])
            # If the activation function is nothing, then there is
            # no derivative to be taken, so it can be represented
            # as 1 for each output in the batch
            else:
                cache["dz" + str(i)] = np.ones(cache["z" + \
                                                     str(i)].shape)
            


            # Multiply the current dz value by the activation
            # derivative from the proceeding layer if the layer
            # is the last layer
            if i == self.numLayers-1:
                cache["dz" + str(i)] = cache["da" + str(i+1)]*\
                                       cache["dz" + str(i)]
            # If the layer is a hidden layer, multiply the current
            # dx value by the proceeding layer's weight and
            # bias derivatives to continue the chain rule.
            else:
                # Change the shape of dz to be (numNodes(i), 1, m)
                cache["dz" + str(i)] = cache["dz" + str(i)].\
                        reshape(self.layers[i].numNodes, 1,\
                        X_train.shape[0])
                # Update dz to continue the chain rule.
                cache["dz" + str(i)] = np.sum(cache["dweights" +\
                      str(i+1)], axis=-2).reshape(cache["dz" + \
                      str(i)].shape) * cache["dbiases" +\
                      str(i+1)] * cache["dz" + str(i)]





            # Now, calculate the derivatives for the weights, 
            # and biaseses. Rememebr to multiply all of these
            # values by the z value derivative to complete the
            # chain rule.


            # Update the shapes of a and dz before taking the
            # derivatives. The value do not change, this just
            # changes the shape of the tensor.
            cache["a" + str(i-1)] = cache["a" + str(i-1)].\
                  reshape(cache["a" + str(i-1)].shape[0], 1,\
                  X_train.shape[0])
            cache["dz" + str(i)] = cache["dz" + str(i)].\
                  reshape(self.layers[i].numNodes, 1,\
                          X_train.shape[0])

            # Get the derivative of the weights. The derivative of
            # the weights is the corresponding input values
            # (a(i-1))*(1/m) dot the previous 
            # derivative values. In this case, the previous
            # derivative dz: values are dweights = a(i-1)*dz
            cache["dweights" + str(i)] = (1/X_train.shape[0])*\
                  np.array([x * cache["dz" + str(i)] \
                    for x in cache["a" +\
                    str(i-1)]]).reshape(cache["a" + str(i-1)].\
                      shape[0], cache["dz" + str(i)].shape[0],\
                      X_train.shape[0])

            # The derivative of a bias is 1*(1/m), since the bias
            # is constant, dot the previous derivative values.
            # In this case, the previous derivative values are dz:
            # dbiases = 1*dz
            cache["dbiases" + str(i)] = 1*(1/X_train.shape[0])*\
                                        cache["dz" + str(i)]


            # Correct the weights and biases by changing all nan
            # values to 0 and keeping the values between -1 and 1
            # to avoid exploding gradients
            cache["dweights" + str(i)]  = np.nan_to_num(\
                  cache["dweights" + str(i)], nan=0.0)
            cache["dbiases" + str(i)]  = np.nan_to_num(\
                  cache["dbiases" + str(i)], nan=0.0)
            cache["dweights" + str(i)] = np.clip(\
                  cache["dweights" + str(i)], -1, 1)
            cache["dbiases" + str(i)] = np.clip(\
                  cache["dbiases" + str(i)], -1, 1)


            # Update the weights and biases for this layer
            self.layers[i].weights -= alpha*np.sum(\
                cache["dweights" + str(i)], axis=-1).T
            self.layers[i].biases -= alpha*np.sum(\
                np.sum(cache["dbiases" + str(i)], axis=-1), axis=-1)


        # Return the updated cache
        return cache
    





# The loss function for the neural network. Since the network
# is using a binary classifier, the loss function is
# binary cross entropy
# Parameters:
# - yhat: Predictions from the neural network
# - y: The values that the predictions should be
def binaryCrossEntropy(yhat, y):
    # Make sure the inputs are numpy arrays
    yhat = np.array(yhat)
    y = np.array(y)
    
    # Compute the reversed values
    yhat_rev = 1-yhat
    
    # Change values of 1 which yaht and 0 within yhat_rev slightly
    # to avoid the loss from becoming nan or infinity.
    yhat_rev = np.where(yhat_rev==0, yhat_rev+0.0000001, yhat_rev)

    # Compute the log terms
    log1 = y*np.log(yhat)
    log2 = (1-y)*np.log(yhat_rev)

    # Return the loss
    return -1*(1/y.shape[1])*(np.sum(log1 + log2))





def main():
    # The learning rate used to update the weights and biases
    alpha = 0.25

    # Get the data from a dataset
    data = sklearn.datasets.make_moons(noise=0.2, n_samples=200,\
                                      shuffle=False, random_state=0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(\
                                                data[0], data[1])


    # Create the neural network with:
    # - numInputs of size 2 (an x1 and x2 value)
    # - an output layer of size 1 (for the classification (yhat))
    # - 2 hidden layers
    # - 16 nodes per layer
    # - A relu activation function for each hidden layer
    # - A sigmoid activation functin for the output layer
    model = NeuralNetwork(2, 3, [16, 16, 1], ["relu", "relu", \
                                              "sigmoid"])

    
    # Holds all loss values and it's correpsonding step value
    # to plot the data
    loss_y = []
    loss_x = []


    # Setup the graphs and variables needed to graph the data.
    gs = gridspec.GridSpec(2, 2)
    plt.figure()
    loss_plot = plt.subplot(gs[0, :])
    real_plot = plt.subplot(gs[1, 0])
    pred_plot = plt.subplot(gs[1, 1])
    #fig, (loss_plot, real_plot, pred_plot) = plt.subplots(3)
    plt.ion()


    # For 100 iterations, feed forward the training data through
    # the network and update the weights using the gradients
    # by going backwards.
    for i in range(0, 20000):
        #####################
        #Forward Propagation#
        #####################

        # Feed forward the datapoints through the network to get
        # the predictions as well as a cache that holds values
        # calculated through the hidden layers
        predictions, cache = model.forward(X_train.T)

        # Get the loss for the current predictions
        cache["loss"] = binaryCrossEntropy(predictions, \
                                           np.array([y_train]))
        loss_y.append(cache["loss"])
        loss_x.append(i)
        if i%50 == 0:
            print("loss at step " + str(i) + " = " + \
                  str(cache["loss"]))

        # Round the predictions to a 0 or a 1
        predictions = np.round(predictions)



        ##################
        #Back Propagation#
        ##################
        cache = model.backward(cache, X_train, y_train, alpha)



        ###################
        #Plotting the Data#
        ###################
        if i%50 == 0:
            plt.cla()
            loss_plot.clear()
            real_plot.clear()
            pred_plot.clear()
            plt.tight_layout(pad=3)#, w_pad=0.5, h_pad=1.0)
            loss_plot.set(xlabel='step', ylabel='loss value')
            loss_plot.set_title("Loss = " + str(np.around(\
                cache["loss"], 5)))
            real_plot.set(xlabel='X', ylabel='y')
            real_plot.set_title("Real Data")
            pred_plot.set(xlabel='X', ylabel='y')
            pred_plot.set_title("Training Data Predictions")
            loss_plot.scatter(loss_x, loss_y)
            real_plot.scatter(X_train[:,0], X_train[:,1],\
                              c=y_train, cmap=ListedColormap(\
                                  ['#FF0000', '#0000FF']),\
                                  edgecolors='k')
            pred_plot.scatter(X_train[:,0], X_train[:,1],\
                              c=predictions, cmap=ListedColormap(\
                                  ['#FF0000', '#0000FF']),\
                                  edgecolors='k')
            plt.pause(0.0001)
            plt.show()
        


    # Puase to show the ending results for 10 seconds
    plt.pause(10)    

    # Make the final predictions on the test set.
    final_preds, _ = model.forward(X_test.T)

    # Print the loss value for the test set
    final_loss = binaryCrossEntropy(final_preds, np.array([y_test]))
    print("Final loss: " + str(final_loss))

    # Graph the test data
    fig, (real_plot, pred_plot) = plt.subplots(2)
    plt.tight_layout(pad=3)
    real_plot.set(xlabel='X', ylabel='y')
    real_plot.set_title("Real Data (Top) v. Predictions (Bottom)")
    pred_plot.set(xlabel='X', ylabel='y')
    pred_plot.set_title("Test Loss = " + str(final_loss))
    real_plot.scatter(X_test[:,0], X_test[:,1], c=y_test,
                      cmap=ListedColormap(['#FF0000', '#0000FF']))
    pred_plot.scatter(X_test[:,0], X_test[:,1], c=final_preds,
                      cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.show()
    plt.pause(30)





# Call the main function
main()
