# Code Explanation
Using only numpy in Python, a neural network with a forward and backward method is used to classify given points (x1, x2) to a color of red or blue. The neural network is initially assigned random weights and biases, which cause it to make terrible predictions. Through the use of gradient descent with backpropagation, the network learns a function that classifies most of the points correctly, resulting in better predictions by the model. Note that not all outcomes will be good due to random weight assignment when initializing the model.



# Example Graphics
While the network is trained, graphics are displayed to show the current model's loss on the training set as well as predictions vs. real data points on the training set (Graphic 1). When the model is finished training, more graphics appear to show the model's loss on the test set as well as predictions vs. real data points on the test set (Graphic 2).
![Alt text](/imgs/Graphic 1.PNG?raw=true "Graphic 1")
![Alt text](/imgs/Graphic 2.PNG?raw=true "Graphic 2")



# Requirements To Run Code
Here are the necessary libraries to run the code:
- numpy
- scipy
- sklearn
- matplotlib
Note: I tested the code using Python 3.8.
