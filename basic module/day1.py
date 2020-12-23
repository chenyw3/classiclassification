import torch
from torch.autograd import Variable
import numpy as np
# input array
X = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])

# output array
y = np.array([[1], [1], [0]])

# sigmoid Function


def sigmoid(x):
    return 1/(1+(np.exp(-x)))

# Derivative of sigmoid function


def derivative_sigmoid(x):
    return x*(1-x)


# Variable initialization
epoch = 5000  # setting training iterations
lr = 0.1  # setting learning rate
inputlayer_neurons = X.shape[1]  # number of features in dataset
hiddenlayer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of output layers neurons
# weights and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):

    # forward propagation
    hidden_layer_input1 = np.dot(X, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    # backward propagation
    E = y - output
    slope_output_layer = derivative_sigmoid(output)
    slope_hidden_layer = derivative_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_output, axis=0, keepdims=True) * lr


print('actual=\n', y, '\n')
print('predict=\n', output)
