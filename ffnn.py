from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import math
from sklearn.metrics import accuracy_score

'''
Building up a Feed-Forward Neural Network from scratch using the make_moons data set
Then using Backpropagation, the weights of the layers get updated and a plot with the evolution of the loss function is displayed
'''

X,y = make_moons(n_samples=50, noise=0.2, random_state=42)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ffnn (X, w0, w1):
    '''
    Feed-Forward Neural Network
    Input needed:
        X: data
        w0: weights of the first layer
        w1: weights of the second layer
    Output:
        ypred0: results from the first layer
        ypred: results from the final layer
    '''
    input_layer = np.hstack((X,np.ones((X.shape[0],1)))) # shape (50,3)
    weight_matrix_layer = w0 # shape (3,2)
    dot_product1 = np.dot(input_layer,weight_matrix_layer) # shape (50,2)
    hidden_layer =  np.hstack((sigmoid(dot_product1),np.ones((X.shape[0],1)))) # shape (50,3)
    weight_matrix_layer_2 = w1 # shape (3,1)
    dot_product2 = np.dot(hidden_layer,weight_matrix_layer_2) # shape (50,1)
    output_layer = sigmoid(dot_product2) # shape (50,1)
    ypred0 = hidden_layer # shape (50,3)
    ypred = output_layer # shape (50,1)

    return ypred0, ypred

def loss(ypred,ytrue):
    '''
    Loss function
    Needs prediction and true values
    Outputs a list with every data point loss
    '''
    log_loss = []
    for p,t in zip(ypred, ytrue):
        p = float(p)
        log_loss.append(-(t*math.log(p))+(1-t)*math.log(1-p))

    return np.array(log_loss)

def backprop(w0,w1,ypred0,ypred, X, y, LR_O, LR_H):
    '''
    This function replicate back propagation and update weights for a neural network with two layers
    Input needed:
        w0: weights of the first layer
        w1: weights of the second layer
        ypred0: results from the first layer
        ypred:results from the final layer
        X: Data
        y: Data labels
        LR_O and LR_H: Learning rates
    Output:
        updates weights
    '''
    # reshape original values and calculate the loss
    ytrue = y.reshape(50,)
    error = loss(ypred, ytrue)

    # derivative of the sigmoid function with respect to ypred * weights
    sig_deriv = (sigmoid(ypred) * ( 1 - sigmoid(ypred))).flatten()
    y_grad = sig_deriv * error

    delta_wo = np.dot(y_grad.transpose(), ypred0) * LR_O

    w0_new = w1 + delta_wo.reshape(3,1)

    return w0_new

# Intial random weights instansiation
w0 = np.random.random([3,2])
w1 = np.random.random([3,1])

# Predict with the initial random weights
ypred0 = ffnn(X,w0,w1)[0]
ypred = ffnn(X,w0,w1)[1]

# New weights with a randomnly chosen Learning rates
new_output_weight = backprop(w0,w1,ypred0,ypred,X, y,0.1, 0.01)

# Calculate new predictions with the recently obtain weights
new_ypred0, new_ypred = ffnn(X,w0,new_output_weight)

# Loop and store all calculated loss results
LOSS_VEC = []
for i in range(10000):
    ypred0, ypred = ffnn(X, w0, w1)
    LOSS_VEC.append(np.abs(loss(ypred,y)).sum())
    w1 = backprop(w0, w1, ypred0, ypred, X, y, 0.1, 0.01)


plt.plot(LOSS_VEC)
plt.title('Loss values over time')
