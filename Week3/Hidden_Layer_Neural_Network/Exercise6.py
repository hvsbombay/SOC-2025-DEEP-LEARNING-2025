# Package imports
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils.testCases_v2 import *
from utils.public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from utils.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def backward_propagation(parameters, cache, X, Y):
   
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) 
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

parameters, cache, t_X, t_Y = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, t_X, t_Y)

print("dW1 = "+ str(grads["dW1"]))
print("db1 = "+ str(grads["db1"]))
print("dW2 = "+ str(grads["dW2"]))
print("db2 = "+ str(grads["db2"]))

backward_propagation_test(backward_propagation)
