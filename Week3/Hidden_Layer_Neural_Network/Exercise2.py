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

def layer_sizes(X, Y):
    # YOUR CODE STARTS HERE
    n_x = X.shape[0]   
    n_h = 4            # hidden layer size, fixed for this example
    n_y = Y.shape[0]   # number of output classes (output layer size)
    # YOUR CODE ENDS HERE
    return (n_x, n_h, n_y)

# Run the test
t_X, t_Y = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(t_X, t_Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

layer_sizes_test(layer_sizes)
