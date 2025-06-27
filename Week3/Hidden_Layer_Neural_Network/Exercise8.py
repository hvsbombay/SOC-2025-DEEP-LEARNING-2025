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

def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes_test_case(X, Y)[0]
    n_y = layer_sizes_test_case(X, Y)[2]
    
    parameters = initialize_parameters_test_case(n_x, n_h, n_y)
    
    for i in range(0,num_iterations):
        A2, cache = forward_propagation_test_case(X, parameters)
        cost = compute_cost_test_case(A2, Y)
        grads = backward_propagation_test_case(parameters, cache, X, Y)
        parameters = update_parameters_test_case(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters

nn_model_test(nn_model)