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

def compute_cost(A2, Y):

    m = Y.shape[1] 
    
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = - np.sum(logprobs) / m

    cost = float(np.squeeze(cost))  # ensure cost is scalar
    return cost

A2, t_Y = compute_cost_test_case()
cost = compute_cost(A2, t_Y)
print("cost = " + str(cost))

compute_cost_test(compute_cost)
