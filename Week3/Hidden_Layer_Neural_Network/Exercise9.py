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

def predict(parameters, X):
    A2, cache = forward_propagation_test(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

parameters, t_X = predict_test_case()

predictions = predict(parameters, t_X)
print("Predictions: " + str(predictions))

predict_test(predict)