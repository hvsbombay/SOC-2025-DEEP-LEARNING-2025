import numpy as np
import math

def numpy_sigmoid(x):
    # np.exp() works element-wise on numpy arrays or single floats
    s = 1 / (1 + np.exp(-x))
    return s

print(numpy_sigmoid(2))

def basic_sigmoid(x):
    # sigmoid(x) = 1 / (1 + e^(-x))
    s = 1 / (1 + math.exp(-x))
    return s

python_list = [1, 2, 3, 4, 5, 6]
sigmoid_list = [basic_sigmoid(x) for x in python_list]
print(sigmoid_list)

numpy_array = np.array([1, 2, 3, 4, 5, 6])
sigmoid_array = numpy_sigmoid(numpy_array)
print(sigmoid_array)
