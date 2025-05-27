import math

def basic_sigmoid(x):
    # sigmoid(x) = 1 / (1 + e^(-x))
    s = 1 / (1 + math.exp(-x))
    return s

print(basic_sigmoid(2))
