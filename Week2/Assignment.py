import numpy as np 
import matplotlib.pyplot as plt

dataset_raw = np.genfromtxt("heart.csv", dtype="str", delimiter=",")
print(dataset_raw.shape)

def init_params(num_features):
    W = np.zeros((num_features, 1))
    b = 0
    return W, b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_prop(W, b, X):
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A

def calculate_loss(A, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return np.squeeze(cost)

def backward_prop(A, X, Y):
    m = X.shape[1]
    dZ = A - Y
    dW = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)
    return dW, db

def update_params(W, b, dW, db, learning_rate):
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def predict(W, b, X):
  A = forward_prop(W, b, X)
  Y_pred = (A>=0.5)*1.0
  return Y_pred