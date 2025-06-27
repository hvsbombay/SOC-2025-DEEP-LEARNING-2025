import numpy as np
import matplotlib.pyplot as plt
import h5py


def loadDataset():
    trainDataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    trainDatasetX = np.array(trainDataset["train_set_x"][:])
    trainDatasetY = np.array(trainDataset["train_set_y"][:])
    
    testDataset = h5py.File("datasets/test_catvnoncat.h5")
    testDatasetX = np.array(testDataset["test_set_x"][:])
    testDatasetY = np.array(testDataset["test_set_y"][:])

    classes = np.array(trainDataset["list_classes"][:])

    trainDatasetY = trainDatasetY.reshape((1, trainDatasetY.shape[0]))
    testDatasetY = testDatasetY.reshape((1, testDatasetY.shape[0]))
    return trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, classes

trainDatasetX, trainDatasetY, testDatasetX, testDatasetY, classes = loadDataset()
print(trainDatasetX.shape, trainDatasetY.shape)

index = 10
plt.imshow(trainDatasetX[index])
plt.show()
print ("y = " + str(trainDatasetY[0, index]) + ", it's a '" + classes[np.squeeze(trainDatasetY[:, index])].decode("utf-8") +  "' picture.")

m_train = trainDatasetX[0].shape   # number of training examples
m_test = testDatasetX[0].shape    # number of test examples
num_px = trainDatasetX[1].shape    # height/width of each image (assuming square)

print("Number of training examples: ", m_train)
print("Number of testing examples: ", m_test)
print("Height of image: ", num_px, "px")
print("Shape of image: ", trainDatasetX[0].shape)
print("Shape of training dataset X: ", trainDatasetX.shape)
print("Shape of training dataset Y: ", trainDatasetY.shape)


XTrainFlatten = trainDatasetX.reshape(m_train, -1).T  # shape: (num_px*num_px*3, m_train)
XTestFlatten = testDatasetX.reshape(m_test, -1).T    # shape: (num_px*num_px*3, m_test)

print("Shape of test dataset X: ", XTestFlatten.shape)
print("Shape of XTestFlatten: ", XTestFlatten.shape)

XTrainFlatten = XTrainFlatten / 255.
XTestFlatten = XTestFlatten / 255.

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initializeParameters(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]  # number of examples

    # Forward propagation
    Z = np.dot(w.T, X) + b           # shape: (1, m)
    A = sigmoid(Z)                   # shape: (1, m)

    # Cost function (logistic loss)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)  # shape: (nx, 1)
    db = (1/m) * np.sum(A - Y)          # scalar

    grad = {"dw": dw,
            "db": db}
    
    return grad, cost

# Gradient descent training loop
def train_model(X, Y, numIterations, learningRate, printCost=False):
    nx = X.shape[0]
    w, b = initializeParameters(nx)
    costs = []

    for i in range(numIterations):
        grad, cost = propagate(w, b, X, Y)
        costs.append(cost)

        # Update parameters
        w -= learningRate * grad["dw"]
        b -= learningRate * grad["db"]

        if printCost and i % 100 == 0:
            print(f"Cost after {i} iterations: {cost}")

    params = {"w": w,
              "b": b}
    
    return params, costs


params, costs = train_model(XTrainFlatten, trainDatasetY, numIterations = 2000, learningRate = 0.005, printCost = True)

plt.plot(costs)
plt.show()


def predict(params, X, Y):
    w = params["w"]
    b = params["b"]
    Z = w.T.dot(X) + b
    A = sigmoid(Z)
    Y_Prediction = (A > 0.5) * 1.0
    return Y_Prediction

#checking accuracy
trainPrediction = predict(params, XTrainFlatten, trainDatasetY)
testPrediction = predict(params, XTestFlatten, testDatasetY)
print(f"train accuracy: {100-np.mean(np.abs(trainPrediction-trainDatasetY))*100}")
print(f"test accuracy: {100-np.mean(np.abs(testPrediction-testDatasetY))*100}")