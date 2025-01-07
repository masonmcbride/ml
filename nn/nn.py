from utils import fetch_mnist
import numpy as np

X_train, Y_train, X_test, Y_test = fetch_mnist()
X_test = X_test[:30]

l1 = np.random.randn(28*28, 128) / np.sqrt((28*28))
l2 = np.random.randn(128, 10) / np.sqrt(128)

def forward(x):
    x = x.dot(l1)
    x = ReLU(x)
    x = x.dot(l2)
    return x

def ReLU(x):
    return np.maximum(x, 0)

def Softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True)) 
    print(e_x)
    return e_x / e_x.sum(axis=1, keepdims=True)

z = X_test.reshape((-1, 28*28))
z = forward(z)
Sz = Softmax(z)

# forward(x) returns size (batch_size, 10) 
Y_test_preds = np.argmax(z, axis=1)
