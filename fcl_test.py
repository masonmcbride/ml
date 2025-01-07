from network2 import *
import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net2 = Network2([
FullyConnectedLayer(784, 80),
FullyConnectedLayer(80, 40),
FullyConnectedLayer(40, 10,Softmax)
])
a = np.random.randn(784, 1)
out = net2.feed_forward(a)
print(out)
