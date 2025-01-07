import network2
import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net2 = network2.Network2([
network2.FullyConnectedLayer(784,80, p_dropout=1.0), \
network2.FullyConnectedLayer(80,40, p_dropout=1.0), \
network2.FullyConnectedLayer(40,10,network2.Softmax)])
net2.SGD(training_data, mini_batch_size=30, epochs=10, lr=0.1, lmbda=0.8, test_data=test_data)
