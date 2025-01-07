from network2 import *
import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net2 = Network2([
ConvPoolLayer(image_shape=(1, 1, 28, 28), 
    kernel=(20,1,5,5), 
    poolsize=(2,2)),
ConvPoolLayer(image_shape=(1, 20, 12, 12),
    kernel=(40, 20, 5, 5),
    poolsize=(2,2)),
FullyConnectedLayer(40*4*4, 100),
FullyConnectedLayer(100,10,Softmax)
])
a = np.random.randn(784, 1)
out = net2.feed_forward(a)
print(out)
