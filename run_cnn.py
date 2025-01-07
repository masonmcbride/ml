from network2 import *
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
miniBatchSize=50
net2 = Network2([
ConvPoolLayer(image_shape=(miniBatchSize,1,28,28),
    kernel=(20,1,5,5),
    poolsize=(2,2)),
ConvPoolLayer(image_shape=(miniBatchSize,20,12,12),
    kernel=(40,20,5,5),
    poolsize=(2,2)),
FullyConnectedLayer(40*4*4,100),
FullyConnectedLayer(100,10,Softmax)
])
net2.SGD(training_data, mini_batch_size=miniBatchSize, epochs=30, lr=0.03, lmbda=0.1, test_data=test_data)
