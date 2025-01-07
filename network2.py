#w:/cs/ml/network2.py
#Network2 this time but more abstract
#Network2 class will just have methods like SGD, manage mini-batches, record data
#Network2 will be supplied layer classes that it will use in a list

import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.special import softmax
import sys

class Network2(object):
    #__init__
    #SGD
    #update_mini_batch
    #backprop 
    #evaluate
    #feed_forward
    
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
    def SGD(self, training_data, mini_batch_size, epochs, lr, lmbda, test_data=None):
        if test_data: test_data_size = len(test_data)
        training_data_size = len(training_data)

        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, training_data_size, mini_batch_size)]
        mini_batches.pop()

        print("training started")
        for j in range(epochs):
            i = 1
            for mini_batch in mini_batches:
                sys.stdout.write('\r')
                a = i//(len(mini_batches)//20)
                sys.stdout.write("{}/{}".format(i, len(mini_batches)))
                sys.stdout.write("[%-20s] %d%% epoch %d " % ('='*a, 5*a, j))
                sys.stdout.flush()
                i+=1

                self.update_mini_batch(mini_batch, lr, lmbda, training_data_size)

            if test_data: 
                print("{} / {}".format(self.evaluate(test_data), test_data_size))

            else:
                print("epoch {} complete.".format(j))


    def update_mini_batch(self, mini_batch, lr, lmbda, training_data_size):

        #TODO compute mini_batch at the same time? not one by one?
        mini_batch_size = len(mini_batch)
        mini_batch = list(zip(*mini_batch))
        x = np.array(mini_batch[0][:]).squeeze(2).T
        y = np.array(mini_batch[1][:]).squeeze(2).T
        nabla_w, nabla_b = self.backprop(x,y)

        for l, nw, nb in zip(self.layers, nabla_w, nabla_b):
            l.weights = (1-(lmbda*lr)/training_data_size)*l.weights - (lr/mini_batch_size)*nw
            l.biases -= (lr/mini_batch_size)*nb

    def backprop(self,x,y):
        activation = x
        activations = [x]

        nabla_w = []
        nabla_b = []
        for l in self.layers:
            if isinstance(l.weights, torch.Tensor):
                nabla_w.append(torch.zeros(l.weights.shape))
                nabla_b.append(torch.zeros(l.biases.shape))
            else:
                nabla_w.append(np.zeros(l.weights.shape))
                nabla_b.append(np.zeros(l.biases.shape))
        zs=[]
        activation_primes = []
        #feed forward and collect data for backprop
        for l in self.layers:
            z = l.forward(activation)
            zs.append(z)
            activation = l.activation_fn.activate(z)
            activations.append(activation)
            activation_prime = l.activation_fn.activate_prime(z)
            activation_primes.append(activation_prime)

        #cross entropy prime * softmax prime = pi-yi
        #pi is soft max probs yi is labels 
        delta = activations[-1] - y 
        #last delta is calculated first and the rest is calculted in loop
        #initial delta = cost_prime(a_out) * activation_prime(previous_out)
        #dw= delta *(some operation) weights db= delta
        #new delta is calculated with da *(some operation) activation_prime(previous_out)
        for l in range(1, self.num_layers):
            da, dw, db = self.layers[-l].backward(delta, activations[-l-1])
            nabla_w[-l] = dw
            nabla_b[-l] = db
            if isinstance(da, np.ndarray) and isinstance(activation_primes[-l-1], torch.Tensor): 
                activation_primes[-l-1] = self.layers[-l].toNarray(activation_primes[-l-1])
            delta = da * activation_primes[-l-1]
        return (nabla_w, nabla_b)
    
    def evaluate(self, test_data):        
        test_results = [(np.argmax(self.feed_forward(x)),y) for x,y in test_data]
        test_results = sum(int(x==y) for (x,y) in test_results)
        return test_results

    def feed_forward(self, a):
        for l in self.layers:
            a = l.forward(a)
            a = l.activation_fn.activate(a)
        return a

#activation function creation
class Relu(object):
    #activate
    #activate_prime

    @staticmethod
    def activate(a):
        return np.maximum(0,a)

    @staticmethod
    def activate_prime(a):
        a[a<=0] = 0
        a[a>0] = 1
        return a

class Softmax(object):
    #activate
    #activate_prime
    
    @staticmethod
    def activate(a):
        return softmax(a, axis=0)

    @staticmethod
    def activate_prime(a):
        pass

class Sigmoid(object):
    #activate
    #activate_prime

    @staticmethod
    def activate(a):
        return 1.0 / (1+np.exp(-a))

    @staticmethod
    def activate_prime(a):
        s = 1.0 / (1+np.exp(-a))
        return s * (1-s)

class FullyConnectedLayer(object):
    #__init__
    #forward
    #backward

    # input(minibatch, in_channels, out_channels)

    def __init__(self, input_neurons, output_neurons, activation_fn=Relu, p_dropout=1.0):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.weights = np.random.randn(output_neurons, input_neurons) / np.sqrt(input_neurons)
        self.biases = np.random.randn(output_neurons, 1)
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

    def forward(self, inpt):
        if isinstance(inpt, torch.Tensor):
            mini_batch_size = inpt.shape[0]
            inpt = inpt.flatten().reshape(mini_batch_size, self.input_neurons).T
        a = np.dot(self.weights, inpt) + self.biases
        p = np.random.binomial(1, self.p_dropout, a.shape)
        a *= p/self.p_dropout
        return a

    def backward(self, delta, activation):
        if isinstance(activation, torch.Tensor):
            mini_batch_size = activation.shape[0]
            activation = activation.flatten().reshape(mini_batch_size, self.input_neurons).T
        dw = np.dot(delta, activation.T)
        db = delta.sum(axis=1).reshape(delta.shape[0], 1)
        da = np.dot(self.weights.T, delta)
        return da, dw, db

    def toNarray(self, a):
        mini_batch_size = a.shape[0]
        a = a.flatten().reshape(self.input_neurons, mini_batch_size)
        return a.numpy()

class ConvPoolLayer(object):
    #__init__
    #forward
    #backward 
    #toConvTensor(for getting around data types for each type of layer)

    #Conv needs input=(mini_batch_size, in_channels, iH, iW) kernel=(out_channels, in_channels, kH, kW)
    def __init__(self, image_shape, kernel, poolsize, activation_fn=Relu):
        self.mini_batch_size, self.in_channels, self.iH, self.iW = image_shape
        self.out_channels, in_channels, self.kH, self.kW = kernel
        self.oH, self.oW = self.iH-self.kH+1, self.iW-self.kW+1
        self.weights = torch.randn(self.out_channels, self.in_channels, self.kH, self.kW) /np.sqrt(self.iH*self.iW) 
        self.biases = torch.randn(self.out_channels)
        self.poolsize = poolsize #(pH, pW)
        self.activation_fn = activation_fn
        self.indices = None

    def forward(self, a):
        #forward will do conv operation and maxpool -> gradient only cares about the max values 
        if len(a.shape) == 2: a = torch.from_numpy(self.toConvTensor(a))
        a = F.conv2d(a, self.weights, bias=self.biases)
        a, self.indices = F.max_pool2d(a, self.poolsize, return_indices=True)
        return a

    def backward(self, delta, activation):
        if isinstance(delta, np.ndarray): 
            delta=torch.from_numpy(delta.reshape(self.mini_batch_size, self.out_channels, self.oH//2, self.oW//2))
        delta = delta.type(torch.FloatTensor)
        ones = torch.ones(self.mini_batch_size,1,self.oH,self.oW)
        delta = F.max_unpool2d(delta, self.indices, self.poolsize)
        dw, db = self.backConv(activation, delta)
        da = F.conv_transpose2d(delta, self.weights)
        dw = dw
        db = db.flatten()

        return da, dw, db

    def toConvTensor(self, x):
        x = x.T
        if x.shape[0] > 1:
            tensor = np.array([x[0].reshape(self.iH, self.iW), x[1].reshape(self.iH, self.iW)])
            for r in x[2:]:
                tensor = np.concatenate((tensor, [r.reshape(self.iH, self.iW)]))
        else:
            tensor = x.reshape(self.iH, self.iW)
            tensor = tensor[None, : , :]
        tensor = tensor[:, None, :, :]
        return tensor.astype(np.float32)   

    def backConv(self, activation, delta):
        dw = torch.zeros(self.weights.shape)
        db = torch.zeros(1, self.out_channels, 1, 1)
        ones = torch.ones(1,1,self.oH,self.oW)
        for act_sliver, del_sliver in zip(activation, delta):
            dws = F.conv2d(act_sliver.unsqueeze(0), del_sliver.unsqueeze(1).repeat(1,self.in_channels,1,1))
            dbs = F.conv2d(ones, del_sliver.unsqueeze(1))
            dw += dws.permute(1,0,2,3).repeat(1,self.in_channels,1,1)
            db += dbs
        return dw, db

def cross_entropy(yHat, y):
    log_likelihood = [ -yi*np.log(pi) for yi, pi in zip(y, yHat) ] 
    loss = np.sum(log_likehood) / len(y[0])
    return loss

def report(a):
    if len(sys.argv) > 1:
        print(a)
