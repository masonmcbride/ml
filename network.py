import numpy as np
import random


class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def update_mini_batch(self, mini_batch, eta):
        
        
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        #for loop sums all errors and stores in nabla_x
        #couldn't this be done with dot product and transposition?
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x,y)

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]


        #nablas are divided by len to get average, multiplied by learning rate and subtracted from weights
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        activation = x
        activations = [x]

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        #feed forward while storing important values for backprop (zs and activated zs)
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        #backward pass
        #delta = cost'(error) * activation'(z's) 
        delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1])
        #delta dotted with previous neuron activations gives gradient of weights
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta
        #last delta is calculated first and the rest is calculted in loop
        #delta stacks as loop contines and is dotted with respective activatied neurons
        #formula delta of layer * activations of layer before = gradient of layer
        #delta of layer = previous delta * previous weights * current activation primes
        #initial delta = cost_prime(activation(z_out)) * activation_primez_out)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T,delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta

        return(nabla_w, nabla_b)


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)

        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

        for j in range(epochs):
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))

            else:
                print("Epoch {} complete".format(j))
    
    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_prime(self, output_activations, y):

        return output_activations-y

    
def sigmoid(z):
    return 1.0 /(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))


"""
Layer needs 3 things
1. Initialize layer - input neurons, output neurons (ex: in=784 out=10)
2. Forward ex: 784->10
3. Backward ex: 10-784 but their derivatives
"""
