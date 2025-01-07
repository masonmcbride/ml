import numpy as np
from scipy import linalg

class ConvPoolLayer(object):
    
    def __init__(self, image_shape, filter_shape, pool_size=(2,2), activation_fn=None):
        """
        image_shape = (minibatch size, num of input features maps, image width, image height)
        filter_shape = (number of feature naps, num of input feature maps, filter width, filter height)
        """


        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.activation_fn = activation_fn

def unroll(kernel, n):
    
    unrolled = np.zeros((n, n+1))
    unrolled[:,:-1] = kernel

    return unrolled.reshape(((n+1)*n, 1))

#kernel is the weights matrix, n=kernel width/height (assuming width = height), m=input image width
#Toeplitz takes two inputs first_col and first_row, col is size n^2 and row is m^2
def Toeplitz2D(kernel, n, m):
    first_col = np.r_[kernel[0][0].reshape((1,1)), np.zeros(((m-n+1)**2-1,1))]
    unrolled = unroll(kernel, n)[:-1]
    first_row = np.r_[unrolled, np.zeros((m**2-len(unrolled),1))]
    print("col={} row={}".format(len(first_col), len(first_row)))
    return linalg.toeplitz(first_col, first_row)
    

kernel = np.array([[1,2],[3,4]])
t2d = Toeplitz2D(kernel, 2, 4) 
print(t2d)
