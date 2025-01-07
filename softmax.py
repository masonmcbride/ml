import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

#TESTS

p = np.array([209,  90, 142,  67,  74,  64, 100])
p = p.reshape((1,7))
p2 = np.array([1,3,1])
print(p)
x = softmax(p)

print(f"softmax: {x} sum: {x.sum(axis=1)}")
