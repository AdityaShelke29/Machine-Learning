# Here is the neural network created using Python and Numpy. 
# Aditya Shelke

import numpy as np
from pandas import *

# array[-1] returns an array of all the elements in array excluding the last one. 

# zip(array1, array2) returns a Zip object, which is an array of tuples where 
# each tuple is corresponding elements from array1 and array2. 

# randn(x, y) returns a 2D array with dimensions x and y of random numbers. 



class Network(object): 
    def __init__(self, sizes): 
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a): 
        for b, w in zip(self.biases, self.weights): 
            a = sigmoid(np.dot(w, a) + b)
            
        return a
    


def sigmoid(z): 
    return 1.0/(1.0 + np.exp(-z))

network = Network([2, 5, 3])

print([2, 5, 3][:-1])
print([2, 5, 3][1:])

for x in zip([2, 5, 3][:-1], [2, 5, 3][1:]): 
    print(x)

print(network.biases)
print("\n")
print(network.weights)