from .array import Array, rand
from math import sqrt

class Linear:
     
    def __init__(self, in_features, out_features, bias=True):
        self.has_bias = bias
        bound = sqrt(1. / in_features)
        self.weight = rand([out_features, in_features]) * 2 * bound - bound
        if self.has_bias:
            self.bias = rand([out_features]) * 2 * bound - bound

    def __call__(self, x):
        output = self.weight @ x  # Assuming x is of shape [batch_size, in_features]
        if self.has_bias:
            output += self.bias.reshape(-1, 1)  
        return output
