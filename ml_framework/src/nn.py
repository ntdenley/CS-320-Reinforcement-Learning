'''
'''
from torch import Tensor
import torch
from math import sqrt

class Linear:
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        bound = sqrt(1/in_features)
        self.weight = (torch.rand((out_features, in_features)) - 0.5)*bound*2
        self.weight = torch.ones_like(self.weight)
        self.weight.requires_grad = True
        if bias:
            self.bias = (torch.rand(out_features) - 0.5)*bound*2
            self.bias = torch.ones_like(self.bias)
            self.bias.requires_grad = True

    def __call__(self, x):
        output = self.weight @ x.T
        if self.has_bias:
            output += self.bias.unsqueeze(1)
        return output.T

from tensor import Tensor
from math import sqrt

class Linear2:
     
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        bound = sqrt(1/in_features)
        self.weight = Tensor.rand([out_features, in_features], -1, 1)*bound
        self.weight = Tensor.fill(self.weight.shape, 1)
        self.weight.init_grad()
        if bias:
            self.bias = Tensor.rand([out_features], -1, 1)*bound
            self.bias = Tensor.fill(self.bias.shape, 1)
            self.bias.init_grad()

    def __call__(self, x):
        output = self.weight @ x.t()
        if self.has_bias:
            output += self.bias.unsqueeze(1)
        return output.t()