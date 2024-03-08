from tensor import Tensor
from math import sqrt

''' a linear layer, very closely modeled on pytorch's linear layer'''
class Linear:
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        bound = sqrt(1/in_features)
        self.weight = Tensor.rand([out_features, in_features], -bound, bound)
        self.weight.init_grad()
        if bias:
            self.bias = Tensor.rand([out_features], -bound, bound)
            self.bias.init_grad()

    def __call__(self, x):
        if self.bias != None:
            return self.weight @ x + self.bias
        else:
            return self.weight @ x

''' a generic way to construct networks out a layers, very similar to pytorch'''
class MyNet():
    def __init__(self):
        self.fc1 = Linear(28*28, 512)
        self.fc2 = Linear(512, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

data = Tensor.rand([28*28], 0, 1)
m = MyNet()
labels = m.forward(data)
print(labels)