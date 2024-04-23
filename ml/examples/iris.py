import ml.array as ml
import ml.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import reduce
from abc import ABC, abstractmethod

def get_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = ml.Array(X_train.tolist())
    X_test = ml.Array(X_test.tolist())
    y_train = ml.Array(y_train.tolist())
    y_test = ml.Array(y_test.tolist())
    return X_train, X_test, y_train, y_test

def test_accuracy(model, X_test, y_test):
    logits_test = model.np_(X_test)
    predictions = np.argmax(logits_test, axis=1)
    accuracy = np.mean((predictions == y_test.eval()))
    print(f'Accuracy: {accuracy.item()}')

class Linear:

    def __init__(self, inp_shape, out_shape):
        self.w = ml.rand([inp_shape,out_shape])
        self.b = ml.rand([out_shape])
        self.params = [self.w, self.b]

    def __call__(self, x):
        return x @ self.w + self.b
        
    def np_(self, x):
        return x @ self.w.eval() + self.b.eval()

# class ANN:

#     def __init__(self):
#         self.funcs = [
#             Linear(4,10),
#             lambda x: x.abs(),
#             Linear(10,3)
#         ]
#         self.layers = list(filter(lambda x: isinstance(x, Linear), self.funcs))

#     def params(self):
#         return reduce(lambda a,b: a.params + b.params, self.layers) 

#     def __call__(self, x):
#         out = x
#         for f in self.funcs:
#             out = f(out)
#         return out
    
#     def np_(self, x):
#         out = x.eval()
#         for f in self.funcs:
#             if f in self.layers:
#                 out = f.np_(out)
#             else:
#                 out = f(out)
#         return out

class ANN:

    def __init__(self):
        self.funcs = [
            Linear(4,10),
            Linear(10,3)
        ]
        self.layers = list(filter(lambda x: isinstance(x, Linear), self.funcs))

    def params(self):
        return reduce(lambda a,b: a.params + b.params, self.layers) 

    def __call__(self, x):
        out = x
        for f in self.funcs:
            out = f(out)
        return out
    
    def np_(self, x):
        out = x.eval()
        for f in self.funcs:
            if f in self.layers:
                out = f.np_(out)
            else:
                out = out
        return out

    
def train(model, loss, optim, iter, reports=10):
    for it in range(iter):
        if (it % (iter//reports) == 0): print(f"{it} loss = ", loss.eval().item())
        loss.eval()
        loss.zero_grad()
        loss.set_reeval()
        for p in model.params():
            p.grad().set_reeval()
        optim.step()
    test_accuracy(model, X_test, y_test)


lr = 0.01
iter = 100

X_train, X_test, y_train, y_test = get_data()
model = ANN()
loss = nn.softmax_cross_entropy_loss(
    model(X_train), 
    nn.one_hot_encode(y_train, 3)
) 
loss.build_backward()
optim = nn.Adam(lr, model.params())

train(model, loss, optim, iter)