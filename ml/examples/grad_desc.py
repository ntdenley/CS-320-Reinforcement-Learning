import ml.array as ml
from ml.graph import ComputeNode, LazyOp
import torch
import time

lr = 1e-3
iter = 1000

shape = 1
start = torch.ones([shape])

def torch_version():
    x = torch.arange(1,shape+1, dtype=torch.float32)
    w = torch.ones([shape])
    w.requires_grad = True

    def f(x,w): return x.dot(w)
    def loss(pred): return (10-pred).square()
    
    for _ in range(iter):
        l = loss(f(x,w))
        # print("\t loss = ", l)
        l.backward()
        with torch.no_grad():
            w -= w.grad * lr
            w.grad.zero_()
    # print("Weights = ", w)
    print("Loss = ", loss(f(x,w)))
    return loss(f(x,w))

def my_version():
    
    x = ml.arange(1,shape+1)
    w = ml.ones([shape])

    f = x.dot(w)
    loss = (10 - f).square()
    loss.build_backward()

    temp_array = loss.node.backend.empty(x.node.shape)

    for _ in range(iter):  
        # print("\tloss = ", loss)
        loss.eval()
        loss.zero_grad()
        loss.set_reeval()
        w.grad().set_reeval()

        loss.node.backend.multiply(lr, w.grad().eval(), temp_array)
        loss.node.backend.negative(temp_array, temp_array)
        loss.node.backend.add(w.node.data, temp_array, w.node.data)

    # print("Weights = ", w)
    print("Loss = ", loss)
    return loss.eval()

st = time.time()
l1 = torch_version()
et = time.time()
print("time = ", et-st)

print()
st = time.time()
l2 = my_version()
et = time.time()
print("time = ", et-st)

print()
print(f"{l1=}")
print(f"{l2=}")