from tensor import Tensor
import torch

'''
todo
- softmax
- one hot encoding
- cross entropy loss
'''

a = Tensor([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
b = torch.Tensor([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
a.init_grad()
b.requires_grad = True

a1 = a.softmax(0)+3
a1.backward()
print(a1)
print(a.grad)

b1 = b.softmax(0)+3
b1.backward(torch.ones_like(b1))
print(b1)
print(b.grad)