from tensor import Tensor

a = Tensor.fill([2,2],0)
b = Tensor.fill([2,2],0.002)

a.init_grad()
b.init_grad()

f = lambda x, y: x ** y

out = f(a,b)
out.sum().backward()

# print(a)
print(b)

# print(a.grad)
print(b.grad)