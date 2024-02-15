from tensor import Tensor

a = Tensor.from_list([[[1,2,3],[4,5,6]],[[2,3,4],[18,1,18]]])
b = Tensor.fill(a.shape, 2)

a.init_grad()
b.init_grad()

recip = b.map_pow(-1)
# recip.sum().backward()

c = a.binary_mul(recip)
c.sum().backward()

print(b.grad)
print(a.map_mul(-1/4))
# # c = ((a.binary_pow(b)).binary_add(b)).sum()
# c = a.binary_div(b)
# print(c)
# c = c.sum()

# c.backward()

# for n in (a,b,c): print(n.info())
