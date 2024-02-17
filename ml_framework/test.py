from tensor import Tensor

# just binary_mul
# a = Tensor.from_list([[[1,2,3],[4,5,6]],[[2,3,4],[18,1,18]]])
# a.init_grad()
# b = Tensor.fill(a.shape, 2)
# b.init_grad()
# out = a.binary_mul(b)
# out.sum().backward()
# print(a.grad)
# print(b.grad)

# binary_div
# init
# a = Tensor.from_list([1,2,3])
# a.init_grad()
# b = Tensor.fill(a.shape, 2)
# b.init_grad()

# # **-1
# recip = b.map_pow(-1)

# # a * (b**-1)
# out = a.binary_mul(recip)
# out.sum().backward()

# print('a')
# print(a.grad)
# print('b')
# print(b.grad)
# print('recip')
# print(recip.grad)
# print('out')
# print(out.grad)

# a = Tensor.from_list([[1,2],[3,4]])
# b = Tensor.from_list([[5,7],[8,9],[10,11]])
# print(a.shape)
# print(b.shape)
# print(a.matmul(b))

a = Tensor.from_list([1,2,3])
b = Tensor.from_list([4,5,6])
print(a.dot(b))