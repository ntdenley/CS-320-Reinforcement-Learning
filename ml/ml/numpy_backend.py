import numpy as np
from .op import BackendInterface

dtype = np.float32

class NumpyBackend(BackendInterface):
    def fromlist(self, shape, list): return np.array(list, dtype=dtype).reshape(shape)
    def empty(self, shape): return np.empty(shape, dtype=dtype)
    def rand(self, shape): return np.random.rand(*shape)
    def full(self, shape, val): return np.full(shape, val, dtype=dtype)
    def fill_inplace(self, inp, val): return np.ndarray.fill(inp, val)
    def reshape(self, inp, shape): return np.reshape(inp, shape)
    def transpose(self, inp): return np.transpose(inp)
    def sum(self, inp, out, axis, keepdims): 
        return np.sum(a=inp, axis=axis, dtype=None, out=out, keepdims=keepdims)
    def max(self, inp, out, axis, keepdims): 
        return np.max(a=inp, axis=axis, out=out, keepdims=keepdims)
    def abs(self, inp, out): return np.abs(inp, out)
    def sqrt(self, inp, out): return np.sqrt(inp, out)
    def square(self, inp, out): return np.square(inp, out)
    def log(self, inp, out): return np.log(inp, out)
    def exp(self, inp, out): return np.exp(inp, out)
    def negative(self, inp, out): return np.negative(inp, out)
    def reciprocal(self, inp, out): return np.reciprocal(inp, out)
    def copy(self, inp, out): return np.copyto(out, inp)
    def sin(self, inp, out): return np.sin(inp, out)
    def cos(self, inp, out): return np.cos(inp, out)
    def tan(self, inp, out): return np.tan(inp, out)
    def add(self, inp1, inp2, out): return np.add(inp1, inp2, out)
    def multiply(self, inp1, inp2, out): return np.multiply(inp1, inp2, out)
    def matmul(self, inp1, inp2, out): return np.matmul(inp1, inp2, out)
    def cmp(self, inp1, inp2, out): 
        out[inp1 > inp2] = 1
        out[inp1 < inp2] = -1
        out[inp1 == inp2] = 0