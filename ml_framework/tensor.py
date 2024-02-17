'''
A tensor with automatic differentation

Sources
- Tutorial for autograd and neural networks: https://www.youtube.com/watch?v=VMj-3S1tku0&t=1s
- Explanation of pytorch tensor internals: http://blog.ezyang.com/2019/05/pytorch-internals/
- Inpsiration for architecture and implementation details: https://github.com/tinygrad/tinygrad
                                                           https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h

Edge cases:
- When division by 0 is encountered anywhere, an error is thrown.
- When log(0) is encountered in a gradient, we add 1e9 to it.
'''

import random
from math import log
from enum import Enum

ops = Enum('ops', ['Unary', 'Binary', 'Map'])
_autograd_stack = []

class Tensor:

    def __init__(self):
        self.data = None
        self.shape = None
        self.stride = None
        self.grad = None

    ''' initialization factory methods '''
    @staticmethod
    def manual_init(data, shape):
        new = Tensor()
        new.data = data
        new.shape = shape
        new.set_stride()
        return new
    
    @staticmethod
    def from_list(passed_list):
        data = []
        shape = []
        ndim = 0
        curdim = 0
        def recursive_helper(passed_in):        
            nonlocal data, shape, ndim, curdim
            if isinstance(passed_in,list):
                if ndim == curdim:
                    ndim += 1
                    shape.append(len(passed_in))
                elif shape[curdim] != len(passed_in):
                    raise Exception(f"Expected {passed_in} to be of length {shape[curdim]}.")
                curdim += 1
                for elem in passed_in: recursive_helper(elem)
                curdim -= 1
            elif isinstance(passed_in,(int,float)): data.append(passed_in)
            else: raise TypeError(f"Expected '{passed_in}' to be an number or list.")
        recursive_helper(passed_list)
        new = Tensor()
        new.data = data
        new.shape = shape
        new.set_stride()
        return new
    
    @staticmethod
    def fill(shape, value):
        new = Tensor()
        new.shape = shape
        new.data = []
        for i in range(new.num_elements()):
            new.data.append(value)
        new.set_stride()
        return new
    
    @staticmethod
    def rand(shape, start, end):
        new = Tensor()
        new.shape = shape
        new.data = []
        for i in range(new.num_elements()):
            new.data.append(random.uniform(start, end))
        new.set_stride()
        return new
    
    # this creates a new tensor which points 
    # to the same data and grad as the old tensor
    @staticmethod
    def soft_copy(tensor):
        new = Tensor()
        new.data = tensor.data
        new.shape = tensor.shape.copy()
        new.stride = tensor.stride.copy()
        new.grad = tensor.grad
        return new
    
    @staticmethod
    def hard_copy(tensor):
        new = Tensor()
        new.data = tensor.data.copy()
        new.shape = tensor.shape.copy()
        new.stride = tensor.stride.copy()
        new.grad = None if tensor.grad == None else tensor.grad.copy()
        return new

    ''' ways to look at tensor '''
    def __repr__(self):
        return TensorViewer(self).to_string()

    def info(self):
        return TensorViewer(self).info()

    ''' reshape works weirdly: it returns a new tensor
        with the same data. '''
    def reshape(self, shape):
        new = Tensor().soft_copy(self)
        new.shape = shape
        new.set_stride()
        return new

    ''' math operations. The domain is always Tensor -> Tensor '''
    # unary
    def reduce(self):
        pass

    def sum(self):
        n = 0
        for d in self.data:
            n += d
        new = Tensor.manual_init([n], [1])
        new.init_grad()
        
        def _backward(out, inp):
            for i in range(len(inp.grad.data)):
                inp.grad.data[i] += out.grad.data[0]

        _autograd_stack.append(new)
        _autograd_stack.append(self)
        _autograd_stack.append(_backward)
        _autograd_stack.append(ops.Unary)

        return new
    
    def map(self, value, op, _backward):
        new = Tensor.manual_init(self.data.copy(), self.shape)
        op(new, value)
        new.init_grad()
        _autograd_stack.append(new)
        _autograd_stack.append(self)
        _autograd_stack.append(value)
        _autograd_stack.append(_backward)
        _autograd_stack.append(ops.Map)
        return new
    
    def map_add(self, value):
        def op(tensor, value):
            for i in range(tensor.num_elements()):
                tensor.data[i] += value
        def _backward(out, inp, value):
            for i in range(out.num_elements()):
                inp.grad.data[i] += out.grad.data[i]
        return self.map(value, op, _backward)
     
    def map_sub(self, value):
        return self.map_add(-1*value)

    def map_mul(self, value):
        def op(tensor, value):
            for i in range(tensor.num_elements()):
                tensor.data[i] *= value
        def _backward(out, inp, value):
            for i in range(out.num_elements()):
                inp.grad.data[i] += out.grad.data[i] * value
        return self.map(value, op, _backward)

    def map_pow(self, value):
        def op(tensor, value):
            for i in range(tensor.num_elements()):
                if value >= 0:
                    tensor.data[i] = tensor.data[i]**value
                elif tensor.data[i] != 0:
                    tensor.data[i] = tensor.data[i]**value
                else:
                    raise Exception("Division by zero encountered.")
        def _backward(out, inp, value):
            for i in range(out.num_elements()):
                inp.grad.data[i] += value * (inp.data[i] ** (value-1)) * out.grad.data[i]
        return self.map(value, op, _backward)
    
    def map_div(self, value):
        return self.map_mul(1/value)
    
    def relu(self):
        def op(tensor, value):
            for i in range(tensor.num_elements()):
                tensor.data[i] = max(tensor.data[i], 0)
        def _backward(out, inp, value):
            for i in range(out.num_elements()):
                if inp.data[i] > 0:
                    inp.grad.data[i] += out.grad.data[i]
        return self.map(0, op, _backward)

    # binary 
    def binary_op(self, other, op, _backward):
        if self.shape != other.shape:
            raise Exception("when performing binary operations between two tensors," + \
                            "they must have the same shape")
        new = Tensor.manual_init(self.data.copy(), self.shape)
        op(new, other)
        new.init_grad()
        _autograd_stack.append(new)
        _autograd_stack.append(self)
        _autograd_stack.append(other)
        _autograd_stack.append(_backward)
        _autograd_stack.append(ops.Binary)
        return new

    def binary_add(self, other):
        def op(a,b):
            for i in range(a.num_elements()):
                a.data[i] += b.data[i]
        def _backward(out, inp1, inp2):
            for i in range(len(inp1.grad.data)):
                inp1.grad.data[i] += out.grad.data[i]
                inp2.grad.data[i] += out.grad.data[i]
        return self.binary_op(other, op, _backward)

    def binary_sub(self, other):
        def op(a, b):
            for i in range(a.num_elements()):
                a.data[i] -= b.data[i]
        def _backward(out, inp1, inp2):
            for i in range(len(inp1.grad.data)):
                inp1.grad.data[i] += out.grad.data[i]
                inp2.grad.data[i] -= out.grad.data[i]
        return self.binary_op(other, op, _backward)

    def binary_mul(self, other):
        def op(a, b):
            for i in range(a.num_elements()):
                a.data[i] *= b.data[i]
        def _backward(out, inp1, inp2):
            for i in range(inp1.num_elements()):
                inp1.grad.data[i] += inp2.data[i] * out.grad.data[i]
                inp2.grad.data[i] += inp1.data[i] * out.grad.data[i]
        return self.binary_op(other, op, _backward)
    
    def binary_pow(self, other):
        def op(a, b):
            for i in range(a.num_elements()):
                a.data[i] = a.data[i] ** b.data[i]
        def _backward(out, inp1, inp2):
            for i in range(len(inp1.grad.data)):
                inp1.grad.data[i] += inp2.data[i] * (inp1.data[i] ** (inp2.data[i]-1)) * out.grad.data[i]
                if inp1.data[i] != 0:
                    inp2.grad.data[i] += log(inp1.data[i]) * (inp1.data[i] ** inp2.data[i]) * out.grad.data[i]
                else:
                    inp2.grad.data[i] += log(1e9) * (inp1.data[i] ** inp2.data[i]) * out.grad.data[i] 
        return self.binary_op(other, op, _backward)

    def binary_div(self, other):
        return self.binary_mul(other.map_pow(-1))

    # matrix operations
    def dot(self, other):
        len1 = len(self.shape)
        len2 = len(other.shape)
        if len1 != 1 or len2 != 1:
            raise Exception(f"Expected two 1D arrays, but got a {len1}D and {len2}D array.")
        return self.binary_mul(other).sum()

    # still working on this
    def matmul(self, other):
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise Exception("For matmul, only 2d arrays are allowed")
        if self.shape[1] != other.shape[0]:
            raise Exception(f"For matmul: {self.shape} x {other.shape} doesn't work")

        new = Tensor.fill(self.shape, 0)
        new.init_grad()
        

        return new

    ''' gradient related methods '''
    def init_grad(self):
        self.grad = Tensor().fill(self.shape, 0)

    def stack_trace(self):
        print("STACK TRACE (this is the bottom):")
        for a in _autograd_stack:
            print(a)
        print("END STACK TRACE")
        print()

    # you can only do this once, since this empties the stack.
    def backward(self):
        # why is this here?
        if self.shape != [1]:
            raise Exception("You can only call .backward of tensors with shape [1]")

        self.stack_trace()

        self.grad = Tensor().fill(self.shape, 1)
        while (len(_autograd_stack) > 0):
            optype = _autograd_stack.pop() 
            fn = _autograd_stack.pop()
            if optype == ops.Unary:
                inp = _autograd_stack.pop()
                out = _autograd_stack.pop()
                fn(out,inp)
            elif optype == ops.Binary:
                inp2 = _autograd_stack.pop()
                inp1 = _autograd_stack.pop()
                out = _autograd_stack.pop()
                fn(out,inp1,inp2)
            elif optype == ops.Map:
                value = _autograd_stack.pop()
                inp = _autograd_stack.pop()
                out = _autograd_stack.pop()
                fn(out,inp,value)
            else:
                raise Exception("something wrong with autograd stack!!")

    ''' utilities '''
    def set_stride(self):
        self.stride = []
        for i in range(len(self.shape)):
            s = 1
            for j in range(i+1, len(self.shape)):
                s *= self.shape[j]
            self.stride.append(s)

    def num_elements(self):
        n = 1
        for s in self.shape:
            n *= s
        return n
    
    ''' wrappers around math functions '''

''' This class takes in a tensor and can represent it as a string '''
class TensorViewer():

    def __init__(self, tensor):
        self.tensor = tensor

    def to_string(self):
        string = ""
        _iter = 0
        depth = 0
        ndim = len(self.tensor.shape)
        col_width, pre_dot, post_dot = self.find_col_width()
        shape = self.tensor.shape.copy()
        
        def recursive_helper(self):
            nonlocal string, _iter, depth, col_width, pre_dot, post_dot, shape
            if (len(shape) == 0): 
                if (col_width > 7):
                    post = min(max(post_dot, pre_dot), 8)
                    string += "%*.*e" % (post+7, post, self.tensor.data[_iter])
                else:
                    string += "%*.*f" % (col_width+1, post_dot, self.tensor.data[_iter])
                _iter += 1
                return
            
            string += "["
            depth += 1
            n = shape.pop(0)

            for _ in range(n): recursive_helper(self)
            
            shape.insert(0,n)
            depth -= 1
            if (string[-1] == ' '):
                while (string[-1] != ']'): string = string[:-1]
            string += ']'
            for i in range(ndim - depth): string += '\n'
            for i in range(depth): string += " "
        
        recursive_helper(self)
        for i in range(ndim): string = string[:-1]
        return string

    def find_col_width(self):
        col_width = 0
        pre_dot = 0
        post_dot = 0
        for d in self.tensor.data:
            pre,post = str(float(d)).split('.')
            if len(pre) > pre_dot: pre_dot=len(pre)
            if len(post) > post_dot: post_dot=len(post)
        col_width = pre_dot+post_dot+1
        return col_width, pre_dot, post_dot
    
    def info(self):
        grad = None if self.tensor.grad == None else self.tensor.grad.data
        return  f"Tensor(" + \
                f"\n\tid        : {id(self.tensor)},"      + \
                f"\n\tdata      : {self.tensor.data},"     + \
                f"\n\tshape     : {self.tensor.shape},"    + \
                f"\n\tstride    : {self.tensor.stride},"   + \
                f"\n\tgrad.data : {grad},"                 + \
                "\n)"
