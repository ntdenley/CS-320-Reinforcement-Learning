from .op import OpType
from .graph import ComputeNode, LazyOp
from .graph_utils import view_graph
from .backward import zero_grad, alloc_grad, build_backward_graph

''' factories '''

def full(shape, val):
    out = ComputeNode(shape)
    out.loadop(LazyOp(OpType.Alloc.Full, shape, val))
    return Array(node=out)

def ones(shape): return full(shape, 1)

def zeros(shape): return full(shape, 0)

def rand(shape):
    out = ComputeNode(shape)
    out.loadop(LazyOp(OpType.Alloc.Rand, shape))
    return Array(node=out)

def linspace(start, end, num):
    return Array([(i+start) * (end-start) / num for i in range(num)])

def arange(start, end):
    return linspace(start, end, (end-start))

''' utilities '''

def parse_nested_list(passed_list):
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
                raise ValueError(f"Expected {passed_in} to be of length {shape[curdim]}.")
            curdim += 1
            for elem in passed_in: recursive_helper(elem)
            curdim -= 1
        elif isinstance(passed_in,(int,float)): data.append(passed_in)
        else: raise TypeError(f"Expected '{passed_in}' to be an number or list.")
    recursive_helper(passed_list)
    return shape, data

def nelem(shape):
    nelem = 1
    for s in shape:
        nelem *= s
    return nelem

def transpose_shape(shape, dim0=0, dim1=1):
    if len(shape) < 2: return shape
    tshape = shape.copy()
    tshape[dim0] = shape[dim1]
    tshape[dim1] = shape[dim0]
    return tshape

def shapes_broadcastable(shape1, shape2):    
    for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError(f"{shape1} and {shape2} not broadcastable")
    return True

def gen_broad_shape(shape1, shape2):
    ndim1, ndim2 = len(shape1), len(shape2)
    larger, smaller = (shape1,shape2) if ndim1 > ndim2 else (shape2, shape1)
    larger, smaller = larger.copy(), smaller.copy()
    while len(smaller) < len(larger):
        smaller.insert(0,1)
    out = [max(s1, s2) for s1, s2 in zip(larger,smaller)]
    return out

def get_sum_shape(shape, axis, keepdims) -> list:
    if axis==None and not keepdims:
        return []
    elif axis==None and keepdims:
        return [1 for _ in shape]
    elif axis != None and not keepdims:
        s = list(shape).copy()
        s = s[0:axis] + s[axis+1:]
        return s
    elif axis != None and keepdims:
        s = list(shape).copy()
        s[axis] = 1
        return s

'''
Array:
1. infers shape to allocate
2. sytactic sugar
3. type checking
'''
class Array:

    def __init__(self, passed_list=None, node=None):
        self.built_backward = False
        if passed_list != None:
            shape, data = parse_nested_list(passed_list)
            self.node = ComputeNode(shape)
            self.node.loadop(LazyOp(OpType.Alloc.FromList, shape, data))
        else:
            self.node = node
        
    def eval(self):
        if not self.node.is_alloced:
            self.node.allocate()
        return self.node.evaluate()

    def view_graph(self, name="computation_graph", view=True, view_data=True):
        view_graph(self.node, name, view, view_data)

    ''' view '''

    def reshape(self, *args):
        if isinstance(args[0], list): 
            shape=args[0]
            assert len(args) == 1, "reshape: too many arguments"
        else:
            shape = list(args)
        
        # replace -1 with thing
        if -1 in shape:
            assert shape.count(-1) == 1, "reshape: no more than 1 -1s allowed"
            placeholder = nelem(self.node.shape)
            for s in shape:
                if s == -1: continue
                assert placeholder % s == 0, f"reshape: {placeholder=}, {s=}; each other besides -1 must divide nelem."
                placeholder /= s
            shape[shape.index(-1)] = int(placeholder)

        n1, n2 = nelem(self.node.shape), nelem(shape)
        assert n1 == n2, f"reshape: {self.node.shape} and {shape} don't have same number of elems"
        new_node = ComputeNode(shape)
        new_node.loadop(LazyOp(OpType.View.Reshape, shape, inputNodes=[self.node]))
        return Array(node=new_node)

    def flatten(self):
        return self.reshape([nelem(self.node.shape)])

    def transpose(self):
        new_node = ComputeNode(transpose_shape(self.node.shape))
        new_node.loadop(LazyOp(OpType.View.Transpose, inputNodes=[self.node]))
        return Array(node=new_node)

    def t(self):
        return self.transpose()

    def unsqueeze(self, axis):
        new_shape = self.node.shape.copy()
        new_shape.insert(axis,1)
        return self.reshape(new_shape)

    def __repr__(self):
        return repr(self.eval())
    
    ''' unary operations '''

    def unary_op(self, optype):
        out = ComputeNode(self.node.shape)
        out.loadop(LazyOp(optype, inputNodes=[self.node]))
        return Array(node=out)

    def sum(self, axis=None, keepdims=False):
        if axis: assert isinstance(axis, int)
        assert isinstance(keepdims, bool)
        out = ComputeNode(get_sum_shape(self.node.shape, axis, keepdims))
        out.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, inputNodes=[self.node]))
        return Array(node=out)
    
    def max(self, axis=None, keepdims=False):
        if axis: assert isinstance(axis, int)
        assert isinstance(keepdims, bool)
        out = ComputeNode(get_sum_shape(self.node.shape, axis, keepdims))
        out.loadop(LazyOp(OpType.Inplace.Max, axis, keepdims, inputNodes=[self.node]))
        return Array(node=out)

    def sqrt(self):
        return self.unary_op(OpType.Inplace.Sqrt)

    def abs(self):
        return self.unary_op(OpType.Inplace.Abs)

    def relu(self):
        return self.abs()

    def square(self):
        return self.unary_op(OpType.Inplace.Square)

    def log(self):
        return self.unary_op(OpType.Inplace.Log)
    
    def exp(self):
        return self.unary_op(OpType.Inplace.Exp)

    def negative(self):
        return self.unary_op(OpType.Inplace.Negative)

    def __neg__(self):
        return self.negative()

    def reciprocal(self):
        return self.unary_op(OpType.Inplace.Reciprocal)

    def sin(self):
        return self.unary_op(OpType.Inplace.Sin)

    def cos(self):
        return self.unary_op(OpType.Inplace.Cos)

    def tan(self):
        return self.unary_op(OpType.Inplace.Tan)

    def softmax(self, axis):
        # max_vals = self.max(axis=axis, keepdims=True)
        # exps = (self - max_vals).exp()  
        exps = self.exp()
        return exps / exps.sum(axis=axis, keepdims=True)

    def sigmoid(self):
        return ((-self).exp() + 1).reciprocal()

    ''' binary operations'''
    
    def binop(self, other, optype):
        if isinstance(other, (int, float)):
            scalar_node = ComputeNode(shape=[1])
            scalar_node.loadop(LazyOp(OpType.Alloc.Full, scalar_node.shape, other))
            other = Array(node=scalar_node)
        assert isinstance(other, Array), "Inplace ops only supported between Arrays!"
        assert shapes_broadcastable(self.node.shape, other.node.shape)
        # select the output shape to be the larger of the two
        new_node = ComputeNode(gen_broad_shape(self.node.shape, other.node.shape))
        new_node.loadop(LazyOp(optype, inputNodes=[self.node,other.node]))
        return Array(node=new_node)

    def cmp(self, other):
        return self.binop(other, OpType.Inplace.Cmp)

    def __eq__(self, other):
        return 1 - self.cmp(other).abs()
    
    def __ne__(self, other):
        return self.cmp(other).abs()
    
    def __lt__(self, other):
        return self.cmp(other) == -1
    
    def __le__(self, other):
        return (self < other) + (self == other)

    def __gt__(self, other):
        return self.cmp(other) == 1
    
    def __ge__(self, other):
        return (self > other) + (self == other)

    def __add__(self, other):
        return self.binop(other, OpType.Inplace.Add)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            scalar_node = ComputeNode(shape=[1])
            scalar_node.loadop(LazyOp(OpType.Alloc.Full, scalar_node.shape, other))
            other = Array(node=scalar_node)
        return self + -other
        # return self.binop(other, OpType.Inplace.Subtract)

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        if isinstance(other, (int,float)):
            return self * (1/other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def __mul__(self, other):
        return self.binop(other, OpType.Inplace.Multiply)

    def __rmul__(self, other):
        return self * other
    
    def dot(self, other):
        assert self.node.shape == other.node.shape, "dot product: shapes must match"
        return (self * other).sum()

    def __matmul__(self, other):
        assert isinstance(other, Array), "matmul only supported between Arrays!"
        assert len(self.node.shape) == len(other.node.shape) == 2, "matmul: 2D tensors only"
        assert self.node.shape[1] == other.node.shape[0], "matmul: inner dims must match"
        new_node = ComputeNode([self.node.shape[0], other.node.shape[1]])
        new_node.loadop(LazyOp(OpType.Inplace.Matmul, inputNodes=[self.node, other.node]))
        return Array(node=new_node)

    ''' backward '''
    def build_backward(self):
        build_backward_graph(self.node)
        alloc_grad(self.node)
        zero_grad(self.node)

    def zero_grad(self):
        zero_grad(self.node)

    def grad(self):
        return Array(node=self.node.grad)
    
    def set_reeval(self):
        self.node.set_reevaluate()