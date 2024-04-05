from .op import OpType
from .graph import ComputeNode, LazyOp
from .graph_utils import view_graph

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

def nelem_from_shape(shape):
    nelem = 1
    for s in shape:
        nelem *= s
    return nelem

def transpose_shape(shape, dim0=0, dim1=1):
    tshape = shape.copy()
    tshape[dim0] = shape[dim1]
    tshape[dim1] = shape[dim0]
    return tshape

def shapes_broadcastable(shape1, shape2):    
    for dim1, dim2 in zip(reversed(shape1), reversed(shape2)):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False
    return True

'''
1. simulate pytorch/mlx api
2. infers shape
3. enforces shape, preventing user from doing illegal shapes
'''
class Array:

    def __init__(self, passed_list=None, node=None):
        if passed_list:
            shape, data = parse_nested_list(passed_list)
            self.node = ComputeNode(shape)
            self.node.loadop(LazyOp(OpType.Alloc.FromList, shape, data))
        else:
            self.node = node
        
    def eval(self):
        if not self.node.is_alloced:
            self.node.allocate()
        return self.node.evaluate()

    def reshape(self, shape):
        n1, n2 = nelem_from_shape(self.node.shape), nelem_from_shape(shape)
        assert n1 == n2, f"Could not perform reshape from shape {self.node.shape} to {shape}"
        new_node = ComputeNode(shape)
        new_node.loadop(LazyOp(OpType.View.Reshape, shape, inputNodes=[self.node]))
        return Array(node=new_node)

    def transpose(self):
        new_node = ComputeNode(transpose_shape(self.node.shape))
        new_node.loadop(LazyOp(OpType.View.Transpose, inputNodes=[self.node]))
        return Array(node=new_node)

    def inplace_normal_op(self, other, optype):
        if isinstance(other, (int, float)):
            scalar_node = ComputeNode(shape=[1])
            scalar_node.loadop(LazyOp(OpType.Alloc.Full, scalar_node.shape, other))
            other = Array(node=scalar_node)
        assert isinstance(other, Array), "Inplace ops only supported between Arrays!"
        assert shapes_broadcastable(self.node.shape, other.node.shape)
        new_node = ComputeNode(self.node.shape)
        new_node.loadop(LazyOp(optype, 
                               inputNodes=[self.node, other.node]))
        return Array(node=new_node)

    def __add__(self, other):
        return self.inplace_normal_op(other, OpType.Inplace.Add)

    def __sub__(self, other):
        return self.inplace_normal_op(other, OpType.Inplace.Subtract)

    def __mul__(self, other):
        return self.inplace_normal_op(other, OpType.Inplace.Multiply)

    def __truediv__(self, other):
        return self.inplace_normal_op(other, OpType.Inplace.Divide)

    def __matmul__(self, other):
        assert isinstance(other, Array), "matmul only supported between Arrays!"
        assert len(self.node.shape) == len(other.node.shape) == 2, "matmul: 2D tensors only"
        assert self.node.shape[1] == other.node.shape[0], "matmul: inner dims must match"
        new_node = ComputeNode([self.node.shape[0], other.node.shape[1]])
        new_node.loadop(LazyOp(OpType.Inplace.Matmul, inputNodes=[self.node, other.node]))
        return Array(node=new_node)