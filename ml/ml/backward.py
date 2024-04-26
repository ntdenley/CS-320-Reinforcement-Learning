from .op import OpType
from .graph import ComputeNode, LazyOp
from .graph_utils import view_graph
from typing import List

def nelem(shape):
    nelem=1
    for s in shape: 
        if isinstance(s, (list,tuple)):
            continue
        elif (isinstance(s, (int,float))):
            nelem *= s
        else:
            raise ValueError(f"weird element {s} encountered in shape {shape}")
    return nelem

def unary_op(inp: ComputeNode, optype: OpType) -> ComputeNode:
    n = ComputeNode(inp.shape)
    n.loadop(LazyOp(optype, inputNodes=[inp]))
    return n

'''
Applies enough sum operations on input_node 
until the shape is equal to out_shape.

1. left-pad out_shape with zeros.
2. for s1, s2, i, in zip(out_shape, inp_shape, range(len(inp_shape))):
        if s1 == s2: continue
        if s1 == 1: sum(keepdim=True)
        if s1 == 0: sum(axis=i)
'''
def sum_squeeze(input_node: ComputeNode, out_shape: List[int]):
    def sum_op(inp_node, axis, keepdims, out_shape):
        out = ComputeNode(out_shape)
        out.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, 
                          inputNodes=[inp_node]))
        return out
    
    if len(out_shape) == 0:
        return sum_op(input_node, None, False, out_shape)

    broad = out_shape.copy()
    while len(broad) < len(input_node.shape):
        broad.insert(0,0)

    out = input_node
    temp_shape = input_node.shape.copy()
    axis_acc = 0
    for s1, s2, i in zip(input_node.shape, broad, range(len(broad))):
        if temp_shape == out_shape: break
        elif s1 == s2: continue
        elif s2 == 1: 
            temp_shape[i-axis_acc] = 1
            out = sum_op(out, i-axis_acc, True, temp_shape.copy())
        elif s2 == 0:
            axis_acc += 1
            temp_shape = temp_shape[1:]
            out = sum_op(out, 0, False, temp_shape.copy())
        elif s1 == 1:
            temp_shape[i-axis_acc] = 1
            out = sum_op(out, i-axis_acc, True, temp_shape.copy())
        else:
            raise ValueError("sum_squeeze: unbroadcastable shapes passed")

    return out

def apply_backward_to_parents(node: ComputeNode) -> None:
    from .array import Array
    if isinstance(node.op.optype, OpType.Alloc): return
    
    # make the grads into accumulator nodes.
    ps = node.op.get_parents()
    for p in ps: 
        if p.grad == None:
            p.grad = ComputeNode(p.shape)
            p.grad.loadop(LazyOp(OpType.Inplace.Accumulate, inputNodes=[]))

    # find new inputs into the accumulators based on op.
    i0, i1 = None, None

    # view
    if node.op.optype == OpType.View.Reshape:
        i0 = (Array(node=node.grad).reshape(ps[0].shape)).node
    elif node.op.optype == OpType.View.Transpose:
        i0 = (Array(node=node.grad).transpose()).node
    # unary
    elif node.op.optype == OpType.Inplace.Sum:
        # in general, we let broadcasting distribute sum's grad.
        # but in this case, since an axis was collapsed, we need to 
        # insert a 1 into the shape for broadcasting to work.
        axis, keepdims = node.op.args[0], node.op.args[1]
        if axis != None and keepdims == False:
            new_shape = node.grad.shape.copy()
            new_shape.insert(axis,1)
            i0 = Array(node=node.grad).reshape(new_shape).node
        else:
            i0 = node.grad
    elif node.op.optype == OpType.Inplace.Max: 
        axis, keepdims = node.op.args[0], node.op.args[1]
        if axis != None and keepdims == False:
            new_shape = node.grad.shape.copy()
            new_shape.insert(axis,1)
            i0 = Array(node=node.grad).reshape(new_shape).node
            i0 = (Array(node=i0) * (Array(node=ps[0]) == Array(node=node).reshape(new_shape))).node
        else:
            i0 = node.grad
            i0 = (Array(node=i0) * (Array(node=ps[0]) == Array(node=node))).node
    elif node.op.optype == OpType.Inplace.Abs:
        i0 = ((Array(node=node.grad) >= 0) * Array(node=node.grad)).node
    elif node.op.optype == OpType.Inplace.Sqrt:
        i0 = (Array(node=node.grad) * (2*Array(node=ps[0]).sqrt()).reciprocal()).node
    elif node.op.optype == OpType.Inplace.Square:
        i0 = (Array(node=node.grad) * Array(node=ps[0])*2).node
    elif node.op.optype == OpType.Inplace.Log:
        i0 = (Array(node=node.grad) * Array(node=ps[0]).reciprocal()).node
    elif node.op.optype == OpType.Inplace.Exp:
        i0 = (Array(node=node.grad) * Array(node=node)).node
    elif node.op.optype == OpType.Inplace.Negative:
        i0 = (-Array(node=node.grad)).node
    elif node.op.optype == OpType.Inplace.Reciprocal:
        i0 = (Array(node=node.grad).negative() * Array(node=ps[0]).reciprocal().square()).node
    elif node.op.optype == OpType.Inplace.Copy:
        i0 = node.grad
    elif node.op.optype == OpType.Inplace.Sin:
        i0 = (Array(node=node.grad) * Array(node=ps[0]).cos()).node
    elif node.op.optype == OpType.Inplace.Cos:
        i0 = (Array(node=node.grad) * -Array(node=ps[0]).sin()).node
    elif node.op.optype == OpType.Inplace.Tan:
        i0 = (Array(node=node.grad) * Array(node=ps[0]).cos().square().reciprocal()).node
    # binary
    elif node.op.optype == OpType.Inplace.Add:
        i0 = node.grad
        i1 = node.grad
    elif node.op.optype == OpType.Inplace.Multiply:
        i0 = (Array(node=ps[1]) * Array(node=node.grad)).node
        i1 = (Array(node=ps[0]) * Array(node=node.grad)).node
    elif node.op.optype == OpType.Inplace.Matmul:
        i0 = (Array(node=node.grad) @ Array(node=ps[1]).transpose()).node
        i1 = (Array(node=ps[0]).transpose() @ Array(node=node.grad)).node
    else:
        raise Exception(f"Backward not implemented for op {node.op.optype}")

    # add i1, i2 into respective accumulators.
    ps[0].grad.op.inputNodes.append(sum_squeeze(i0, ps[0].shape))
    if i1 != None:
        ps[1].grad.op.inputNodes.append(sum_squeeze(i1, ps[1].shape))

    for p in ps: apply_backward_to_parents(p)

def build_backward_graph(node):
    n = ComputeNode(node.shape)
    n.loadop(LazyOp(OpType.Alloc.Full, node.shape, 1))
    node.grad = n
    apply_backward_to_parents(node)

def alloc_grad(node):
    if node.grad:
        node.grad.allocate()
        for p in node.op.get_parents():
            alloc_grad(p)

def _zero_grad(node):
    if node.grad and node.grad.is_alloced:
        node.backend.fill_inplace(node.grad.data, 0)
        for p in node.op.get_parents():
            _zero_grad(p)

def zero_grad(node):
    if node.grad and node.grad.is_alloced:
        node.backend.fill_inplace(node.grad.data, 1)
        for p in node.op.get_parents():
            _zero_grad(p)