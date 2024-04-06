from .op import OpType
from .graph import ComputeNode, LazyOp
from .graph_utils import view_graph

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

def expand_or_squeeze_unary_op(inp: ComputeNode, out_shape, optype: OpType) -> ComputeNode:
    # if equal shape, return the simple case
    if len(out_shape) == len(inp.shape)  and nelem(out_shape) == nelem(inp.shape):
        n = ComputeNode(inp.shape)
        n.loadop(LazyOp(optype, inputNodes=[inp]))
        return n
    # if out is larger, let broadcasting take over
    if len(out_shape) > len(inp.shape) or nelem(out_shape) > nelem(inp.shape):
        n = ComputeNode(out_shape)
        n.loadop(LazyOp(optype, inputNodes=[inp]))
        return n
    # otherwise, out is smaller, and we need to sum.
    # one sum per dimension shaved off, so maybe several.
    else:
        # if output 0 dimensions, just do sum
        if len(out_shape) == 0:
            n = ComputeNode(inp.shape)
            n.loadop(LazyOp(optype, inputNodes=[inp]))
            keepdims = False
            axis = None
            out = ComputeNode(out_shape)
            out.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, inputNodes=[n]))
            return out
        else:
            if len(inp.shape) - len(out_shape) == 1:
                keepdims = False
                axis = 0
                n = ComputeNode(inp.shape)
                n.loadop(LazyOp(optype, inputNodes=[inp]))
                out = ComputeNode(out_shape)
                out.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, inputNodes=[n]))
                return out
            else:
                axis = 0
                keepdims = False
                n = ComputeNode(inp.shape)
                n.loadop(LazyOp(optype, inputNodes=[inp]))
                out = ComputeNode(inp.shape[1:])
                out.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, inputNodes=[n]))
                out2 = ComputeNode(out_shape)
                out2.loadop(LazyOp(OpType.Inplace.Sum, axis, keepdims, inputNodes=[out]))
                return out2
 

def expand_binary_op(inp1: ComputeNode, inp2:  ComputeNode, optype: OpType) -> ComputeNode:
    # ensures that the output node has the shape of the larger of the input nodes.
    ndim1, ndim2 = len(inp1.shape), len(inp2.shape)
    larger, smaller = (inp1,inp2) if ndim1 > ndim2 else (inp2,inp1)
    out = ComputeNode(larger.shape)
    out.loadop(LazyOp(optype, inputNodes=[larger, smaller]))
    return out

def apply_backward_to_parents(node: ComputeNode) -> None:
    if isinstance(node.op.optype, OpType.Alloc): return
    ps = node.op.get_parents()
    # view
    if node.op.optype == OpType.View.Reshape:
        pass
    elif node.op.optype == OpType.View.Transpose:
        pass
    # unary
    elif node.op.optype == OpType.Inplace.Sum:
        ps[0].grad = expand_or_squeeze_unary_op(node.grad, ps[0].shape, OpType.Inplace.Copy)
    elif node.op.optype == OpType.Inplace.Sqrt:
        pass
    elif node.op.optype == OpType.Inplace.Square:
        pass
    elif node.op.optype == OpType.Inplace.Negative:
        pass
    elif node.op.optype == OpType.Inplace.Copy:
        pass
    elif node.op.optype == OpType.Inplace.Sin:
        pass
    elif node.op.optype == OpType.Inplace.Cos:
        pass
    elif node.op.optype == OpType.Inplace.Tan:
        pass
    # binary
    elif node.op.optype == OpType.Inplace.Add:
        ps[0].grad, ps[1].grad = ComputeNode(ps[0].shape), ComputeNode(ps[1].shape)
        ps[0].grad.loadop(LazyOp(OpType.Inplace.Copy, inputNodes=[node.grad]))
        ps[1].grad.loadop(LazyOp(OpType.Inplace.Copy, inputNodes=[node.grad]))
    elif node.op.optype == OpType.Inplace.Subtract:
        pass
    elif node.op.optype == OpType.Inplace.Multiply:
        n0 = expand_binary_op(ps[1], node.grad, OpType.Inplace.Multiply)
        ps[0].grad = expand_or_squeeze_unary_op(n0, ps[0].shape, OpType.Inplace.Copy)
        n1 = expand_binary_op(ps[0], node.grad, OpType.Inplace.Multiply)
        ps[1].grad = expand_or_squeeze_unary_op(n1, ps[1].shape, OpType.Inplace.Copy)
    elif node.op.optype == OpType.Inplace.Divide:
        pass
    elif node.op.optype == OpType.Inplace.Matmul:
        pass
    else:
        raise Exception(f"Backward not implemented for op {node.op.optype}")

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