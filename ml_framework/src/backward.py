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
        # keep doing sums until... out.shape matches out_shape
        n = ComputeNode(inp.shape)
        n.loadop(LazyOp(optype, inputNodes=[inp]))
        keepdims = False #len(out_shape) == len(inp.shape)
        out = ComputeNode(())
        out.loadop(LazyOp(OpType.Inplace.Sum, None, keepdims, inputNodes=[n]))
        return out

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

# shape = [4,4]

# # a = ComputeNode(shape=shape)
# # a.loadop(LazyOp(OpType.Alloc.FromList, a.shape, [1,2,3,4]))
# a = ComputeNode(shape=shape)
# a.loadop(LazyOp(OpType.Alloc.Full, a.shape, 2))

# b = ComputeNode(shape=[1])
# b.loadop(LazyOp(OpType.Alloc.FromList, [1], [4]))

# c = ComputeNode(shape=shape)
# c.loadop(LazyOp(OpType.Inplace.Multiply, inputNodes=[a,b]))

# d = ComputeNode(shape=())
# d.loadop(LazyOp(OpType.Inplace.Sum, None, False, inputNodes=[c]))

# build_backward_graph(d)
# d.allocate()
# d.evaluate()
# alloc_grad(d)
# zero_grad(d)

# a.grad.evaluate()
# b.grad.evaluate()

# print(a.evaluate())

# view_graph(d, "forward")