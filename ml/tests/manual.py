from ml.backward import build_backward_graph, alloc_grad, zero_grad
from ml.graph import LazyOp, ComputeNode
from ml.op import OpType
from ml.array import Array
from ml.nn import Linear
import ml.array as ml

a = ml.Array([1,2,3])
b = ml.Array([100,200,300])

out = (a + b).sqrt() 

out.eval()
out.view_graph()
