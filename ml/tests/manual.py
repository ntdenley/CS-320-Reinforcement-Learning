from ml.backward import build_backward_graph, alloc_grad, zero_grad
from ml.graph import LazyOp, ComputeNode
from ml.op import OpType
from ml.array import Array
from ml.nn import Linear
import ml.array as ml
import numpy as np
import torch