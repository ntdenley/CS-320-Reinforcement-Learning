from .numpy_backend import NumpyBackend
from .op import OpType

class LazyOp:
    
    def __init__(self, optype, *args, inputNodes=None):
        self.optype = optype
        self.args = args
        self.inputNodes = inputNodes

    def get_parents(self):
        if not self.inputNodes: return []
        return self.inputNodes
        
    def __call__(self, backend, out=None):
        method = getattr(backend, self.optype.name.lower())
        if isinstance(self.optype, OpType.Alloc):
            return method(*self.args)
        if isinstance(self.optype, OpType.View):
            evaled_nodes = [n.evaluate() for n in self.inputNodes]
            return method(*evaled_nodes, *self.args)
        if isinstance(self.optype, OpType.Inplace):
            evaled_nodes = [n.evaluate() for n in self.inputNodes]
            method(*evaled_nodes, out, *self.args)

class ComputeNode:
    
    def __init__(self, shape, backend=NumpyBackend()):
        self.shape = shape
        self.backend = backend
        self.data = None
        self.op = None
        self.has_op = False
        self.is_alloced = False
        self.is_evaled = False

        self.requires_grad = False
        self.grad = None

    def loadop(self, op):
        self.op = op
        self.has_op = True

    def allocate(self):
        assert self.has_op, "ComputeNode needs op before alloc"
        self.is_alloced = True
        if isinstance(self.op.optype, OpType.Alloc):
            self.data = self.op(self.backend)
        elif isinstance(self.op.optype, OpType.View):
            for parent in self.op.get_parents():
                parent.allocate()
        elif isinstance(self.op.optype, OpType.Inplace):
            self.data = self.backend.empty(self.shape)
            for parent in self.op.get_parents():
                parent.allocate()

    def evaluate(self):
        assert self.is_alloced, "ComputeNode needs alloc before eval."
        if self.is_evaled: return self.data
        self.is_evaled = True
        if isinstance(self.op.optype, OpType.Alloc):
            return self.data
        elif isinstance(self.op.optype, OpType.View):
            self.data = self.op(self.backend)
            return self.data
        elif isinstance(self.op.optype, OpType.Inplace):
            self.op(self.backend, self.data)
            return self.data

    def set_reevaluate(self):
        if isinstance(self.op.optype, OpType.Inplace):
            self.is_evaled = False
        for parent in self.op.get_parents():
            parent.set_reevaluate()