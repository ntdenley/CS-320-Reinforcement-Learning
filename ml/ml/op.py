from abc import ABC, abstractmethod
from enum import Enum, auto

class OpType:
    class Alloc(Enum): # allocate memory in which to store some data.
        Empty = auto()
        FromList = auto()
        Full = auto()
        Rand = auto()
    class View(Enum): # creates a new view on some tensor, some alloc.
        Reshape = auto()
        Transpose = auto()
    class Inplace(Enum): # applies transformations between 1 or 2 input tensors, with an output tensor.
        Add = auto()
        Abs = auto()
        Accumulate = auto() # nondifferetiable
            # Cmp is a weird operation that i've created.
            # cmp a b: if a < b: out = -1
            #          if a == b:out =  0
            #          if a > b: out =  1
            # broadcasting is still supported here.
        Cmp = auto() 
        Copy = auto()
        Cos = auto()
        Exp = auto()
        Fillinplace = auto()
        Log = auto()
        Max = auto()
        Multiply = auto()
        Matmul = auto()
        Negative = auto()
        Reciprocal = auto()
        Sin = auto()
        Sum = auto()
        Sqrt = auto()
        Square = auto()
        Tan = auto()

'''
Requirements:
- Inplace operations must be inplace operations
- All operations must expect broadcastable tensors, or scalars, 
    and prepare to do broadcasting
- Matmul is meant only for 2D tensors

Guaruntees:
- Will be given broadcastable tensors or one scalar + one tensors
- Matmul will be given tensors of valid shapes
'''
class BackendInterface(ABC):
    @abstractmethod
    def fromlist(self, shape, list): pass
    
    @abstractmethod 
    def empty(self, shape): pass

    @abstractmethod
    def rand(self, shape): pass
    
    @abstractmethod
    def full(self, shape, val): pass

    @abstractmethod
    def fill_inplace(self, inp, val): pass
        
    ''' differentable '''
    @abstractmethod
    def reshape(self, inp, shape): pass
    
    @abstractmethod
    def transpose(self, inp): pass
        
    @abstractmethod
    def sum(self, inp, out, axis, keepdims): pass

    @abstractmethod
    def max(self, inp, out, axis, keepdims): pass

    @abstractmethod
    def abs(self, inp, out): pass
    
    @abstractmethod
    def sqrt(self, inp, out): pass
        
    @abstractmethod
    def square(self, inp, out): pass
    
    @abstractmethod
    def log(self, inp, out): pass

    @abstractmethod
    def exp(self, inp, out): pass
    
    @abstractmethod
    def negative(self, inp, out): pass

    @abstractmethod
    def reciprocal(self, inp, out): pass

    @abstractmethod
    def copy(self, inp, out): pass
    
    @abstractmethod
    def sin(self, inp, out): pass
    
    @abstractmethod
    def cos(self, inp, out): pass
    
    @abstractmethod
    def tan(self, inp, out): pass
        
    @abstractmethod
    def add(self, inp1, inp2, out): pass
    
    @abstractmethod
    def multiply(self, inp1, inp2, out): pass
    
    @abstractmethod
    def matmul(self, inp1, inp2, out): pass

    @abstractmethod
    def cmp(self, inp1, inp2, out): pass

    ''' methods that build on base ones. '''
    def accumulate(self, *args, out): 
        # can accept any number of nodes, all of which will add to this one.
        for a in args: self.add(a, out, out)

    