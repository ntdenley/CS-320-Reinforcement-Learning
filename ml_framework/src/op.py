from abc import ABC, abstractmethod
from enum import Enum, auto

class OpType:
    class Alloc(Enum): # allocate memory in which to store some data.
        FromList = auto()
        Empty = auto()
        Rand = auto()
        Full = auto()
    class View(Enum): # creates a new view on some tensor, some alloc.
        Reshape = auto()
        Transpose = auto()
    class Inplace(Enum): # applies transformations between allocated tensors.
        Sum = auto()
        Sqrt = auto()
        Square = auto()
        Negative = auto()
        Copy = auto()
        Fillinplace = auto()
        Accumulate = auto()
        Sin = auto()
        Cos = auto()
        Tan = auto()
        Add = auto()
        Subtract = auto()
        Multiply = auto()
        Divide = auto()
        Matmul = auto()

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
    def sqrt(self, inp, out): pass
        
    @abstractmethod
    def square(self, inp, out): pass
    
    @abstractmethod
    def negative(self, inp, out): pass
    
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
    def subtract(self, inp1, inp2, out): pass
    
    @abstractmethod
    def multiply(self, inp1, inp2, out): pass
    
    @abstractmethod
    def divide(self, inp1, inp2, out): pass
    
    @abstractmethod
    def matmul(self, inp1, inp2, out): pass
    
    ''' methods that build on base ones. '''
    def accumulate(self, inp, out): self.add(inp, out, out)