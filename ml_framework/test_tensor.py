from tensor import Tensor
import torch
import random
from math import log10

''' compares two tensors (a of type Tensor and b of type torch.Tensor)
    to see if they have the same shape and elements, within the specified
    precision '''
def compare(a, b, sigfigs, debug=False):
    for i in range(len(a.shape)):
        if a.shape[i] != b.shape[i]:
            return False
    if debug:
        print("\n%25s | %25s | %16s | %16s" % \
            ("MINE", "PYTORCH", "ERROR", "GOOD SIGFIGS"))
    for i in range(len(a.data)):
        val_a = float(a.data[i])
        val_b = float(b.storage()[i])
        error = abs(val_a - val_b)
        correct_sigfigs = int(log10(abs(val_a/error))) if error != 0 else 100
        if (abs(val_a) < 1e-9 and abs(val_b) < 1e-9): correct_sigfigs = 100
        if debug:
            print("%25.10f | %25.10f | %16.10f | %16d" % \
                  (val_a, val_b, error, correct_sigfigs))
        if correct_sigfigs < sigfigs:
            return False
    return True

''' generates a random Tensor and torch.Tensor that are almost the same '''
def gen_random_tensors():
    ndims = random.randint(1,4)
    shape = [random.randint(1,10) for i in range(ndims)]
    a = Tensor.rand(shape, -1, 1) * 10000
    b = torch.Tensor(a.data)
    b = b.reshape(a.shape)
    return a, b

''' run the specified function on random tensors and compare them an iter
    number of times. '''
def forward_test(f, sigfigs, iters, debug):
    for i in range(iters):
        a, b = gen_random_tensors()
        if not compare(f(a), f(b), sigfigs, debug):
            return False
    return True

def backward_test(f, sigfigs, iters, debug):
    for i in range(iters):
        a,b = gen_random_tensors()
        a.init_grad()
        b.requires_grad=True
        f(a).sum().backward()
        f(b).sum().backward()
        if not compare(a.grad, b.grad, sigfigs, debug):
            return False
    return True

def full_test(f, sigfigs=5, iters=100, debug=False):
    if not forward_test(f, sigfigs, iters, debug):
        print("Failed forward test")
        return False
    if not backward_test(f, sigfigs, iters, debug):
        print("Failed backward test")
        return False
    return True

'''
Todo
- dot
- matmul
- binary pow
- more comprehensive test for binary (not x * x)
'''
# to get warnings out of the way
full_test(lambda x: x)
print()
print()

# params
sigfigs = 6
iters = 10
debug = False

# map function tests
print("MAP TESTS")
vals = [1,2,3,-33,0.321,.00412,-0.0242]
for v in vals:
    add = lambda x : x + v
    sub = lambda x : x - v
    mul = lambda x : x * v
    div = lambda x : x / v
    pow = lambda x : (x.relu()+1) ** v

    print(f"Testing add on {v}")
    if not full_test(add, sigfigs, iters, debug): break
    print(f"Testing sub on {v}")
    if not full_test(sub, sigfigs, iters, debug): break
    print(f"Testing mul on {v}")
    if not full_test(mul, sigfigs, iters, debug): break
    print(f"Testing div on {v}"); 
    if not full_test(div, sigfigs, iters, debug): break
    print(f"Testing pow on {v}")
    if not full_test(pow, sigfigs, iters, debug): break

print("\n\nBINARY TESTS")
add = lambda x : x + x
sub = lambda x : x - x
mul = lambda x : x * x
div = lambda x : x / x
pow = lambda x : (x.relu()+1) ** x
def binary_tests():
    print(f"Testing binary add")
    if not full_test(add, sigfigs, iters, debug): return
    print(f"Testing binary sub")
    if not full_test(sub, sigfigs, iters, debug): return
    print(f"Testing binary mul")
    if not full_test(mul, sigfigs, iters, debug): return
    print(f"Testing binary div"); 
    if not full_test(div, sigfigs, iters, debug): return
    # print(f"Testing binary pow")
    # if not full_test(pow, sigfigs, iters, debug): return
binary_tests()