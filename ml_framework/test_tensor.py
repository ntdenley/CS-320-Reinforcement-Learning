from tensor import Tensor
import torch
import random
from math import log10
from matplotlib import pyplot as plt
from collections import Counter

# params

# tensor generation
mindims = 2
maxdims = 2
minshape = 40
maxshape = 40
multiplier = 10000
map_vals = [1,2,3,-33,0.321,.00412,-0.0242]

# testing conditions
sigfigs = 1
iters = 1
debug = True
graph = True

# what test to do
test_map = False
test_binary = False
test_matmul = True # only set to true is mindims = maxdims = 2

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
    ndims = random.randint(mindims,maxdims)
    shape = [random.randint(minshape,maxshape) for i in range(ndims)]
    a = Tensor.rand(shape, -1, 1) * multiplier
    b = torch.Tensor(a.data)
    b = b.reshape(a.shape)
    
    # switching torch.tensor to float64 results in more tests passed for sigfigs=6,
    # which suggests our tensor is more precise (though much less efficient) than 
    # pytorch, so passing tests with 5-6 sig figs is good enough.
    b = b.to(torch.float64)

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

def graph_sigfigs(f, iters=10):
    def compare(a, b, sig_track):
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
            sig_track.append(correct_sigfigs)
    
    # this function, which takes an array of numbers and graphs their distribution,
    # was created by chatGPT.
    def graph_distribution(sig_track, title):
        number_counts = Counter(sig_track)
        total_elements = sum(number_counts.values())
        percentages = {number: (count / total_elements) * 100 for number, count in number_counts.items()}
        sorted_numbers = sorted(percentages.items())  # Sort by the number
        numbers, percentages = zip(*sorted_numbers)  # Unzip the sorted pairs
        plt.bar(numbers, percentages)
        plt.title(title)
        plt.xlabel('SigFigs')
        plt.ylabel('Percentage')
        for i, percentage in enumerate(percentages):
            plt.text(numbers[i], percentage, f"{percentage:.2f}%", ha='center', va='bottom')
        plt.show()

    # forward pass
    sig_track = []
    for i in range(iters):
        a, b = gen_random_tensors()
        compare(f(a), f(b), sig_track)
    graph_distribution(sig_track, 'Distribution of Accuracy up to SigFigs (Forward pass)')

    sig_track = []
    for i in range(iters):
        a,b = gen_random_tensors()
        a.init_grad()
        b.requires_grad=True
        f(a).sum().backward()
        f(b).sum().backward()
        compare(a.grad, b.grad, sig_track)
    graph_distribution(sig_track, 'Distribution of Accuracy up to SigFigs (Backward pass)')
     

'''
Todo
- matmul
- binary pow
- more comprehensive test for binary (not x * x)
'''
# to get warnings out of the way
full_test(lambda x: x)
print()
print()

print(f"TESTING WITH sigfigs = {sigfigs}, iters = {iters}\n")

# map function tests
if test_map:
    print("MAP TESTS")
    for v in map_vals:
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

if test_binary:
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

if test_matmul:
    print("\n\nMATMUL TEST")
    matmul = lambda x: x @ (x.reshape([x.shape[1], x.shape[0]]))
    if graph: 
        graph_sigfigs(matmul, iters)
    else:
        full_test(matmul, sigfigs, iters, debug)