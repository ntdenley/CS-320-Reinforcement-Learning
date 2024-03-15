
'''
Speed records

time (us)            Mine              Pytorch
matmul      |      2,469,114    |        32
binary add  |         23,736    |        11
map add     |          1,004    |        17
'''

import pytest
import random
from math import log10
import torch
from torch import zeros
from torch import Tensor as tTensor
from ..src.tensor import Tensor

pytestmark = pytest.mark.skip("for some reason")

# testing parameters
mindims = 2
maxdims = 2
minshape = 100
maxshape = 100
multiplier = 50

class TestTensor:

    def setup_method(self, method):
        ndims = random.randint(mindims,maxdims)
        shape = [random.randint(minshape,maxshape) for i in range(ndims)]
        self.a1 = Tensor.rand(shape, -1, 1) * multiplier
        self.a2 = Tensor.rand(shape, -1, 1) * multiplier
        self.b1 = tTensor(self.a1.data).reshape(self.a1.shape).to("cpu")
        self.b2 = tTensor(self.a2.data).reshape(self.a2.shape).to("cpu")

    def mine_matmul(self):
        self.a1 @ self.a2

    def theirs_matmul(self):
        self.b1 @ self.b2

    def test_mine_matmul(self, benchmark):
        result = benchmark(self.mine_matmul)
    
    def test_theirs_matmul(self, benchmark):
        result = benchmark(self.theirs_matmul)

    def mine_add(self):
        self.a1 + self.a2

    def theirs_add(self):
        self.b1 + self.b2

    def test_mine_add(self, benchmark):
        result = benchmark(self.mine_add)
    
    def test_theirs_add(self, benchmark):
        result = benchmark(self.theirs_add)

    def mine_add_map(self):
        self.a1 + 1032.03

    def theirs_add_map(self):
        self.b1 + 1032.03

    def test_mine_add_map(self, benchmark):
        result = benchmark(self.mine_add_map)
    
    def test_theirs_add_map(self, benchmark):
        result = benchmark(self.theirs_add_map)