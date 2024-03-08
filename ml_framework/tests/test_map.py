'''
Tests map operation
- sum, log, exp, relu, +, -, *, /, **
'''

import pytest
import random
from math import log10
from torch import zeros
from torch import Tensor as tTensor
from ..src.tensor import Tensor

# testing parameters
mindims = 2
maxdims = 2
minshape = 40
maxshape = 40
multiplier = 50

min_single_sigfig = 3
min_avg_sigfig = 5

class TestTensor:

    def setup_method(self, method):
        ndims = random.randint(mindims,maxdims)
        shape = [random.randint(minshape,maxshape) for i in range(ndims)]
        self.a = Tensor.rand(shape, -1, 1) * multiplier
        self.b = tTensor(self.a.data).reshape(self.a.shape)

    # are these two tensors equal?
    def tensors_eq(self, a, b, single_sig_threshold=min_single_sigfig):
        assert len(a.shape) == len(b.shape), f"ndim mismatch: a.ndim = {len(a.shape)}, b.ndim = {len(b.shape)}"
        assert a.shape == list(b.shape), f"Shape mismatch: a.shape = {a.shape}, b.shape = {b.shape}"
        sig_history = []
        for val_a, val_b in zip(a.data, b.flatten()):
            error = abs(val_a - val_b)
            if error <= 1e-9: continue
            correct_sigfigs = int(log10(abs(val_a/error)))
            if correct_sigfigs < single_sig_threshold:
                return correct_sigfigs, False
            sig_history.append(correct_sigfigs)
        if len(sig_history) == 0: return 1e9, True
        avg_sigs = sum(sig_history)/len(sig_history)
        return avg_sigs, True
    
    def generic_forward(self, f, name):
        sigs, result = self.tensors_eq(f(self.a), f(self.b))
        assert result, f"{name} forward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    def generic_backward(self, f, name):
        self.a.init_grad()
        self.b.requires_grad = True
        f(self.a).sum().backward()
        f(self.b).sum().backward()
        sigs, result = self.tensors_eq(self.a.grad, self.b.grad)
        assert result, f"{name} backward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} backward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    # operations for which no exceptions, divby0, inf, -inf, 
    # or other funny business is expected
    basic_ops = [
        (lambda x: x.sum().reshape([1]),       "sum"),
        (lambda x: x.exp(),                    "exp"),
        (lambda x: (x.relu()+1e-9).log(),       "normal log"),
        (lambda x: x.relu(),                   "relu"),
        (lambda x: x + 100,                    "map add"),
        (lambda x: x - 0.00321,                "map sub"),
        (lambda x: -x,                         "neg"),
        (lambda x: x * 32,                     "map mul"),
        (lambda x: x * 1e20,                   "large map mul"),
        (lambda x: x * -0.0031,                "neg map mul"),
        (lambda x: x * 0,                      "zero map mul"),
        (lambda x: x.relu() ** 3.1415,         "relu pow pi"),
        (lambda x: (x.relu()+1e-9) ** 0,        "relu pow 0"),
        (lambda x: (x+1e-9) ** -3,              "pow neg three"),
        (lambda x: x / 3,                      "map div"),
        (lambda x: x / -0.004,                 "neg map div"),
        (lambda x: 1e10 / x,                   "large div by x")
    ]

    @pytest.mark.parametrize("f, name", basic_ops)
    def test_basic_forward(self, f, name):
        self.generic_forward(f, name)

    @pytest.mark.parametrize("f, name", basic_ops)
    def test_basic_backward(self, f, name):
        self.generic_backward(f, name)
    
    def test_log_zero(self):
        self.a = Tensor.fill([10], 0)
        self.b = zeros([10])+1e9
        f = lambda x: x.log()
        self.generic_forward(f, "Log of zero")

    def test_log_neg(self):
        self.a = Tensor.fill([10], -1)
        with pytest.raises(ValueError):
            self.a.log()
        
    def test_pow_zero_zero(self):
        self.a = Tensor.fill([10], 0)
        self.b = zeros([10])
        f = lambda x: x ** 0
        self.generic_forward(f, "map 0^0")

    def test_pow_zero_neg(self):
        self.a = Tensor.fill([10], 0)
        with pytest.raises(ZeroDivisionError):
            self.a ** -1

    def test_pow_neg_frac(self):
        self.a = Tensor.fill([10], -1)
        with pytest.raises(ValueError):
            self.a ** 0.5

    def test_div_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            self.a / 0