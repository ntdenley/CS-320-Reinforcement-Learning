import pytest
import random
from math import log10
import torch
import numpy as np
from ml.array import Array

# testing parameters
mindims = 2
maxdims = 2
minshape = 4
maxshape = 4
multiplier = 50

min_single_sigfig = 2
min_avg_sigfig = 5

class TestArray:

    def setup_method(self, method):
        ndims = random.randint(mindims,maxdims)
        shape = [random.randint(minshape,maxshape) for i in range(ndims)]
        self.b1 = torch.rand(shape) * multiplier
        self.b2 = torch.rand(shape) * multiplier
        self.a1 = Array(self.b1.tolist())
        self.a2 = Array(self.b2.tolist())

    def tensors_eq(self, a, b, single_sig_threshold=min_single_sigfig):
        assert isinstance(a, Array) and isinstance(b, torch.Tensor)
        assert len(a.node.shape) == len(b.shape), f"ndim mismatch: a.ndim = {len(a.node.shape)}, b.ndim = {len(b.shape)}"
        assert a.node.shape == list(b.shape), f"Shape mismatch: a.shape = {a.node.shape}, b.shape = {b.shape}"
        sig_history = []
        for val_a, val_b in zip(a.eval().flatten(), b.flatten()):
            if val_a == 0 and val_b != 0:
                correct_sigfigs = int(log10(abs(1/val_b)))
            elif val_a != 0 and val_b == 0:
                correct_sigfigs = int(log10(abs(1/val_a)))
            else:
                error = abs(val_a - val_b)
                if error <= 10**(-(min_avg_sigfig+1)): continue # deal with zeros
                correct_sigfigs = int(log10(abs(val_a/error)))
            if correct_sigfigs < single_sig_threshold:
                return correct_sigfigs, False
            sig_history.append(correct_sigfigs)
        if len(sig_history) == 0: return 1e9, True
        avg_sigs = sum(sig_history)/len(sig_history)
        return avg_sigs, True

    def shape_specific_test(self, tensor_a_shape, tensor_b_shape, operation, op_name):
        b1 = torch.rand(tensor_a_shape) * multiplier
        b2 = torch.rand(tensor_b_shape) * multiplier
        a1 = Array(b1.tolist())
        a2 = Array(b2.tolist())

        # Forward
        sigs, result = self.tensors_eq(operation(a1, a2), operation(b1, b2))
        assert result, f"{op_name} forward failed: some elements agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{op_name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

        # Backward
        b1.requires_grad = True
        b2.requires_grad = True
        operation(a1, a2).build_backward()
        operation(b1, b2).sum().backward()
        sigs_a, result_a = self.tensors_eq(a1.grad(), b1.grad)
        sigs_b, result_b = self.tensors_eq(a2.grad(), b2.grad)
        assert result_a and result_b, f"{op_name} backward failed: some elements agreed on only min({sigs_a}, {sigs_b}) / {min_single_sigfig} sigs"
        assert min(sigs_a, sigs_b) >= min_avg_sigfig, f"{op_name} backward failed: on average agreed on only min({sigs_a}, {sigs_b}) / {min_avg_sigfig} sigs"
    
    def generic_forward_unary(self, f, name):
        sigs, result = self.tensors_eq(f(self.a1), f(self.b1))
        print("\n\n", name)
        print("f(a1):")
        print(f(self.a1).eval())
        print("f(b1):")
        print(f(self.b1))
        assert result, f"{name} forward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    def generic_backward_unary(self, f, name):
        self.b1.requires_grad = True
        f(self.a1).build_backward()
        f(self.b1).sum().backward()
        print("\n\n", name)
        print("a1:")
        print(self.a1.eval())
        print(self.a1.grad().eval())
        print("b1:")
        print(self.b1)
        print(self.b1.grad)
        sigs, result = self.tensors_eq(self.a1.grad(), self.b1.grad)
        assert result, f"{name} backward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} backward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs" 

    def generic_forward_binary(self, f, name):
        sigs, result = self.tensors_eq(f(self.a1, self.a2), f(self.b1, self.b2))
        assert result, f"{name} forward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    def generic_backward_binary(self, f, name):
        self.b1.requires_grad=True
        self.b2.requires_grad=True
        out = f(self.a1, self.a2)
        out.build_backward()
        self.a1.grad().eval()
        self.a2.grad().eval()
        f(self.b1, self.b2).sum().backward()
        
        sigs, result = self.tensors_eq(self.a1.grad(), self.b1.grad) \
                    and self.tensors_eq(self.a2.grad(), self.b2.grad)  
        assert result, f"{name} backward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} backward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    @pytest.mark.parametrize("tensor_a_shape, tensor_b_shape, operation, op_name, expect_failure", [
        ([2, 2], [1], lambda x, y: x + y, "simple broadcast add", False),
        ([5, 1, 7], [1, 6, 7], lambda x, y: x + y, "simple broadcast add", False),
        ([1], [8, 3, 2], lambda x, y: x * y, "scalar broadcast mul", False),
        ([4, 4], [3, 3], lambda x, y: x - y, "mismatch shape broadcast sub", True),
        ([1, 100], [100, 1], lambda x, y: x / y, "large broadcast div", False),
        ([2, 3, 1], [2, 1, 4], lambda x, y: x * y, "multi-dim broadcast mul", False),
        ([10], [1, 10], lambda x, y: x - y, "row vs column broadcast sub", False),
        ([3, 5], [5], lambda x, y: x + y, "add row vector", False),
        ([3, 1], [1, 4], lambda x, y: x / y, "div column by row", False),
        ([15, 1, 6, 1], [7, 1, 5], lambda x, y: x + y, "complex broadcast add", False),
        ([8, 1, 6, 1], [7, 1, 1], lambda x, y: x - y, "complex broadcast sub with singleton dims", False),
        ([1], [1], lambda x, y: x * y, "scalar broadcast mul trivial", False),
        ([2, 2], [3, 2, 2], lambda x, y: x + y, "add with leading broadcast dimension", False),
        ([2, 2, 2], [2], lambda x, y: x / y, "div with trailing broadcast dimension", False),
        ([5, 1, 1], [1, 4, 1], lambda x, y: x * y, "mul with middle broadcast dimension", False),
        ([1, 3, 1], [3, 1, 3], lambda x, y: x - y, "sub with alternating broadcast dimensions", False),
        ([1, 3, 1], [3, 1, 3], lambda x, y: (x)/1e10  * (y+x+y).log(), "combo", False),
    ])
    def test_broadcast_operations(self, tensor_a_shape, tensor_b_shape, operation, op_name, expect_failure):
        if expect_failure:
            with pytest.raises(ValueError):
                self.shape_specific_test(tensor_a_shape, tensor_b_shape, operation, op_name)
        else:
            self.shape_specific_test(tensor_a_shape, tensor_b_shape, operation, op_name)

    binary_ops = [
        (lambda x, y: x + y, "add"),
        (lambda x, y: x - y, "sub"),
        (lambda x, y: x * y, "mul"),
        (lambda x, y: x / y, "div"),
        (lambda x, y: x * x * y + x + y * y / x,                    "combo1"),
        (lambda x, y: (x.flatten()).dot((y.flatten())),             "dot"),
        (lambda x, y: x @ y,                                        "matmul")
    ]

    @pytest.mark.parametrize("f, name", binary_ops)
    def test_binary_forward(self, f, name):
        self.generic_forward_binary(f, name)
    
    @pytest.mark.parametrize("f, name", binary_ops)
    def test_binary_backward(self, f, name):
        self.generic_backward_binary(f, name)

    unary_ops = [
        (lambda x: x, "                         noop"),
        (lambda x: x.sqrt(), "                  sqrt"),
        (lambda x: x.square(), "                square"),
        (lambda x: x.negative(), "              negative"),
        (lambda x: x.reciprocal(),             "reciprocal"),
        (lambda x: x.sin(),                    "sin"),
        (lambda x: x.cos(),                    "cos"),
        (lambda x: x.tan(),                    "tan"),
        (lambda x: x.sum(),                    "sum"),
        (lambda x: x.sum(axis=0),              "sum0"),
        (lambda x: x.sum(axis=1),              "sum1"),
        (lambda x: x.sum(axis=None, keepdims=True),    "sum"),
        (lambda x: x.sum(axis=1, keepdims=True),       "sum"),
        (lambda x: x.max(),                            "max"),
        (lambda x: x.max(axis=0) if isinstance(x, Array) 
                    else x.max(axis=0).values,                      "max0"),
        (lambda x: x.max(axis=1) if isinstance(x, Array) 
                    else x.max(axis=1).values,                      "max1"),
        (lambda x: x.max(axis=0, keepdims=True) if isinstance(x, Array) 
                    else x.max(axis=0, keepdims=True).values,       "max1keepdims"),
        (lambda x: x.exp(),                    "exp"),
        (lambda x: x.log(),                     "log"),
        (lambda x: x.relu(),                   "relu"),
        (lambda x: x.sqrt().exp(),             "unary+unary"),
        (lambda x: x.relu().sqrt(),            "relu+sqrt"),
        (lambda x: x/x,                         "div self"),
        (lambda x: x.square()/(x.square()),    "square self"),
        (lambda x: x.sqrt()/(x.sqrt()),        "sqrt self"),
        (lambda x: x.softmax(0),               "softmax0"),
        (lambda x: x.softmax(1),               "softmax1"),
        (lambda x: x + 100,                    "map add"),
        (lambda x: x - 0.00321,                "map sub"),
        (lambda x: 1 - x,                      "map rsub"),
        (lambda x: -x,                         "neg"),
        (lambda x: x * 32,                     "map mul"),
        (lambda x: 32 * x,                     "map rsmul"),
        (lambda x: x * 1e20,                   "large map mul"),
        (lambda x: x * -0.0031,                "neg map mul"),
        (lambda x: x * 0,                      "zero map mul"),
        (lambda x: x / 3,                      "map div"),
        (lambda x: x / -0.004,                 "neg map div"),
        (lambda x: 1e10 / x,                   "large div by x"),
        #i'll throw views in here to ensure deriv works
        (lambda x: x.reshape(-1,2) * 32,       "reshape"),
        (lambda x: x.flatten() * 32,           "reshape"),
        (lambda x: x.t() * 32,                 "transpose"),
        (lambda x: x.unsqueeze(1) * 32,        "unsqueeze"),
        (lambda x: x.sigmoid(),                "sigmoid")
    ]

    @pytest.mark.parametrize("f, name", unary_ops)
    def test_basic_forward(self, f, name):
        self.generic_forward_unary(f, name)

    @pytest.mark.parametrize("f, name", unary_ops)
    def test_basic_backward(self, f, name):
        self.generic_backward_unary(f, name)
    
    def test_div_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            self.a1 / 0

    @pytest.mark.parametrize("input1, input2, expected_eq, expected_ne, expected_lt, expected_le, expected_gt, expected_ge", [
        (
            [1, 2, 3],       # input1
            [3, 2, 1],       # input2
            [0, 1, 0],       # expected_eq (1 if equal, 0 otherwise)
            [1, 0, 1],       # expected_ne (1 if not equal, 0 otherwise)
            [1, 0, 0],       # expected_lt (-1 if less, 0 otherwise)
            [1, 1, 0],       # expected_le (-1 or 0 if less or equal, 1 if greater)
            [0, 0, 1],       # expected_gt (1 if greater, 0 otherwise)
            [0, 1, 1],       # expected_ge (1 or 0 if greater or equal, -1 if less)
        ),
        (
            [4, 5, 6],
            [4, 6, 6],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
        ),
    ])
    def test_comparison_operations(self, input1, input2, expected_eq, expected_ne, expected_lt, expected_le, expected_gt, expected_ge):
        a1 = Array(input1)
        a2 = Array(input2)
        
        result_eq = (a1 == a2).eval()
        result_ne = (a1 != a2).eval()
        result_lt = (a1 < a2).eval()
        result_le = (a1 <= a2).eval()
        result_gt = (a1 > a2).eval()
        result_ge = (a1 >= a2).eval()
        
        np.testing.assert_array_equal(result_eq, expected_eq, err_msg="Equality test failed")
        np.testing.assert_array_equal(result_ne, expected_ne, err_msg="Inequality test failed")
        np.testing.assert_array_equal(result_lt, expected_lt, err_msg="Less than test failed")
        np.testing.assert_array_equal(result_le, expected_le, err_msg="Less or equal test failed")
        np.testing.assert_array_equal(result_gt, expected_gt, err_msg="Greater than test failed")
        np.testing.assert_array_equal(result_ge, expected_ge, err_msg="Greater or equal test failed")