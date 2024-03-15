# this entire file was created by ChatGPT
import pytest
import random
from math import log10
from torch import zeros, randn as torch_randn, ones as torch_ones
from torch import Tensor as tTensor
from ..src.tensor import Tensor

# Constants for tests
multiplier = 50
min_single_sigfig = 2
min_avg_sigfig = 5

class TestTensorBroadcasting:
    @staticmethod
    def tensors_eq(a, b, single_sig_threshold=min_single_sigfig):
        if len(b.shape) == 0: b = b.reshape([1])  # PyTorch scalar tensors are 0D, this library treats them as 1D
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
        avg_sigs = sum(sig_history) / len(sig_history)
        return avg_sigs, True

    def generic_test(self, tensor_a_shape, tensor_b_shape, operation, op_name):
        a1 = Tensor.rand(tensor_a_shape, -1, 1) * multiplier
        a2 = Tensor.rand(tensor_b_shape, -1, 1) * multiplier
        b1 = tTensor(a1.data).reshape(a1.shape)
        b2 = tTensor(a2.data).reshape(a2.shape)

        # Forward
        sigs, result = self.tensors_eq(operation(a1, a2), operation(b1, b2))
        assert result, f"{op_name} forward failed: some elements agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{op_name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

        # Backward
        a1.init_grad()
        a2.init_grad()
        b1.requires_grad = True
        b2.requires_grad = True
        operation(a1, a2).sum().backward()
        operation(b1, b2).sum().backward()
        sigs_a, result_a = self.tensors_eq(a1.grad, b1.grad)
        sigs_b, result_b = self.tensors_eq(a2.grad, b2.grad)
        assert result_a and result_b, f"{op_name} backward failed: some elements agreed on only min({sigs_a}, {sigs_b}) / {min_single_sigfig} sigs"
        assert min(sigs_a, sigs_b) >= min_avg_sigfig, f"{op_name} backward failed: on average agreed on only min({sigs_a}, {sigs_b}) / {min_avg_sigfig} sigs"

    @pytest.mark.parametrize("tensor_a_shape, tensor_b_shape, operation, op_name, expect_failure", [
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
        ([1, 3, 1], [3, 1, 3], lambda x, y: (x+2-y+x)/1e10  * (y.relu()+1e-9).log(), "combo", False),
    ])

    def test_broadcast_operations(self, tensor_a_shape, tensor_b_shape, operation, op_name, expect_failure):
        if expect_failure:
            with pytest.raises(ValueError):
                self.generic_test(tensor_a_shape, tensor_b_shape, operation, op_name)
        else:
            self.generic_test(tensor_a_shape, tensor_b_shape, operation, op_name)

