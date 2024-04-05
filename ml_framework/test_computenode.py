from src.graph import ComputeNode, LazyOp
from src.op import OpType
import random
import numpy as np
from math import log10
import pytest

mindims = 2
maxdims = 2
minshape = 40
maxshape = 40

min_single_sigfig = 2
min_avg_sigfig = 5

def get_nelem(shape):
    nelem = 1
    for s in shape: nelem *= s
    return nelem

class TestComputeNode:

    def setup_method(self, method):        
        ndims = random.randint(mindims,maxdims)
        self.shape = [random.randint(minshape,maxshape) for i in range(ndims)]
        self.nelem = get_nelem(self.shape)
        
        self.np1 = np.random.rand(self.nelem)
        self.np2 = np.random.rand(self.nelem)

        self.cn1 = ComputeNode(self.shape)
        self.cn1.loadop(LazyOp(OpType.Alloc.FromList, self.shape, self.np1))
        self.cn2 = ComputeNode(self.shape)
        self.cn2.loadop(LazyOp(OpType.Alloc.FromList, self.shape, self.np2))

        self.np1 = self.np1.reshape(self.shape)
        self.np2 = self.np2.reshape(self.shape)

    def tensors_eq(self, a, b, single_sig_threshold=min_single_sigfig):
        # if len(b.shape) == 0: b = b.reshape([1]) # this needs to be here because pytorch handles scalar tensors as 0 dimensional, whereas i say they are 1d
        assert len(a.shape) == len(b.shape), f"ndim mismatch: a.ndim = {len(a.shape)}, b.ndim = {len(b.shape)}"
        assert a.shape == b.shape, f"Shape mismatch: a.shape = {a.shape}, b.shape = {b.shape}"
        sig_history = []
        for val_a, val_b in zip(a.flatten(), b.flatten()):
            error = abs(val_a - val_b)
            if error <= 1e-9: continue
            correct_sigfigs = int(log10(abs(val_a/error)))
            if correct_sigfigs < single_sig_threshold:
                return correct_sigfigs, False
            sig_history.append(correct_sigfigs)
        if len(sig_history) == 0: return 1e9, True
        avg_sigs = sum(sig_history)/len(sig_history)
        return avg_sigs, True
        
    def make_op(self, optype, *args, inputNodes):
        out = ComputeNode(self.shape)
        out.loadop(LazyOp(optype, *args, inputNodes = inputNodes))
        return out
    
    unary_inplace_ops = [
        (OpType.Inplace.Sqrt),
        (OpType.Inplace.Square),
        (OpType.Inplace.Sin),
        (OpType.Inplace.Cos),
        (OpType.Inplace.Tan)
    ]

    @pytest.mark.parametrize("optype", unary_inplace_ops)
    def test_unary_inplace_op(self, optype):
        name = optype.name.lower()
        #cn
        out_cp = self.make_op(optype, inputNodes=[self.cn1])
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    

    
    binary_inplace_ops = [
        (OpType.Inplace.Add),
        (OpType.Inplace.Subtract),
        (OpType.Inplace.Divide),
        (OpType.Inplace.Multiply),
        (OpType.Inplace.Matmul),
    ]

    @pytest.mark.parametrize("optype", binary_inplace_ops)
    def test_binary_inplace_op(self, optype):
        name = optype.name.lower()
        #cn
        out_cp = self.make_op(optype, inputNodes=[self.cn1, self.cn2])
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1, self.np2)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    reduce_ops = [
        (OpType.Inplace.Sum),
    ]

    @pytest.mark.parametrize("optype", reduce_ops)
    def test_reduce_simple(self, optype):
        name = optype.name.lower()
        #cn
        out_cp = ComputeNode(())
        out_cp.loadop(LazyOp(optype, None, False, inputNodes=[self.cn1]))
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    
 
    @pytest.mark.parametrize("optype", reduce_ops)
    def test_reduce_keepdims(self, optype):
        name = optype.name.lower()
        #cn
        out_cp = ComputeNode([1,1])
        out_cp.loadop(LazyOp(optype, None, True, inputNodes=[self.cn1]))
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1, keepdims=True)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    

    @pytest.mark.parametrize("optype", reduce_ops)
    def test_reduce_axis(self, optype):
        name = optype.name.lower()
        #cn
        out_cp = ComputeNode(self.shape[:-1])
        out_cp.loadop(LazyOp(optype, 0, False, inputNodes=[self.cn1]))
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1, axis=0)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    def test_reshape(self, optype=OpType.View.Reshape):
        name = optype.name.lower()
        #cn
        out_cp = ComputeNode(self.np1.flatten().shape)
        out_cp.loadop(LazyOp(optype, self.np1.flatten().shape, inputNodes=[self.cn1]))
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1, self.np1.flatten().shape)
        
        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    def test_transpose(self, optype=OpType.View.Transpose):
        name = optype.name.lower()
        #cn
        out_cp = ComputeNode([self.shape[1], self.shape[0]])
        out_cp.loadop(LazyOp(optype, inputNodes=[self.cn1]))
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    @pytest.mark.parametrize("optype", binary_inplace_ops)
    def test_reevaluate(self, optype):
        name = optype.name.lower()
            # first pass
        #cn
        out_cp = self.make_op(optype, inputNodes=[self.cn1, self.cn2])
        out_cp.allocate()

        #np
        method = getattr(np, name)
        out_np = method(self.np1, self.np2)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

            # subtle change, and second pass
        self.change = np.random.rand(self.nelem).reshape(self.shape)
        #cn
        self.cn1.data -= self.change
        out_cp.set_reevaluate()
        #np
        self.np1 -= self.change
        out_np = method(self.np1, self.np2)

        sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
        assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"