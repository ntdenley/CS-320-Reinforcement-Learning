from src.graph import ComputeNode, LazyOp
from src.op import OpType
from src.backward import build_backward_graph, zero_grad, alloc_grad
from src.graph_utils import view_graph
import torch 
from math import log10
import pytest

min_single_sigfig = 2
min_avg_sigfig = 5

def get_nelem(shape):
    nelem = 1
    for s in shape: nelem *= s
    return nelem

class TestComputeNode:

    def tensors_eq(self, a, b, single_sig_threshold=min_single_sigfig):
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
        
    # unary_inplace_ops = [
    #     (OpType.Inplace.Sqrt),
    #     (OpType.Inplace.Square),
    #     (OpType.Inplace.Sin),
    #     (OpType.Inplace.Cos),
    #     (OpType.Inplace.Tan)
    # ]

    # @pytest.mark.parametrize("optype", unary_inplace_ops)
    # def test_unary_inplace_op(self, optype):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = self.make_op(optype, inputNodes=[self.cn1])
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    

    binary_inplace_ops = [
        # (OpType.Inplace.Add),
        # (OpType.Inplace.Subtract),
        # (OpType.Inplace.Divide),
        (OpType.Inplace.Multiply),
        # (OpType.Inplace.Matmul),
    ]

    s=2

    binary_inplace_ops_shapes = [
        ([s,s],[s,s],[s,s]),
        ([s],[s,s],[s,s]),
        ([s,s],[s],[s,s]),
        ([1],[s,s],[s,s]),
        # ([s,s],[1],[s,s]),
    ]

    @pytest.mark.parametrize("optype", binary_inplace_ops)
    @pytest.mark.parametrize("shape1, shape2, outshape", binary_inplace_ops_shapes)
    def test_binary_inplace_op(self, optype, shape1, shape2, outshape):
        np1 = torch.rand(shape1)
        np2 = torch.rand(shape2)

        cn1 = ComputeNode(shape1)
        cn2 = ComputeNode(shape2)
        cn1.loadop(LazyOp(OpType.Alloc.FromList, shape1, np1.flatten()))
        cn2.loadop(LazyOp(OpType.Alloc.FromList, shape2, np2.flatten()))

        #cn
        out_cp = ComputeNode(outshape)
        out_cp.loadop(LazyOp(optype, inputNodes = [cn1, cn2]))
        out_cp.allocate()
        out_cp.evaluate()
        build_backward_graph(out_cp)
        alloc_grad(out_cp)
        zero_grad(out_cp)
        view_graph(out_cp)
        # cp1_grad = cn1.grad.evaluate()
        # cp2_grad = cn2.grad.evaluate()

        #torch
        name = optype.name.lower()
        method = getattr(torch, name)
        np1.requires_grad = True
        np2.requires_grad = True
        out_np = method(np1, np2)
        out_np.sum().backward()
        np1_grad = np1.grad
        np2_grad = np2.grad

        # print("NORMAL")
        # print(cn1.evaluate())
        # print(np1)
        # print("GRAD")
        # print(cp1_grad)
        # print(np1_grad)

        # sigs, success = self.tensors_eq(cp1_grad, np1_grad)
        # assert success, f"{name} failed cp1 vs torch1: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        # assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
        # sigs, success = self.tensors_eq(cp2_grad, np2_grad)
        # assert success, f"{name} failed cp2 vs torch2: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        # assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    # reduce_ops = [
    #     (OpType.Inplace.Sum),
    # ]

    # @pytest.mark.parametrize("optype", reduce_ops)
    # def test_reduce_simple(self, optype):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = ComputeNode(())
    #     out_cp.loadop(LazyOp(optype, None, False, inputNodes=[self.cn1]))
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    
 
    # @pytest.mark.parametrize("optype", reduce_ops)
    # def test_reduce_keepdims(self, optype):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = ComputeNode([1,1])
    #     out_cp.loadop(LazyOp(optype, None, True, inputNodes=[self.cn1]))
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1, keepdims=True)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"    

    # @pytest.mark.parametrize("optype", reduce_ops)
    # def test_reduce_axis(self, optype):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = ComputeNode(self.shape[:-1])
    #     out_cp.loadop(LazyOp(optype, 0, False, inputNodes=[self.cn1]))
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1, axis=0)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    # def test_reshape(self, optype=OpType.View.Reshape):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = ComputeNode(self.np1.flatten().shape)
    #     out_cp.loadop(LazyOp(optype, self.np1.flatten().shape, inputNodes=[self.cn1]))
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1, self.np1.flatten().shape)
        
    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    # def test_transpose(self, optype=OpType.View.Transpose):
    #     name = optype.name.lower()
    #     #cn
    #     out_cp = ComputeNode([self.shape[1], self.shape[0]])
    #     out_cp.loadop(LazyOp(optype, inputNodes=[self.cn1]))
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    # @pytest.mark.parametrize("optype", binary_inplace_ops)
    # def test_reevaluate(self, optype):
    #     name = optype.name.lower()
    #         # first pass
    #     #cn
    #     out_cp = self.make_op(optype, inputNodes=[self.cn1, self.cn2])
    #     out_cp.allocate()

    #     #np
    #     method = getattr(np, name)
    #     out_np = method(self.np1, self.np2)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"

    #         # subtle change, and second pass
    #     self.change = np.random.rand(self.nelem).reshape(self.shape)
    #     #cn
    #     self.cn1.data -= self.change
    #     out_cp.set_reevaluate()
    #     #np
    #     self.np1 -= self.change
    #     out_np = method(self.np1, self.np2)

    #     sigs, success = self.tensors_eq(out_cp.evaluate(), out_np)
    #     assert success, f"{name} failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
    #     assert sigs >= min_avg_sigfig, f"{name} failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
