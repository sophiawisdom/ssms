import triton
import triton.language as tl
import torch
import math
import numpy as np

from torch.nn import functional as F
from einops import rearrange
def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')
    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)
        return torch.einsum('lsr,blr->bsl', w2_bfly, out1).reshape(batch, s * l)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        # out1 = F.linear(x, w1_dense)
        out1 = x @ w1_dense
        print(f"before {out1.shape=} {l=}")
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)
        print(f"after {out1.shape=}")
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        # out2 = F.linear(out1, w2_dense)
        out2 = out1 @ w2_dense
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out1, out2

@torch.jit.script
def optimized_forward(x, w1_bfly, w2_bfly):
    batch_dim = x.shape[0]
    n = x.shape[1]
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    # out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
    out1 = torch.bmm(x_reshaped, w1_bfly, out=out1)
    out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    # out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    out2 = torch.bmm(out1, w2_bfly, out=out2)
    out2 = out2.permute(1, 2, 0).reshape(batch_dim, s * l)
    return out2

@triton.jit
def blockdiag_triton_matmul(x_ptr, weight_ptr, out_ptr, ROOT_N: tl.constexpr, BATCH_SIZE: tl.constexpr):
    matrix_index = tl.program_id(axis=0) # which block are we on in the block-diagonal
    initial_two_block_indices = (tl.arange(0, BATCH_SIZE) * ROOT_N * ROOT_N)[:,None] + matrix_index * ROOT_N + tl.arange(0, ROOT_N)[None,:]
    x_block = tl.load(x_ptr + initial_two_block_indices)
    weights = tl.load(weight_ptr + matrix_index * ROOT_N * ROOT_N + (tl.arange(0, ROOT_N)[:,None] * ROOT_N) + tl.arange(0, ROOT_N)[None,:])
    matmul_result = tl.dot(x_block, weights, allow_tf32=False).to(weights.dtype)
    tl.store(out_ptr + initial_two_block_indices, matmul_result)

@triton.jit
def blockdiag_permute(x_ptr, weight_ptr, out_ptr, ROOT_N: tl.constexpr, BATCH_SIZE: tl.constexpr):
    matrix_index = tl.program_id(axis=0) # which block are we on in the block-diagonal
    initial_two_block_indices = (tl.arange(0, BATCH_SIZE) * ROOT_N * ROOT_N)[:,None] + matrix_index * ROOT_N + tl.arange(0, ROOT_N)[None,:]
    x_block = tl.load(x_ptr + initial_two_block_indices) # block is the batched permuted X

    weights = tl.load(weight_ptr + matrix_index * ROOT_N * ROOT_N + (tl.arange(0, ROOT_N)[:,None] * ROOT_N) + tl.arange(0, ROOT_N)[None,:])
    matmul_result = tl.dot(x_block, weights, allow_tf32=False).to(weights.dtype)

    indices = matrix_index*ROOT_N + tl.arange(0, ROOT_N) # e.g. {4,5,6,7}
    permuted_indices = (indices % ROOT_N) * ROOT_N + indices // ROOT_N # -> {1, 5, 9, 13}
    permuted_two_block_indices = (tl.arange(0, BATCH_SIZE) * ROOT_N * ROOT_N)[:,None] + permuted_indices[None,:]

    tl.store(out_ptr + permuted_two_block_indices, matmul_result)

def triton_blockdiag(x, w1, w2):
    BATCH = x.shape[0]
    N = x.shape[1]
    root_n = int(math.sqrt(N))
    out = torch.empty(BATCH, N, device="cuda", dtype=torch.bfloat16)
    out2 = torch.empty(BATCH, N, device="cuda", dtype=torch.bfloat16)
    blockdiag_permute[(root_n,)](x, w1, out, root_n, BATCH)
    blockdiag_permute[(root_n,)](out, w2, out2, root_n, BATCH)
    return out2

@triton.testing.perf_report(triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            2**(i*2) for i in range(4, 9) # 256 to 65536
        ],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        y_log=True,
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', "dense"],  # possible values for `line_arg`
        line_names=['Triton', "torch", "dense"],  # label name for the lines
        styles=[('red', 'solid'), ("blue", "solid"), ("green", "solid")],  # line styles
        ylabel='s',  # label name for the y-axis
        plot_name=f'monarch-performance',  # name for the plot. Used also as a file name for saving the plot.
        args={"BATCH": 64},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_monarch(N, provider, BATCH):
    root_n = int(math.sqrt(N))
    x = torch.randn(BATCH, N, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
    w1_dense = torch.block_diag(*torch.unbind(w1, dim=0))
    w2 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(BATCH, N, device="cuda", dtype=torch.bfloat16)
    out2 = torch.empty(BATCH, N, device="cuda", dtype=torch.bfloat16)
    dense = torch.randn((N, N), device="cuda", dtype=torch.bfloat16)

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_blockdiag(x, w1, w2))
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: optimized_forward(x, w1, w2))
    elif provider == "dense":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, dense))
    else:
        raise ValueError("got unknown provider", provider)
    
    return ms, max_ms, min_ms

benchmark_monarch.run(print_data=True)

# torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(5)
N = (65536)
root_n = int(math.sqrt(N))
BATCH = 32
x = torch.randn(BATCH, N, device='cuda', dtype=torch.bfloat16)
w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
w1_dense = torch.block_diag(*torch.unbind(w1, dim=0))
w2 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
out = torch.zeros(BATCH, N, device="cuda", dtype=torch.bfloat16)
out2 = torch.zeros(BATCH, N, device="cuda", dtype=torch.bfloat16)
print("about to call ours!")
blockdiag_permute[(root_n,)](x, w1, out, root_n, BATCH)
torch.cuda.synchronize()
blockdiag_permute[(root_n,)](out, w2, out2, root_n, BATCH)
torch.cuda.synchronize()
theirs_initial, theirs = blockdiag_butterfly_multiply_reference(x, w1, w2, version=3)
theirs_manual = x @ w1_dense
theirs_manual_permute = rearrange(theirs, 'b (r l) -> b (l r)', l=16)
theirs_optimized = optimized_forward(x, w1, w2)
breakpoint()