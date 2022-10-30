import triton
import triton.language as tl
import torch
import math

def forward(x, w1_bfly, w2_bfly):
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q
    times = []
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
    out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    print(", ".join([f"{(times[i]-times[i-1])*1000:.2f}ms" for i in range(1, len(times))]))
    return out2

@triton.jit
def ours(x_ptr, w1_bfly_ptr, w2_bfly_ptr, out_ptr, ROOT_N: tl.constexpr, N: tl.constexpr, BATCH_SIZE: tl.constexpr):
    # load permuted X (batched!)

    # X is (N, BATCH_SIZE)

    matrix_index = tl.program_id(axis=0) # which matrix are we on
    indices = matrix_index*ROOT_N + tl.arange(0, ROOT_N)
    permuted_indices = (indices % ROOT_N) * ROOT_N + indices // ROOT_N
    print("permuted indices", permuted_indices)
    indices_into_x = (permuted_indices * BATCH_SIZE)
    print("indices into x", indices_into_x)
    permuted_two_block_indices = indices_into_x[None,:] + tl.arange(0, BATCH_SIZE)[:,None]
    print("two_block_indices", permuted_two_block_indices)
    block = tl.load(x_ptr + permuted_two_block_indices) # block is the batched permuted X

    w1_bfly = tl.load(w1_bfly_ptr + matrix_index * N + tl.arange(0, ROOT_N)[None,:] * ROOT_N + tl.arange(0, ROOT_N)[:,None])
    print("w1_bfly is", w1_bfly)
    matmul_result = tl.dot(block, w1_bfly)
    print("matmul result is", matmul_result)

    initial_two_block_indices = (indices * BATCH_SIZE)[None,:] + tl.arange(0, BATCH_SIZE)[:,None]
    tl.store(x_ptr + permuted_two_block_indices, matmul_result)

    initial_two_block_indices = (indices * BATCH_SIZE)[None,:] + tl.arange(0, BATCH_SIZE)[:,None]
    intermediate_value = tl.load(x_ptr + initial_two_block_indices)
    w2_bfly = tl.load(w2_bfly_ptr + matrix_index * N + tl.arange(0, ROOT_N)[None,:] * ROOT_N + tl.arange(0, ROOT_N)[:,None])
    final_result = tl.dot(intermediate_value, w2_bfly)
    print("final_result", final_result)
    tl.store(out_ptr + initial_two_block_indices, final_result)

# ours()

# for each program, X is (BATCH, ROOT_N), w1 is (ROOT_N, ROOT_N), w2 is (ROOT_N, ROOT_N)
# so for n=16384, it's (16, 128) @ (128, 128)

N = (256)
root_n = int(math.sqrt(N))
BATCH = 16
x = torch.randn(BATCH, N, device='cuda', dtype=torch.bfloat16)
w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
w2 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
out = torch.zeros(BATCH, N, device="cuda", dtype=torch.int32)
print("about to call ours!")
ours[(root_n,)](x, w1, w2, out, root_n, N, BATCH)
breakpoint()