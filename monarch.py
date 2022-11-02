import triton
import triton.language as tl
import torch
import math
import sys

print("compiling monarch")
result = subprocess.run(["ptxas", "--gpu-name=sm_80", "monarch.ptx", "-o", "monarch.o"])
if result.returncode != 0:
    print("FAILED TO COMPILE!!! RESULT IS", result)
    sys.exit(1)
print("completed compiling")

print("about to compile")
monarch = load(name="monarch", sources=["monarch.cpp", "monarch.cu"], extra_ldflags=["-lcuda"])
print("compiled")

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

# ours()

# for each program, X is (BATCH, ROOT_N), w1 is (ROOT_N, ROOT_N), w2 is (ROOT_N, ROOT_N)
# so for n=16384, it's (16, 128) @ (128, 128)

N = (256)
root_n = int(math.sqrt(N))
BATCH = 16
x = torch.randn(BATCH, N, device='cuda', dtype=torch.bfloat16)
w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
w2 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
out = torch.zeros(BATCH, N, device="cuda", dtype=torch.float32)
print("about to call ours!")
ours[(root_n,)](x, w1, w2, out, root_n, N, BATCH)
breakpoint()