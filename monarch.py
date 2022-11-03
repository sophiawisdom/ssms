import torch
import math
import subprocess
from torch.utils.cpp_extension import load
import sys

# pip3 install ninja
# sudo pip3 install pybind11[global]

print("compiling monarch")
result = subprocess.run(["ptxas", "--gpu-name=sm_80", "monarch.ptx", "-g", "-o", "monarch.o"])
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
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
    out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
    out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
    out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
    out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2

def forward_w1(x, w1_bfly):
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = np.prod(batch_shape)
    k, q, p = w1_bfly.shape
    assert k * p == n
    x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
    out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
    out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
    out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
    return out1

# ours()

# for each program, X is (BATCH, ROOT_N), w1 is (ROOT_N, ROOT_N), w2 is (ROOT_N, ROOT_N)
# so for n=16384, it's (16, 128) @ (128, 128)

N = (16384)
root_n = int(math.sqrt(N))
BATCH = 16
x = torch.ones(BATCH, N, device='cuda', dtype=torch.bfloat16)
w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
w1 = torch.arange(root_n * N).reshape((root_n, root_n, root_n)) 
x_small = x.reshape(16*16384)[:2048].reshape(16, 128)
true_out = x_small @ w1[0]
print("about to call ours!")
out = monarch.forward(x, w1)
small_out = out.reshape((16*16384))[:2048].reshape(16, 128)
print("Completed!")
print(f"{out.sum()=} {true_out.sum()=}")
breakpoint()