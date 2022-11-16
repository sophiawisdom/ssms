import torch
import math
import os
import subprocess
from torch.utils.cpp_extension import load
import sys

# pip3 install ninja
# sudo pip3 install pybind11[global]

print("compiling monarch")
result = subprocess.run(["ptxas", "--gpu-name=sm_80", "monarch.ptx", "-g", "-lineinfo", "-o", "monarch.o"])
if result.returncode != 0:
    print("FAILED TO COMPILE!!! RESULT IS", result)
    sys.exit(1)
print("completed compiling")

print("about to compile")
monarch = load(name="monarch", sources=["monarch.cpp", "monarch.cu"], extra_ldflags=["-lcuda"])
print("compiled")

# ours()

# for each program, X is (BATCH, ROOT_N), w1 is (ROOT_N, ROOT_N), w2 is (ROOT_N, ROOT_N)
# so for n=16384, it's (16, 128) @ (128, 128)

N = (16384)
root_n = int(math.sqrt(N))
BATCH = 16
x = torch.ones(BATCH, N, device='cuda', dtype=torch.bfloat16)
# x = torch.arange(BATCH*N).reshape((BATCH, N)).to(device="cuda", dtype=torch.bfloat16)
# w1 = torch.randn(root_n, root_n, root_n, device='cuda', dtype=torch.bfloat16)
# w1 = torch.arange(root_n * N).reshape((root_n, root_n, root_n)).to(dtype=torch.bfloat16, device="cuda")
w1 = torch.ones(root_n, root_n, root_n, dtype=torch.bfloat16, device="cuda")
x_small = x.reshape(16*16384)[:2048].reshape(16, 128)
x_small[:,2] = 0
true_out = (x_small @ w1[0]).to(dtype=torch.float32)
print("about to call ours!")
out = torch.zeros(BATCH, N, dtype=torch.float32, device="cuda")
out_new = monarch.forward(x, w1, out)
small_out = out.reshape((16*16384))[:2048].reshape(16, 128)
print(f"Completed! {torch.allclose(small_out, true_out)=}")
print(f"{out.sum()=} {true_out.sum()=}")
breakpoint()
