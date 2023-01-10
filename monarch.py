import torch
import math
import os
import subprocess
from torch.utils.cpp_extension import load
import sys

# pip3 install ninja
# sudo pip3 install pybind11[global]

print("compiling monarch")
result = subprocess.run(["ptxas", "--gpu-name=sm_80", "monarch.ptx", "-lineinfo", "-o", "monarch.o"])
if result.returncode != 0:
    print("FAILED TO COMPILE!!! RESULT IS", result)
    sys.exit(1)
print("completed compiling")

print("about to compile")
monarch = load(name="monarch", sources=["monarch.cpp", "monarch.cu"], extra_ldflags=["-lcuda"], extra_cflags=["-g"])
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
x_small[:,0] = 0
x_small[:,1] = 1
x_small[:,2] = 2
x_small[:,3] = 3
x_small[:,4] = 4
w1[:,2] = 0
true_out = (x_small @ w1[0])
print("about to call ours!")

# print("X we're passing to monarch is", x.reshape(16*16384)[:2048].reshape(16, 128))
# print("w1 is", w1)
outs = []
true_outs = []
print("index\tptx\ttrue")
for i in range(16):
    x_small[:,4] = i
    true_out = (x_small @ w1[0])
    true_outs.append(true_out)
    out = monarch.forward(x, w1)
    small_out = out.reshape((16*16384))[:2048].reshape(16, 128)
    outs.append(small_out)
    print(f"{i}\t{int(out.to(dtype=torch.float64).sum())}\t{int(true_out.to(dtype=torch.float64).sum())}")
breakpoint()


# si 59?

# python registers_before = [(gdb.execute(f"cuda thread {j}"), [int(gdb.newest_frame().read_register(f"R{i}")) for i in range(40)])[1] for j in range(32)]
# python registers_after = [(gdb.execute(f"cuda thread {j}"), [int(gdb.newest_frame().read_register(f"R{i}")) for i in range(40)])[1] for j in range(32)]
# python diffs = [[hex(after_val)[2:].rjust(8, '0')[4:] + hex(after_val)[2:].rjust(8, '0')[:4] for reg_index, (before_val, after_val) in enumerate(zip(registers_before[thread], registers_after[thread])) if before_val != after_val] for thread in range(32)]

# print in threads then registers
# python print('\n'.join([f'{index}\t' + '\t'.join([b[:4] + " " + b[4:] for b in a]) for index, a in enumerate(diffs)]))

# print in matrix format
for i in range(4):
    print(f"matrix {i}")
    for j in range(0, 32, 4):
        string = ''.join([diffs[j][i] + diffs[j+1][i] + diffs[j+2][i] + diffs[j+3][i]])
        print(' '.join([string[a:a+4] for a in range(0, len(string), 4)]))
    print("\n\n")

# si 62 to get one after first ldsm
# si 63 to get before ldsm
# si 64 to get one after first hmma
import struct
registers_after = [(gdb.execute(f"cuda thread {j}"), [int(gdb.newest_frame().read_register(f"R{i}") & 0xffffffff) for i in range(40)])[1] for j in range(32)]


# diffs here are [thread][register]
x_diffs = [[hex(registers_after[thread][reg_index])[2:].rjust(8, '0')[4:] + hex(registers_after[thread][reg_index])[2:].rjust(8, '0')[:4] for reg_index in (8, 9, 10, 11)] for thread in range(32)]
w1_diffs = [[hex(registers_after[thread][reg_index])[2:].rjust(8, '0')[4:] + hex(registers_after[thread][reg_index])[2:].rjust(8, '0')[:4] for reg_index in (12, 13)] for thread in range(32)]
# out is fp32
out_diffs = [[hex(registers_after[thread][reg_index])[2:].rjust(8, '0') for reg_index in (16, 17, 18, 19)] for thread in range(32)]

def print_matrix(matrix):
    for i in range(len(matrix[0])):
        print(f"matrix {i}")
        for j in range(0, 32, 4):
            string = ''.join([matrix[j][i] + matrix[j+1][i] + matrix[j+2][i] + matrix[j+3][i]])
            print(' '.join([string[a:a+4] for a in range(0, len(string), 4)]))
        print("\n\n")

def print_matrix_bf16(matrix):
    for i in range(len(matrix[0])):
        print(f"matrix {i}")
        for j in range(0, 32, 4):
            string = ''.join([matrix[j][i] + matrix[j+1][i] + matrix[j+2][i] + matrix[j+3][i]])
            print(' '.join([str(struct.unpack('>f', bytes.fromhex(string[a:a+4] + "0000"))[0]) for a in range(0, len(string), 4)]))
        print("\n\n")

def print_matrix_fp32(matrix):
    for i in range(0, len(matrix[0]), 2): # i here is indexing into register
        print(f"matrix {i}")
        for j in range(0, 32, 4): # j here is indexing into thread. for f32 each row is four threads https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
            # T0R0, T0R1, T1R0, T1R1, etc.
            strings = [str(struct.unpack('>f', bytes.fromhex(matrix[j+(k>>1)][i+(k&1)]))[0]) for k in range(8)]
            print(' '.join(strings))
        print("\n\n")
