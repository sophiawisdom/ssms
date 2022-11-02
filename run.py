import torch
import sys
sys.path.append("/home/ubuntu/.local/lib/python3.8/site-packages")
import time
import math
import subprocess
from torch.utils.cpp_extension import load
import triton
import triton.language as tl

print("compiling siso")
result = subprocess.run(["ptxas", "--gpu-name=sm_80", "siso.ptx", "-o", "siso.o"])
if result.returncode != 0:
    print("FAILED TO COMPILE!!! RESULT IS", result)
    sys.exit(1)
print("completed compiling")

print("about to compile")
siso = load(name="siso", sources=["siso.cpp", "siso.cu"], extra_ldflags=["-lcuda"])
print("compiled")

@torch.jit.script
def torch_diag(sequence, A_DIAG, B, C, N_HEADS: int, STATE_SIZE: int, SEQUENCE_LENGTH: int):
    torch_diag_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
    for i in range(N_HEADS):
        state = torch.zeros((STATE_SIZE,), dtype=torch.float32, device="cuda")
        for j in range(SEQUENCE_LENGTH):
            first_part = (A_DIAG[i] * state)
            second_part = B[i] * sequence[i][j]
            state = first_part + second_part
            torch_diag_outputs[(i, j)] = torch.sum(C[i] * state)
    return torch_diag_outputs

@triton.testing.perf_report(triton.testing.Benchmark(
        x_names=['SEQUENCE_LENGTH'],  # argument names to use as an x-axis for the plot
        x_vals=[
            2**i for i in range(4, 14)
        ],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        y_log=True,
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['ptx'],  # possible values for `line_arg`
        line_names=['Triton'],  # label name for the lines
        styles=[('red', 'solid'), ("blue", "solid"),],  # line styles
        ylabel='elem/s',  # label name for the y-axis
        plot_name=f'ssm-performance @ N=64, N_HEADS=131072',  # name for the plot. Used also as a file name for saving the plot.
        args={"N_HEADS": 131072, "STATE_SIZE": 64},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_unbatched(SEQUENCE_LENGTH, provider, STATE_SIZE, N_HEADS):
    A = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    B = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    C = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    sequence = torch.ones((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")

    if provider == "ptx":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: siso.forward(sequence, A, B, C, SEQUENCE_LENGTH))
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_ssm_batched(sequence, A, B, C, 1, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH))
    else:
        raise ValueError("got unknown provider", provider)
    elems = lambda ms: SEQUENCE_LENGTH * N_HEADS * 1000/(ms)
    return elems(ms), elems(max_ms), elems(min_ms)

print("Got to running first test")
N_HEADS = 256
STATE_SIZE = 32
SEQUENCE_LENGTH = 512
A = torch.randn((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
print("Created A")
B = torch.randn((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
print("Created B")
C = torch.randn((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
print("Created A,B,C")
sequence = torch.ones((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
print("Created sequence")
output = siso.forward(sequence, A, B, C, SEQUENCE_LENGTH)
print("PTX sum output is", output.sum())

torch_output = torch_diag(sequence, A, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)
print("torch sum output is", torch_output.sum())

breakpoint()

if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
    benchmark_unbatched.run(print_data=True)
sys.exit(0)
