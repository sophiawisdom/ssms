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

@triton.jit
def ssm_kernel_perhead(u_ptr, a_ptr, b_ptr, c_ptr, output_ptr, U_LENGTH: tl.constexpr, N: tl.constexpr, N_HEADS: tl.constexpr):
    i = tl.program_id(axis=0)
    outputs = tl.zeros((U_LENGTH,), dtype=tl.float32)
    X = tl.zeros((N,), dtype=tl.float32)
    A = tl.load(a_ptr + i * N + tl.arange(0, N))
    B = tl.load(b_ptr + i * N + tl.arange(0, N))
    C = tl.load(c_ptr + i * N + tl.arange(0, N))
    for j in range(U_LENGTH):
        u_t = tl.load(u_ptr + j)
        X = X*A + B*u_t
        value = tl.sum(X*C, axis=0)
        tl.store(output_ptr + (i * U_LENGTH + j), value)

@triton.autotune(
    configs=[
        triton.Config({},num_warps=1)
    ],
    key=[],
)
@triton.jit
def ssm_kernel_batched_perhead_switched_loops(u_ptr, a_ptr, b_ptr, c_ptr, output_ptr, BATCH_SIZE:tl.constexpr, U_LENGTH: tl.constexpr, N: tl.constexpr, N_HEADS: tl.constexpr):
    i = tl.program_id(axis=0) # which head we're on
    A = tl.load(a_ptr + i * N + tl.arange(0, N))
    B = tl.load(b_ptr + i * N + tl.arange(0, N))
    C = tl.load(c_ptr + i * N + tl.arange(0, N))
    for k in range(BATCH_SIZE):
        X = tl.zeros((N,), dtype=tl.float32)
        for j in range(U_LENGTH):
            u_k = tl.load(u_ptr + j * BATCH_SIZE + k)
            X = X * A + B*u_k # X*A is N multiplies, B*u_k is N multiplies, adding is N adds
            output_idx = (j * BATCH_SIZE * N_HEADS + k * N_HEADS + i)
            tl.store(output_ptr + output_idx, tl.sum(X*C, axis=0)) # X*C is N multiplies, summing is N adds
            # all told 2N FMAs and N multiplies

def triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH: int):
    triton_outputs = torch.empty((N_HEADS, len(sequence)), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs, asm

runs = 0
def triton_ssm_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH):
    global runs
    triton_outputs = torch.zeros((SEQUENCE_LENGTH, BATCH_SIZE, N_HEADS), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_batched_perhead_switched_loops[(N_HEADS,)](sequence, A, B, C, triton_outputs, BATCH_SIZE, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
    if SEQUENCE_LENGTH == 8192:
        runs += 1
        # print("runs", runs)
        if runs == 8:
            print(asm.asm["ptx"])
        # print(asm.asm["ptx"])
    return triton_outputs, asm

def make_benchmark(STATE_SIZE, N_HEADS, BATCH_SIZE):
    return triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals=[
            2**i for i in range(1, 14)
        ],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        y_log=True,
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton'],  # possible values for `line_arg`
        line_names=['Triton'],  # label name for the lines
        styles=[('red', 'solid'), ("blue", "solid"),],  # line styles
        ylabel='elem/s',  # label name for the y-axis
        plot_name=f'ssm-performance @ N={STATE_SIZE}, N_HEADS={N_HEADS}, BATCH_SIZE={BATCH_SIZE}',  # name for the plot. Used also as a file name for saving the plot.
        args={"N_HEADS": N_HEADS, "STATE_SIZE": STATE_SIZE, "BATCH_SIZE": BATCH_SIZE},  # values for function arguments not in `x_names` and `y_name`
    )

benchmarks = []
for k in range(4):
    benchmarks.append(make_benchmark(64, 1024, 16 * 2**k))

@triton.testing.perf_report([make_benchmark(64, 1024, 1)])
def benchmark(size, provider, STATE_SIZE, BATCH_SIZE, N_HEADS):
    SEQUENCE_LENGTH = size
    state = torch.zeros((STATE_SIZE))
    A = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    B = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    C = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), device="cuda", dtype=torch.float32)
    sequence = torch.ones((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
    # print(f"{provider} handling ({N_HEADS}, {STATE_SIZE}, {BATCH_SIZE}, {SEQUENCE_LENGTH})")

    if provider == "torch_script":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_diag_script_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, size))
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_ssm_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, size))
    else:
        raise ValueError("got unknown provider", provider)
    elems = lambda ms: size * BATCH_SIZE * N_HEADS * 1000/(ms)
    return elems(ms), elems(max_ms), elems(min_ms)

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


N_HEADS = 131072
STATE_SIZE = 64
SEQUENCE_LENGTH = 512
A = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
B = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
C = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
sequence = torch.ones((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
output = siso.forward(sequence, A, B, C, SEQUENCE_LENGTH)
print("sum output is", output.sum())

if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
    benchmark_unbatched.run(print_data=True)
sys.exit(1)
