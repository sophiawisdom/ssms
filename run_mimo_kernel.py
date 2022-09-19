from torch.utils.cpp_extension import load
import subprocess
import sys
import triton

print("compiling mimo_done")
result = subprocess.run(["/usr/local/cuda-11.7/bin/ptxas", "--gpu-name=sm_80", "mimo_done.ptx", "-o", "mimo_done.o"])
if result.returncode != 0:
    print("FAILED TO COMPILE!!! RESULT IS", result)
    sys.exit(1)
print("completed compiling")

print("about to compile")
mimo = load(name="ptxer", sources=["ptxer.cpp", "mimo.cu"], extra_ldflags=["-lcuda"])
print("compiled")

STATE_SIZE = 64
SEQUENCE_LENGTH = 8192
N_HEADS = 1
BATCH_SIZE= 16
L = 8

'''
STATE_SIZE = 128
SEQUENCE_LENGTH = 64
N_HEADS = 1
BATCH_SIZE= 32
L = 16
'''

import numpy as np
import torch
from scipy.linalg import toeplitz

np.random.seed(5)

A = np.ones(STATE_SIZE, dtype=np.float64)
B = np.ones((STATE_SIZE, 1), dtype=np.float64)
C = np.ones((1, STATE_SIZE), dtype=np.float64)
sequence = np.ones((SEQUENCE_LENGTH, BATCH_SIZE, 1), dtype=np.float64)
A = np.random.rand(STATE_SIZE)
B = np.random.rand(STATE_SIZE, 1)
C = np.random.rand(1, STATE_SIZE)
sequence = np.random.rand(SEQUENCE_LENGTH, BATCH_SIZE, 1)

B_hat = torch.tensor(np.block([np.diag(A ** i) @ B for i in range(L - 1, -1, -1)]), device="cuda", dtype=torch.bfloat16)
C_hat = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda", dtype=torch.bfloat16)
C_hat_flipped = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda", dtype=torch.bfloat16).T.contiguous()
C_hat_flipter = torch.cat(tuple(C_hat_flipped[i:i+8].T for i in range(0, 64, 8)))
D_hat = torch.tensor(toeplitz([0] + [(C @ (np.diag(A ** i)) @ B).squeeze((0, 1))
                        for i in range(L - 2, -1, -1)], r=np.zeros(L)), device="cuda", dtype=torch.bfloat16)
D_hat_flipped = D_hat.T.contiguous()

A_hat = torch.tensor(A ** L, device="cuda", dtype=torch.float32)
sequence_hat = torch.tensor(sequence.reshape(SEQUENCE_LENGTH//L, BATCH_SIZE, L), device="cuda", dtype=torch.bfloat16)

state = torch.zeros((STATE_SIZE,), device="cuda", dtype=torch.float32)
batched_outputs = torch.zeros((SEQUENCE_LENGTH//L, BATCH_SIZE, L), device="cuda", dtype=torch.bfloat16)
states = []
for i, u in enumerate(sequence_hat):
    first_new_state = A_hat * state
    new_contribution = (u @ B_hat.T)
    state = first_new_state + new_contribution
    states.append(state)
    output_state = state.to(dtype=torch.bfloat16) @ C_hat_flipped
    output_dhat = u @ D_hat
    output = output_state + output_dhat
    batched_outputs[i] = output
print(batched_outputs.sum())

@torch.jit.script
def pytorch_mimo(sequence_hat, A_hat, B_hat, C_hat, D_hat, sequence_length: int, STATE_SIZE: int):
    state = torch.zeros((STATE_SIZE,), device="cuda", dtype=torch.float32)
    batched_outputs = torch.empty_like(sequence_hat)
    for i, u in enumerate(sequence_hat):
        state = A_hat * state + (u @ B_hat)
        batched_outputs[i] = state.to(dtype=torch.bfloat16) @ C_hat + u @ D_hat
    return batched_outputs

print(f"{sequence_hat.shape=}")
output = mimo.forward(sequence_hat, A_hat, B_hat, C_hat_flipter, D_hat_flipped, SEQUENCE_LENGTH//L)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQUENCE_LENGTH'],  # argument names to use as an x-axis for the plot
        x_vals=[
            2 ** i for i in range(3, 17)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['pytorch', 'ptx'],
        # label name for the lines
        line_names=['pytorch', 'ptx'],
        # line styles
        styles=[('green', '-'), ('red', '-')],
        ylabel="Elements/s",  # label name for the y-axis
        plot_name="ssm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(SEQUENCE_LENGTH, provider):
    A = np.random.rand(STATE_SIZE)
    B = np.random.rand(STATE_SIZE, 1)
    C = np.random.rand(1, STATE_SIZE)
    sequence = np.random.rand(SEQUENCE_LENGTH, BATCH_SIZE, 1)

    B_hat = torch.tensor(np.block([np.diag(A ** i) @ B for i in range(L - 1, -1, -1)]), device="cuda", dtype=torch.bfloat16)
    C_hat = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda", dtype=torch.bfloat16)
    C_hat_flipped = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda", dtype=torch.bfloat16).T.contiguous()
    C_hat_flipter = torch.cat(tuple(C_hat_flipped[i:i+8].T for i in range(0, 64, 8)))
    D_hat = torch.tensor(toeplitz([0] + [(C @ (np.diag(A ** i)) @ B).squeeze((0, 1))
                            for i in range(L - 2, -1, -1)], r=np.zeros(L)), device="cuda", dtype=torch.bfloat16)
    D_hat_flipped = D_hat.T.contiguous()

    print("sequence length is", SEQUENCE_LENGTH)

    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: pytorch_mimo(sequence_hat, A_hat, B_hat.T, C_hat_flipped, D_hat, SEQUENCE_LENGTH//L, STATE_SIZE))
    elif provider == "ptx":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: mimo.forward(sequence_hat, A_hat, B_hat, C_hat_flipter, D_hat_flipped, SEQUENCE_LENGTH//L))
    perf = lambda ms: SEQUENCE_LENGTH * BATCH_SIZE / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

print("Batched outputs:")
# expected_output = states[0].to(torch.bfloat16) @ C_hat_flipped
expected_output = batched_outputs[1]
# print(batched_outputs[0], batched_outputs.sum())
print(expected_output, expected_output.sum())

print("Output:")
print(output[1], output.sum())

print(f"Difference: {100 * float(torch.abs(expected_output - output[1]).sum())/float(torch.abs(expected_output).sum()):.4f}%")

benchmark.run(print_data=True)

breakpoint()