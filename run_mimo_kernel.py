from torch.utils.cpp_extension import load
import subprocess

print("compiling mimo_done")
print("ptx compile result: ", subprocess.run(["/usr/local/cuda-11.7/bin/ptxas", "--gpu-name=sm_80", "-g", "mimo_done.ptx", "-o", "mimo_done.o"]))
print("completed compiling")

print("about to compile")
mimo = load(name="ptxer", sources=["ptxer.cpp", "mimo.cu"], extra_ldflags=["-lcuda"])
print("compiled")

STATE_SIZE = 64
SEQUENCE_LENGTH = 64
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

A = np.ones(STATE_SIZE,)
B = np.ones((STATE_SIZE, 1))
C = np.ones((1, STATE_SIZE))
sequence = np.ones((SEQUENCE_LENGTH, BATCH_SIZE, 1))
A = np.random.rand(STATE_SIZE,)
# B = np.random.rand(STATE_SIZE, 1)
# C = np.random.rand(1, STATE_SIZE)
# sequence = np.random.rand(SEQUENCE_LENGTH, BATCH_SIZE, 1)


B_hat = torch.tensor(np.block([np.diag(A ** i) @ B for i in range(L - 1, -1, -1)]), device="cuda", dtype=torch.bfloat16)
C_hat = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda", dtype=torch.bfloat16)
D_hat = torch.tensor(toeplitz([0] + [(C @ (np.diag(A ** i)) @ B).squeeze((0, 1))
                        for i in range(L - 2, -1, -1)], r=np.zeros(L)), device="cuda", dtype=torch.bfloat16)

state = torch.zeros((STATE_SIZE,), device="cuda", dtype=torch.float32)
batched_outputs = torch.zeros((SEQUENCE_LENGTH//L, BATCH_SIZE, L), device="cuda", dtype=torch.bfloat16)
A_hat = torch.tensor(A ** L, device="cuda", dtype=torch.float32)
sequence_hat = torch.tensor(sequence.reshape(SEQUENCE_LENGTH//L, BATCH_SIZE, L), device="cuda", dtype=torch.bfloat16)
sequence_hat[:,8:] = 2 # switch off arrays

print(f"{A_hat.shape=} {state.shape=} {B_hat.shape=} {sequence_hat.shape=}")
print("initial state sum", state.sum())
for i, u in enumerate(sequence_hat):
    first_new_state = A_hat * state
    new_contribution = (u @ B_hat.T)
    print("first_new_state sum", first_new_state.sum(), "new contribution sum", new_contribution.sum())
    state = first_new_state + new_contribution
    output_state = state.to(dtype=torch.bfloat16) @ C_hat.T
    output_dhat = u @ D_hat
    output = output_state + output_dhat
    batched_outputs[i] = output
print(batched_outputs.sum())

print(f"{sequence_hat.shape=}")

output = mimo.forward(sequence_hat, A_hat, B_hat, C_hat, D_hat)

print("Batched outputs:")
print(batched_outputs[0], batched_outputs.sum())

print("Output:")
print(output[0], output.sum())
breakpoint()