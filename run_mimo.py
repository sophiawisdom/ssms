import resource
print("stack before", resource.getrlimit(resource.RLIMIT_STACK))
# resource.setrlimit(resource.RLIMIT_STACK, (512*1024*1024, -1))
print("stack after", resource.getrlimit(resource.RLIMIT_STACK))

import torch
import sys
import time
import math
import triton
import triton.language as tl

# A: {N}
# B: {N, L}
# C: {L, N}
# D: {L, L}
# X: {BATCH_SIZE, N}
# U: {SEQUENCE_LENGTH//L, BATCH_SIZE, L}
# triton X: {N, BATCH_SIZE}
# triton U: {SEQUENCE_LENGTH//L, BATCH_SIZE, L}

@triton.jit
def ssm_kernel_mimo(u_ptr, a_ptr, b_ptr, c_ptr, d_ptr, output_ptr, BATCH_SIZE:tl.constexpr, L: tl.constexpr, U_LENGTH: tl.constexpr, N: tl.constexpr, N_HEADS: tl.constexpr):
    i = tl.program_id(axis=0) # which head we're on
    A = tl.reshape(tl.load(a_ptr + i * N + tl.arange(0, N)), (128, 1)) # A ** L
    B_hat = tl.reshape(tl.load(b_ptr + i * N + tl.arange(0, N*L)), (N, L)) # [STATE_SIZE, L]
    C_hat = tl.reshape(tl.load(c_ptr + i * N + tl.arange(0, N*L)), (N, L)) # [STATE_SIZE, L]
    D_hat = tl.reshape(tl.load(d_ptr + tl.arange(0, L*L)), (L, L)) # [L, L]
    print("A is", A, a_ptr)
    print("B_hat shape is", B_hat, b_ptr)
    print("C_hat shape is", C_hat, c_ptr)
    print("D_hat shape is", D_hat, d_ptr)

    print("output_ptr is of size", output_ptr)
    X = tl.zeros((N, BATCH_SIZE), dtype=tl.float64)
    for j in range(U_LENGTH):
        # u is {U_LENGTH, BATCH_SIZE, L}
        us = tl.reshape(tl.load(u_ptr + j * L * BATCH_SIZE + tl.arange(0, L*BATCH_SIZE)), (BATCH_SIZE, L))
        print("us is", us, "X is", X)
        second_part = tl.dot(B_hat, us, trans_b=True)
        print("second_part is", second_part)
        first_part = X*tl.broadcast_to(A, [N, BATCH_SIZE])
        # tl.store(X + k * BATCH_SIZE + tl.arange(0, BATCH_SIZE))
        # immediate_state = tl.load(X + k * BATCH_SIZE + tl.arange(0, BATCH_SIZE))
        print("first_part is", first_part, "second_part is", second_part)
        X = first_part + second_part
        print("new X is", X)
        output_first = tl.dot(C_hat, X, trans_a=True)
        print("output_first is", output_first)
        output_second = tl.dot(D_hat, us, trans_b=True)
        print("output_second is", output_second)
        output_combined = output_first + output_second
        print("output_combined is", output_combined)
        # output is of size {U_LENGTH, N_HEADS, L, BATCH_SIZE}
        output_ptr_specific = (j * N_HEADS * L * BATCH_SIZE + i * L * BATCH_SIZE)
        l_offsets = tl.arange(0, L)
        batch_offsets = tl.arange(0, BATCH_SIZE)
        print("offsets is", l_offsets, "batch_offsets", batch_offsets)
        output_ptrs = output_ptr + BATCH_SIZE * l_offsets[:, None] + batch_offsets[None, :]
        print("outputs is", output_ptrs)
        tl.store(output_ptrs, output_combined)


def triton_mimo_batched(sequence, A, B, C, D, L, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH):
    triton_outputs = torch.zeros((SEQUENCE_LENGTH, N_HEADS, L), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_batched_perhead_switched_loops[(N_HEADS,)](sequence, A, B, C, D, triton_outputs, L, BATCH_SIZE, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
    return triton_outputs, asm

from scipy.linalg import toeplitz
import numpy as np

STATE_SIZE = 128
SEQUENCE_LENGTH = 64
N_HEADS = 1
BATCH_SIZE= 32
L = 16

A = np.ones(STATE_SIZE,)
B = np.ones((STATE_SIZE, 1))
C = np.ones((1, STATE_SIZE))
sequence = np.ones((SEQUENCE_LENGTH, 1))

A_hat = torch.tensor(np.diag(A**L), device="cuda")
B_hat = torch.tensor(np.block([np.diag(A ** i) @ B for i in range(L - 1, -1, -1)]), device="cuda")
C_hat = torch.tensor(np.block([[C @ np.diag(A ** i)] for i in range(L)]), device="cuda")
D_hat = torch.tensor(toeplitz([0] + [(C @ (np.diag(A ** i)) @ B).squeeze((0, 1))
                        for i in range(L - 2, -1, -1)], r=np.zeros(L)), device="cuda")

# state = torch.zeros((STATE_SIZE,), device="cuda")
# batched_outputs = torch.zeros((SEQUENCE_LENGTH//L, L), device="cuda")
A_hat = torch.tensor(A ** L, device="cuda")
sequence_hat = torch.tensor(sequence.reshape(-1, L), device="cuda")

# Equivalent torch implementation
'''
print(f"{A_hat.shape=} {state.shape=} {B_hat.shape=} {sequence_hat.shape=}")
for i, u in enumerate(sequence_hat):
    state = A_hat * state + (B_hat @ u)
    output = C_hat @ state + D_hat @ u
    batched_outputs[i] = output
print(batched_outputs.sum())
'''

triton_outputs = torch.zeros((SEQUENCE_LENGTH, 1, L), device="cuda", dtype=torch.float32)
print("ABOUT TO DO TRITON")
ssm_kernel_mimo[(N_HEADS,)](sequence_hat, A_hat, B_hat, C_hat, D_hat, triton_outputs, BATCH_SIZE, L, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
print("we did triton!")
print(triton_outputs.sum())
# triton_mimo_batched
