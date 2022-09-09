import resource
print("stack before", resource.getrlimit(resource.RLIMIT_STACK))
resource.setrlimit(resource.RLIMIT_STACK, (512*1024*1024, -1))
print("stack after", resource.getrlimit(resource.RLIMIT_STACK))

import torch
import sys
sys.path.append("/home/sophiawisdom/.local/lib/python3.8")
import time
import math
import triton
import triton.language as tl

'''
state = torch.zeros((STATE_SIZE))
A = torch.eye(STATE_SIZE, dtype=torch.float32, device="cuda")[None, :, :].repeat(N_HEADS, 1, 1)
B = torch.randn((N_HEADS, STATE_SIZE, 1), dtype=torch.float32, device="cuda")
C = torch.randn((N_HEADS, 1, STATE_SIZE), dtype=torch.float32, device="cuda")
A *= torch.randn(N_HEADS, STATE_SIZE, STATE_SIZE, device="cuda")
# TODO: do this correctly
A_DIAG = torch.stack([torch.diagonal(A[i]) for i in range(len(A))])
print("A_DIAG shape is", A_DIAG.shape, A_DIAG.dtype)

sequence = torch.rand(SEQUENCE_LENGTH, dtype=torch.float32, device="cuda")[:, None]

print("A shape", A.shape, "B shape", B.shape, "C shape", C.shape, "A_DIAG shape", A_DIAG.shape)
print("sequence shape is", sequence.shape)

torch_no_diag_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
for i in range(N_HEADS):
    state = torch.zeros((STATE_SIZE,), dtype=torch.float32, device="cuda")
    for j in range(SEQUENCE_LENGTH):
        state = A[i] @ state + B[i] @ sequence[j]
        torch_no_diag_outputs[(i, j)] = C[i] @ state

torch_diag_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
for i in range(N_HEADS):
    state = torch.zeros((STATE_SIZE,), dtype=torch.float32, device="cuda")
    for j in range(SEQUENCE_LENGTH):
        first_part = (A_DIAG[i] * state)
        second_part = B[i].reshape(-1) * sequence[j]
        state = first_part + second_part
        torch_diag_outputs[(i, j)] = torch.sum(C[i] @ state, axis=0)
'''

def torch_no_diag(sequence, A, B, C, N_HEADS: int, STATE_SIZE: int, SEQUENCE_LENGTH: int):
    torch_no_diag_outputs = torch.empty((len(sequence), N_HEADS), dtype=torch.float32, device="cuda")
    return torch_no_diag_outputs
    state = torch.zeros((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    for j in range(SEQUENCE_LENGTH):
        print(f"A.shape", A.shape, "state shape", state.shape, "b shape", B.shape, "sequence shape", sequence[j].shape)
        state_move = A @ state
        input_contribution = torch.einsum('hij,bj->bhi', B, sequence[j])
        print("state_move shape", state_move.shape, "input_contribution shape", input_contribution.shape)
        state = state_move + input_contribution
        torch_no_diag_outputs[j] = C @ state
    return torch_no_diag_outputs

torch_no_diag_script = torch.jit.script(torch_no_diag)

def torch_diag(sequence, A_DIAG, B, C, N_HEADS: int, STATE_SIZE: int, SEQUENCE_LENGTH: int):
    torch_diag_outputs = torch.empty((N_HEADS, len(sequence)), dtype=torch.float32, device="cuda")
    for i in range(N_HEADS):
        state = torch.zeros((STATE_SIZE,), dtype=torch.float32, device="cuda")
        for j in range(len(sequence)):
            first_part = (A_DIAG[i] * state)
            second_part = B[i].reshape(-1) * sequence[j]
            state = first_part + second_part
            torch_diag_outputs[(i, j)] = torch.sum(C[i] @ state)
    return torch_diag_outputs

def torch_diag_batched_on_batches(sequence, A, B, C, BATCH_SIZE: int, N_HEADS: int, STATE_SIZE: int, SEQUENCE_LENGTH: int):
    torch_diag_outputs = torch.empty((SEQUENCE_LENGTH, BATCH_SIZE, N_HEADS), dtype=torch.float32, device="cuda")
    state = torch.zeros((BATCH_SIZE, N_HEADS, STATE_SIZE, 1), dtype=torch.float32, device="cuda")
    # print("sequence is of shape", sequence.shape)
    for j, u in enumerate(sequence):
        # print("u is of shape", u.shape)
        # print(f"{A[None, :, :, None].shape=} {B[None, :, :, None].shape=} {C.shape=}")
        # print("B total shape", B.shape, "B new shape", B[None, :, :, None].shape, "u shape", u.shape, "u new shape", u[:, None, None, None].shape)
        state = (A[None, :, :, None] * state + (B[None, :, :, None] @ u[:, None, None, None]))
        # print("batched state is", state.shape, "C shape is", C.shape, "weird c", C[None, :, None, :].shape, "atted", (C[None, :, None, :] @ state).shape)
        torch_diag_outputs[j] = (C[None, :, None, :] @ state).squeeze(2).squeeze(2)
    return torch_diag_outputs

# torch_diag_script_unbatched = torch.jit.script(torch_diag)
torch_diag_script = torch.jit.script(torch_diag)
torch_diag_script_batched = torch.jit.script(torch_diag_batched_on_batches)

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

@triton.jit
def ssm_kernel_batched_perhead(u_ptr, a_ptr, b_ptr, c_ptr, output_ptr, BATCH_SIZE:tl.constexpr, U_LENGTH: tl.constexpr, N: tl.constexpr, N_HEADS: tl.constexpr):
    i = tl.program_id(axis=0) # which head we're on
    Xs = tl.zeros((BATCH_SIZE, N), dtype=tl.float32)
    A = tl.load(a_ptr + i * N + tl.arange(0, N))
    B = tl.load(b_ptr + i * N + tl.arange(0, N))
    C = tl.load(c_ptr + i * N + tl.arange(0, N))
    for j in range(U_LENGTH):
        for k in range(BATCH_SIZE):
            u_k = tl.load(u_ptr + j * BATCH_SIZE + k)
            Xs[k] = Xs[k] * A + B*u_k
            tl.store(output_ptr + (i * U_LENGTH + j * BATCH_SIZE + k), tl.dot(Xs[k], C))

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
            X = X * A + B*u_k
            tl.store(output_ptr + (j * BATCH_SIZE * N_HEADS + k * N_HEADS + i), tl.sum(X*C, axis=0))

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

'''
triton_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), device="cuda", dtype=torch.float32)
assert sequence.is_cuda and A.is_cuda and B.is_cuda and C.is_cuda and triton_outputs.is_cuda
ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
'''

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
        print("runs", runs)
        if runs == 8:
            print(asm.asm["ptx"])
        # print(asm.asm["ptx"])
    return triton_outputs, asm

def triton_mimo_batched(sequence, A, B, C, D, L, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH):
    triton_outputs = torch.zeros((SEQUENCE_LENGTH, N_HEADS, L), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_batched_perhead_switched_loops[(N_HEADS,)](sequence, A, B, C, D, triton_outputs, L, BATCH_SIZE, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
    return triton_outputs, asm

# benchmark.run(print_data=True, save_path="/home/sophiawisdom")

'''
A = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
B = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
C = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")

sequence = torch.ones((SEQUENCE_LENGTH, BATCH_SIZE), dtype=torch.float32, device="cuda")
torch_diag_output = torch_diag_script_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)

triton_output, asm = triton_ssm_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)

print(f"{triton_output.shape=} {torch_diag_output.shape=}")

print("diff between torch and triton is", (torch_diag_output - triton_output).abs().sum())
torch.testing.assert_close(torch_diag_output, triton_output)
'''

'''
sys.exit(0)

for i in range(128):
    torch.cuda.synchronize()
    t0 = time.time()
    output, asm = triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Took {t1-t0:.2f}s on iteration #{i}")
    breakpoint()
sys.exit(0)
'''

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
for i in range(4):
    for j in range(4):
        for k in range(4):
            benchmarks.append(make_benchmark(512 * 2 ** i, 256 * 2**j, 16 * 2**k))

@triton.testing.perf_report([make_benchmark(4096, 2048, 64)])
def benchmark(size, provider, STATE_SIZE, BATCH_SIZE, N_HEADS):
    SEQUENCE_LENGTH = size
    state = torch.zeros((STATE_SIZE))
    A = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    B = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    C = torch.ones((N_HEADS, STATE_SIZE), dtype=torch.float32, device="cuda")
    outputs = torch.zeros((SEQUENCE_LENGTH, BATCH_SIZE, N_HEADS), device="cuda", dtype=torch.float32)
    sequence = torch.ones((SEQUENCE_LENGTH, BATCH_SIZE), dtype=torch.float32, device="cuda")
    # print(f"{provider} handling ({N_HEADS}, {STATE_SIZE}, {BATCH_SIZE}, {SEQUENCE_LENGTH})")

    if provider == "torch_script":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_diag_script_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, size))
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_ssm_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, size))
    else:
        raise ValueError("got unknown provider", provider)
    elems = lambda ms: size * BATCH_SIZE/(ms)
    return elems(ms), elems(max_ms), elems(min_ms)

# benchmark.run(print_data=True, save_path="/home/sophiawisdom")

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
'''
print(f"{A_hat.shape=} {state.shape=} {B_hat.shape=} {sequence_hat.shape=}")
for i, u in enumerate(sequence_hat):
    state = A_hat * state + (B_hat @ u)
    output = C_hat @ state + D_hat @ u
    batched_outputs[i] = output
print(batched_outputs.sum())
'''

triton_outputs = torch.zeros((SEQUENCE_LENGTH, 1, L), device="cuda", dtype=torch.float32)
time.sleep(1)
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
print("ABOUT TO DO TRITON")
time.sleep(10)
ssm_kernel_mimo[(N_HEADS,)](sequence_hat, A_hat, B_hat, C_hat, D_hat, triton_outputs, BATCH_SIZE, L, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
print("we did triton!")
print(triton_outputs.sum())
# triton_mimo_batched
