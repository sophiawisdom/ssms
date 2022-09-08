import torch
import sys
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

'''
triton_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), device="cuda", dtype=torch.float32)
assert sequence.is_cuda and A.is_cuda and B.is_cuda and C.is_cuda and triton_outputs.is_cuda
ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
'''

def triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH: int):
    triton_outputs = torch.empty((N_HEADS, len(sequence)), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs, asm

def triton_ssm_batched(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH):
    triton_outputs = torch.zeros((SEQUENCE_LENGTH, BATCH_SIZE, N_HEADS), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_batched_perhead_switched_loops[(N_HEADS,)](sequence, A, B, C, triton_outputs, BATCH_SIZE, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs, asm

def triton_ssm_batched_switched_loops(sequence, A, B, C, BATCH_SIZE, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH):
    triton_outputs = torch.zeros((SEQUENCE_LENGTH, BATCH_SIZE, N_HEADS), device="cuda", dtype=torch.float32)
    asm = ssm_kernel_batched_perhead_switched_loops[(N_HEADS,)](sequence, A, B, C, triton_outputs, BATCH_SIZE, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs, asm

# benchmark.run(print_data=True, save_path="/home/sophiawisdom")

STATE_SIZE = 64
N_HEADS = 32
BATCH_SIZE = 32

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

@triton.testing.perf_report(benchmarks)
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

benchmark.run(print_data=True, save_path="/home/sophiawisdom")