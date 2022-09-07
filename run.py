import torch
import sys
import triton
import triton.language as tl

true_states = []
true_outputs = []

STATE_SIZE = 2048
SEQUENCE_LENGTH = 32
N_HEADS = 1024

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

def torch_diag_batched(sequence, A_DIAG, B, C, N_HEADS: int, STATE_SIZE: int, SEQUENCE_LENGTH: int):
    torch_diag_outputs = torch.empty((len(sequence), N_HEADS), dtype=torch.float32, device="cuda")
    state = torch.zeros((N_HEADS, STATE_SIZE,), dtype=torch.float32, device="cuda")
    for j in range(len(sequence)):
        first_part = (A_DIAG * state)
        second_part = B.reshape((B.shape[0], B.shape[1])) * sequence[j]
        state = first_part + second_part
        torch_diag_outputs[j] = torch.sum(C @ state)
    return torch_diag_outputs

torch_diag_script = torch.jit.script(torch_diag_batched)

@triton.jit
def ssm_kernel_perhead(u_ptr, a_ptr, b_ptr, c_ptr, output_ptr, U_LENGTH: tl.constexpr, N: tl.constexpr, N_HEADS: tl.constexpr):
    i = tl.program_id(axis=0)
    outputs = tl.zeros((U_LENGTH,), dtype=tl.float32)
    X = tl.zeros((N,), dtype=tl.float32)
    A = tl.load(a_ptr + i * N + tl.arange(0, N))
    B = tl.load(b_ptr + i * N + tl.arange(0, N))
    C = tl.load(c_ptr + i * N + tl.arange(0, N))
    u = tl.load(u_ptr + tl.arange(0, U_LENGTH))
    for j in range(U_LENGTH):
        u_t = tl.load(u_ptr + j)
        X = X*A + B*u_t
        value = tl.sum(X*C, axis=0)
        tl.store(output_ptr + (i * U_LENGTH + j), value)

'''
triton_outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), device="cuda", dtype=torch.float32)
assert sequence.is_cuda and A.is_cuda and B.is_cuda and C.is_cuda and triton_outputs.is_cuda
ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, SEQUENCE_LENGTH, STATE_SIZE, N_HEADS)
'''

def triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH: int):
    triton_outputs = torch.empty((N_HEADS, len(sequence)), device="cuda", dtype=torch.float32)
    ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals=[
            2**i for i in range(2, 14)
        ],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch_script'],  # possible values for `line_arg`
        line_names=['Triton', 'TorchScript'],  # label name for the lines
        styles=[('red', 'solid'), ("blue", "solid"),],  # line styles
        ylabel='elem/s',  # label name for the y-axis
        plot_name=f'ssm-performance @ N={STATE_SIZE}, N_HEADS={N_HEADS}',  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(size, provider):
    state = torch.zeros((STATE_SIZE))
    A = torch.eye(STATE_SIZE, dtype=torch.float32, device="cuda")[None, :, :].repeat(N_HEADS, 1, 1)
    B = torch.randn((N_HEADS, STATE_SIZE, 1), dtype=torch.float32, device="cuda")
    C = torch.randn((N_HEADS, 1, STATE_SIZE), dtype=torch.float32, device="cuda")
    A *= torch.randn(N_HEADS, STATE_SIZE, STATE_SIZE, device="cuda")
    A_DIAG = torch.stack([torch.diagonal(A[i]) for i in range(len(A))])
    outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
    sequence = torch.rand(size, dtype=torch.float32, device="cuda")[:, None]

    print("STATE_SIZE", STATE_SIZE, "called for length", size, "and provider", provider)

    if provider == "no_diag_script":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_no_diag_script(sequence, A, B, C, N_HEADS, STATE_SIZE, size))
    elif provider == "torch_script":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_diag_script(sequence, A_DIAG, B, C.reshape((C.shape[0], C.shape[2])).T, N_HEADS, STATE_SIZE, size))
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, size))
    else:
        raise ValueError("got unknown provider", provider)
    elems = lambda ms: size/(ms/1000)
    return elems(ms), elems(max_ms), elems(min_ms)

benchmark.run(print_data=True, save_path="/home/sophiawisdom")

sys.exit(0)
