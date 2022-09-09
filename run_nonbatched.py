import torch
import triton
import triton.language as tl

STATE_SIZE = 2048
SEQUENCE_LENGTH = 8192
N_HEADS = 1024

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

def triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH: int):
    triton_outputs = torch.empty((N_HEADS, len(sequence)), device="cuda", dtype=sequence.dtype)
    asm = ssm_kernel_perhead[(N_HEADS,)](sequence, A_DIAG, B, C, triton_outputs, len(sequence), STATE_SIZE, N_HEADS)
    return triton_outputs, asm

state = torch.zeros((STATE_SIZE))
A = torch.eye(STATE_SIZE, dtype=torch.float32, device="cuda")[None, :, :].repeat(N_HEADS, 1, 1)
B = torch.ones((N_HEADS, STATE_SIZE, 1), dtype=torch.float32, device="cuda")
C = torch.ones((N_HEADS, 1, STATE_SIZE), dtype=torch.float32, device="cuda")
# A *= torch.ones(N_HEADS, STATE_SIZE, STATE_SIZE, device="cuda")
A_DIAG = torch.stack([torch.diagonal(A[i]) for i in range(len(A))])
outputs = torch.zeros((N_HEADS, SEQUENCE_LENGTH), dtype=torch.float32, device="cuda")
sequence = torch.ones(SEQUENCE_LENGTH, dtype=torch.float32, device="cuda")[:, None]

B_bf16 = B.to(dtype=torch.bfloat16)
C_bf16 = C.to(dtype=torch.bfloat16)
A_DIAG_bf16 = A_DIAG.to(dtype=torch.bfloat16)
sequence_bf16 = sequence.to(dtype=torch.bfloat16)

for i in range(8):
    # triton_outputs, asm = triton_ssm(sequence_bf16, A_DIAG_bf16, B_bf16, C_bf16, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)
    triton_outputs, asm = triton_ssm(sequence, A_DIAG, B, C, N_HEADS, STATE_SIZE, SEQUENCE_LENGTH)
    print(triton_outputs.sum())

breakpoint()
