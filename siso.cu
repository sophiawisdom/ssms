#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <builtin_types.h>

#include <vector>

CUmodule ptx_module;
CUfunction siso_forward;
CUfunction siso_backward;
CUdevice   device;
CUcontext  context;
int major = 0, minor = 0;

// #define DEBUG

__attribute__((constructor))
static void initialize_cuda() {
    CUresult err = cuInit(0);

    err = cuCtxCreate(&context, 0, device);

    cuDeviceGet(&device, 0);
    cuDeviceComputeCapability(&major, &minor, device);
    err = cuModuleLoad(&ptx_module, "siso.o");
    if (err != CUDA_SUCCESS) {
        cudaError_t error = cudaGetLastError();
        const char *name = cudaGetErrorName(error);
        const char *string = cudaGetErrorString(error);
        fprintf(stderr, "* Error loading PTX module. Error name \"%s\" string \"%s\" err is %d\n", name, string, err);
        exit(-1);
    }

    err = cuModuleGetFunction(&siso_forward, ptx_module, "siso_forward_32");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get forward function\n");
    }

    err = cuModuleGetFunction(&siso_backward, ptx_module, "siso_backward_32");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get backward function\n");
    }

    printf("SISO initialized!\n");
}

unsigned int old_sequence_length = 0;

torch::Tensor siso_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    unsigned int sequence_length
) {
#ifdef DEBUG
  int start = clock();
#endif

  const auto num_heads = a.size(0); // {N_HEADS, STATE_SIZE}
  assert(a.size(1) == 32); // STATE_SIZE must be 32 at the moment.
  assert(sequence_length % 4 == 0); // sequence length must be divisible by 4

  auto output = torch::empty_like(u);

  assert(num_heads % 8 == 0);

  // printf("a sizes 0 is %d\n", a.sizes()[0]);
  unsigned int n_heads = a.sizes()[0];
  
  void * argBuffer[6];
  int argBufferSize = 5*8 + 4; // 5 pointers and an int
  argBuffer[0] = u.data_ptr();
  argBuffer[1] = a.data_ptr();
  argBuffer[2] = b.data_ptr();
  argBuffer[3] = c.data_ptr();
  argBuffer[4] = output.data_ptr();
  int *argBufferView = (int *)&argBuffer;
  argBufferView[10] = sequence_length;

#ifdef DEBUG
  if (old_sequence_length != sequence_length) {
    printf("sequence_length %u\n", sequence_length);
    old_sequence_length = sequence_length;
  }

  for (int i = 0; i < sizeof(argBuffer)/sizeof(void *) + 1; i++) {
    printf("argBuffer for #%d is %p\n", i, argBuffer[i]);
  }

  printf("about to cuLaunchKernel, sequence_length is %d\n", sequence_length);
  int end = clock();
  printf("Took %d us to get to cuLaunchKernel\n", end-start);
  printf("launching with %d grid\n", num_heads/8);
#endif

  void *config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argBufferView,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
    CU_LAUNCH_PARAM_END,
  };

  int error = cuLaunchKernel(siso_forward,
  num_heads/8, 1, 1, // grid x, y, z
  4, 8, 1, // block x, y, z
  0, 0, NULL, config);
  if (error != CUDA_SUCCESS) {
    cudaError_t lastErr = cudaGetLastError();
    const char *name = cudaGetErrorName(lastErr);
    const char *string = cudaGetErrorString(lastErr);
    fprintf(stderr, "* Error with cuLaunchKernel. Error name \"%s\" string \"%s\" err is %d\n", name, string, error);
  }
#ifdef DEBUG
  printf("cuLaunchKernel result is %d\n", error);
#endif

  return output;
}

void siso_cuda_backward(
    torch::Tensor gradients,
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor grad_a,
    torch::Tensor b,
    torch::Tensor grad_b,
    torch::Tensor c,
    torch::Tensor grad_c,
    unsigned int sequence_length) {
      return;
}
