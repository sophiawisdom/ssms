#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <builtin_types.h>

#include <vector>

CUmodule ptx_module;
CUfunction kernel_function;
CUdevice   device;
CUcontext  context;
int major = 0, minor = 0;

__attribute__((constructor))
static void initialize_cuda() {
    CUresult err = cuInit(0);

    err = cuCtxCreate(&context, 0, device);

    cuDeviceGet(&device, 0);
    cuDeviceComputeCapability(&major, &minor, device);
    err = cuModuleLoad(&ptx_module, "mimo_done.o");
    if (err != CUDA_SUCCESS) {
        cudaError_t error = cudaGetLastError();
        const char *name = cudaGetErrorName(error);
        const char *string = cudaGetErrorString(error);
        fprintf(stderr, "* Error loading PTX module. Error name \"%s\" string \"%s\" err is %d\n", name, string, err);
        exit(-1);
    }

    err = cuModuleGetFunction(&kernel_function, ptx_module, "mimo_kernel");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get function");
    }
}

unsigned int old_sequence_length = 0;

torch::Tensor mimo_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    unsigned int sequence_length
) {

  const auto num_heads = a.size(0); // {N_HEADS, STATE_SIZE}
  const auto batch_size = u.size(1);

  auto output = torch::zeros_like(u);

  printf("a sizes 0 is %d\n", a.sizes()[0]);
  unsigned int n_heads = a.sizes()[0];

  void * argBuffer[6];
  int argBufferSize = 7*8; // 6 pointers
  argBuffer[0] = u.data_ptr();
  argBuffer[1] = a.data_ptr();
  argBuffer[2] = b.data_ptr();
  argBuffer[3] = c.data_ptr();
  argBuffer[4] = d.data_ptr();
  argBuffer[5] = output.data_ptr();
  argBuffer[6] = (void *)sequence_length;

  if (old_sequence_length != sequence_length) {
    printf("sequence_length %u\n", sequence_length);
    old_sequence_length = sequence_length;
  }

  for (int i = 0; i < sizeof(argBuffer)/sizeof(void *) + 1; i++) {
    // printf("argBuffer for #%d is %p\n", i, argBuffer[i]);
  }

  void *config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
    CU_LAUNCH_PARAM_END,
  };
  // printf("about to cuLaunchKernel, sequence_length is %d %p\n", sequence_length, (void *)sequence_length);
  int error = cuLaunchKernel(kernel_function,
  n_heads, 1, 1, // grid x, y, z
  32, 1, 1, // block x, y, z
  0, 0, NULL, config);
  if (error != CUDA_SUCCESS) {
    cudaError_t lastErr = cudaGetLastError();
    const char *name = cudaGetErrorName(lastErr);
    const char *string = cudaGetErrorString(lastErr);
    fprintf(stderr, "* Error with cuLaunchKernel. Error name \"%s\" string \"%s\" err is %d\n", name, string, error);
  }
  // printf("cuLaunchKernel result is %d\n", error);

  return output;
}