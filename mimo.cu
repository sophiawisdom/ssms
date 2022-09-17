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

template <typename scalar_t>
__global__ void mimo_cuda_forward_kernel(
    scalar_t * __restrict__ u,
    scalar_t * __restrict__ a,
    scalar_t * __restrict__ b,
    scalar_t * __restrict__ c,
    scalar_t * __restrict__ d,
    scalar_t * __restrict__ out
) {
    out[threadIdx.x + blockIdx.x * blockDim.x] = 6;
}

torch::Tensor mimo_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d
) {

  const auto num_heads = a.size(0); // {N_HEADS, STATE_SIZE}
  const auto batch_size = u.size(1);

  auto output = torch::zeros_like(u);

  void * argBuffer[6];
  int argBufferSize = 6*8; // 6 pointers
  argBuffer[0] = u.data_ptr();
  argBuffer[1] = a.data_ptr();
  argBuffer[2] = b.data_ptr();
  argBuffer[3] = c.data_ptr();
  argBuffer[4] = d.data_ptr();
  argBuffer[5] = output.data_ptr();
  for (int i = 0; i < 6; i++) {
    printf("argBuffer for #%d is %p\n", i, argBuffer[i]);
  }

  void *config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
    CU_LAUNCH_PARAM_END,
  };
  printf("about to cuLaunchKernel\n");
  int error = cuLaunchKernel(kernel_function,
  1, 1, 1, // grid x, y, z
  32, 1, 1, // block x, y, z
  0, 0, NULL, config);
  if (error != CUDA_SUCCESS) {
    cudaError_t lastErr = cudaGetLastError();
    const char *name = cudaGetErrorName(lastErr);
    const char *string = cudaGetErrorString(lastErr);
    fprintf(stderr, "* Error with cuLaunchKernel. Error name \"%s\" string \"%s\" err is %d\n", name, string, error);
  }
  printf("cuLaunchKernel result is %d\n", error);
  // mimo_cuda_forward_kernel<<<num_heads, 32>>>(u.data<float>(), a.data<float>(), b.data<float>(), c.data<float>(), d.data<float>(), output.data<float>());

  return output;
}