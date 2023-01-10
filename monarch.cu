#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <builtin_types.h>
#include <mma.h>

#include <vector>

CUmodule ptx_module;
CUfunction kernel_function;
CUdevice   device;
CUcontext  context;
int major = 0, minor = 0;

using namespace nvcuda;

__global__ void monarch_impl(__nv_bfloat16 *x_ptr, __nv_bfloat16 *w1_ptr, float *out_ptr) {
  // ROOT_N = 128 (N=16384), BATCH_SIZE = 16
  int warp = threadIdx.x / 32;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> w1_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;

  // xs are the same for every warp, weights are different
  __nv_bfloat16 *per_warp_w1 = w1_ptr + warp*32;

  wmma::fill_fragment(out_frag, 0.0f); // initialize out to 0

  for (int i = 0; i < 8; i++) {
      wmma::load_matrix_sync(x_frag, x_ptr + (i * 16 * sizeof(__nv_bfloat16)), 128); // load which 16x16 chunk
      wmma::load_matrix_sync(w1_frag, per_warp_w1 + (i * 2048 * sizeof(__nv_bfloat16)), 128);
      wmma::mma_sync(out_frag, x_frag, w1_frag, out_frag);
  }

  float *per_warp_out = out_ptr + warp*32;
  wmma::store_matrix_sync(per_warp_out, out_frag, 128, wmma::mem_row_major);
}

__attribute__((constructor))
static void initialize_cuda() {
    CUresult err = cuInit(0);

    err = cuCtxCreate(&context, 0, device);

    cuDeviceGet(&device, 0);
    cuDeviceComputeCapability(&major, &minor, device);
    err = cuModuleLoad(&ptx_module, "monarch.o");
    if (err != CUDA_SUCCESS) {
        cudaError_t error = cudaGetLastError();
        const char *name = cudaGetErrorName(error);
        const char *string = cudaGetErrorString(error);
        fprintf(stderr, "* Error loading PTX module. Error name \"%s\" string \"%s\" err is %d\n", name, string, err);
        exit(-1);
    }

    err = cuModuleGetFunction(&kernel_function, ptx_module, "monarch_kernel");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get function\n");
    }

    printf("Monarch initialized!\n");
}

torch::Tensor monarch_cuda_forward(
    torch::Tensor x,
    torch::Tensor w1_bfly
) {
  auto output = torch::zeros_like(x);

  unsigned int root_n = w1_bfly.sizes()[0];

  // monarch_impl<<<256, root_n>>>((__nv_bfloat16 *)x.data_ptr(), (__nv_bfloat16 *) w1_bfly.data_ptr(), (float *) out.data_ptr());
  // return out;

  void * argBuffer[3];
  int argBufferSize = sizeof(argBuffer);
  argBuffer[0] = x.data_ptr();
  argBuffer[1] = w1_bfly.data_ptr();
  argBuffer[2] = output.data_ptr();

  // printf("argBufferSize %d\n", argBufferSize);

  for (int i = 0; i < sizeof(argBuffer)/sizeof(void *) + 1; i++) {
#ifdef DEBUG
    printf("argBuffer for #%d is %p\n", i, argBuffer[i]);
#endif
  }

  void *config[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
    CU_LAUNCH_PARAM_END,
  };
#ifdef DEBUG
  printf("about to cuLaunchKernel, sequence_length is %d\n", sequence_length);
#endif
  // printf("launching with %d grid\n", root_n);
  int error = cuLaunchKernel(kernel_function,
  // root_n, 1, 1, // grid x, y, z
  1, 1, 1,
  // 32, 8, 1, // block x, y, z
  32, 8, 1,
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
