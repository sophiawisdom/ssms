#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor monarch_cuda_forward(
    torch::Tensor x,
    torch::Tensor w1_bfly);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor monarch_forward(
    torch::Tensor x,
    torch::Tensor w1_bfly) {
  CHECK_INPUT(x);
  CHECK_INPUT(w1_bfly);


  return {monarch_cuda_forward(x, w1_bfly)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &monarch_forward, "monarch forward (CUDA)");
}