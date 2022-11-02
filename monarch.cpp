#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor monarch_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    unsigned int sequence_length);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor monarch_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    unsigned int sequence_length) {
  CHECK_INPUT(u);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);


  return {monarch_cuda_forward(u, a, b, c, sequence_length)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &monarch_forward, "monarch forward (CUDA)");
}