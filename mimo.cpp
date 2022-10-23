#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor mimo_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    unsigned int sequence_length);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mimo_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    unsigned int sequence_length) {
  CHECK_INPUT(u);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(d);


  return {mimo_cuda_forward(u, a, b, c, d, sequence_length)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mimo_forward, "Mimo forward (CUDA)");
}