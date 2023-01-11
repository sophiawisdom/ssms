#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor siso_cuda_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    unsigned int sequence_length);

void siso_cuda_backward(
    torch::Tensor gradients,
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor grad_a,
    torch::Tensor b,
    torch::Tensor grad_b,
    torch::Tensor c,
    torch::Tensor grad_c,
    unsigned int sequence_length);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor siso_forward(
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    unsigned int sequence_length) {
  CHECK_INPUT(u);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);


  return {siso_cuda_forward(u, a, b, c, sequence_length)};
}

void siso_backward(
    torch::Tensor gradients,
    torch::Tensor u,
    torch::Tensor a,
    torch::Tensor grad_a,
    torch::Tensor b,
    torch::Tensor grad_b,
    torch::Tensor c,
    torch::Tensor grad_c,
    unsigned int sequence_length) {
  CHECK_INPUT(gradients);
  CHECK_INPUT(u);
  CHECK_INPUT(a);
  CHECK_INPUT(grad_a);
  CHECK_INPUT(b);
  CHECK_INPUT(grad_b);
  CHECK_INPUT(c);
  CHECK_INPUT(grad_c);

  siso_cuda_backward(gradients, u, a, grad_a, b, grad_b, c, grad_c, sequence_length);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &siso_forward, "SISO forward (CUDA)");
  m.def("backward", &siso_backward, "SISO backward (CUDA)");
}