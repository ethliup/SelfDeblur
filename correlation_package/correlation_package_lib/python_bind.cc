#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "corr_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<corr_cuda>(m, "corr_cuda")
            .def(py::init<>())
            .def("forward", &corr_cuda::forward)
            .def("backward", &corr_cuda::backward);
}



