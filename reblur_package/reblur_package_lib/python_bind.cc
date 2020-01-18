#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "flow_blurrer.h"
#include "flow_forward_warp_mask.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Flow_blurrer>(m, "Flow_blurrer_layer")
            .def(py::init<at::Tensor, at::Tensor, int, int, int, int>())
            .def("forward", &Flow_blurrer::forward)
            .def("backward", &Flow_blurrer::backward);

    py::class_<Flow_forward_warp_mask>(m, "Flow_forward_warp_mask")
            .def(py::init<at::Tensor>())
            .def("get_mask", &Flow_forward_warp_mask::get_mask);
}



