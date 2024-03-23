#include "tensor_base.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(tensor_ext, m) {
    m.attr("test_var") = 42;
    m.def("subtract", [](int i, int j) { return i - j; }, "sub two numbers");
}